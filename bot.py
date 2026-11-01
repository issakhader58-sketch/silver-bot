"""
Silver bot: UT Bot signals + a learning machine that keeps learning between runs.
Runs on GitHub Actions every ~5 minutes. Learning is saved in state.json.

Nothing here places trades. It only sends alerts and tracks accuracy.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# SETTINGS (change these, then commit the file)
# ============================================================
SYMBOL = "SI=F"          # Yahoo Finance silver futures
KEY_VALUE = 2.5          # UT Bot sensitivity (higher = fewer signals)
ATR_PERIOD = 10          # UT Bot ATR period
HORIZON = 5              # bars (minutes) until a signal is judged
COST_PCT = 0.05          # a win needs a move bigger than this % (set to your spread + fees)
LR = 0.02                # learning rate
DECAY = 0.0005           # weight decay
MIN_TRAIN = 300          # training samples before the ML judges anything
CONF = 0.55              # ML confidence needed to approve a UT Bot signal
STRONG_ATR = 4.0         # a "strong move" = price travels this many ATRs one way...
STRONG_WINDOW = 15       # ...within this many minutes (and barely goes the other way)
EARLY_LIFT = 2.5         # ML fires an early signal when it thinks a strong move is this many times likelier than normal
EARLY_COOLDOWN_MIN = 20  # minutes between early signals
EARLY_MIN_SCORED = 100   # early signals that must be scored before they can go LIVE
EARLY_MIN_PRECISION = 50 # % of early signals that must be followed by a strong move (also needs 2x the normal rate)
CONFIRM_MIN = 3         # how many TRUSTED confirmations must agree before an early signal
CONF_MIN_SAMPLES = 200  # a confirmation must be seen this many times before it can be trusted
CONF_MIN_LIFT = 1.3     # ...and strong moves must follow it this many times more often than normal
NOTIFY_MAX_AGE_MIN = 15  # only alert on signals newer than this
MINTICK = 0.005
WARMUP = 100
NF = 9
NAMES = ["short-term momentum", "trend vs EMA20", "UT Bot trend direction", "RSI direction",
         "volume spike", "strong candles in a row", "breakout of last 10 bars", "volatility expanding"]
NC = len(NAMES)

STATE_FILE = Path("state.json")
LOG_FILE = Path("signals.csv")
STATUS_FILE = Path("status.md")


# ============================================================
# DATA
# ============================================================
def fetch():
    try:
        import yfinance as yf
        df = yf.download(SYMBOL, period="7d", interval="1m", progress=False, auto_adjust=False)
    except Exception as e:  # network or Yahoo problem
        print("fetch failed:", e)
        return None
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.columns = ["open", "high", "low", "close", "volume"]
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    # drop the bar that is still forming
    now = pd.Timestamp.now(tz="UTC")
    if len(df) and df.index[-1] + pd.Timedelta(minutes=1) > now:
        df = df.iloc[:-1]
    return df


# ============================================================
# UT BOT + FEATURES
# ============================================================
def rma(s, n):
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def clip3(s):
    return s.replace([np.inf, -np.inf], 0).fillna(0).clip(-3, 3)


def indicators(df):
    c, h, l, v, o = df["close"], df["high"], df["low"], df["volume"], df["open"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = rma(tr, ATR_PERIOD)

    close = c.to_numpy(float)
    atr_np = atr.to_numpy(float)
    n = len(close)
    stop = np.zeros(n)
    pos = np.zeros(n, dtype=int)
    for i in range(n):
        prev = stop[i - 1] if i > 0 else 0.0
        s = close[i]
        sp = close[i - 1] if i > 0 else np.nan
        nl = KEY_VALUE * atr_np[i]
        if s > prev and sp > prev:
            stop[i] = max(prev, s - nl)
        elif s < prev and sp < prev:
            stop[i] = min(prev, s + nl)
        elif s > prev:
            stop[i] = s - nl
        else:
            stop[i] = s + nl
        if i > 0 and sp < prev and s > prev:
            pos[i] = 1
        elif i > 0 and sp > prev and s < prev:
            pos[i] = -1
        else:
            pos[i] = pos[i - 1] if i > 0 else 0

    prev_stop = np.concatenate([[0.0], stop[:-1]])
    prev_close = np.concatenate([[np.nan], close[:-1]])
    with np.errstate(invalid="ignore"):
        buy = (close > stop) & (prev_close <= prev_stop)
        sell = (close < stop) & (prev_stop <= prev_close)

    # features (same idea as the Pine version)
    idx = df.index
    atrv = pd.Series(np.maximum(atr_np, MINTICK), index=idx)
    stop_s = pd.Series(stop, index=idx)
    delta = c.diff()
    up = delta.clip(lower=0)
    dn = (-delta).clip(lower=0)
    rs = rma(up, 14) / rma(dn, 14).replace(0, np.nan)
    rsi = (100 - 100 / (1 + rs)).fillna(50)
    ema20 = c.ewm(span=20, adjust=False).mean()
    avg_vol = v.rolling(20).mean()
    atr_avg = atr.rolling(50).mean()
    vol_rel = (v / avg_vol - 1).where(avg_vol > 0, 0.0)
    vol_reg = (atr / atr_avg - 1).where(atr_avg > 0, 0.0)

    feats = [
        (c - c.shift(1)) / atrv,
        (c - c.shift(5)) / atrv / 2.2,
        (c - c.shift(15)) / atrv / 3.9,
        (c - stop_s) / atrv / 2.0,
        (rsi - 50) / 25,
        vol_rel,
        vol_reg,
        (ema20 - ema20.shift(3)) / atrv,
    ]
    X = np.column_stack([clip3(f).to_numpy(float) for f in feats] + [np.ones(n)])

    # confirmations: simple yes/no conditions the bot learns to trust (or not)
    pos_s = pd.Series(pos, index=idx)
    roll_hi = h.shift(1).rolling(10).max()
    roll_lo = l.shift(1).rolling(10).min()
    body2 = (c - o) + (c.shift(1) - o.shift(1))
    vol_spike = (avg_vol > 0) & (v > 1.5 * avg_vol)
    expand = atr > 1.1 * atr.shift(5)
    cu = [
        (c - c.shift(3)) > 0.8 * atrv,
        (ema20 > ema20.shift(3)) & (c > ema20),
        pos_s == 1,
        rsi > 55,
        vol_spike,
        (c > o) & (c.shift(1) > o.shift(1)) & (body2 > atrv),
        c > roll_hi,
        expand,
    ]
    cd = [
        (c - c.shift(3)) < -0.8 * atrv,
        (ema20 < ema20.shift(3)) & (c < ema20),
        pos_s == -1,
        rsi < 45,
        vol_spike,
        (c < o) & (c.shift(1) < o.shift(1)) & (body2 < -atrv),
        c < roll_lo,
        expand,
    ]
    CU = np.column_stack([x.fillna(False).to_numpy(bool) for x in cu])
    CD = np.column_stack([x.fillna(False).to_numpy(bool) for x in cd])
    return buy, sell, X, close, atr_np, h.to_numpy(float), l.to_numpy(float), CU, CD


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))


# ============================================================
# STATE
# ============================================================
VERSION = 3


def new_state():
    return {
        "v": VERSION,
        "w": [0.0] * NF,        # direction model (approves or skips UT Bot signals)
        "wU": [0.0] * NF,       # strong-UP-move model
        "wD": [0.0] * NF,       # strong-DOWN-move model
        "n_train": 0,
        "last_ts": None,
        "ut": {"n": 0, "w": 0},     # plain UT Bot signals
        "mlA": {"n": 0, "w": 0},    # UT Bot signals the ML approved
        "early": {"n": 0, "w": 0},  # early signals followed by a strong move
        "recent": [],               # last 100 early-signal results (1 = strong move followed)
        "base": {"n": 0, "up": 0, "dn": 0},  # how often strong moves normally happen
        "conf": {"up": [{"n": 0, "h": 0} for _ in range(NC)],   # how often a strong move followed each confirmation
                 "dn": [{"n": 0, "h": 0} for _ in range(NC)]},
        "pending": [],
        "early_up_on": False,
        "early_dn_on": False,
        "early_last_ts": 0,
        "early_open": [],           # early signals waiting for UT Bot to confirm
        "lead": [],                 # how much earlier than UT Bot they were
    }


def load_state():
    if STATE_FILE.exists():
        try:
            st = json.loads(STATE_FILE.read_text())
            if st.get("v") == VERSION:
                return st
            print("old state format, starting fresh")
        except Exception:
            print("state file unreadable, starting fresh")
    return new_state()


def save_state(st):
    STATE_FILE.write_text(json.dumps(st))


def base_rates(st):
    n = max(st["base"]["n"], 1)
    return max(st["base"]["up"] / n, 0.02), max(st["base"]["dn"] / n, 0.02)


def conf_lift(st, side, c):
    """How many times more often a strong move follows this confirmation than normal."""
    d = st["conf"][side][c]
    bu, bd = base_rates(st)
    base = bu if side == "up" else bd
    return ((d["h"] + base) / (d["n"] + 1)) / base


def conf_trusted(st, side, c):
    return st["conf"][side][c]["n"] >= CONF_MIN_SAMPLES and conf_lift(st, side, c) >= CONF_MIN_LIFT


def conf_summary(st):
    out = []
    for side, label in (("up", "Bullish"), ("dn", "Bearish")):
        good = [f"{NAMES[c]} ({conf_lift(st, side, c):.1f}x)" for c in range(NC) if conf_trusted(st, side, c)]
        out.append(f"{label} confirmations it trusts: " + (", ".join(good) if good else "none yet"))
    return out


def early_stats(st):
    r = st["recent"]
    n = len(r)
    prec = (100.0 * sum(r) / n) if n else None
    bu, bd = base_rates(st)
    base_pct = 100.0 * (bu + bd) / 2
    live = bool(n >= EARLY_MIN_SCORED and prec is not None
                and prec >= EARLY_MIN_PRECISION and prec >= 2 * base_pct)
    return prec, n, base_pct, live


# ============================================================
# CORE: go through every new bar, score, learn, find signals
# ============================================================
def process(df, st):
    buy, sell, X, close, atr_np, high, low, CU, CD = indicators(df)
    n = len(df)
    tsec = np.array([int(t.timestamp()) for t in df.index])
    backfill = st["last_ts"] is None
    if backfill:
        start = WARMUP
    else:
        start = max(WARMUP, int(np.searchsorted(tsec, st["last_ts"], side="right")))
    if start >= n:
        return [], False

    w = np.array(st["w"], dtype=float)
    wU = np.array(st["wU"], dtype=float)
    wD = np.array(st["wD"], dtype=float)
    H = HORIZON * 60
    HE = STRONG_WINDOW * 60
    events = []

    for i in range(start, n):
        t = int(tsec[i])

        # 1) judge older signals that are now old enough
        keep = []
        for p in st["pending"]:
            age = t - p["ts"]
            span = H if p["kind"] == "ut" else HE
            if age < span:
                keep.append(p)
                continue
            if age > span * 3:  # crossed a market gap, skip it
                continue
            if p["kind"] == "ut":
                mv = close[i] - p["price"]
                cst = p["price"] * COST_PCT / 100.0
                win = (mv > cst) if p["side"] == 1 else (-mv > cst)
                for k in p["score"]:
                    st[k]["n"] += 1
                    st[k]["w"] += int(win)
            else:
                k0 = int(np.searchsorted(tsec, p["ts"]))
                seg_hi, seg_lo = high[k0 + 1:i + 1], low[k0 + 1:i + 1]
                if len(seg_hi) == 0:
                    continue
                upm = float(seg_hi.max()) - p["price"]
                dnm = p["price"] - float(seg_lo.min())
                thr = STRONG_ATR * p["atr"]
                strong = (upm >= thr and dnm < 0.5 * upm) if p["side"] == 1 else (dnm >= thr and upm < 0.5 * dnm)
                st["early"]["n"] += 1
                st["early"]["w"] += int(strong)
                st["recent"].append(int(strong))
                st["recent"] = st["recent"][-100:]
        st["pending"] = keep

        # 2a) learn the direction model (for approving UT Bot signals)
        j = i - HORIZON
        if j >= 0 and t - tsec[j] <= H * 3:
            mv = close[i] - close[j]
            cst = close[j] * COST_PCT / 100.0
            if abs(mv) > cst:
                y = 1.0 if mv > 0 else 0.0
                err = y - sigmoid(float(w @ X[j]))
                w = w * (1 - DECAY) + LR * err * X[j]
                st["n_train"] += 1

        # 2b) learn the strong-move models (for early signals)
        je = i - STRONG_WINDOW
        if je >= 0 and t - tsec[je] <= HE * 3:
            upm = float(high[je + 1:i + 1].max()) - close[je]
            dnm = close[je] - float(low[je + 1:i + 1].min())
            thr = STRONG_ATR * max(atr_np[je], MINTICK)
            yU = 1.0 if (upm >= thr and dnm < 0.5 * upm) else 0.0
            yD = 1.0 if (dnm >= thr and upm < 0.5 * dnm) else 0.0
            wU = wU * (1 - DECAY) + LR * (yU - sigmoid(float(wU @ X[je]))) * X[je]
            wD = wD * (1 - DECAY) + LR * (yD - sigmoid(float(wD @ X[je]))) * X[je]
            st["base"]["n"] += 1
            st["base"]["up"] += int(yU)
            st["base"]["dn"] += int(yD)
            # learn what each confirmation is worth
            for c in range(NC):
                if CU[je, c]:
                    st["conf"]["up"][c]["n"] += 1
                    st["conf"]["up"][c]["h"] += int(yU)
                if CD[je, c]:
                    st["conf"]["dn"][c]["n"] += 1
                    st["conf"]["dn"][c]["h"] += int(yD)

        trained = st["n_train"] >= MIN_TRAIN
        p_up = float(sigmoid(float(w @ X[i])))

        # 3) UT Bot signal + does the ML approve it?
        side = 1 if buy[i] else (-1 if sell[i] else 0)
        if side != 0:
            approved = bool(trained and ((side == 1 and p_up >= CONF) or (side == -1 and p_up <= 1 - CONF)))
            score = ["ut"] + (["mlA"] if approved else [])
            st["pending"].append({"kind": "ut", "ts": t, "side": side, "price": float(close[i]), "score": score})
            # did this confirm an earlier ML signal?
            st["early_open"] = [e for e in st["early_open"] if t - e["ts"] <= 30 * 60]
            confirms = None
            for e in list(st["early_open"]):
                if e["side"] == side:
                    confirms = {"min": (t - e["ts"]) / 60.0, "edge": (float(close[i]) - e["price"]) * side}
                    st["lead"].append(confirms)
                    st["lead"] = st["lead"][-100:]
                    st["early_open"].remove(e)
                    break
            events.append({"ts": t, "kind": "UT Bot", "side": side, "price": float(close[i]),
                           "approved": approved, "p_up": p_up, "confirms": confirms})

        # 4) early signal: ML thinks a strong move is starting AND trusted confirmations agree
        if st["base"]["n"] >= MIN_TRAIN:
            bu, bd = base_rates(st)
            pU = float(sigmoid(float(wU @ X[i])))
            pD = float(sigmoid(float(wD @ X[i])))
            actU = [c for c in range(NC) if CU[i, c] and conf_trusted(st, "up", c)]
            actD = [c for c in range(NC) if CD[i, c] and conf_trusted(st, "dn", c)]
            condU = len(actU) >= CONFIRM_MIN and pU >= EARLY_LIFT * bu and pD < bd
            condD = len(actD) >= CONFIRM_MIN and pD >= EARLY_LIFT * bd and pU < bu
            ok_time = t - st["early_last_ts"] >= EARLY_COOLDOWN_MIN * 60
            fire = 0
            if condU and not st["early_up_on"] and ok_time:
                fire = 1
            elif condD and not st["early_dn_on"] and ok_time:
                fire = -1
            st["early_up_on"], st["early_dn_on"] = bool(condU), bool(condD)
            if fire != 0:
                st["early_last_ts"] = t
                atr_e = float(max(atr_np[i], MINTICK))
                st["pending"].append({"kind": "early", "ts": t, "side": fire, "price": float(close[i]), "atr": atr_e})
                st["early_open"].append({"ts": t, "side": fire, "price": float(close[i])})
                events.append({"ts": t, "kind": "Early", "side": fire, "price": float(close[i]),
                               "approved": False, "p_up": pU if fire == 1 else pD, "confirms": None,
                               "why": [NAMES[c] for c in (actU if fire == 1 else actD)]})

    st["w"] = [float(x) for x in w]
    st["wU"] = [float(x) for x in wU]
    st["wD"] = [float(x) for x in wD]
    st["last_ts"] = int(tsec[n - 1])
    return events, backfill


# ============================================================
# OUTPUT
# ============================================================
def acc(c):
    return None if c["n"] == 0 else 100.0 * c["w"] / c["n"]


def fmt(c):
    a = acc(c)
    return "-" if a is None else f"{a:.1f}% (n={c['n']})"


def pct(c):
    a = acc(c)
    return "-" if a is None else f"{a:.1f}%"


def accuracy_line(st):
    prec, n, base_pct, live = early_stats(st)
    ptxt = "-" if prec is None else f"{prec:.0f}% (n={n})"
    return f"Accuracy | UT Bot {fmt(st['ut'])} | ML-approved {fmt(st['mlA'])} | Early hit rate {ptxt}"


def notify(text):
    import requests
    tok = os.environ.get("TELEGRAM_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    print(text)
    if not tok or not chat:
        return
    try:
        requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                      data={"chat_id": chat, "text": text}, timeout=15)
    except Exception as e:
        print("telegram failed:", e)


def write_status(st):
    prec, n, base_pct, live = early_stats(st)
    trained = st["n_train"] >= MIN_TRAIN
    learning_note = "" if trained else " - still learning"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if live:
        early_note = "LIVE (alerts are on)"
    else:
        early_note = (f"PRACTICE (tracked only, no alerts). Goes live after {EARLY_MIN_SCORED} scored early signals "
                      f"with a {EARLY_MIN_PRECISION}%+ hit rate that is at least 2x the normal rate.")
    lead = st["lead"]
    if lead:
        avg_min = sum(x["min"] for x in lead) / len(lead)
        avg_edge = sum(x["edge"] for x in lead) / len(lead)
        lead_note = (f"Early signals that UT Bot later confirmed: {len(lead)}. "
                     f"On average {avg_min:.1f} minutes earlier, with a price edge of {avg_edge:+.3f} per ounce.")
    else:
        lead_note = "Early signals that UT Bot later confirmed: none yet."
    ptxt = "-" if prec is None else f"{prec:.1f}%"
    lines = [
        "# Silver bot accuracy",
        "",
        f"Updated: {stamp}",
        "",
        "| Signals | Accuracy | Scored |",
        "|---|---|---|",
        f"| UT Bot (all signals) | {pct(st['ut'])} | {st['ut']['n']} |",
        f"| UT Bot + ML approved | {pct(st['mlA'])} | {st['mlA']['n']} |",
        f"| ML early signals (last 100) | {ptxt} | {n} |",
        "",
        f"ML training samples: {st['n_train']} (starts judging at {MIN_TRAIN}){learning_note}",
        "",
        f"Early signals: {early_note}",
        "",
        lead_note,
        "",
        *conf_summary(st),
        "",
        f"(A confirmation is trusted after {CONF_MIN_SAMPLES} sightings, once strong moves follow it {CONF_MIN_LIFT}x more often than normal. "
        f"An early signal needs {CONFIRM_MIN} trusted confirmations to agree.)",
        "",
        f"A strong move = price travels {STRONG_ATR:g} ATRs one way within {STRONG_WINDOW} minutes and barely goes the other way. "
        f"This normally happens about {base_pct:.0f}% of the time.",
        "",
        f"A UT Bot win = price moves more than {COST_PCT}% your way within {HORIZON} minutes.",
    ]
    STATUS_FILE.write_text("\n".join(lines) + "\n")


def log_events(events, live):
    new = not LOG_FILE.exists()
    with LOG_FILE.open("a") as f:
        if new:
            f.write("time_utc,kind,side,price,flag,p\n")
        for e in events:
            tm = datetime.fromtimestamp(e["ts"], timezone.utc).strftime("%Y-%m-%d %H:%M")
            flag = e["approved"] if e["kind"] == "UT Bot" else ("live" if live else "practice")
            f.write(f"{tm},{e['kind']},{'BUY' if e['side'] == 1 else 'SELL'},{e['price']:.3f},{flag},{e['p_up']:.3f}\n")


def main():
    st = load_state()
    df = fetch()
    if df is None or len(df) < WARMUP + STRONG_WINDOW + 20:
        print("not enough data right now (market closed or Yahoo problem)")
        return
    before = st["last_ts"]
    events, backfill = process(df, st)
    if st["last_ts"] == before:
        print("no new bars")
        return
    save_state(st)
    write_status(st)
    if backfill:
        print("first run: learned from history, no alerts sent")
        return
    live = early_stats(st)[3]
    log_events(events, live)
    now = datetime.now(timezone.utc).timestamp()
    for e in events:
        if now - e["ts"] > NOTIFY_MAX_AGE_MIN * 60:
            continue
        word = "BUY" if e["side"] == 1 else "SELL"
        emoji = "\U0001F7E2" if e["side"] == 1 else "\U0001F534"
        if e["kind"] == "UT Bot":
            ml = "ML approves" if e["approved"] else "ML says skip"
            msg = f"{emoji} {word} silver {e['price']:.3f}\nUT Bot signal | {ml} (P up {e['p_up'] * 100:.0f}%)"
            if e["confirms"]:
                c = e["confirms"]
                msg += f"\nConfirms the earlier ML signal ({c['min']:.0f} min earlier, price edge {c['edge']:+.3f})"
        else:
            if not live:
                continue  # practice mode: tracked only
            msg = (f"{emoji} EARLY {word} silver {e['price']:.3f}\nML thinks a strong move is starting (before UT Bot)"
                   f"\nConfirmed by: {', '.join(e['why'])}")
        notify(msg + "\n" + accuracy_line(st))


if __name__ == "__main__":
    main()





