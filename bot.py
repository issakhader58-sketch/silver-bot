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
OWN_CONF = 0.60          # confidence needed for the ML's own early signals
OWN_COOLDOWN_MIN = 15    # minutes between the ML's own signals
MIN_PROVEN = 200         # approved signals needed before "ML proven" can show
MARGIN = 5.0             # ML must beat UT Bot by this many % points
NOTIFY_MAX_AGE_MIN = 15  # only alert on signals newer than this
MINTICK = 0.005
WARMUP = 100
NF = 9

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
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
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
    return buy, sell, X, close


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))


# ============================================================
# STATE
# ============================================================
def new_state():
    return {
        "w": [0.0] * NF,
        "n_train": 0,
        "last_ts": None,
        "ut": {"n": 0, "w": 0},     # plain UT Bot signals
        "mlA": {"n": 0, "w": 0},    # UT Bot signals the ML approved
        "mlO": {"n": 0, "w": 0},    # the ML's own signals
        "pending": [],
        "own_pos": 0,
        "own_last_ts": 0,
    }


def load_state():
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            print("state file unreadable, starting fresh")
    return new_state()


def save_state(st):
    STATE_FILE.write_text(json.dumps(st))


# ============================================================
# CORE: go through every new bar, score, learn, find signals
# ============================================================
def process(df, st):
    buy, sell, X, close = indicators(df)
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
    H = HORIZON * 60
    events = []

    for i in range(start, n):
        t = int(tsec[i])

        # 1) judge older signals that are now old enough
        keep = []
        for p in st["pending"]:
            age = t - p["ts"]
            if age < H:
                keep.append(p)
                continue
            if age <= H * 3:  # skip ones that crossed a market gap
                mv = close[i] - p["price"]
                cst = p["price"] * COST_PCT / 100.0
                win = (mv > cst) if p["side"] == 1 else (-mv > cst)
                for k in p["score"]:
                    st[k]["n"] += 1
                    st[k]["w"] += int(win)
        st["pending"] = keep

        # 2) learn from what price did since HORIZON bars ago
        j = i - HORIZON
        if j >= 0 and t - tsec[j] <= H * 3:
            mv = close[i] - close[j]
            cst = close[j] * COST_PCT / 100.0
            if abs(mv) > cst:
                y = 1.0 if mv > 0 else 0.0
                err = y - sigmoid(float(w @ X[j]))
                w = w * (1 - DECAY) + LR * err * X[j]
                st["n_train"] += 1

        trained = st["n_train"] >= MIN_TRAIN
        p_up = float(sigmoid(float(w @ X[i])))

        # 3) UT Bot signal + does the ML approve it?
        side = 1 if buy[i] else (-1 if sell[i] else 0)
        if side != 0:
            approved = bool(trained and ((side == 1 and p_up >= CONF) or (side == -1 and p_up <= 1 - CONF)))
            score = ["ut"] + (["mlA"] if approved else [])
            st["pending"].append({"ts": t, "side": side, "price": float(close[i]), "score": score})
            events.append({"ts": t, "kind": "UT Bot", "side": side, "price": float(close[i]),
                           "approved": approved, "p_up": p_up})

        # 4) the ML's own signal (tracked only, one per move)
        if trained and t - st["own_last_ts"] >= OWN_COOLDOWN_MIN * 60:
            own = 0
            if p_up >= OWN_CONF and st["own_pos"] != 1:
                own = 1
            elif p_up <= 1 - OWN_CONF and st["own_pos"] != -1:
                own = -1
            if own != 0:
                st["own_pos"] = own
                st["own_last_ts"] = t
                st["pending"].append({"ts": t, "side": own, "price": float(close[i]), "score": ["mlO"]})
                events.append({"ts": t, "kind": "ML own", "side": own, "price": float(close[i]),
                               "approved": True, "p_up": p_up})

    st["w"] = [float(x) for x in w]
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


def accuracy_line(st):
    return f"Accuracy | UT Bot {fmt(st['ut'])} | ML-approved {fmt(st['mlA'])} | ML-own {fmt(st['mlO'])}"


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


def pct(c):
    a = acc(c)
    return "-" if a is None else f"{a:.1f}%"


def write_status(st):
    ua, ma = acc(st["ut"]), acc(st["mlA"])
    proven = (st["mlA"]["n"] >= MIN_PROVEN and ua is not None and ma is not None and ma >= ua + MARGIN)
    trained = st["n_train"] >= MIN_TRAIN
    learning_note = "" if trained else " - still learning"
    proven_note = "YES" if proven else f"not yet (needs {MIN_PROVEN} approved signals and a {MARGIN:.0f} point lead)"
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Silver bot accuracy",
        "",
        f"Updated: {stamp}",
        "",
        "| Signals | Accuracy | Scored |",
        "|---|---|---|",
        f"| UT Bot (all signals) | {pct(st['ut'])} | {st['ut']['n']} |",
        f"| UT Bot + ML approved | {pct(st['mlA'])} | {st['mlA']['n']} |",
        f"| ML own signals | {pct(st['mlO'])} | {st['mlO']['n']} |",
        "",
        f"ML training samples: {st['n_train']} (starts judging at {MIN_TRAIN}){learning_note}",
        "",
        f"ML proven better than UT Bot: {proven_note}",
        "",
        f"A win = price moves more than {COST_PCT}% your way within {HORIZON} minutes.",
    ]
    STATUS_FILE.write_text("\n".join(lines) + "\n")


def log_events(events):
    new = not LOG_FILE.exists()
    with LOG_FILE.open("a") as f:
        if new:
            f.write("time_utc,kind,side,price,ml_approved,p_up\n")
        for e in events:
            tm = datetime.fromtimestamp(e["ts"], timezone.utc).strftime("%Y-%m-%d %H:%M")
            f.write(f"{tm},{e['kind']},{'BUY' if e['side'] == 1 else 'SELL'},{e['price']:.3f},{e['approved']},{e['p_up']:.3f}\n")


def main():
    st = load_state()
    df = fetch()
    if df is None or len(df) < WARMUP + HORIZON + 20:
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
    log_events(events)
    now = datetime.now(timezone.utc).timestamp()
    for e in events:
        if now - e["ts"] > NOTIFY_MAX_AGE_MIN * 60:
            continue
        icon = "BUY" if e["side"] == 1 else "SELL"
        emoji = "\U0001F7E2" if e["side"] == 1 else "\U0001F534"
        if e["kind"] == "UT Bot":
            ml = "ML approves" if e["approved"] else "ML says skip"
            msg = f"{emoji} {icon} silver {e['price']:.3f}\nUT Bot signal | {ml} (P up {e['p_up'] * 100:.0f}%)"
        else:
            msg = f"{emoji} {icon} silver {e['price']:.3f}\nML's own early signal (P up {e['p_up'] * 100:.0f}%)"
        notify(msg + "\n" + accuracy_line(st))


if __name__ == "__main__":
    main()
