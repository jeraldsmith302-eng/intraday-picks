"""
India Intraday Stock Agent (NSE)
Scans liquid NSE stocks every morning, scores them on trend, momentum,
volume and breakout setup, and prints the top BUY candidates with
entry / stop-loss / target levels.

Install:  pip install yfinance pandas numpy requests
Run:      python india_intraday_agent.py
Schedule: run daily at ~9:00 AM IST (before market opens at 9:15)

Educational tool only. Not SEBI-registered investment advice.
"""

import datetime as dt
import os
import requests
import numpy as np
import pandas as pd
import yfinance as yf

# Telegram setup: set these as environment variables (or paste them here)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


def send_telegram(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram not configured; skipping alert.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=15)
        if not r.ok:
            print("Telegram error:", r.text)
    except Exception as e:
        print("Telegram failed:", e)

def write_html(picks, today, path="index.html"):
    """Write a static, mobile-friendly web page with today's picks."""
    from html import escape
    cards = ""
    for _, p in picks.iterrows():
        risk = p["entry"] - p["stop_loss"]
        rr = (p["target"] - p["entry"]) / risk if risk > 0 else 0
        cards += f"""<div class="c"><div class="h"><b>{escape(p['stock'])}</b><span>Score {p['score']}</span></div>
<div class="g"><div><small>Buy above</small>{p['entry']}</div><div><small>Stop loss</small><i class="d">{p['stop_loss']}</i></div>
<div><small>Target</small><i class="u">{p['target']}</i></div><div><small>R:R</small>{rr:.1f}</div></div>
<p>{escape(p['reasons'])}</p></div>"""
    if not cards:
        cards = '<p class="e">No high-conviction setups today. Better to stay out.</p>'
    page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Intraday Picks {today}</title>
<style>:root{{--bg:#f6f7f9;--cd:#fff;--tx:#14181f;--mu:#667085;--bd:#e3e6eb;--up:#0a8f4e;--dn:#d92d20;color-scheme:light dark}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0e1116;--cd:#171b22;--tx:#eef0f4;--mu:#9aa3b2;--bd:#2a303a;--up:#35c47c;--dn:#ff6b5e}}}}
body{{margin:0;background:var(--bg);color:var(--tx);font:15px/1.5 system-ui,sans-serif}}main{{max-width:720px;margin:auto;padding:20px 16px}}
h1{{font-size:22px;margin:0}}.s,.n{{color:var(--mu);font-size:13px}}.c{{background:var(--cd);border:1px solid var(--bd);border-radius:14px;padding:14px;margin:12px 0}}
.h{{display:flex;justify-content:space-between;font-size:18px}}.h span{{font-size:12px;color:var(--mu)}}
.g{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;text-align:center;margin-top:8px}}.g div{{background:var(--bg);border-radius:10px;padding:8px 2px}}
small{{display:block;color:var(--mu);font-size:11px}}i{{font-style:normal}}.u{{color:var(--up)}}.d{{color:var(--dn)}}p{{color:var(--mu);font-size:13px;margin:8px 0 0}}.e{{text-align:center}}</style></head>
<body><main><h1>Intraday Picks</h1><div class="s">Scan date: {today} &middot; updated automatically every market day</div>
{cards}<div class="n">Educational only, not SEBI-registered investment advice. Intraday trading carries a high risk of loss.
Enter only after the entry level breaks (after 9:30 AM), use the stop-loss, and exit by 3:15 PM. Risk max 1% of capital per trade.</div></main></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)


UNIVERSE = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "SBIN", "BHARTIARTL",
    "ITC", "LT", "AXISBANK", "KOTAKBANK", "HINDUNILVR", "BAJFINANCE",
    "MARUTI", "SUNPHARMA", "TATASTEEL", "NTPC", "POWERGRID",
    "ONGC", "ADANIENT", "ADANIPORTS", "COALINDIA", "JSWSTEEL", "HCLTECH",
    "WIPRO", "TECHM", "M&M", "TITAN", "ULTRACEMCO", "BAJAJFINSV", "INDUSINDBK",
    "HINDALCO", "GRASIM", "CIPLA", "DRREDDY", "EICHERMOT", "BPCL", "TRENT",
    "BEL", "HAL", "VEDL", "PNB", "BANKBARODA", "IOC", "GAIL", "DLF", "ETERNAL", "ASIANPAINT",
]

TOP_N = 5
MIN_SCORE = 5


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down.replace(0, np.nan))


def analyse(df):
    df = df.dropna()
    if len(df) < 60:
        return None
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]

    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    r = rsi(c)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    vol_ratio = v.iloc[-1] / v.iloc[-21:-1].mean()
    high20 = h.iloc[-21:-1].max()

    last, prev_high = c.iloc[-1], h.iloc[-1]
    atr_pct = atr.iloc[-1] / last * 100

    score, reasons = 0, []
    if last > ema20.iloc[-1] > ema50.iloc[-1]:
        score += 2; reasons.append("uptrend (price > EMA20 > EMA50)")
    if 50 < r.iloc[-1] < 70:
        score += 1; reasons.append(f"RSI {r.iloc[-1]:.0f} healthy")
    elif r.iloc[-1] >= 75:
        score -= 2; reasons.append(f"RSI {r.iloc[-1]:.0f} overbought")
    if vol_ratio > 1.5:
        score += 2; reasons.append(f"volume {vol_ratio:.1f}x avg")
    elif vol_ratio > 1.2:
        score += 1; reasons.append(f"volume {vol_ratio:.1f}x avg")
    if hist.iloc[-1] > 0 and hist.iloc[-1] > hist.iloc[-2]:
        score += 1; reasons.append("MACD momentum rising")
    if last >= 0.98 * high20:
        score += 2; reasons.append("near 20-day breakout")
    if 1.2 <= atr_pct <= 3.5:
        score += 1; reasons.append(f"good intraday range (ATR {atr_pct:.1f}%)")

    # Intraday plan: buy on break above yesterday's high, 1:2 risk-reward
    entry = round(prev_high * 1.001, 2)
    sl = round(entry - 0.6 * atr.iloc[-1], 2)
    target = round(entry + 1.2 * atr.iloc[-1], 2)

    return dict(score=score, close=round(last, 2), entry=entry, stop_loss=sl,
                target=target, reasons="; ".join(reasons))


def run():
    tickers = [f"{s}.NS" for s in UNIVERSE]
    data = yf.download(tickers, period="8mo", interval="1d", group_by="ticker",
                       auto_adjust=True, progress=False, threads=True)
    rows = []
    for s in UNIVERSE:
        try:
            res = analyse(data[f"{s}.NS"])
            if res:
                rows.append({"stock": s, **res})
        except Exception:
            continue

    out = pd.DataFrame(rows).sort_values("score", ascending=False)
    picks = out[out["score"] >= MIN_SCORE].head(TOP_N)

    today = dt.date.today().isoformat()
    print(f"\n=== Intraday BUY candidates for {today} ===\n")
    if picks.empty:
        print("No high-conviction setups today. Better to stay out.")
    for _, p in picks.iterrows():
        print(f"{p['stock']}  (score {p['score']})")
        print(f"  Buy above : {p['entry']}")
        print(f"  Stop loss : {p['stop_loss']}")
        print(f"  Target    : {p['target']}")
        print(f"  Why       : {p['reasons']}\n")
    out.to_csv(f"picks_{today}.csv", index=False)
    write_html(picks, today)

    msg = [f"📈 Intraday BUY candidates - {today}\n"]
    if picks.empty:
        msg.append("No high-conviction setups today. Better to stay out.")
    for _, p in picks.iterrows():
        msg.append(f"{p['stock']} (score {p['score']})\n"
                   f"Buy above: {p['entry']}\n"
                   f"SL: {p['stop_loss']} | Target: {p['target']}\n"
                   f"Why: {p['reasons']}\n")
    msg.append("Enter only after 9:30 AM trigger. Exit by 3:15 PM. "
               "Risk max 1% per trade. Not investment advice.")
    send_telegram("\n".join(msg))
    print("Rules: enter only if price crosses the entry level after 9:30 AM, "
          "risk max 1% of capital per trade, exit all positions by 3:15 PM.")
    print("Not investment advice. Intraday trading carries high risk of loss.")


if __name__ == "__main__":
    run()
