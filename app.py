
import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timezone

st.set_page_config(page_title="Crypto AI Pro", page_icon="📈", layout="wide")
st.title("📈 Crypto AI Pro — Futures Scanner")
st.caption("تحليل فني متعدد المنصات والفريمات | إشارات تعليمية وليست توصيات مالية")

EXCHANGES = {
    "Binance": ccxt.binance,
    "Bybit": ccxt.bybit,
    "OKX": ccxt.okx,
    "Bitget": ccxt.bitget,
    "KuCoin Futures": ccxt.kucoinfutures,
    "Gate.io": ccxt.gateio,
}
TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d", "1w"]

def make_exchange(name):
    cls = EXCHANGES[name]
    ex = cls({"enableRateLimit": True, "options": {"defaultType": "swap"}})
    return ex

def resolve_symbol(ex, base):
    markets = ex.load_markets()
    candidates = [
        s for s, m in markets.items()
        if m.get("swap") and m.get("quote") == "USDT"
        and (m.get("base") == base or s.startswith(base + "/"))
    ]
    # Prefer linear USDT perpetual contract
    for s in candidates:
        m = markets[s]
        if m.get("linear") and m.get("settle") == "USDT":
            return s
    if candidates:
        return candidates[0]
    raise ValueError(f"لم يتم العثور على عقد USDT دائم لـ {base} على هذه المنصة.")

def candles(ex, symbol, timeframe, limit=250):
    rows = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    if not rows:
        raise ValueError("لم تصل بيانات الشموع.")
    df = pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def analyze(df):
    c, h, l = df.close, df.high, df.low
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()
    rr = rsi(c)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    mid = c.rolling(20).mean()
    std = c.rolling(20).std()
    atr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1).rolling(14).mean()

    i = len(df)-1
    price = float(c.iloc[i])
    a = float(atr.iloc[i]) if pd.notna(atr.iloc[i]) else price*0.01
    rv = float(rr.iloc[i]) if pd.notna(rr.iloc[i]) else 50
    score_long = 0
    score_short = 0
    score_long += 20 if price > ema20.iloc[i] else 0
    score_short += 20 if price < ema20.iloc[i] else 0
    score_long += 15 if ema20.iloc[i] > ema50.iloc[i] else 0
    score_short += 15 if ema20.iloc[i] < ema50.iloc[i] else 0
    score_long += 15 if ema50.iloc[i] > ema200.iloc[i] else 0
    score_short += 15 if ema50.iloc[i] < ema200.iloc[i] else 0
    score_long += 15 if macd.iloc[i] > signal.iloc[i] else 0
    score_short += 15 if macd.iloc[i] < signal.iloc[i] else 0
    score_long += 10 if 50 < rv < 70 else 0
    score_short += 10 if 30 < rv < 50 else 0
    vol_avg = df.volume.rolling(20).mean().iloc[i]
    score_long += 10 if df.volume.iloc[i] > vol_avg and price > c.iloc[i-1] else 0
    score_short += 10 if df.volume.iloc[i] > vol_avg and price < c.iloc[i-1] else 0
    score_long += 15 if price > mid.iloc[i] else 0
    score_short += 15 if price < mid.iloc[i] else 0

    side = "LONG" if score_long >= score_short else "SHORT"
    score = max(score_long, score_short)
    # ATR-based illustrative risk levels
    if side == "LONG":
        sl, t1, t2, t3 = price-1.5*a, price+1.5*a, price+2.5*a, price+3.5*a
    else:
        sl, t1, t2, t3 = price+1.5*a, price-1.5*a, price-2.5*a, price-3.5*a

    if score < 45:
        label = "انتظار — لا توجد إشارة قوية"
    elif score < 65:
        label = "إشارة متوسطة"
    else:
        label = "إشارة قوية نسبياً"
    return {
        "price": price, "rsi": rv, "ema20": float(ema20.iloc[i]),
        "ema50": float(ema50.iloc[i]), "ema200": float(ema200.iloc[i]),
        "macd": "إيجابي" if macd.iloc[i] > signal.iloc[i] else "سلبي",
        "side": side, "score": score, "label": label,
        "sl": sl, "tp1": t1, "tp2": t2, "tp3": t3,
        "atr": a, "updated": df.time.iloc[i]
    }

with st.sidebar:
    st.header("إعدادات التحليل")
    exchange_name = st.selectbox("منصة العقود الآجلة", list(EXCHANGES.keys()))
    base = st.text_input("رمز العملة", "BTC").upper().strip().replace("/USDT","")
    timeframe = st.selectbox("الفريم الأساسي", TIMEFRAMES, index=3)
    multi = st.checkbox("تحليل جميع الفريمات", value=True)
    st.info("تُستخدم بيانات السوق العامة فقط. لا تدخل مفاتيح API أو بيانات محفظتك.")

if st.button("🔍 ابدأ التحليل", type="primary"):
    try:
        with st.spinner("جارٍ الاتصال بالمنصة وجلب بيانات العقود..."):
            ex = make_exchange(exchange_name)
            symbol = resolve_symbol(ex, base)
            st.subheader(f"{exchange_name}  ·  {symbol}")
            frames = TIMEFRAMES if multi else [timeframe]
            results = []
            for tf in frames:
                try:
                    df = candles(ex, symbol, tf)
                    res = analyze(df)
                    res["timeframe"] = tf
                    results.append(res)
                except Exception as e:
                    st.warning(f"تعذر تحليل {tf}: {e}")
            if not results:
                st.error("لم نتمكن من جلب بيانات قابلة للتحليل.")
            else:
                primary = next((x for x in results if x["timeframe"] == timeframe), results[0])
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("السعر", f'{primary["price"]:,.6g}')
                c2.metric("الاتجاه", primary["side"])
                c3.metric("قوة الإشارة", f'{primary["score"]}/100')
                c4.metric("RSI", f'{primary["rsi"]:.1f}')
                st.subheader("خطة الصفقة — تقديرية")
                st.success(f'{primary["label"]} | {primary["side"]}')
                table = pd.DataFrame([
                    ["الدخول المرجعي", primary["price"]],
                    ["وقف الخسارة", primary["sl"]],
                    ["الهدف الأول TP1", primary["tp1"]],
                    ["الهدف الثاني TP2", primary["tp2"]],
                    ["الهدف الثالث TP3", primary["tp3"]],
                    ["ATR(14)", primary["atr"]],
                ], columns=["البند","السعر"])
                st.dataframe(table, use_container_width=True, hide_index=True)
                st.subheader("توافق الفريمات")
                view = pd.DataFrame([{
                    "الفريم": r["timeframe"], "الاتجاه": r["side"],
                    "القوة /100": r["score"], "RSI": round(r["rsi"],1),
                    "MACD": r["macd"], "التقييم": r["label"]
                } for r in results])
                st.dataframe(view, use_container_width=True, hide_index=True)
                st.subheader("البيانات الفنية")
                st.write({
                    "EMA 20": primary["ema20"], "EMA 50": primary["ema50"],
                    "EMA 200": primary["ema200"], "MACD": primary["macd"],
                    "آخر شمعة": str(primary["updated"])
                })
                st.caption("ملاحظة: درجة القوة هي نتيجة قواعد فنية تجريبية وليست احتمال نجاح. لا تستخدمها وحدها لاتخاذ قرار.")
    except Exception as e:
        st.error(f"حدث خطأ: {e}")
        st.info("قد لا تدعم المنصة هذا العقد أو قد تكون واجهة API غير متاحة مؤقتاً.")

st.divider()
st.warning("مخاطر العقود الآجلة مرتفعة وقد تؤدي إلى خسارة كامل الهامش أو أكثر بحسب المنصة والرافعة. الأداة لا تنفذ صفقات، ولا تضمن الربح. اختبر الإشارات تاريخياً وعلى حساب تجريبي، وتحقق من الرسوم والانزلاق السعري قبل التداول.")
st.caption("Crypto AI Pro • Prototype v1 • بيانات عامة عبر CCXT")
