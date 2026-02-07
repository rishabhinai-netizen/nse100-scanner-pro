"""
NSE SCANNER PRO v2.0 — World-Class Trading Platform
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import pytz
import json
import os

from stock_universe import get_stock_universe, get_sector, NIFTY_50
from data_engine import (
    fetch_batch_daily, fetch_nifty_data,
    Indicators, BreezeEngine, now_ist, IST
)
from scanners import (
    STRATEGY_PROFILES, ALL_SCANNERS, DAILY_SCANNERS, INTRADAY_PROXY_SCANNERS,
    run_scanner, run_all_scanners, check_market_health, ScanResult
)
from risk_manager import RiskManager
from enhancements import (
    plot_candlestick, compute_sector_performance, plot_sector_heatmap,
    compute_rs_rankings, plot_rs_scatter,
    load_journal, save_journal, add_journal_entry, compute_journal_analytics, plot_equity_curve,
    check_weekly_alignment, compute_market_breadth, plot_breadth_gauge,
    format_fii_dii_summary
)

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="NSE Scanner Pro", page_icon="🎯", layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get Help": None, "Report a bug": None, "About": None}
)

# ============================================================================
# CSS
# ============================================================================
st.markdown("""
<style>
    #MainMenu, footer, header, .stDeployButton,
    [data-testid="stToolbar"], [data-testid="stStatusWidget"],
    .viewerBadge_container__r5tak, .styles_viewerBadge__CvC9N,
    ._profileContainer_51w34_53, ._profilePreview_51w34_63,
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_link__qRIco,
    .styles_viewerBadge__1yB5_, .viewerBadge_text__1JaDK,
    #stDecoration { display: none !important; visibility: hidden !important; }
    
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1d23, #252830);
        border: 1px solid #333; border-radius: 8px; padding: 8px 12px; overflow: hidden;
    }
    div[data-testid="stMetric"] label { font-size: 0.75rem !important; color: #888 !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.1rem !important; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .pc { background: linear-gradient(135deg,#1a1d23,#252830); border: 1px solid #333;
          border-radius: 8px; padding: 10px 14px; margin: 4px 0; }
    .pc .lb { font-size: 0.72rem; color: #888; margin-bottom: 2px; }
    .pc .vl { font-size: 1rem; font-weight: 600; color: #fafafa; }
    .pc .vl.g { color: #00d26a; } .pc .vl.r { color: #ff4757; } .pc .vl.o { color: #FF6B35; }
    .sc { background: #1a1d23; border: 1px solid #333; border-radius: 10px; padding: 14px; margin: 6px 0; }
    .sc:hover { border-color: #FF6B35; }
    .bg { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.7em; }
    .bg-s { background: #1e3a5f; color: #5dade2; } .bg-i { background: #3e2723; color: #ff8a65; }
    .bg-p { background: #1b5e20; color: #81c784; } .bg-o { background: #4a148c; color: #ce93d8; }
    .mg { color: #00d26a !important; } .mr { color: #ff4757 !important; } .mo { color: #FF6B35 !important; }
    .ws { border-left: 3px solid #FF6B35; padding: 8px 14px; margin: 6px 0;
          background: #1a1d23; border-radius: 0 8px 8px 0; }
    .bb { padding: 8px 14px; border-radius: 8px; margin: 8px 0; font-size: 0.85rem; }
    .bb-on { background: #0d3320; border: 1px solid #1b5e20; color: #81c784; }
    .bb-off { background: #1a1d23; border: 1px solid #333; color: #888; }
    .dataframe { font-size: 0.8rem !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE
# ============================================================================
for k, v in {
    "watchlist": [], "scan_results": {}, "market_health": None,
    "data_loaded": False, "stock_data": {}, "enriched_data": {},
    "nifty_data": None, "capital": 500000,
    "breeze_connected": False, "breeze_engine": None, "breeze_msg": "",
    "workflow_checks": {}, "universe_size": "nifty200",
    "telegram_token": "", "telegram_chat_id": "",
    "journal": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.journal is None:
    st.session_state.journal = load_journal()

# ============================================================================
# BREEZE AUTO-CONNECT
# ============================================================================
def try_breeze():
    if st.session_state.breeze_connected:
        return
    try:
        ak = st.secrets.get("BREEZE_API_KEY", "")
        asc = st.secrets.get("BREEZE_API_SECRET", "")
        st_ = st.secrets.get("BREEZE_SESSION_TOKEN", "")
        if ak and asc and st_ and ak != "your_api_key_here":
            e = BreezeEngine()
            ok, msg = e.connect(ak, asc, st_)
            st.session_state.breeze_connected = ok
            st.session_state.breeze_msg = msg
            if ok: st.session_state.breeze_engine = e
    except Exception as ex:
        st.session_state.breeze_msg = f"Breeze: {str(ex)[:80]}"
try_breeze()

# ============================================================================
# HELPERS
# ============================================================================
def pc(label, value, css=""):
    if isinstance(value, (int, float)):
        v = f"₹{value:,.0f}" if value >= 10000 else (f"₹{value:,.1f}" if value >= 100 else f"₹{value:,.2f}")
    else: v = str(value)
    st.markdown(f'<div class="pc"><div class="lb">{label}</div><div class="vl {css}">{v}</div></div>', unsafe_allow_html=True)

def fp(v):
    if v >= 10000: return f"₹{v:,.0f}"
    elif v >= 100: return f"₹{v:,.1f}"
    else: return f"₹{v:,.2f}"

def send_tg(msg):
    tk = st.session_state.telegram_token or st.secrets.get("TELEGRAM_BOT_TOKEN", "")
    ci = st.session_state.telegram_chat_id or st.secrets.get("TELEGRAM_CHAT_ID", "")
    if not tk or not ci: return False
    try:
        import requests
        return requests.post(f"https://api.telegram.org/bot{tk}/sendMessage",
                             json={"chat_id": ci, "text": msg, "parse_mode": "HTML"}, timeout=10).status_code == 200
    except: return False

def fmt_alert(r):
    return (f"🎯 <b>{r.strategy}</b> — {r.signal}\n📈 <b>{r.symbol}</b> ({r.sector})\n"
            f"💰 CMP: {fp(r.cmp)} | Entry: {fp(r.entry)}\n🛑 SL: {fp(r.stop_loss)} | T1: {fp(r.target_1)}\n"
            f"📊 Conf: {r.confidence}% | R:R 1:{r.risk_reward:.1f}")

def results_df(results):
    return pd.DataFrame([{
        "Symbol": r.symbol, "Signal": r.signal, "CMP": fp(r.cmp), "Entry": fp(r.entry),
        "Type": r.entry_type, "SL": fp(r.stop_loss), "T1": fp(r.target_1), "T2": fp(r.target_2),
        "R:R": f"1:{r.risk_reward:.1f}", "Risk%": f"{r.risk_pct:.1f}%",
        "Conf%": f"{r.confidence}%", "RS": int(r.rs_rating), "RSI": round(r.rsi,1),
        "Sector": r.sector, "Hold": r.hold_type,
    } for r in results])

def load_data():
    syms = get_stock_universe(st.session_state.universe_size)
    pb = st.progress(0, "Starting...")
    def cb(p, t): pb.progress(min(p, 0.95), t)
    data = fetch_batch_daily(syms, "1y", cb)
    pb.progress(0.96, "Fetching Nifty 50 index...")
    nifty = fetch_nifty_data()
    # Enrich all data
    pb.progress(0.97, "Computing indicators...")
    enriched = {}
    for s, df in data.items():
        try:
            enriched[s] = Indicators.enrich_dataframe(df)
        except:
            enriched[s] = df
    pb.progress(1.0, f"✅ {len(data)} stocks loaded & enriched!")
    return data, nifty, enriched

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("## 🎯 NSE Scanner Pro")
    st.caption("v2.0 — World-Class Edition")
    st.markdown("---")
    page = st.radio("", [
        "📊 Dashboard", "🔍 Scanner Hub", "📈 Charts & RS",
        "📐 Trade Planner", "⭐ Watchlist", "📓 Trade Journal",
        "📋 Daily Workflow", "🔔 Alerts", "⚙️ Settings"
    ], label_visibility="collapsed")
    st.markdown("---")
    ist = now_ist()
    is_mkt = time(9,15) <= ist.time() <= time(15,30) and ist.weekday() < 5
    st.caption(f"{'🟢' if is_mkt else '🔴'} {ist.strftime('%d %b, %I:%M %p IST')}")
    if st.session_state.market_health:
        mh = st.session_state.market_health
        nv = mh.get("nifty_close", 0)
        st.markdown(f"**{mh['regime']}** | Nifty ₹{nv:,.0f}" if isinstance(nv, (int,float)) else f"**{mh['regime']}**")
    sigs = sum(len(v) for v in st.session_state.scan_results.values())
    st.caption(f"Signals: {sigs} | Watch: {len(st.session_state.watchlist)} | Journal: {len(st.session_state.journal)}")
    st.markdown("---")
    if st.session_state.breeze_connected:
        st.markdown('<div class="bb bb-on">✅ Breeze Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="bb bb-off">⚪ Breeze Off</div>', unsafe_allow_html=True)


# ============================================================================
# DASHBOARD
# ============================================================================
def page_dashboard():
    st.markdown("# 📊 Market Dashboard")
    ist = now_ist()
    is_mkt = time(9,15) <= ist.time() <= time(15,30) and ist.weekday() < 5
    st.markdown(f"**{'🟢 MARKET OPEN' if is_mkt else '🔴 MARKET CLOSED'}** — {ist.strftime('%d %b %Y, %I:%M %p IST')}")
    
    if st.session_state.breeze_connected:
        st.success("✅ **Breeze API Connected** — Live intraday for ORB, VWAP Reclaim, Lunch Low")
    elif st.session_state.breeze_msg:
        st.warning(f"Breeze: {st.session_state.breeze_msg}")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.session_state.universe_size = st.selectbox("Universe", ["nifty50","nifty200","nifty500"], index=1,
            format_func=lambda x: {"nifty50":"Nifty 50","nifty200":"Nifty 200","nifty500":"Nifty 500"}[x])
    with c2:
        if st.button("🔄 Load / Refresh Data", type="primary", use_container_width=True):
            st.cache_data.clear()
            data, nifty, enriched = load_data()
            st.session_state.stock_data = data
            st.session_state.nifty_data = nifty
            st.session_state.enriched_data = enriched
            st.session_state.data_loaded = True
            if nifty is not None:
                st.session_state.market_health = check_market_health(nifty)
            st.rerun()
    
    if not st.session_state.data_loaded:
        st.info("👆 Click **Load / Refresh Data** to start scanning.")
        cols = st.columns(4)
        for i, (k, p) in enumerate(STRATEGY_PROFILES.items()):
            with cols[i%4]:
                bt = " 🔌" if p.get("requires_intraday") else ""
                st.markdown(f'<div class="sc"><strong>{p["icon"]} {p["name"]}</strong>{bt}<br>'
                    f'<span class="mg">Win {p["win_rate"]}%</span> · <span class="mo">+{p["expectancy"]}%</span></div>',
                    unsafe_allow_html=True)
        return
    
    # === Market Health ===
    mh = st.session_state.market_health
    if mh:
        st.markdown("### 🏥 Market Health")
        c1,c2,c3,c4 = st.columns(4)
        with c1: pc("Regime", mh["regime"])
        with c2: pc("Health Score", f"{mh['score']}/{mh['max_score']}")
        nv = mh.get("nifty_close", 0)
        with c3: pc("Nifty 50", f"₹{nv:,.0f}" if isinstance(nv,(int,float)) else str(nv))
        with c4: pc("Position Size", f"{mh['position_multiplier']*100:.0f}%")
        with st.expander("Details"):
            for d in mh.get("details",[]): st.markdown(f"  {d}")
    
    # === Market Breadth ===
    st.markdown("### 📊 Market Breadth")
    breadth = compute_market_breadth(st.session_state.enriched_data or st.session_state.stock_data)
    if breadth:
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        with c1: pc("Advancing", str(breadth["advancing"]), "g")
        with c2: pc("Declining", str(breadth["declining"]), "r")
        with c3: pc("A/D Ratio", str(breadth["ad_ratio"]))
        with c4: pc("> 200 SMA", f"{breadth['above_200sma_pct']}%")
        with c5: pc("52W High", str(breadth["new_52w_high"]), "g")
        with c6: pc("52W Low", str(breadth["new_52w_low"]), "r")
        
        gauge = plot_breadth_gauge(breadth)
        if gauge: st.plotly_chart(gauge, use_container_width=True)
    
    # === Sector Heatmap ===
    st.markdown("### 🗺️ Sector Rotation")
    sector_df = compute_sector_performance(
        st.session_state.enriched_data or st.session_state.stock_data, get_sector)
    if not sector_df.empty:
        fig = plot_sector_heatmap(sector_df)
        if fig: st.plotly_chart(fig, use_container_width=True)
    
    # === Summary + Quick Scan ===
    c1,c2,c3 = st.columns(3)
    with c1: pc("Stocks Loaded", str(len(st.session_state.stock_data)))
    with c2: pc("Active Signals", str(sigs))
    with c3: pc("Watchlist", str(len(st.session_state.watchlist)))
    
    st.markdown("### ⚡ Quick Scan")
    if st.button("🚀 Run All Swing Scanners", type="primary"):
        with st.spinner("Scanning..."):
            results = run_all_scanners(st.session_state.stock_data, st.session_state.nifty_data, True)
            st.session_state.scan_results = results
            for s, signals in results.items():
                for r in signals: send_tg(fmt_alert(r))
            st.rerun()
    
    if st.session_state.scan_results:
        st.markdown("### 📋 Latest Results")
        for strategy, results in st.session_state.scan_results.items():
            if not results: continue
            p = STRATEGY_PROFILES.get(strategy, {})
            with st.expander(f"{p.get('icon','')} {p.get('name',strategy)} — {len(results)}", expanded=True):
                st.dataframe(results_df(results), use_container_width=True, hide_index=True)


# ============================================================================
# SCANNER HUB
# ============================================================================
def page_scanner_hub():
    st.markdown("# 🔍 Scanner Hub")
    if not st.session_state.data_loaded:
        st.warning("⚠️ Load data from Dashboard first.")
        return
    
    cols = st.columns(4)
    selected = None
    for i, (k, p) in enumerate(STRATEGY_PROFILES.items()):
        with cols[i%4]:
            nb = p.get("requires_intraday", False)
            bc = {"Swing":"bg-s","Intraday":"bg-i","Positional":"bg-p","Overnight":"bg-o"}.get(p["type"],"bg-s")
            dt = ('<small style="color:#00d26a">🔴 LIVE</small>' if st.session_state.breeze_connected 
                  else '<small style="color:#ff8a65">📊 Proxy</small>') if nb else '<small style="color:#5dade2">📊 Daily</small>'
            st.markdown(f'<div class="sc"><strong>{p["icon"]} {p["name"]}</strong><br>'
                f'<span class="bg {bc}">{p["type"]}</span> <small style="color:#888">{p["hold"]}</small><br>'
                f'<span class="mg">Win {p["win_rate"]}%</span> · <span class="mo">+{p["expectancy"]}%</span><br>{dt}</div>',
                unsafe_allow_html=True)
            if st.button("Scan", key=f"s_{k}", use_container_width=True): selected = k
    
    st.markdown("---")
    c1,c2 = st.columns(2)
    with c1:
        if st.button("🚀 All Swing", type="primary", use_container_width=True): selected = "ALL_SWING"
    with c2:
        if st.button("🔄 All (incl. Intraday)", use_container_width=True): selected = "ALL"
    
    if selected:
        n = st.session_state.nifty_data
        if selected == "ALL_SWING":
            with st.spinner("Scanning..."): st.session_state.scan_results = run_all_scanners(st.session_state.stock_data, n, True)
        elif selected == "ALL":
            with st.spinner("Scanning..."): st.session_state.scan_results = run_all_scanners(st.session_state.stock_data, n, False)
        else:
            with st.spinner(f"Running {STRATEGY_PROFILES[selected]['name']}..."):
                st.session_state.scan_results[selected] = run_scanner(selected, st.session_state.stock_data, n)
        for s, sigs in st.session_state.scan_results.items():
            for r in sigs: send_tg(fmt_alert(r))
        st.rerun()
    
    if not st.session_state.scan_results:
        st.info("Select a strategy and click Scan.")
        return
    
    for strategy, results in st.session_state.scan_results.items():
        if not results: continue
        p = STRATEGY_PROFILES.get(strategy, {})
        st.markdown(f"#### {p.get('icon','')} {p.get('name',strategy)} — {len(results)} Signal(s)")
        st.dataframe(results_df(results), use_container_width=True, hide_index=True)
        
        for r in results:
            with st.expander(f"📋 {r.symbol} — {r.signal} | {fp(r.cmp)}"):
                c1,c2,c3,c4,c5,c6 = st.columns(6)
                with c1: pc("CMP", fp(r.cmp))
                with c2: pc("Entry", fp(r.entry))
                with c3: pc("Stop Loss", fp(r.stop_loss), "r")
                with c4: pc("Target 1", fp(r.target_1), "g")
                with c5: pc("Target 2", fp(r.target_2), "g")
                with c6: pc("Conf", f"{r.confidence}%", "o")
                
                st.markdown(f"**{r.entry_type}** | R:R 1:{r.risk_reward:.1f} | RS {r.rs_rating:.0f} | RSI {r.rsi:.0f} | {r.sector}")
                
                # Multi-timeframe check
                if r.symbol in (st.session_state.enriched_data or {}):
                    mtf = check_weekly_alignment(st.session_state.enriched_data[r.symbol])
                    if mtf["aligned"]:
                        st.success(f"✅ Weekly timeframe confirms ({mtf['score']}/{mtf['max_score']})")
                    else:
                        st.warning(f"⚠️ Weekly not aligned ({mtf['score']}/{mtf['max_score']})")
                    with st.expander("Weekly Details"):
                        for reason in mtf.get("reasons", []): st.markdown(f"  {reason}")
                
                # Chart
                if r.symbol in (st.session_state.enriched_data or {}):
                    fig = plot_candlestick(
                        st.session_state.enriched_data[r.symbol], r.symbol,
                        entry=r.entry, stop_loss=r.stop_loss,
                        target1=r.target_1, target2=r.target_2, signal=r.signal
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("**Why:**")
                for reason in r.reasons: st.markdown(f"  • {reason}")
                
                c1,c2,c3 = st.columns(3)
                with c1:
                    if st.button("⭐ Watchlist", key=f"a_{strategy}_{r.symbol}"):
                        ent = {"symbol":r.symbol,"strategy":strategy,"cmp":r.cmp,
                               "entry":r.entry,"stop":r.stop_loss,"target1":r.target_1,
                               "target2":r.target_2,"confidence":r.confidence,
                               "date":r.timestamp,"entry_type":r.entry_type}
                        if not any(w["symbol"]==r.symbol and w["strategy"]==strategy for w in st.session_state.watchlist):
                            st.session_state.watchlist.append(ent)
                            st.success(f"Added!")
                with c2:
                    if st.button("📱 Telegram", key=f"t_{strategy}_{r.symbol}"):
                        if send_tg(fmt_alert(r)): st.success("Sent!")
                        else: st.warning("Setup Telegram first")
                with c3:
                    if st.button("📓 Journal", key=f"j_{strategy}_{r.symbol}"):
                        add_journal_entry({
                            "symbol":r.symbol,"strategy":strategy,"signal":r.signal,
                            "entry":r.entry,"stop":r.stop_loss,"target1":r.target_1,
                            "cmp":r.cmp,"confidence":r.confidence,"status":"open",
                            "entry_date":r.timestamp,"reasons":r.reasons[:3]
                        })
                        st.session_state.journal = load_journal()
                        st.success("Added to journal!")


# ============================================================================
# CHARTS & RS (NEW)
# ============================================================================
def page_charts_rs():
    st.markdown("# 📈 Charts & Relative Strength")
    if not st.session_state.data_loaded:
        st.warning("Load data from Dashboard first.")
        return
    
    tab1, tab2, tab3 = st.tabs(["📊 Stock Chart", "💪 RS Rankings", "🗺️ Sector Rotation"])
    
    with tab1:
        enriched = st.session_state.enriched_data or st.session_state.stock_data
        symbols = sorted(enriched.keys())
        sel = st.selectbox("Stock", symbols, index=0)
        days = st.slider("Days", 30, 250, 90)
        
        if sel in enriched:
            fig = plot_candlestick(enriched[sel], sel, days=days)
            st.plotly_chart(fig, use_container_width=True)
            
            # Key metrics
            df = enriched[sel]
            lat = df.iloc[-1]
            c1,c2,c3,c4,c5 = st.columns(5)
            with c1: pc("CMP", fp(lat["close"]))
            with c2: pc("RSI", f"{lat.get('rsi_14', 0):.0f}")
            with c3: pc("52W High", fp(lat.get("high_52w", 0)))
            with c4: pc("From 52WH", f"{lat.get('pct_from_52w_high', 0):.1f}%", "r" if lat.get("pct_from_52w_high",0) < -10 else "g")
            with c5:
                mtf = check_weekly_alignment(df)
                pc("Weekly", f"{'✅' if mtf['aligned'] else '❌'} {mtf['score']}/4")
    
    with tab2:
        st.markdown("### 💪 Relative Strength Rankings (vs Nifty)")
        rs_df = compute_rs_rankings(
            st.session_state.enriched_data or st.session_state.stock_data,
            st.session_state.nifty_data, get_sector
        )
        if not rs_df.empty:
            fig = plot_rs_scatter(rs_df)
            if fig: st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### ⭐ Top 20 RS Leaders")
            st.dataframe(rs_df.head(20)[["Symbol","Sector","CMP","1M %","3M %","RS Score","RS Rank"]],
                         use_container_width=True, hide_index=True)
            
            st.markdown("#### 🔴 Bottom 20 RS Laggards")
            st.dataframe(rs_df.tail(20)[["Symbol","Sector","CMP","1M %","3M %","RS Score","RS Rank"]],
                         use_container_width=True, hide_index=True)
    
    with tab3:
        st.markdown("### 🗺️ Sector Rotation Analysis")
        sector_df = compute_sector_performance(
            st.session_state.enriched_data or st.session_state.stock_data, get_sector)
        if not sector_df.empty:
            fig = plot_sector_heatmap(sector_df)
            if fig: st.plotly_chart(fig, use_container_width=True)
            st.dataframe(sector_df.reset_index().rename(columns={
                "index":"Sector","stocks":"Stocks","avg_1w":"1W Avg%","avg_1m":"1M Avg%","avg_3m":"3M Avg%"
            }), use_container_width=True, hide_index=True)


# ============================================================================
# TRADE PLANNER
# ============================================================================
def page_trade_planner():
    st.markdown("# 📐 Trade Planner")
    c1,c2 = st.columns(2)
    with c1:
        capital = st.number_input("Capital (₹)", value=st.session_state.capital, step=50000, min_value=10000)
        st.session_state.capital = capital
        risk_pct = st.slider("Risk %", 0.5, 3.0, 2.0, 0.25)
        sigs = [(f"{r.symbol} ({s}) {fp(r.cmp)}", s, r) for s, res in st.session_state.scan_results.items() for r in res]
        mode = st.radio("Input", ["Scanner","Manual"], horizontal=True)
        if mode == "Scanner" and sigs:
            sel = st.selectbox("Signal", [s[0] for s in sigs])
            r = sigs[[s[0] for s in sigs].index(sel)][2]
            entry, sl, short = r.entry, r.stop_loss, r.signal == "SHORT"
            st.info(f"**{r.symbol}** {r.signal} | CMP {fp(r.cmp)} | Conf {r.confidence}%")
        else:
            entry = st.number_input("Entry (₹)", value=100.0, step=1.0)
            sl = st.number_input("Stop Loss (₹)", value=95.0, step=1.0)
            short = st.checkbox("Short")
    with c2:
        mult = st.session_state.market_health.get("position_multiplier", 1.0) if st.session_state.market_health else 1.0
        if mult < 1: st.warning(f"⚠️ Positions at {mult*100:.0f}%")
        if entry > 0 and sl > 0 and entry != sl:
            pos = RiskManager.calculate_position(capital, risk_pct, entry, sl, mult)
            tgt = RiskManager.calculate_targets(entry, sl, short)
            c1,c2 = st.columns(2)
            with c1: pc("Shares", f"{pos.shares:,}"); pc("Position", f"₹{pos.position_value:,.0f}")
            with c2: pc("Risk Amt", f"₹{pos.risk_amount:,.0f}"); pc("% Portfolio", f"{pos.pct_of_portfolio:.1f}%")
            for w in pos.warnings: st.warning(w)
            st.markdown("### 🎯 Targets")
            c1,c2,c3 = st.columns(3)
            with c1: pc("T1 (1.5R)", fp(tgt.t1), "g")
            with c2: pc("T2 (2.5R)", fp(tgt.t2), "g")
            with c3: pc("T3 (4R)", fp(tgt.t3), "g")
            st.markdown(f"**Trail at:** {fp(tgt.trailing_trigger)} → SL to breakeven")
            rps = tgt.risk_per_share
            for lb, m in [("Stop Loss",-1),("T1",1.5),("T2",2.5),("T3",4)]:
                pnl = pos.shares * m * rps
                st.markdown(f"{'🟢' if pnl>0 else '🔴'} **{lb}:** ₹{pnl:+,.0f}")


# ============================================================================
# WATCHLIST
# ============================================================================
def page_watchlist():
    st.markdown("# ⭐ Watchlist")
    if not st.session_state.watchlist:
        st.info("Empty. Add from Scanner Hub.")
        return
    rows = [{"#":i+1,"Symbol":w["symbol"],"Strategy":w["strategy"],"CMP":fp(w.get("cmp",w["entry"])),
             "Entry":fp(w["entry"]),"SL":fp(w["stop"]),"T1":fp(w["target1"]),
             "Risk%":f"{abs(w['entry']-w['stop'])/w['entry']*100:.1f}%","Conf":f"{w['confidence']}%"
    } for i, w in enumerate(st.session_state.watchlist)]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    
    if st.session_state.data_loaded:
        for w in st.session_state.watchlist:
            if w["symbol"] in st.session_state.stock_data:
                cmp = st.session_state.stock_data[w["symbol"]]["close"].iloc[-1]
                pnl = (cmp/w["entry"]-1)*100
                hit = cmp <= w["stop"] if w["entry"] > w["stop"] else cmp >= w["stop"]
                st.markdown(f"**{w['symbol']}** ({w['strategy']}) — {'🔴 STOP HIT' if hit else ('🟢' if pnl>0 else '🟡')} "
                    f"CMP {fp(cmp)} <span class='{'mg' if pnl>0 else 'mr'}'>P&L {pnl:+.1f}%</span>", unsafe_allow_html=True)
    
    c1,c2 = st.columns(2)
    with c1:
        syms = [f"{w['symbol']} ({w['strategy']})" for w in st.session_state.watchlist]
        to_rm = st.selectbox("Remove", syms)
        if st.button("🗑️ Remove"):
            st.session_state.watchlist.pop(syms.index(to_rm)); st.rerun()
    with c2:
        if st.button("🗑️ Clear All"):
            st.session_state.watchlist = []; st.rerun()


# ============================================================================
# TRADE JOURNAL (NEW)
# ============================================================================
def page_journal():
    st.markdown("# 📓 Trade Journal")
    
    journal = st.session_state.journal
    analytics = compute_journal_analytics(journal)
    
    if analytics and analytics.get("closed_trades", 0) > 0:
        st.markdown("### 📊 Performance Analytics")
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        with c1: pc("Total Trades", str(analytics["total_trades"]))
        with c2: pc("Win Rate", f"{analytics['win_rate']}%", "g" if analytics["win_rate"]>55 else "r")
        with c3: pc("Total P&L", f"₹{analytics['total_pnl']:+,.0f}", "g" if analytics["total_pnl"]>0 else "r")
        with c4: pc("Profit Factor", str(analytics["profit_factor"]))
        with c5: pc("Avg Win", f"₹{analytics['avg_win']:,.0f}", "g")
        with c6: pc("Avg Loss", f"₹{analytics['avg_loss']:,.0f}", "r")
        
        fig = plot_equity_curve(analytics)
        if fig: st.plotly_chart(fig, use_container_width=True)
        
        # Strategy breakdown
        if analytics.get("strategy_stats"):
            st.markdown("### 📈 By Strategy")
            rows = []
            for s, d in analytics["strategy_stats"].items():
                wr = d["wins"]/d["trades"]*100 if d["trades"] else 0
                rows.append({"Strategy":s,"Trades":d["trades"],"Wins":d["wins"],
                             "Losses":d["losses"],"Win%":f"{wr:.0f}%","P&L":f"₹{d['pnl']:+,.0f}"})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    
    # Add new trade
    st.markdown("### ➕ Add Trade")
    with st.form("add_trade"):
        c1,c2,c3 = st.columns(3)
        with c1:
            sym = st.text_input("Symbol", placeholder="RELIANCE")
            strat = st.selectbox("Strategy", list(STRATEGY_PROFILES.keys()))
            sig = st.selectbox("Signal", ["BUY","SHORT"])
        with c2:
            entry_p = st.number_input("Entry ₹", min_value=0.0, step=1.0)
            sl_p = st.number_input("Stop Loss ₹", min_value=0.0, step=1.0)
            tgt_p = st.number_input("Target ₹", min_value=0.0, step=1.0)
        with c3:
            qty = st.number_input("Qty", min_value=1, value=1)
            status = st.selectbox("Status", ["open","closed"])
            exit_p = st.number_input("Exit ₹ (if closed)", min_value=0.0, step=1.0)
        notes = st.text_area("Notes / Lessons", placeholder="What went right or wrong?")
        
        if st.form_submit_button("Add Trade"):
            pnl = (exit_p - entry_p) * qty if status == "closed" and exit_p > 0 else 0
            if sig == "SHORT" and status == "closed" and exit_p > 0:
                pnl = (entry_p - exit_p) * qty
            add_journal_entry({
                "symbol":sym.upper(), "strategy":strat, "signal":sig,
                "entry":entry_p, "stop":sl_p, "target1":tgt_p,
                "qty":qty, "status":status, "exit":exit_p if exit_p > 0 else None,
                "pnl":pnl, "notes":notes, "entry_date": str(now_ist().date()),
                "exit_date": str(now_ist().date()) if status=="closed" else None,
            })
            st.session_state.journal = load_journal()
            st.success(f"Trade added! P&L: ₹{pnl:+,.0f}" if pnl else "Trade added!")
            st.rerun()
    
    # Trade list
    if journal:
        st.markdown("### 📋 All Trades")
        rows = []
        for e in reversed(journal):
            rows.append({
                "#":e.get("id",""), "Symbol":e.get("symbol",""), "Strategy":e.get("strategy",""),
                "Signal":e.get("signal",""), "Entry":f"₹{e.get('entry',0):,.0f}",
                "SL":f"₹{e.get('stop',0):,.0f}", "Qty":e.get("qty",1),
                "Status":e.get("status","open"),
                "P&L":f"₹{e.get('pnl',0):+,.0f}" if e.get("status")=="closed" else "—",
                "Notes":e.get("notes","")[:40],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        
        # Close open trade
        open_trades = [e for e in journal if e.get("status") == "open"]
        if open_trades:
            st.markdown("### 🔒 Close Open Trade")
            labels = [f"#{e['id']} {e['symbol']} ({e['strategy']}) Entry ₹{e['entry']:,.0f}" for e in open_trades]
            sel = st.selectbox("Select trade", labels)
            idx = labels.index(sel)
            trade = open_trades[idx]
            exit_p = st.number_input("Exit Price ₹", min_value=0.0, step=1.0, key="close_exit")
            if st.button("Close Trade"):
                if exit_p > 0:
                    pnl = (exit_p - trade["entry"]) * trade.get("qty", 1)
                    if trade.get("signal") == "SHORT":
                        pnl = (trade["entry"] - exit_p) * trade.get("qty", 1)
                    for e in journal:
                        if e.get("id") == trade["id"]:
                            e["status"] = "closed"
                            e["exit"] = exit_p
                            e["pnl"] = pnl
                            e["exit_date"] = str(now_ist().date())
                    save_journal(journal)
                    st.session_state.journal = journal
                    st.success(f"Closed! P&L: ₹{pnl:+,.0f}")
                    st.rerun()


# ============================================================================
# DAILY WORKFLOW
# ============================================================================
def page_workflow():
    st.markdown("# 📋 Daily Workflow")
    ist = now_ist()
    key = ist.strftime("%Y-%m-%d")
    if key not in st.session_state.workflow_checks:
        st.session_state.workflow_checks[key] = {}
    ch = st.session_state.workflow_checks[key]
    
    wf = [
        ("8:30 AM","🌅 Pre-Market",[("mh","Load Data → Market Health"),("gc","SGX Nifty / US Futures / Asia"),
            ("fii","FII/DII data"),("news","News check")]),
        ("9:15 AM","🔔 Market Open",[("obs","Watch 15-min candle — DON'T TRADE"),("gap","Note gaps on watchlist")]),
        ("9:30 AM","🔓 ORB 🔌",[("orb","Run ORB scanner"),("orbt","Execute ORB trades")]),
        ("10:00 AM","📈 VWAP 🔌",[("vwap","Run VWAP Reclaim"),("vwapt","Execute VWAP trades")]),
        ("12:30 PM","🍽️ Lunch Low 🔌",[("lunch","Run Lunch Low scanner"),("trail","Trail morning stops")]),
        ("3:00 PM","⭐ ATH",[("ath","Run Last 30 Min ATH"),("athb","BUY at 3:25 PM")]),
        ("3:30 PM+","📋 Swing",[("vcp","VCP scan"),("ema","21 EMA scan"),("brk","52WH scan"),
            ("sht","Short scan"),("wl","Update watchlist"),("jnl","Journal trades")]),
        ("Weekend","📅 Review",[("wpnl","Weekly P&L"),("sec","Sector rotation"),("heat","Heat check"),("f500","Full 500 scan")]),
    ]
    total = sum(len(t) for _,_,t in wf)
    done = sum(1 for v in ch.values() if v)
    st.progress(done/total if total else 0)
    st.caption(f"{done}/{total}")
    for tm, title, tasks in wf:
        st.markdown(f'<div class="ws"><strong>⏰ {tm} IST</strong> — {title}</div>', unsafe_allow_html=True)
        for tid, lb in tasks:
            ch[tid] = st.checkbox(lb, value=ch.get(tid, False), key=f"wf_{key}_{tid}")
    st.session_state.workflow_checks[key] = ch


# ============================================================================
# ALERTS
# ============================================================================
def page_alerts():
    st.markdown("# 🔔 Alerts")
    st.markdown("### 📱 Telegram Setup")
    st.markdown("1. Telegram → `@BotFather` → `/newbot` → copy **Bot Token**\n"
                "2. `@userinfobot` → `/start` → copy **Chat ID**")
    c1,c2 = st.columns(2)
    with c1:
        st.session_state.telegram_token = st.text_input("Bot Token",
            value=st.session_state.telegram_token or st.secrets.get("TELEGRAM_BOT_TOKEN",""), type="password")
    with c2:
        st.session_state.telegram_chat_id = st.text_input("Chat ID",
            value=st.session_state.telegram_chat_id or st.secrets.get("TELEGRAM_CHAT_ID",""))
    if st.button("🧪 Test Alert"):
        if send_tg("🎯 <b>NSE Scanner Pro</b>\n✅ Test — Telegram connected!"):
            st.success("✅ Sent! Check Telegram.")
        else: st.error("❌ Check credentials.")
    st.markdown("---")
    st.markdown("**Auto-alerts** fire when Quick Scan or All Scanners find signals. "
                "Or click 📱 on any signal in Scanner Hub.")
    st.markdown("**Streamlit Secrets** (persist across sessions):")
    st.code('TELEGRAM_BOT_TOKEN = "123456:ABCdef..."\nTELEGRAM_CHAT_ID = "987654321"', language="toml")


# ============================================================================
# SETTINGS
# ============================================================================
def page_settings():
    st.markdown("# ⚙️ Settings")
    st.session_state.capital = st.number_input("Capital (₹)", value=st.session_state.capital, step=50000)
    
    st.markdown("### 🔌 Breeze API")
    if st.session_state.breeze_connected:
        st.success("✅ **Breeze Connected!**")
        st.markdown("**Enabled:** Real-time 5-min candles for ORB, VWAP Reclaim, Lunch Low.\n\n"
                     "**VCP, EMA21, 52WH, Short, ATH** use daily data (Breeze not needed).")
    else:
        if st.session_state.breeze_msg: st.error(st.session_state.breeze_msg)
        st.markdown("**What Breeze unlocks:**\n\n"
            "| Scanner | Without | With Breeze |\n|---|---|---|\n"
            "| ORB | Daily proxy | **Real 15-min breakout** |\n"
            "| VWAP Reclaim | Estimated | **Live VWAP** |\n"
            "| Lunch Low | EOD hammer | **Real-time reversal** |\n"
            "| VCP/EMA21/52WH/Short/ATH | ✅ Full | ✅ Same |")
        st.markdown("---")
        st.warning("⚠️ Paste ONLY these 3 lines in **Streamlit Settings → Secrets** (no backticks!):")
        st.code('BREEZE_API_KEY = "your_key"\nBREEZE_API_SECRET = "your_secret"\nBREEZE_SESSION_TOKEN = "daily_token"', language="toml")
        st.info("⚠️ **Session Token expires daily.** Regenerate each morning from ICICI Direct portal.")
        
        with st.expander("🔧 Manual Connect (testing)"):
            with st.form("bf"):
                ak = st.text_input("API Key", type="password")
                asc = st.text_input("API Secret", type="password")
                st_ = st.text_input("Session Token", type="password")
                if st.form_submit_button("Connect"):
                    if ak and asc and st_:
                        e = BreezeEngine()
                        ok, msg = e.connect(ak, asc, st_)
                        if ok:
                            st.success(msg); st.session_state.breeze_connected = True
                            st.session_state.breeze_engine = e; st.session_state.breeze_msg = msg
                        else: st.error(msg); st.session_state.breeze_msg = msg
    
    st.session_state.universe_size = st.selectbox("Universe", ["nifty50","nifty200","nifty500"],
        index=["nifty50","nifty200","nifty500"].index(st.session_state.universe_size),
        format_func=lambda x: {"nifty50":"Nifty 50","nifty200":"Nifty 200","nifty500":"Nifty 500"}[x])


# ============================================================================
# ROUTER
# ============================================================================
{"📊 Dashboard": page_dashboard, "🔍 Scanner Hub": page_scanner_hub,
 "📈 Charts & RS": page_charts_rs, "📐 Trade Planner": page_trade_planner,
 "⭐ Watchlist": page_watchlist, "📓 Trade Journal": page_journal,
 "📋 Daily Workflow": page_workflow, "🔔 Alerts": page_alerts,
 "⚙️ Settings": page_settings}[page]()
