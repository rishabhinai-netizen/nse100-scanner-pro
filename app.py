"""
NSE SCANNER PRO — Elite Stock Scanner
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import pytz
import json
import os

# Local imports
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

# ============================================================================
# PAGE CONFIG — No GitHub/source info exposed
# ============================================================================
st.set_page_config(
    page_title="NSE Scanner Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": None,
    }
)

# ============================================================================
# HIDE ALL STREAMLIT BRANDING, GITHUB LINKS, FOOTER, MENU
# ============================================================================
st.markdown("""
<style>
    /* Hide Streamlit header, footer, GitHub corner, hamburger menu items */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    .stDeployButton { display: none; }
    [data-testid="stToolbar"] { display: none; }
    .viewerBadge_container__r5tak { display: none; }
    .styles_viewerBadge__CvC9N { display: none; }
    ._profileContainer_51w34_53 { display: none; }
    ._profilePreview_51w34_63 { display: none; }
    [data-testid="stStatusWidget"] { display: none; }
    /* GitHub icon in corner */
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_link__qRIco,
    .styles_viewerBadge__1yB5_, .viewerBadge_text__1JaDK{ display: none; }
    /* Source code link */
    #stDecoration { display: none; }
    
    /* Dark theme enhancements */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1d23, #252830);
        border: 1px solid #333;
        border-radius: 10px;
        padding: 12px 16px;
    }
    .strategy-card {
        background: #1a1d23;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
    }
    .strategy-card:hover { border-color: #FF6B35; }
    .metric-green { color: #00d26a !important; }
    .metric-red { color: #ff4757 !important; }
    .metric-orange { color: #FF6B35 !important; }
    .badge-swing { background: #1e3a5f; color: #5dade2; display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.75em; }
    .badge-intraday { background: #3e2723; color: #ff8a65; display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.75em; }
    .badge-positional { background: #1b5e20; color: #81c784; display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.75em; }
    .badge-overnight { background: #4a148c; color: #ce93d8; display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.75em; }
    .workflow-step {
        border-left: 3px solid #FF6B35;
        padding: 8px 16px;
        margin: 8px 0;
        background: #1a1d23;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE
# ============================================================================
defaults = {
    "watchlist": [],
    "scan_results": {},
    "market_health": None,
    "data_loaded": False,
    "stock_data": {},
    "nifty_data": None,
    "capital": 500000,
    "breeze_connected": False,
    "breeze_engine": None,
    "workflow_checks": {},
    "universe_size": "nifty200",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================================
# BREEZE AUTO-CONNECT (from Streamlit Secrets)
# ============================================================================
def try_breeze_connect():
    """Try connecting Breeze from Streamlit secrets on app load."""
    if st.session_state.breeze_connected:
        return
    try:
        if "BREEZE_API_KEY" in st.secrets:
            engine = BreezeEngine()
            success, msg = engine.connect_from_secrets()
            if success:
                st.session_state.breeze_connected = True
                st.session_state.breeze_engine = engine
    except Exception:
        pass

try_breeze_connect()

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("## 🎯 NSE Scanner Pro")
    st.caption("8 Elite Trading Strategies")
    st.markdown("---")
    
    page = st.radio(
        "Navigate",
        ["📊 Dashboard", "🔍 Scanner Hub", "📐 Trade Planner",
         "⭐ Watchlist", "📋 Daily Workflow", "⚙️ Settings"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # IST time
    ist_now = now_ist()
    market_open = time(9, 15)
    market_close = time(15, 30)
    is_market_hours = market_open <= ist_now.time() <= market_close and ist_now.weekday() < 5
    status_icon = "🟢" if is_market_hours else "🔴"
    st.caption(f"{status_icon} {ist_now.strftime('%d %b %Y, %I:%M %p IST')}")
    
    if st.session_state.market_health:
        mh = st.session_state.market_health
        st.markdown(f"**Market:** {mh['regime']}")
        st.markdown(f"**Nifty:** ₹{mh.get('nifty_close', 'N/A'):,}")
    
    total_signals = sum(len(v) for v in st.session_state.scan_results.values())
    st.markdown(f"**Signals:** {total_signals} | **Watch:** {len(st.session_state.watchlist)}")
    
    st.markdown("---")
    breeze_status = "🟢 Breeze Connected" if st.session_state.breeze_connected else "⚪ Breeze: Off"
    st.caption(breeze_status)
    st.caption(f"Data: yfinance | Universe: {st.session_state.universe_size}")


# ============================================================================
# DATA LOADING WITH PROGRESS
# ============================================================================
def load_data_with_progress():
    """Load data with progress bar (batched, rate-limited)."""
    symbols = get_stock_universe(st.session_state.universe_size)
    
    progress_bar = st.progress(0, text="Initializing...")
    
    def update_progress(pct, text):
        progress_bar.progress(min(pct, 0.95), text=text)
    
    data = fetch_batch_daily(symbols, period="1y", progress_callback=update_progress)
    
    progress_bar.progress(0.96, text="Fetching Nifty 50 index...")
    nifty = fetch_nifty_data()
    
    progress_bar.progress(1.0, text=f"✅ Loaded {len(data)} stocks!")
    
    return data, nifty


# ============================================================================
# HELPER: Display a scan result row
# ============================================================================
def display_result_row(r: ScanResult):
    """Display a single scan result with CMP, entry, SL, targets."""
    signal_color = "#00d26a" if r.signal == "BUY" else "#ff4757"
    entry_note = f" ({r.entry_type})" if r.entry_type != "AT CMP" else ""
    
    st.markdown(f"""
    **{r.symbol}** <span style="color:#888">({r.sector})</span> — 
    <span style="color:{signal_color};font-weight:700">{r.signal}</span>{entry_note} | 
    CMP: **₹{r.cmp:,.2f}** | Entry: ₹{r.entry:,.2f} | SL: ₹{r.stop_loss:,.2f} | 
    T1: ₹{r.target_1:,.2f} | T2: ₹{r.target_2:,.2f} | 
    Conf: {r.confidence}% | RS: {r.rs_rating:.0f} | Risk: {r.risk_pct:.1f}%
    """, unsafe_allow_html=True)


def results_to_dataframe(results: list) -> pd.DataFrame:
    """Convert scan results to a display DataFrame with CMP."""
    rows = []
    for r in results:
        rows.append({
            "Symbol": r.symbol,
            "Signal": r.signal,
            "CMP ₹": f"{r.cmp:,.2f}",
            "Entry ₹": f"{r.entry:,.2f}",
            "Entry Type": r.entry_type,
            "Stop Loss ₹": f"{r.stop_loss:,.2f}",
            "Target 1 ₹": f"{r.target_1:,.2f}",
            "Target 2 ₹": f"{r.target_2:,.2f}",
            "R:R": f"1:{r.risk_reward:.1f}",
            "Risk %": f"{r.risk_pct:.1f}%",
            "Conf": f"{r.confidence}%",
            "RS": round(r.rs_rating),
            "Vol": f"{r.volume_ratio}x",
            "RSI": r.rsi,
            "Sector": r.sector,
            "Hold": r.hold_type,
        })
    return pd.DataFrame(rows)


# ============================================================================
# PAGE: DASHBOARD
# ============================================================================
def page_dashboard():
    st.markdown("# 📊 Market Dashboard")
    
    ist_now = now_ist()
    is_mkt = time(9, 15) <= ist_now.time() <= time(15, 30) and ist_now.weekday() < 5
    status = "🟢 MARKET OPEN" if is_mkt else "🔴 MARKET CLOSED"
    st.markdown(f"**{status}** — {ist_now.strftime('%d %b %Y, %I:%M %p IST')}")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.session_state.universe_size = st.selectbox(
            "Stock Universe",
            ["nifty50", "nifty200", "nifty500"],
            index=1,
            format_func=lambda x: {"nifty50": "Nifty 50 (fast)", "nifty200": "Nifty 200 (balanced)", "nifty500": "Nifty 500 (full, slower)"}[x]
        )
    with col2:
        if st.button("🔄 Load / Refresh Data", type="primary", use_container_width=True):
            st.cache_data.clear()
            data, nifty = load_data_with_progress()
            st.session_state.stock_data = data
            st.session_state.nifty_data = nifty
            st.session_state.data_loaded = True
            if nifty is not None:
                st.session_state.market_health = check_market_health(nifty)
            st.rerun()
    
    if not st.session_state.data_loaded:
        st.info("👆 Click **Load / Refresh Data** to start scanning.")
        st.markdown("### Strategy Overview")
        cols = st.columns(4)
        for i, (key, p) in enumerate(STRATEGY_PROFILES.items()):
            with cols[i % 4]:
                st.markdown(f"""<div class="strategy-card">
                    <strong>{p['icon']} {p['name']}</strong><br>
                    <small style="color:#888">{p['type']} • {p['hold']}</small><br>
                    <span class="metric-green">Win: {p['win_rate']}%</span> •
                    <span class="metric-orange">PF: {p['profit_factor']}</span>
                </div>""", unsafe_allow_html=True)
        return
    
    # Market Health
    st.markdown("### 🏥 Market Health Check")
    mh = st.session_state.market_health
    if mh:
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Regime", mh["regime"])
        with c2: st.metric("Health Score", f"{mh['score']}/{mh['max_score']}")
        with c3: st.metric("Nifty", f"₹{mh.get('nifty_close', 0):,.2f}")
        with c4: st.metric("Position Size", f"{mh['position_multiplier']*100:.0f}%")
        
        with st.expander("Health Details"):
            for d in mh.get("details", []):
                st.markdown(f"  {d}")
            if mh["position_multiplier"] < 0.5:
                st.warning("⚠️ Bearish — only trade VCP & 21 EMA Bounce at 50% size")
    
    # Summary
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Stocks Loaded", len(st.session_state.stock_data))
    with c2: st.metric("Active Signals", sum(len(v) for v in st.session_state.scan_results.values()))
    with c3: st.metric("Watchlist", len(st.session_state.watchlist))
    
    # Quick Scan
    st.markdown("### ⚡ Quick Scan")
    if st.button("🚀 Run All Swing Scanners", type="primary"):
        with st.spinner("Scanning..."):
            results = run_all_scanners(st.session_state.stock_data, st.session_state.nifty_data, daily_only=True)
            st.session_state.scan_results = results
            st.rerun()
    
    # Results
    if st.session_state.scan_results:
        st.markdown("### 📋 Latest Results")
        for strategy, results in st.session_state.scan_results.items():
            p = STRATEGY_PROFILES.get(strategy, {})
            with st.expander(f"{p.get('icon', '📌')} {p.get('name', strategy)} — {len(results)} signal(s)", expanded=True):
                for r in results[:5]:
                    display_result_row(r)


# ============================================================================
# PAGE: SCANNER HUB
# ============================================================================
def page_scanner_hub():
    st.markdown("# 🔍 Scanner Hub")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Load data first from Dashboard.")
        return
    
    # Strategy cards
    st.markdown("### Select Strategy")
    cols = st.columns(4)
    selected = None
    
    for i, (key, p) in enumerate(STRATEGY_PROFILES.items()):
        with cols[i % 4]:
            badge = {"Swing": "badge-swing", "Intraday": "badge-intraday",
                     "Positional": "badge-positional", "Overnight": "badge-overnight"}.get(p["type"], "badge-swing")
            
            intraday_note = ""
            if p.get("requires_intraday"):
                intraday_note = "<br><small style='color:#ff8a65'>📊 Daily proxy mode</small>" if not st.session_state.breeze_connected else "<br><small style='color:#00d26a'>🔴 Live intraday</small>"
            
            st.markdown(f"""<div class="strategy-card">
                <strong>{p['icon']} {p['name']}</strong><br>
                <span class="{badge}">{p['type']}</span> <small style="color:#888">• {p['hold']}</small><br>
                <small>{p['description'][:70]}...</small><br>
                <span class="metric-green">Win: {p['win_rate']}%</span> •
                <span class="metric-orange">Exp: +{p['expectancy']}%</span>
                {intraday_note}
            </div>""", unsafe_allow_html=True)
            
            if st.button(f"Scan", key=f"scan_{key}", use_container_width=True):
                selected = key
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🚀 All Swing Scanners", type="primary", use_container_width=True):
            selected = "ALL_SWING"
    with c2:
        if st.button("🔄 All Scanners (incl. Intraday Proxy)", use_container_width=True):
            selected = "ALL"
    
    if selected:
        nifty = st.session_state.nifty_data
        if selected == "ALL_SWING":
            with st.spinner("Running swing scanners..."):
                st.session_state.scan_results = run_all_scanners(st.session_state.stock_data, nifty, True)
        elif selected == "ALL":
            with st.spinner("Running all scanners..."):
                st.session_state.scan_results = run_all_scanners(st.session_state.stock_data, nifty, False)
        else:
            with st.spinner(f"Running {STRATEGY_PROFILES[selected]['name']}..."):
                st.session_state.scan_results[selected] = run_scanner(selected, st.session_state.stock_data, nifty)
        st.rerun()
    
    # Results
    st.markdown("### 📊 Scan Results")
    if not st.session_state.scan_results:
        st.info("Select a strategy above and click Scan.")
        return
    
    for strategy, results in st.session_state.scan_results.items():
        if not results:
            continue
        p = STRATEGY_PROFILES.get(strategy, {})
        st.markdown(f"#### {p.get('icon', '📌')} {p.get('name', strategy)} — {len(results)} Signal(s)")
        
        st.dataframe(results_to_dataframe(results), use_container_width=True, hide_index=True)
        
        for r in results:
            with st.expander(f"📋 {r.symbol} — Details"):
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("CMP", f"₹{r.cmp:,.2f}")
                    st.metric("Entry", f"₹{r.entry:,.2f}")
                with c2:
                    st.metric("Stop Loss", f"₹{r.stop_loss:,.2f}")
                    st.metric("Risk %", f"{r.risk_pct:.1f}%")
                with c3:
                    st.metric("Target 1", f"₹{r.target_1:,.2f}")
                    st.metric("Target 2", f"₹{r.target_2:,.2f}")
                with c4:
                    st.metric("Confidence", f"{r.confidence}%")
                    st.metric("RS Rating", f"{r.rs_rating:.0f}")
                
                st.markdown(f"**Entry Type:** {r.entry_type}")
                st.markdown("**Why this trade:**")
                for reason in r.reasons:
                    st.markdown(f"  • {reason}")
                
                if st.button(f"⭐ Add to Watchlist", key=f"add_{strategy}_{r.symbol}"):
                    entry = {"symbol": r.symbol, "strategy": strategy,
                             "cmp": r.cmp, "entry": r.entry, "stop": r.stop_loss,
                             "target1": r.target_1, "target2": r.target_2,
                             "confidence": r.confidence, "date": r.timestamp,
                             "entry_type": r.entry_type}
                    if not any(w["symbol"] == r.symbol and w["strategy"] == strategy for w in st.session_state.watchlist):
                        st.session_state.watchlist.append(entry)
                        st.success(f"Added {r.symbol}!")


# ============================================================================
# PAGE: TRADE PLANNER
# ============================================================================
def page_trade_planner():
    st.markdown("# 📐 Trade Planner")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Trade Parameters")
        capital = st.number_input("Capital (₹)", value=st.session_state.capital, step=50000, min_value=10000)
        st.session_state.capital = capital
        risk_pct = st.slider("Risk per Trade (%)", 0.5, 3.0, 2.0, 0.25)
        
        signals = []
        for strat, results in st.session_state.scan_results.items():
            for r in results:
                signals.append((f"{r.symbol} ({strat}) — CMP ₹{r.cmp:,.0f}", strat, r))
        
        mode = st.radio("Input", ["From Scanner", "Manual"], horizontal=True)
        
        if mode == "From Scanner" and signals:
            sel = st.selectbox("Signal", [s[0] for s in signals])
            idx = [s[0] for s in signals].index(sel)
            r = signals[idx][2]
            entry, stop_loss, is_short = r.entry, r.stop_loss, r.signal == "SHORT"
            st.info(f"**{r.symbol}** — {r.signal} | CMP: ₹{r.cmp:,.2f} | Conf: {r.confidence}%")
        else:
            entry = st.number_input("Entry (₹)", value=100.0, step=1.0)
            stop_loss = st.number_input("Stop Loss (₹)", value=95.0, step=1.0)
            is_short = st.checkbox("Short Trade")
    
    with c2:
        st.markdown("### Position Size")
        mult = st.session_state.market_health.get("position_multiplier", 1.0) if st.session_state.market_health else 1.0
        if mult < 1: st.warning(f"⚠️ Market regime: positions at {mult*100:.0f}%")
        
        if entry > 0 and stop_loss > 0 and entry != stop_loss:
            pos = RiskManager.calculate_position(capital, risk_pct, entry, stop_loss, mult)
            targets = RiskManager.calculate_targets(entry, stop_loss, is_short)
            
            st.metric("Shares", f"{pos.shares:,}")
            st.metric("Position Value", f"₹{pos.position_value:,.0f}")
            st.metric("Risk Amount", f"₹{pos.risk_amount:,.0f}")
            st.metric("% of Portfolio", f"{pos.pct_of_portfolio:.1f}%")
            for w in pos.warnings: st.warning(w)
            
            st.markdown("### 🎯 Targets")
            tc1, tc2, tc3 = st.columns(3)
            with tc1: st.metric("T1 (1.5R)", f"₹{targets.t1:,.2f}")
            with tc2: st.metric("T2 (2.5R)", f"₹{targets.t2:,.2f}")
            with tc3: st.metric("T3 (4R)", f"₹{targets.t3:,.2f}")
            
            st.markdown(f"**Trail SL at:** ₹{targets.trailing_trigger:,.2f} → move SL to breakeven")
            
            st.markdown("### 💰 P&L Scenarios")
            r_per = targets.risk_per_share
            for label, mult_r in [("Stop Loss", -1), ("T1 (1.5R)", 1.5), ("T2 (2.5R)", 2.5), ("T3 (4R)", 4)]:
                pnl = pos.shares * mult_r * r_per
                icon = "🟢" if pnl > 0 else "🔴"
                st.markdown(f"{icon} **{label}:** ₹{pnl:+,.0f}")


# ============================================================================
# PAGE: WATCHLIST
# ============================================================================
def page_watchlist():
    st.markdown("# ⭐ Watchlist")
    
    if not st.session_state.watchlist:
        st.info("Empty. Add stocks from Scanner Hub.")
        return
    
    rows = []
    for i, w in enumerate(st.session_state.watchlist):
        risk = abs(w["entry"] - w["stop"])
        rows.append({
            "#": i + 1, "Symbol": w["symbol"], "Strategy": w["strategy"],
            "CMP ₹": f"{w.get('cmp', w['entry']):,.2f}",
            "Entry ₹": f"{w['entry']:,.2f}", "SL ₹": f"{w['stop']:,.2f}",
            "T1 ₹": f"{w['target1']:,.2f}", "Risk %": f"{(risk/w['entry']*100):.1f}%",
            "Conf": f"{w['confidence']}%", "Date": w.get("date", ""),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    
    # Live status
    if st.session_state.data_loaded:
        st.markdown("### 📊 Live Status")
        for w in st.session_state.watchlist:
            if w["symbol"] in st.session_state.stock_data:
                df = st.session_state.stock_data[w["symbol"]]
                cmp = df["close"].iloc[-1]
                pnl = (cmp / w["entry"] - 1) * 100
                hit_stop = cmp <= w["stop"] if w["entry"] > w["stop"] else cmp >= w["stop"]
                status = "🔴 STOP HIT" if hit_stop else ("🟢 Profit" if pnl > 0 else "🟡 Loss")
                color = "metric-green" if pnl > 0 else "metric-red"
                st.markdown(f"**{w['symbol']}** ({w['strategy']}) — {status} | CMP: ₹{cmp:,.2f} | <span class='{color}'>P&L: {pnl:+.1f}%</span>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        syms = [f"{w['symbol']} ({w['strategy']})" for w in st.session_state.watchlist]
        to_rm = st.selectbox("Remove", syms)
        if st.button("🗑️ Remove"):
            st.session_state.watchlist.pop(syms.index(to_rm))
            st.rerun()
    with c2:
        if st.button("🗑️ Clear All"):
            st.session_state.watchlist = []
            st.rerun()


# ============================================================================
# PAGE: DAILY WORKFLOW (IST timings, corrected)
# ============================================================================
def page_daily_workflow():
    st.markdown("# 📋 Daily Trading Workflow")
    
    ist_now = now_ist()
    today_key = ist_now.strftime("%Y-%m-%d")
    if today_key not in st.session_state.workflow_checks:
        st.session_state.workflow_checks[today_key] = {}
    checks = st.session_state.workflow_checks[today_key]
    
    workflow = [
        {
            "time": "8:30 AM IST",
            "title": "🌅 Pre-Market Prep",
            "tasks": [
                ("market_health", "Open Dashboard → Load Data → Check Market Health"),
                ("global_cues", "Check global cues: US futures, SGX Nifty, Asia markets"),
                ("fii_dii", "Check FII/DII data (moneycontrol.com/markets/fii-dii-activity)"),
                ("news", "Review major news/events for today"),
            ]
        },
        {
            "time": "9:15 AM IST",
            "title": "🔔 Market Opens",
            "tasks": [
                ("watch_open", "Watch first 15 min candle — DO NOT trade, just observe"),
                ("gap_check", "Note gap-up/gap-down in watchlist stocks"),
            ]
        },
        {
            "time": "9:30-9:45 AM IST",
            "title": "🔓 ORB Window (after first 15-min candle closes)",
            "tasks": [
                ("orb_scan", "Run ORB scanner → check for breakouts above 15-min high"),
                ("orb_trade", "Execute ORB trades with volume confirmation (1.5x+ avg)"),
            ]
        },
        {
            "time": "10:00-10:30 AM IST",
            "title": "📈 VWAP Reclaim Window",
            "tasks": [
                ("vwap_scan", "Run VWAP Reclaim scanner → stocks reclaiming VWAP from below"),
                ("vwap_trade", "Execute VWAP reclaim trades — stop below today's low"),
            ]
        },
        {
            "time": "12:30-1:30 PM IST",
            "title": "🍽️ Lunch Low Window",
            "tasks": [
                ("lunch_scan", "Run Lunch Low scanner → hammer reversals at support"),
                ("trail_morning", "Trail morning trade stops to breakeven if profitable"),
            ]
        },
        {
            "time": "2:30-3:00 PM IST",
            "title": "📊 Pre-Close Analysis",
            "tasks": [
                ("review_intraday", "Review all intraday trades — close losing ones"),
                ("prep_ath", "Prepare for Last 30 Min ATH scan"),
            ]
        },
        {
            "time": "3:00-3:15 PM IST",
            "title": "⭐ Last 30 Min ATH (Best overnight play)",
            "tasks": [
                ("ath_scan", "Run Last 30 Min ATH scanner → stocks near 52WH with 2x volume"),
                ("ath_buy", "BUY top signals at 3:25 PM (last 5 minutes) — hold overnight"),
            ]
        },
        {
            "time": "3:30 PM+ IST (Post-Market)",
            "title": "📋 Swing Scanner Session",
            "tasks": [
                ("vcp_scan", "Run VCP scanner (best expectancy: +5.12%)"),
                ("ema_scan", "Run 21 EMA Bounce scanner"),
                ("breakout_scan", "Run 52-Week High Breakout scanner"),
                ("short_scan", "Run Failed Breakout Short scanner (avoid in strong bull)"),
                ("watchlist_update", "Update watchlist with next-day candidates"),
                ("journal", "Record today's trades in journal with lessons"),
            ]
        },
        {
            "time": "Weekend",
            "title": "📅 Weekly Review",
            "tasks": [
                ("weekly_pnl", "Calculate weekly P&L and win rate"),
                ("sector_check", "Check sector rotation — which sectors are leading?"),
                ("heat_check", "Check portfolio heat — must be < 10% of capital"),
                ("full_vcp_scan", "Run VCP on full Nifty 500 for next week's candidates"),
            ]
        },
    ]
    
    total = sum(len(s["tasks"]) for s in workflow)
    done = sum(1 for v in checks.values() if v)
    st.progress(done / total if total > 0 else 0)
    st.markdown(f"**Progress:** {done}/{total} tasks")
    
    for step in workflow:
        st.markdown(f"""<div class="workflow-step">
            <strong>⏰ {step['time']}</strong> — {step['title']}
        </div>""", unsafe_allow_html=True)
        for tid, label in step["tasks"]:
            checks[tid] = st.checkbox(label, value=checks.get(tid, False), key=f"wf_{today_key}_{tid}")
    
    st.session_state.workflow_checks[today_key] = checks
    
    st.markdown("---")
    st.markdown("### 🛡️ Risk Rules (NON-NEGOTIABLE)")
    for rule, val in [
        ("Max risk per trade", "2% of capital"),
        ("Max positions", "5-10"), ("Max sector exposure", "30%"),
        ("Portfolio heat", "< 10% of capital"), ("Max single position", "20% of capital"),
        ("Time stop", "Exit if no move in 10 days"), ("Max loss per trade", "-7% absolute"),
    ]:
        st.markdown(f"  **{rule}:** {val}")


# ============================================================================
# PAGE: SETTINGS
# ============================================================================
def page_settings():
    st.markdown("# ⚙️ Settings")
    
    # Capital
    st.markdown("### 💰 Capital")
    st.session_state.capital = st.number_input("Capital (₹)", value=st.session_state.capital, step=50000)
    
    # Breeze API
    st.markdown("### 🔌 ICICI Breeze API")
    
    if st.session_state.breeze_connected:
        st.success("✅ Breeze API is connected and active!")
        st.markdown("Intraday scanners (ORB, VWAP Reclaim, Lunch Low) are using live data.")
    else:
        st.markdown("""
        **How to connect Breeze API (one-time setup + daily token):**
        
        **Step 1 — Get API credentials (one-time):**
        1. Go to [api.icicidirect.com](https://api.icicidirect.com)
        2. Login → Create App → Copy **API Key** and **API Secret**
        
        **Step 2 — Add to Streamlit Secrets (one-time per deployment):**
        
        **For Streamlit Cloud:** Go to your app → Settings → Secrets → paste:
        ```
        BREEZE_API_KEY = "your_api_key"
        BREEZE_API_SECRET = "your_api_secret"
        BREEZE_SESSION_TOKEN = "your_session_token"
        ```
        
        **For local:** Create `.streamlit/secrets.toml` file with the same content.
        
        **Step 3 — Daily token refresh:**
        ⚠️ The **Session Token expires daily**. Every morning before trading:
        1. Login to [api.icicidirect.com](https://api.icicidirect.com)
        2. Generate new session token
        3. Update the `BREEZE_SESSION_TOKEN` in Streamlit Secrets
        4. Refresh the app
        
        This is an ICICI requirement — there's no way around the daily token refresh.
        """)
        
        # Manual connect option
        with st.expander("Manual Connect (for testing)"):
            with st.form("breeze_form"):
                api_key = st.text_input("API Key", type="password")
                api_secret = st.text_input("API Secret", type="password")
                session_token = st.text_input("Session Token", type="password")
                
                if st.form_submit_button("Connect"):
                    if api_key and api_secret and session_token:
                        engine = BreezeEngine()
                        ok, msg = engine.connect(api_key, api_secret, session_token)
                        if ok:
                            st.success(msg)
                            st.session_state.breeze_connected = True
                            st.session_state.breeze_engine = engine
                        else:
                            st.error(msg)
    
    # Universe
    st.markdown("### 📊 Stock Universe")
    st.session_state.universe_size = st.selectbox(
        "Default Universe",
        ["nifty50", "nifty200", "nifty500"],
        index=["nifty50", "nifty200", "nifty500"].index(st.session_state.universe_size),
        format_func=lambda x: {"nifty50": "Nifty 50 (50 stocks, fastest)",
                                "nifty200": "Nifty 200 (200 stocks, balanced)",
                                "nifty500": "Nifty 500 (500 stocks, full coverage)"}[x]
    )
    
    st.markdown("### 📈 Strategy Rankings")
    st.markdown("""
    | # | Strategy | Win Rate | Expectancy | Hold |
    |---|----------|----------|------------|------|
    | 1 | 🏆 VCP (Minervini) | 67.2% | +5.12% | 15-40 days |
    | 2 | 🚀 52-Week High Breakout | 58.8% | +5.82% | 20-60 days |
    | 3 | 📉 Failed Breakout Short | 64.2% | +3.12% | 3-10 days |
    | 4 | 🔄 21 EMA Bounce | 62.5% | +2.14% | 5-15 days |
    | 5 | ⭐ Last 30 Min ATH | 68.4% | +0.89% | Overnight |
    | 6 | 🔓 ORB | 58.2% | +0.47% | 2-6 hours |
    | 7 | 📈 VWAP Reclaim | 61.8% | +0.39% | 2-4 hours |
    | 8 | 🍽️ Lunch Low Buy | 56.3% | +0.28% | 2-3 hours |
    """)


# ============================================================================
# ROUTER
# ============================================================================
pages = {
    "📊 Dashboard": page_dashboard,
    "🔍 Scanner Hub": page_scanner_hub,
    "📐 Trade Planner": page_trade_planner,
    "⭐ Watchlist": page_watchlist,
    "📋 Daily Workflow": page_daily_workflow,
    "⚙️ Settings": page_settings,
}
pages[page]()
