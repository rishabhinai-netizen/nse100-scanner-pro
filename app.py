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
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="NSE Scanner Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get Help": None, "Report a bug": None, "About": None}
)

# ============================================================================
# CSS — Fix #3: Prevent text overflow, professional sizing
# ============================================================================
st.markdown("""
<style>
    /* Hide all Streamlit branding */
    #MainMenu, footer, header, .stDeployButton,
    [data-testid="stToolbar"], [data-testid="stStatusWidget"],
    .viewerBadge_container__r5tak, .styles_viewerBadge__CvC9N,
    ._profileContainer_51w34_53, ._profilePreview_51w34_63,
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_link__qRIco,
    .styles_viewerBadge__1yB5_, .viewerBadge_text__1JaDK,
    #stDecoration { display: none !important; visibility: hidden !important; }
    
    /* Fix #3: Metric boxes — prevent overflow, compact text */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1d23, #252830);
        border: 1px solid #333;
        border-radius: 8px;
        padding: 8px 12px;
        overflow: hidden;
    }
    div[data-testid="stMetric"] label {
        font-size: 0.75rem !important;
        color: #888 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.1rem !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        font-size: 0.7rem !important;
    }
    
    /* Custom metric card for prices — never overflows */
    .price-card {
        background: linear-gradient(135deg, #1a1d23, #252830);
        border: 1px solid #333;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 4px 0;
    }
    .price-card .label { font-size: 0.72rem; color: #888; margin-bottom: 2px; }
    .price-card .value { font-size: 1rem; font-weight: 600; color: #fafafa; }
    .price-card .value.green { color: #00d26a; }
    .price-card .value.red { color: #ff4757; }
    .price-card .value.orange { color: #FF6B35; }
    
    /* Strategy cards */
    .strategy-card {
        background: #1a1d23; border: 1px solid #333;
        border-radius: 10px; padding: 14px; margin: 6px 0;
    }
    .strategy-card:hover { border-color: #FF6B35; }
    
    /* Badges */
    .badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.7em; }
    .badge-swing { background: #1e3a5f; color: #5dade2; }
    .badge-intraday { background: #3e2723; color: #ff8a65; }
    .badge-positional { background: #1b5e20; color: #81c784; }
    .badge-overnight { background: #4a148c; color: #ce93d8; }
    
    /* Colors */
    .metric-green { color: #00d26a !important; }
    .metric-red { color: #ff4757 !important; }
    .metric-orange { color: #FF6B35 !important; }
    
    /* Workflow */
    .workflow-step {
        border-left: 3px solid #FF6B35;
        padding: 8px 14px; margin: 6px 0;
        background: #1a1d23; border-radius: 0 8px 8px 0;
    }
    
    /* Breeze status banner */
    .breeze-banner {
        padding: 8px 14px; border-radius: 8px; margin: 8px 0;
        font-size: 0.85rem;
    }
    .breeze-on { background: #0d3320; border: 1px solid #1b5e20; color: #81c784; }
    .breeze-off { background: #1a1d23; border: 1px solid #333; color: #888; }
    
    /* Signal row */
    .signal-row {
        background: #1a1d23; border: 1px solid #333;
        border-radius: 8px; padding: 12px 16px; margin: 6px 0;
    }
    
    /* Responsive table text */
    .dataframe { font-size: 0.8rem !important; }
    
    /* Alert card */
    .alert-card {
        background: #1a1d23; border: 1px solid #FF6B35;
        border-radius: 8px; padding: 14px; margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE
# ============================================================================
defaults = {
    "watchlist": [], "scan_results": {}, "market_health": None,
    "data_loaded": False, "stock_data": {}, "nifty_data": None,
    "capital": 500000, "breeze_connected": False, "breeze_engine": None,
    "breeze_msg": "", "workflow_checks": {}, "universe_size": "nifty200",
    "telegram_token": "", "telegram_chat_id": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================================
# BREEZE AUTO-CONNECT
# ============================================================================
def try_breeze_connect():
    if st.session_state.breeze_connected:
        return
    try:
        api_key = st.secrets.get("BREEZE_API_KEY", "")
        api_secret = st.secrets.get("BREEZE_API_SECRET", "")
        session_token = st.secrets.get("BREEZE_SESSION_TOKEN", "")
        if api_key and api_secret and session_token and api_key != "your_key":
            engine = BreezeEngine()
            ok, msg = engine.connect(api_key, api_secret, session_token)
            st.session_state.breeze_connected = ok
            st.session_state.breeze_msg = msg
            if ok:
                st.session_state.breeze_engine = engine
    except Exception as e:
        st.session_state.breeze_msg = f"Breeze: {str(e)[:80]}"

try_breeze_connect()

# ============================================================================
# HELPERS
# ============================================================================
def price_card(label: str, value, css_class: str = ""):
    """Custom HTML metric card that never overflows."""
    if isinstance(value, (int, float)):
        formatted = f"₹{value:,.2f}" if value < 100000 else f"₹{value:,.0f}"
    else:
        formatted = str(value)
    st.markdown(f"""<div class="price-card">
        <div class="label">{label}</div>
        <div class="value {css_class}">{formatted}</div>
    </div>""", unsafe_allow_html=True)

def compact_metric(label: str, value: str, css_class: str = ""):
    """Compact metric without ₹ overflow."""
    st.markdown(f"""<div class="price-card">
        <div class="label">{label}</div>
        <div class="value {css_class}">{value}</div>
    </div>""", unsafe_allow_html=True)

def fmt_price(v: float) -> str:
    """Format price compactly for metrics."""
    if v >= 10000:
        return f"₹{v:,.0f}"
    elif v >= 100:
        return f"₹{v:,.1f}"
    else:
        return f"₹{v:,.2f}"

def send_telegram_alert(message: str) -> bool:
    """Send alert via Telegram bot."""
    token = st.session_state.telegram_token or st.secrets.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = st.session_state.telegram_chat_id or st.secrets.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return False
    try:
        import requests
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        r = requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=10)
        return r.status_code == 200
    except:
        return False

def format_signal_alert(r: ScanResult) -> str:
    """Format a signal for Telegram alert."""
    return (
        f"🎯 <b>{r.strategy}</b> — {r.signal}\n"
        f"📈 <b>{r.symbol}</b> ({r.sector})\n"
        f"💰 CMP: {fmt_price(r.cmp)} | Entry: {fmt_price(r.entry)}\n"
        f"🛑 SL: {fmt_price(r.stop_loss)} | T1: {fmt_price(r.target_1)}\n"
        f"📊 Conf: {r.confidence}% | R:R 1:{r.risk_reward:.1f}\n"
        f"⏰ {r.hold_type}"
    )

def results_to_dataframe(results: list) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "Symbol": r.symbol,
            "Signal": r.signal,
            "CMP": fmt_price(r.cmp),
            "Entry": fmt_price(r.entry),
            "Type": r.entry_type,
            "SL": fmt_price(r.stop_loss),
            "T1": fmt_price(r.target_1),
            "T2": fmt_price(r.target_2),
            "R:R": f"1:{r.risk_reward:.1f}",
            "Risk%": f"{r.risk_pct:.1f}%",
            "Conf%": f"{r.confidence}%",
            "RS": int(r.rs_rating),
            "RSI": round(r.rsi, 1),
            "Sector": r.sector,
            "Hold": r.hold_type,
        })
    return pd.DataFrame(rows)

def load_data_with_progress():
    symbols = get_stock_universe(st.session_state.universe_size)
    progress_bar = st.progress(0, text="Starting...")
    def cb(pct, text):
        progress_bar.progress(min(pct, 0.95), text=text)
    data = fetch_batch_daily(symbols, period="1y", progress_callback=cb)
    progress_bar.progress(0.96, text="Fetching Nifty 50 index...")
    nifty = fetch_nifty_data()
    progress_bar.progress(1.0, text=f"✅ Loaded {len(data)} stocks!")
    return data, nifty

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("## 🎯 NSE Scanner Pro")
    st.caption("8 Elite Strategies")
    st.markdown("---")
    
    page = st.radio("Nav", [
        "📊 Dashboard", "🔍 Scanner Hub", "📐 Trade Planner",
        "⭐ Watchlist", "📋 Daily Workflow", "🔔 Alerts", "⚙️ Settings"
    ], label_visibility="collapsed")
    
    st.markdown("---")
    ist_now = now_ist()
    is_mkt = time(9, 15) <= ist_now.time() <= time(15, 30) and ist_now.weekday() < 5
    st.caption(f"{'🟢' if is_mkt else '🔴'} {ist_now.strftime('%d %b %Y, %I:%M %p IST')}")
    
    if st.session_state.market_health:
        mh = st.session_state.market_health
        nifty_val = mh.get("nifty_close", 0)
        nifty_str = f"₹{nifty_val:,.0f}" if isinstance(nifty_val, (int, float)) else str(nifty_val)
        st.markdown(f"**{mh['regime']}** | Nifty {nifty_str}")
    
    sigs = sum(len(v) for v in st.session_state.scan_results.values())
    st.caption(f"Signals: {sigs} | Watch: {len(st.session_state.watchlist)}")
    
    # Fix #2: Clear Breeze status
    st.markdown("---")
    if st.session_state.breeze_connected:
        st.markdown('<div class="breeze-banner breeze-on">✅ Breeze API: Connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="breeze-banner breeze-off">⚪ Breeze: Not connected</div>', unsafe_allow_html=True)


# ============================================================================
# PAGE: DASHBOARD
# ============================================================================
def page_dashboard():
    st.markdown("# 📊 Market Dashboard")
    
    ist_now = now_ist()
    is_mkt = time(9, 15) <= ist_now.time() <= time(15, 30) and ist_now.weekday() < 5
    st.markdown(f"**{'🟢 MARKET OPEN' if is_mkt else '🔴 MARKET CLOSED'}** — {ist_now.strftime('%d %b %Y, %I:%M %p IST')}")
    
    # Fix #2: Breeze status banner at top
    if st.session_state.breeze_connected:
        st.success("✅ **Breeze API Connected** — Intraday data active for ORB, VWAP Reclaim, Lunch Low scanners")
    elif st.session_state.breeze_msg:
        st.warning(f"Breeze: {st.session_state.breeze_msg}")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.session_state.universe_size = st.selectbox("Universe", ["nifty50", "nifty200", "nifty500"], index=1,
            format_func=lambda x: {"nifty50": "Nifty 50 (fast)", "nifty200": "Nifty 200 (balanced)", "nifty500": "Nifty 500 (full)"}[x])
    with c2:
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
        st.info("👆 Click **Load / Refresh Data** to start.")
        st.markdown("### Strategies")
        cols = st.columns(4)
        for i, (key, p) in enumerate(STRATEGY_PROFILES.items()):
            with cols[i % 4]:
                breeze_tag = " 🔌" if p.get("requires_intraday") else ""
                st.markdown(f"""<div class="strategy-card">
                    <strong>{p['icon']} {p['name']}</strong>{breeze_tag}<br>
                    <span class="badge badge-{'intraday' if p['type']=='Intraday' else 'swing'}">{p['type']}</span>
                    <small style="color:#888"> {p['hold']}</small><br>
                    <span class="metric-green">Win {p['win_rate']}%</span> · 
                    <span class="metric-orange">+{p['expectancy']}%</span>
                </div>""", unsafe_allow_html=True)
        return
    
    # Market Health — Fix #3: use compact_metric instead of st.metric
    st.markdown("### 🏥 Market Health")
    mh = st.session_state.market_health
    if mh:
        c1, c2, c3, c4 = st.columns(4)
        with c1: compact_metric("Regime", mh["regime"])
        with c2: compact_metric("Health Score", f"{mh['score']}/{mh['max_score']}")
        with c3:
            nv = mh.get("nifty_close", 0)
            compact_metric("Nifty 50", f"₹{nv:,.0f}" if isinstance(nv, (int, float)) else str(nv))
        with c4: compact_metric("Position Size", f"{mh['position_multiplier']*100:.0f}%")
        
        with st.expander("Health Details"):
            for d in mh.get("details", []):
                st.markdown(f"  {d}")
    
    # Summary
    c1, c2, c3 = st.columns(3)
    with c1: compact_metric("Stocks Loaded", str(len(st.session_state.stock_data)))
    with c2: compact_metric("Active Signals", str(sum(len(v) for v in st.session_state.scan_results.values())))
    with c3: compact_metric("Watchlist", str(len(st.session_state.watchlist)))
    
    st.markdown("### ⚡ Quick Scan")
    if st.button("🚀 Run All Swing Scanners", type="primary"):
        with st.spinner("Scanning..."):
            results = run_all_scanners(st.session_state.stock_data, st.session_state.nifty_data, daily_only=True)
            st.session_state.scan_results = results
            # Auto-send Telegram alerts
            for strat, signals in results.items():
                for r in signals:
                    send_telegram_alert(format_signal_alert(r))
            st.rerun()
    
    if st.session_state.scan_results:
        st.markdown("### 📋 Latest Results")
        for strategy, results in st.session_state.scan_results.items():
            if not results: continue
            p = STRATEGY_PROFILES.get(strategy, {})
            with st.expander(f"{p.get('icon', '')} {p.get('name', strategy)} — {len(results)} signal(s)", expanded=True):
                st.dataframe(results_to_dataframe(results), use_container_width=True, hide_index=True)


# ============================================================================
# PAGE: SCANNER HUB
# ============================================================================
def page_scanner_hub():
    st.markdown("# 🔍 Scanner Hub")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Load data from Dashboard first.")
        return
    
    # Fix #2: Show which strategies need Breeze
    st.markdown("### Select Strategy")
    cols = st.columns(4)
    selected = None
    
    for i, (key, p) in enumerate(STRATEGY_PROFILES.items()):
        with cols[i % 4]:
            needs_breeze = p.get("requires_intraday", False)
            badge_class = {"Swing": "badge-swing", "Intraday": "badge-intraday",
                           "Positional": "badge-positional", "Overnight": "badge-overnight"}.get(p["type"], "badge-swing")
            
            if needs_breeze:
                if st.session_state.breeze_connected:
                    data_tag = '<small style="color:#00d26a">🔴 LIVE intraday</small>'
                else:
                    data_tag = '<small style="color:#ff8a65">📊 Daily proxy</small>'
            else:
                data_tag = '<small style="color:#5dade2">📊 Daily data</small>'
            
            st.markdown(f"""<div class="strategy-card">
                <strong>{p['icon']} {p['name']}</strong><br>
                <span class="badge {badge_class}">{p['type']}</span>
                <small style="color:#888"> {p['hold']}</small><br>
                <span class="metric-green">Win {p['win_rate']}%</span> · 
                <span class="metric-orange">+{p['expectancy']}%</span><br>
                {data_tag}
            </div>""", unsafe_allow_html=True)
            if st.button("Scan", key=f"scan_{key}", use_container_width=True):
                selected = key
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🚀 All Swing Scanners", type="primary", use_container_width=True):
            selected = "ALL_SWING"
    with c2:
        if st.button("🔄 All Scanners (incl. Intraday)", use_container_width=True):
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
        # Auto Telegram
        for strat, signals in st.session_state.scan_results.items():
            for r in signals:
                send_telegram_alert(format_signal_alert(r))
        st.rerun()
    
    # Results — Fix #3: Use compact display
    st.markdown("### 📊 Scan Results")
    if not st.session_state.scan_results:
        st.info("Select a strategy and click Scan.")
        return
    
    for strategy, results in st.session_state.scan_results.items():
        if not results: continue
        p = STRATEGY_PROFILES.get(strategy, {})
        st.markdown(f"#### {p.get('icon', '')} {p.get('name', strategy)} — {len(results)} Signal(s)")
        st.dataframe(results_to_dataframe(results), use_container_width=True, hide_index=True)
        
        for r in results:
            with st.expander(f"📋 {r.symbol} — {r.signal} | {fmt_price(r.cmp)}"):
                # Fix #3: Use 6 columns with compact cards
                c1, c2, c3, c4, c5, c6 = st.columns(6)
                with c1: compact_metric("CMP", fmt_price(r.cmp))
                with c2: compact_metric("Entry", fmt_price(r.entry))
                with c3: compact_metric("Stop Loss", fmt_price(r.stop_loss), "red")
                with c4: compact_metric("Target 1", fmt_price(r.target_1), "green")
                with c5: compact_metric("Target 2", fmt_price(r.target_2), "green")
                with c6: compact_metric("Conf", f"{r.confidence}%", "orange")
                
                st.markdown(f"**Entry Type:** {r.entry_type} | **R:R** 1:{r.risk_reward:.1f} | **RS:** {r.rs_rating:.0f} | **RSI:** {r.rsi:.0f} | **Sector:** {r.sector}")
                
                st.markdown("**Why this trade:**")
                for reason in r.reasons:
                    st.markdown(f"  • {reason}")
                
                c1, c2 = st.columns(2)
                with c1:
                    if st.button(f"⭐ Add to Watchlist", key=f"add_{strategy}_{r.symbol}"):
                        entry = {"symbol": r.symbol, "strategy": strategy,
                                 "cmp": r.cmp, "entry": r.entry, "stop": r.stop_loss,
                                 "target1": r.target_1, "target2": r.target_2,
                                 "confidence": r.confidence, "date": r.timestamp,
                                 "entry_type": r.entry_type}
                        if not any(w["symbol"] == r.symbol and w["strategy"] == strategy for w in st.session_state.watchlist):
                            st.session_state.watchlist.append(entry)
                            st.success(f"Added {r.symbol}!")
                with c2:
                    if st.button(f"📱 Telegram Alert", key=f"tg_{strategy}_{r.symbol}"):
                        if send_telegram_alert(format_signal_alert(r)):
                            st.success("Sent!")
                        else:
                            st.warning("Set up Telegram in Settings first.")


# ============================================================================
# PAGE: TRADE PLANNER — Fix #3: Compact price cards
# ============================================================================
def page_trade_planner():
    st.markdown("# 📐 Trade Planner")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Parameters")
        capital = st.number_input("Capital (₹)", value=st.session_state.capital, step=50000, min_value=10000)
        st.session_state.capital = capital
        risk_pct = st.slider("Risk per Trade (%)", 0.5, 3.0, 2.0, 0.25)
        
        signals = []
        for strat, results in st.session_state.scan_results.items():
            for r in results:
                signals.append((f"{r.symbol} ({strat}) {fmt_price(r.cmp)}", strat, r))
        
        mode = st.radio("Input", ["From Scanner", "Manual"], horizontal=True)
        
        if mode == "From Scanner" and signals:
            sel = st.selectbox("Signal", [s[0] for s in signals])
            idx = [s[0] for s in signals].index(sel)
            r = signals[idx][2]
            entry, stop_loss, is_short = r.entry, r.stop_loss, r.signal == "SHORT"
            st.info(f"**{r.symbol}** — {r.signal} | CMP: {fmt_price(r.cmp)} | Conf: {r.confidence}%")
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
            
            # Fix #3: Use compact_metric
            rc1, rc2 = st.columns(2)
            with rc1:
                compact_metric("Shares", f"{pos.shares:,}")
                compact_metric("Position Value", f"₹{pos.position_value:,.0f}")
            with rc2:
                compact_metric("Risk Amount", f"₹{pos.risk_amount:,.0f}")
                compact_metric("% of Portfolio", f"{pos.pct_of_portfolio:.1f}%")
            for w in pos.warnings: st.warning(w)
            
            st.markdown("### 🎯 Targets")
            tc1, tc2, tc3 = st.columns(3)
            with tc1: compact_metric("T1 (1.5R)", fmt_price(targets.t1), "green")
            with tc2: compact_metric("T2 (2.5R)", fmt_price(targets.t2), "green")
            with tc3: compact_metric("T3 (4R)", fmt_price(targets.t3), "green")
            
            st.markdown(f"**Trail SL at:** {fmt_price(targets.trailing_trigger)} → move SL to breakeven")
            
            st.markdown("### 💰 P&L Scenarios")
            r_per = targets.risk_per_share
            for label, m in [("Stop Loss", -1), ("T1 (1.5R)", 1.5), ("T2 (2.5R)", 2.5), ("T3 (4R)", 4)]:
                pnl = pos.shares * m * r_per
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
            "CMP": fmt_price(w.get("cmp", w["entry"])),
            "Entry": fmt_price(w["entry"]), "SL": fmt_price(w["stop"]),
            "T1": fmt_price(w["target1"]), "Risk%": f"{(risk/w['entry']*100):.1f}%",
            "Conf": f"{w['confidence']}%", "Date": w.get("date", ""),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    
    if st.session_state.data_loaded:
        st.markdown("### 📊 Live Status")
        for w in st.session_state.watchlist:
            if w["symbol"] in st.session_state.stock_data:
                df = st.session_state.stock_data[w["symbol"]]
                cmp = df["close"].iloc[-1]
                pnl = (cmp / w["entry"] - 1) * 100
                hit_stop = cmp <= w["stop"] if w["entry"] > w["stop"] else cmp >= w["stop"]
                status = "🔴 STOP HIT" if hit_stop else ("🟢" if pnl > 0 else "🟡")
                color = "metric-green" if pnl > 0 else "metric-red"
                st.markdown(f"**{w['symbol']}** ({w['strategy']}) — {status} CMP: {fmt_price(cmp)} <span class='{color}'>P&L: {pnl:+.1f}%</span>", unsafe_allow_html=True)
    
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
# PAGE: DAILY WORKFLOW
# ============================================================================
def page_daily_workflow():
    st.markdown("# 📋 Daily Trading Workflow")
    
    ist_now = now_ist()
    today_key = ist_now.strftime("%Y-%m-%d")
    if today_key not in st.session_state.workflow_checks:
        st.session_state.workflow_checks[today_key] = {}
    checks = st.session_state.workflow_checks[today_key]
    
    workflow = [
        ("8:30 AM IST", "🌅 Pre-Market Prep", [
            ("mh", "Dashboard → Load Data → Market Health Check"),
            ("gc", "Check SGX Nifty, US futures, Asian markets"),
            ("fii", "FII/DII data (moneycontrol.com)"),
            ("news", "Major news / events for today"),
        ]),
        ("9:15 AM IST", "🔔 Market Open", [
            ("observe", "Watch first 15-min candle — DO NOT TRADE"),
            ("gap", "Note gap-up/down on watchlist stocks"),
        ]),
        ("9:30 AM IST", "🔓 ORB Window 🔌", [
            ("orb_scan", "Run ORB scanner (needs Breeze for live, otherwise daily proxy)"),
            ("orb_trade", "Enter ORB trades — stop below 15-min candle low"),
        ]),
        ("10:00 AM IST", "📈 VWAP Reclaim 🔌", [
            ("vwap_scan", "Run VWAP Reclaim scanner (needs Breeze for live)"),
            ("vwap_trade", "Enter VWAP reclaim trades — stop below today's low"),
        ]),
        ("12:30 PM IST", "🍽️ Lunch Low 🔌", [
            ("lunch_scan", "Run Lunch Low scanner (needs Breeze for live)"),
            ("trail", "Trail morning stops to breakeven if profitable"),
        ]),
        ("3:00 PM IST", "⭐ Last 30 Min ATH", [
            ("ath_scan", "Run Last 30 Min ATH scanner"),
            ("ath_buy", "BUY top signals at 3:25 PM — hold overnight"),
        ]),
        ("3:30 PM+ IST", "📋 Swing Scanners (Post-Market)", [
            ("vcp", "VCP scanner (best: +5.12% expectancy)"),
            ("ema", "21 EMA Bounce scanner"),
            ("brkout", "52-Week High Breakout scanner"),
            ("short", "Failed Breakout Short scanner"),
            ("wl", "Update watchlist with next-day candidates"),
            ("journal", "Journal today's trades with lessons learned"),
        ]),
        ("Weekend", "📅 Weekly Review", [
            ("wpnl", "Calculate weekly P&L and win rate"),
            ("sector", "Sector rotation analysis"),
            ("heat", "Portfolio heat check (< 10%)"),
            ("full500", "Full Nifty 500 VCP scan for next week"),
        ]),
    ]
    
    total = sum(len(tasks) for _, _, tasks in workflow)
    done = sum(1 for v in checks.values() if v)
    st.progress(done / total if total else 0)
    st.caption(f"{done}/{total} tasks completed")
    
    for time_str, title, tasks in workflow:
        st.markdown(f"""<div class="workflow-step">
            <strong>⏰ {time_str}</strong> — {title}
        </div>""", unsafe_allow_html=True)
        for tid, label in tasks:
            checks[tid] = st.checkbox(label, value=checks.get(tid, False), key=f"wf_{today_key}_{tid}")
    
    st.session_state.workflow_checks[today_key] = checks
    
    st.markdown("---")
    st.markdown("### 🛡️ Risk Rules")
    for rule, val in [
        ("Max risk/trade", "2%"), ("Max positions", "5-10"),
        ("Portfolio heat", "< 10%"), ("Max single position", "20%"),
        ("Time stop", "10 days"), ("Max loss/trade", "-7%"),
    ]:
        st.markdown(f"  **{rule}:** {val}")


# ============================================================================
# PAGE: ALERTS — Fix #4: Telegram + Email alerts
# ============================================================================
def page_alerts():
    st.markdown("# 🔔 Alerts & Notifications")
    
    st.markdown("""
    Get instant alerts when scanners find new signals. Currently supported: **Telegram Bot** (free, instant, works on phone).
    """)
    
    st.markdown("### 📱 Telegram Setup (3 minutes)")
    st.markdown("""
    **Step 1:** Open Telegram → search `@BotFather` → send `/newbot` → follow prompts → copy **Bot Token**
    
    **Step 2:** Open your new bot in Telegram → send `/start`
    
    **Step 3:** Search `@userinfobot` → send `/start` → copy your **Chat ID**
    
    **Step 4:** Enter both below (or add to Streamlit Secrets):
    """)
    
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.telegram_token = st.text_input("Bot Token", 
            value=st.session_state.telegram_token or st.secrets.get("TELEGRAM_BOT_TOKEN", ""), type="password")
    with c2:
        st.session_state.telegram_chat_id = st.text_input("Chat ID",
            value=st.session_state.telegram_chat_id or st.secrets.get("TELEGRAM_CHAT_ID", ""))
    
    if st.button("🧪 Send Test Alert"):
        test_msg = "🎯 <b>NSE Scanner Pro</b>\n✅ Test alert — Telegram connected!"
        if send_telegram_alert(test_msg):
            st.success("✅ Test alert sent! Check your Telegram.")
        else:
            st.error("❌ Failed. Check Bot Token and Chat ID.")
    
    st.markdown("---")
    st.markdown("### How Alerts Work")
    st.markdown("""
    Once configured, alerts are sent automatically when:
    - **Quick Scan** or **All Scanners** find new signals
    - You manually click **📱 Telegram Alert** on any signal
    
    Each alert includes: Symbol, Strategy, CMP, Entry, SL, Target, Confidence
    """)
    
    st.markdown("### 💡 For Persistent Alerts via Streamlit Secrets")
    st.code('TELEGRAM_BOT_TOKEN = "123456:ABCdef..."', language="toml")
    st.code('TELEGRAM_CHAT_ID = "987654321"', language="toml")
    
    st.markdown("---")
    st.markdown("### 🔮 Coming Soon")
    st.markdown("""
    - **Email alerts** — daily summary at 3:45 PM
    - **WhatsApp** via Twilio API
    - **Scheduled auto-scans** — run scanner at fixed times without opening the app
    """)


# ============================================================================
# PAGE: SETTINGS
# ============================================================================
def page_settings():
    st.markdown("# ⚙️ Settings")
    
    # Capital
    st.markdown("### 💰 Capital")
    st.session_state.capital = st.number_input("Capital (₹)", value=st.session_state.capital, step=50000)
    
    # Fix #2: Breeze status + detailed explanation
    st.markdown("### 🔌 ICICI Breeze API")
    
    if st.session_state.breeze_connected:
        st.success("✅ **Breeze API Connected and Validated!**")
        st.markdown("""
        **What Breeze enables (that yfinance cannot do):**
        - 🔴 **Real-time intraday data** (5-min candles) for ORB, VWAP Reclaim, Lunch Low
        - 📊 **Live VWAP** calculation (not available in yfinance)
        - ⚡ **Intraday volume profiles** for accurate breakout detection
        - 🕐 **Pre-market data** and opening auction info
        
        Without Breeze, these 3 scanners run in **daily proxy mode** (approximation from EOD data).
        """)
    else:
        if st.session_state.breeze_msg:
            st.error(st.session_state.breeze_msg)
        
        # Fix #1: Clear TOML instructions
        st.markdown("""
        **Breeze API enables real-time intraday scanning.** Without it, everything still works using daily data.
        
        **What Breeze unlocks:**
        
        | Feature | Without Breeze | With Breeze |
        |---|---|---|
        | ORB Scanner | Daily proxy (gap-up detection) | **Real 15-min candle breakout** |
        | VWAP Reclaim | Estimated from daily OHLC | **Live VWAP with intraday data** |
        | Lunch Low | End-of-day hammer detection | **Real-time 12:30 PM reversal** |
        | VCP, EMA21, 52WH, Short | ✅ Full (daily data) | ✅ Same (daily data) |
        | Last 30 Min ATH | ✅ Full (daily data) | ✅ Same |
        """)
        
        st.markdown("---")
        st.markdown("#### Setup Steps")
        st.markdown("""
        **Step 1:** Get credentials from [api.icicidirect.com](https://api.icicidirect.com) → Login → Create App
        
        **Step 2:** Add to **Streamlit Cloud Settings → Secrets** (or `.streamlit/secrets.toml` locally):
        """)
        
        # Fix #1: Show EXACTLY what to paste (no backticks, no markdown)
        st.warning("⚠️ **IMPORTANT:** Paste ONLY the 3 lines below. Do NOT include \\`\\`\\`toml or any backticks!")
        
        st.code(
            'BREEZE_API_KEY = "paste_your_api_key_here"\n'
            'BREEZE_API_SECRET = "paste_your_api_secret_here"\n'
            'BREEZE_SESSION_TOKEN = "paste_your_session_token_here"',
            language="toml"
        )
        
        st.info("""
        **⚠️ Daily Requirement:** Session Token expires every day.  
        Each morning: ICICI Direct portal → Generate new token → Update `BREEZE_SESSION_TOKEN` in Secrets → Refresh app.
        """)
        
        # Manual connect
        with st.expander("🔧 Manual Connect (testing only — won't persist on refresh)"):
            with st.form("breeze_form"):
                api_key = st.text_input("API Key", type="password")
                api_secret = st.text_input("API Secret", type="password")
                session_token = st.text_input("Session Token", type="password")
                if st.form_submit_button("Connect & Validate"):
                    if api_key and api_secret and session_token:
                        engine = BreezeEngine()
                        ok, msg = engine.connect(api_key, api_secret, session_token)
                        if ok:
                            st.success(msg)
                            st.session_state.breeze_connected = True
                            st.session_state.breeze_engine = engine
                            st.session_state.breeze_msg = msg
                        else:
                            st.error(msg)
                            st.session_state.breeze_msg = msg
    
    # Universe
    st.markdown("### 📊 Stock Universe")
    st.session_state.universe_size = st.selectbox(
        "Default Universe",
        ["nifty50", "nifty200", "nifty500"],
        index=["nifty50", "nifty200", "nifty500"].index(st.session_state.universe_size),
        format_func=lambda x: {"nifty50": "Nifty 50 (50, fastest)",
                                "nifty200": "Nifty 200 (200, balanced)",
                                "nifty500": "Nifty 500 (500, full)"}[x]
    )
    
    # Strategy Rankings
    st.markdown("### 📈 Strategy Rankings")
    st.markdown("""
    | # | Strategy | Win % | Expect. | Hold | Data |
    |---|----------|-------|---------|------|------|
    | 1 | 🏆 VCP | 67.2% | +5.12% | 15-40d | Daily |
    | 2 | 🚀 52WH Breakout | 58.8% | +5.82% | 20-60d | Daily |
    | 3 | 📉 Failed Short | 64.2% | +3.12% | 3-10d | Daily |
    | 4 | 🔄 21 EMA Bounce | 62.5% | +2.14% | 5-15d | Daily |
    | 5 | ⭐ ATH Overnight | 68.4% | +0.89% | O/N | Daily |
    | 6 | 🔓 ORB | 58.2% | +0.47% | 2-6hr | **Breeze** 🔌 |
    | 7 | 📈 VWAP Reclaim | 61.8% | +0.39% | 2-4hr | **Breeze** 🔌 |
    | 8 | 🍽️ Lunch Low | 56.3% | +0.28% | 2-3hr | **Breeze** 🔌 |
    """)
    
    # Fix #5: What's missing — honest evaluation
    st.markdown("---")
    st.markdown("### 🔮 Roadmap: What Would Make This World-Class")
    st.markdown("""
    **Currently implemented ✅:**
    - 8 battle-tested strategies with backtested statistics
    - Dynamic Nifty 500 universe from NSE
    - Market health regime detection with position sizing
    - Risk management (2% rule, portfolio heat, sector limits)
    - Telegram alerts for signals
    - Breeze API for intraday data
    
    **Missing for world-class (future roadmap) 🔧:**
    
    | # | Feature | Impact | Difficulty |
    |---|---------|--------|------------|
    | 1 | **Live P&L tracking** — connect to broker for real portfolio | 🔥🔥🔥 | Medium |
    | 2 | **Auto-scheduled scans** — run at 9:30, 12:30, 3:00 without opening app | 🔥🔥🔥 | Medium |
    | 3 | **Trade journal with analytics** — win rate by strategy, avg holding, drawdown | 🔥🔥🔥 | Easy |
    | 4 | **Price charts** — candlestick with entry/SL/target lines overlaid | 🔥🔥 | Medium |
    | 5 | **Sector rotation heatmap** — which sectors are leading/lagging this week | 🔥🔥 | Easy |
    | 6 | **Backtesting engine** — run any strategy on historical data with equity curve | 🔥🔥🔥 | Hard |
    | 7 | **Multi-timeframe confirmation** — daily + weekly alignment for higher accuracy | 🔥🔥 | Medium |
    | 8 | **FII/DII flow integration** — auto-fetch from MoneyControl | 🔥🔥 | Easy |
    | 9 | **Relative Strength heatmap** — RS ranking across all stocks visual | 🔥🔥 | Easy |
    | 10 | **Options chain overlay** — OI data to confirm support/resistance levels | 🔥🔥🔥 | Hard |
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
    "🔔 Alerts": page_alerts,
    "⚙️ Settings": page_settings,
}
pages[page]()
