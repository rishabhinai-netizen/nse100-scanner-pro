"""
╔══════════════════════════════════════════════════════════════╗
║           NSE SCANNER PRO — Elite Stock Scanner              ║
║    8 Battle-Tested Strategies | Real-Time Analysis           ║
║    Built for Rishabh | TradeAudit Pro                        ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import json
import os
import time as time_module

# Local imports
from stock_universe import NIFTY_100, NIFTY_50, NIFTY_NEXT_50, get_sector
from data_engine import (
    fetch_daily_data, fetch_batch_daily, fetch_nifty_data, 
    Indicators, BreezeEngine
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
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    /* Global */
    .stApp { }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1d23, #252830);
        border: 1px solid #333;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 4px 0;
    }
    .metric-label { color: #888; font-size: 0.8em; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 1.8em; font-weight: 700; color: #fafafa; }
    .metric-green { color: #00d26a !important; }
    .metric-red { color: #ff4757 !important; }
    .metric-orange { color: #FF6B35 !important; }
    
    /* Strategy cards */
    .strategy-card {
        background: #1a1d23;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
        transition: border-color 0.2s;
    }
    .strategy-card:hover { border-color: #FF6B35; }
    .strategy-name { font-size: 1.2em; font-weight: 600; }
    .strategy-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75em;
        font-weight: 600;
    }
    .badge-swing { background: #1e3a5f; color: #5dade2; }
    .badge-intraday { background: #3e2723; color: #ff8a65; }
    .badge-positional { background: #1b5e20; color: #81c784; }
    .badge-overnight { background: #4a148c; color: #ce93d8; }
    
    /* Result rows */
    .result-buy { border-left: 4px solid #00d26a; padding-left: 12px; }
    .result-short { border-left: 4px solid #ff4757; padding-left: 12px; }
    
    /* Confidence bar */
    .conf-bar { height: 6px; border-radius: 3px; background: #333; }
    .conf-fill { height: 6px; border-radius: 3px; }
    
    /* Workflow steps */
    .workflow-step {
        border-left: 3px solid #FF6B35;
        padding: 8px 16px;
        margin: 8px 0;
        background: #1a1d23;
        border-radius: 0 8px 8px 0;
    }
    
    /* Hide streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1d23, #252830);
        border: 1px solid #333;
        border-radius: 10px;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INIT
# ============================================================================
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
if "scan_results" not in st.session_state:
    st.session_state.scan_results = {}
if "market_health" not in st.session_state:
    st.session_state.market_health = None
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
if "stock_data" not in st.session_state:
    st.session_state.stock_data = {}
if "capital" not in st.session_state:
    st.session_state.capital = 500000
if "breeze_connected" not in st.session_state:
    st.session_state.breeze_connected = False
if "workflow_checks" not in st.session_state:
    st.session_state.workflow_checks = {}

# ============================================================================
# SIDEBAR — Navigation + Settings
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
    
    # Quick stats
    if st.session_state.market_health:
        mh = st.session_state.market_health
        st.markdown(f"**Market:** {mh['regime']}")
        st.markdown(f"**Nifty:** ₹{mh.get('nifty_close', 'N/A')}")
    
    total_signals = sum(len(v) for v in st.session_state.scan_results.values())
    st.markdown(f"**Active Signals:** {total_signals}")
    st.markdown(f"**Watchlist:** {len(st.session_state.watchlist)} stocks")
    
    st.markdown("---")
    st.caption("Data: yfinance (daily) • Breeze API (intraday)")
    breeze_status = "🟢 Connected" if st.session_state.breeze_connected else "⚪ Not connected"
    st.caption(f"Breeze: {breeze_status}")


# ============================================================================
# DATA LOADING
# ============================================================================
@st.cache_data(ttl=300, show_spinner=False)
def load_all_data(universe: str = "nifty100"):
    """Load data for the selected universe."""
    symbols = NIFTY_100 if universe == "nifty100" else NIFTY_50
    
    data = {}
    # Batch download with yfinance
    yf_symbols = [f"{s}.NS" for s in symbols]
    
    try:
        import yfinance as yf
        raw = yf.download(yf_symbols, period="1y", auto_adjust=True, 
                          threads=True, progress=False)
        
        if raw is not None and not raw.empty:
            for symbol, yf_sym in zip(symbols, yf_symbols):
                try:
                    if len(symbols) == 1:
                        df = raw.copy()
                    else:
                        # Handle multi-level columns from yfinance
                        if isinstance(raw.columns, pd.MultiIndex):
                            df = raw.xs(yf_sym, level=1, axis=1).copy()
                        else:
                            df = raw.copy()
                    
                    if df is not None and not df.empty:
                        df = df.dropna(how="all")
                        if len(df) >= 50:
                            df.columns = [c.lower() for c in df.columns]
                            if all(c in df.columns for c in ["open", "high", "low", "close", "volume"]):
                                df = df[["open", "high", "low", "close", "volume"]]
                                df["symbol"] = symbol
                                data[symbol] = df
                except Exception:
                    continue
    except Exception:
        # Fallback to individual downloads
        for symbol in symbols[:20]:  # Limit for speed
            df = fetch_daily_data(symbol)
            if df is not None:
                data[symbol] = df
    
    return data


def load_data_with_progress():
    """Load data with a progress indicator."""
    with st.spinner("📡 Fetching data for Nifty 100 stocks..."):
        data = load_all_data("nifty100")
        nifty = fetch_nifty_data()
    return data, nifty


# ============================================================================
# PAGE: DASHBOARD
# ============================================================================
def page_dashboard():
    st.markdown("# 📊 Market Dashboard")
    
    col_load, col_time = st.columns([1, 3])
    with col_load:
        if st.button("🔄 Load / Refresh Data", type="primary", use_container_width=True):
            st.cache_data.clear()
            data, nifty = load_data_with_progress()
            st.session_state.stock_data = data
            st.session_state.data_loaded = True
            
            # Auto-run market health
            if nifty is not None:
                st.session_state.market_health = check_market_health(nifty)
                st.session_state.nifty_data = nifty
            
            st.rerun()
    
    with col_time:
        now = datetime.now()
        market_open = time(9, 15)
        market_close = time(15, 30)
        is_market_hours = market_open <= now.time() <= market_close and now.weekday() < 5
        status = "🟢 MARKET OPEN" if is_market_hours else "🔴 MARKET CLOSED"
        st.markdown(f"**{status}** — {now.strftime('%d %b %Y, %I:%M %p IST')}")
    
    if not st.session_state.data_loaded:
        st.info("👆 Click **Load / Refresh Data** to start. This fetches 1 year of daily data for Nifty 100 stocks.")
        
        # Show strategy overview while waiting
        st.markdown("### Strategy Overview")
        cols = st.columns(4)
        for i, (key, profile) in enumerate(STRATEGY_PROFILES.items()):
            with cols[i % 4]:
                st.markdown(f"""
                <div class="strategy-card">
                    <div class="strategy-name">{profile['icon']} {profile['name']}</div>
                    <small style="color:#888">{profile['type']} • {profile['hold']}</small><br>
                    <span class="metric-green">Win: {profile['win_rate']}%</span> • 
                    <span class="metric-orange">PF: {profile['profit_factor']}</span>
                </div>
                """, unsafe_allow_html=True)
        return
    
    # Market Health
    st.markdown("### 🏥 Market Health Check")
    mh = st.session_state.market_health
    
    if mh:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Market Regime", mh["regime"])
        with c2:
            st.metric("Health Score", f"{mh['score']}/{mh['max_score']}")
        with c3:
            st.metric("Nifty Close", f"₹{mh.get('nifty_close', 'N/A')}")
        with c4:
            st.metric("Position Sizing", f"{mh['position_multiplier']*100:.0f}%",
                      help="Multiply your standard position size by this")
        
        with st.expander("Health Check Details", expanded=False):
            for detail in mh.get("details", []):
                st.markdown(f"  {detail}")
            
            if mh["position_multiplier"] < 0.5:
                st.warning("⚠️ Bearish regime — only trade VCP and 21 EMA Bounce with 50% reduced size")
    
    # Data summary
    st.markdown("### 📊 Data Summary")
    loaded = len(st.session_state.stock_data)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Stocks Loaded", f"{loaded}/100")
    with c2:
        total_signals = sum(len(v) for v in st.session_state.scan_results.values())
        st.metric("Active Signals", total_signals)
    with c3:
        st.metric("Watchlist", len(st.session_state.watchlist))
    
    # Quick scan
    st.markdown("### ⚡ Quick Scan")
    if st.button("🚀 Run All Swing Scanners Now", type="primary"):
        with st.spinner("Scanning Nifty 100 with all swing strategies..."):
            nifty = st.session_state.get("nifty_data")
            results = run_all_scanners(
                st.session_state.stock_data, 
                nifty_df=nifty,
                daily_only=True
            )
            st.session_state.scan_results = results
            st.rerun()
    
    # Show latest results summary
    if st.session_state.scan_results:
        st.markdown("### 📋 Latest Scan Results")
        for strategy, results in st.session_state.scan_results.items():
            profile = STRATEGY_PROFILES.get(strategy, {})
            icon = profile.get("icon", "📌")
            with st.expander(f"{icon} {profile.get('name', strategy)} — {len(results)} signal(s)", expanded=True):
                for r in results[:5]:
                    signal_color = "#00d26a" if r.signal == "BUY" else "#ff4757"
                    st.markdown(f"""
                    **{r.symbol}** ({r.sector}) — 
                    <span style="color:{signal_color};font-weight:700">{r.signal}</span> | 
                    Entry: ₹{r.entry:,.2f} | SL: ₹{r.stop_loss:,.2f} | 
                    T1: ₹{r.target_1:,.2f} | T2: ₹{r.target_2:,.2f} | 
                    Confidence: {r.confidence}% | RS: {r.rs_rating:.0f}
                    """, unsafe_allow_html=True)


# ============================================================================
# PAGE: SCANNER HUB
# ============================================================================
def page_scanner_hub():
    st.markdown("# 🔍 Scanner Hub")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Load data first from the Dashboard page.")
        return
    
    # Strategy selector
    st.markdown("### Select Strategy")
    
    # Create cards for each strategy
    cols = st.columns(4)
    selected_strategy = None
    
    for i, (key, profile) in enumerate(STRATEGY_PROFILES.items()):
        with cols[i % 4]:
            badge_class = {
                "Swing": "badge-swing",
                "Intraday": "badge-intraday",
                "Positional": "badge-positional",
                "Overnight": "badge-overnight",
            }.get(profile["type"], "badge-swing")
            
            intraday_note = ""
            if profile.get("requires_intraday"):
                if not st.session_state.breeze_connected:
                    intraday_note = "<br><small style='color:#ff8a65'>📊 Daily proxy (connect Breeze for live)</small>"
                else:
                    intraday_note = "<br><small style='color:#00d26a'>🔴 Live intraday mode</small>"
            
            st.markdown(f"""
            <div class="strategy-card">
                <div class="strategy-name">{profile['icon']} {profile['name']}</div>
                <span class="strategy-badge {badge_class}">{profile['type']}</span>
                <span style="color:#888;font-size:0.8em"> • {profile['hold']}</span><br>
                <small style="color:#888">{profile['description'][:80]}...</small><br>
                <span class="metric-green" style="font-size:0.9em">Win: {profile['win_rate']}%</span> • 
                <span class="metric-orange" style="font-size:0.9em">Exp: +{profile['expectancy']}%</span>
                {intraday_note}
            </div>
            """, unsafe_allow_html=True)
            
            if st.button(f"Scan {profile['name']}", key=f"scan_{key}", use_container_width=True):
                selected_strategy = key
    
    # Run all button
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Run ALL Swing Scanners", type="primary", use_container_width=True):
            selected_strategy = "ALL_SWING"
    with col2:
        if st.button("🔄 Run ALL Scanners (incl. Intraday Proxy)", use_container_width=True):
            selected_strategy = "ALL"
    
    # Execute scan
    if selected_strategy:
        nifty = st.session_state.get("nifty_data")
        
        if selected_strategy == "ALL_SWING":
            with st.spinner("Running all swing scanners on Nifty 100..."):
                results = run_all_scanners(st.session_state.stock_data, nifty, daily_only=True)
                st.session_state.scan_results = results
        elif selected_strategy == "ALL":
            with st.spinner("Running ALL scanners on Nifty 100..."):
                results = run_all_scanners(st.session_state.stock_data, nifty, daily_only=False)
                st.session_state.scan_results = results
        else:
            with st.spinner(f"Running {STRATEGY_PROFILES[selected_strategy]['name']} scanner..."):
                results = run_scanner(selected_strategy, st.session_state.stock_data, nifty)
                st.session_state.scan_results[selected_strategy] = results
        
        st.rerun()
    
    # Display results
    st.markdown("### 📊 Scan Results")
    
    if not st.session_state.scan_results:
        st.info("No scan results yet. Select a strategy above and click Scan.")
        return
    
    for strategy, results in st.session_state.scan_results.items():
        if not results:
            continue
        
        profile = STRATEGY_PROFILES.get(strategy, {})
        icon = profile.get("icon", "📌")
        name = profile.get("name", strategy)
        
        st.markdown(f"#### {icon} {name} — {len(results)} Signal(s)")
        
        # Create DataFrame for display
        rows = []
        for r in results:
            rows.append({
                "Symbol": r.symbol,
                "Signal": r.signal,
                "Entry ₹": f"{r.entry:,.2f}",
                "Stop Loss ₹": f"{r.stop_loss:,.2f}",
                "Target 1 ₹": f"{r.target_1:,.2f}",
                "Target 2 ₹": f"{r.target_2:,.2f}",
                "R:R": r.risk_reward,
                "Conf %": r.confidence,
                "RS": round(r.rs_rating),
                "Vol": f"{r.volume_ratio}x",
                "RSI": r.rsi,
                "Sector": r.sector,
                "Hold": r.hold_type,
            })
        
        df_display = pd.DataFrame(rows)
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        # Detailed view for each result
        for r in results:
            with st.expander(f"📋 {r.symbol} — Details & Reasoning"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Entry", f"₹{r.entry:,.2f}")
                    st.metric("Stop Loss", f"₹{r.stop_loss:,.2f}")
                with c2:
                    st.metric("Target 1 (1.5R)", f"₹{r.target_1:,.2f}")
                    st.metric("Target 2 (2.5R)", f"₹{r.target_2:,.2f}")
                with c3:
                    st.metric("Confidence", f"{r.confidence}%")
                    st.metric("Risk %", f"{r.risk_pct:.1f}%")
                
                st.markdown("**Why this trade:**")
                for reason in r.reasons:
                    st.markdown(f"  • {reason}")
                
                # Add to watchlist button
                if st.button(f"⭐ Add {r.symbol} to Watchlist", key=f"add_{strategy}_{r.symbol}"):
                    entry = {"symbol": r.symbol, "strategy": strategy, 
                             "entry": r.entry, "stop": r.stop_loss,
                             "target1": r.target_1, "target2": r.target_2,
                             "confidence": r.confidence, "date": r.timestamp}
                    if entry not in st.session_state.watchlist:
                        st.session_state.watchlist.append(entry)
                        st.success(f"Added {r.symbol} to watchlist!")


# ============================================================================
# PAGE: TRADE PLANNER
# ============================================================================
def page_trade_planner():
    st.markdown("# 📐 Trade Planner")
    st.markdown("Calculate position size, targets, and risk before entering any trade.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Trade Parameters")
        capital = st.number_input("Total Capital (₹)", value=st.session_state.capital, 
                                  step=50000, min_value=10000)
        st.session_state.capital = capital
        
        risk_pct = st.slider("Risk per Trade (%)", 0.5, 3.0, 2.0, 0.25,
                            help="Maximum 2% recommended. Higher risk = more shares but more to lose.")
        
        # Auto-fill from scan results or manual
        symbols_with_signals = []
        for strategy, results in st.session_state.scan_results.items():
            for r in results:
                symbols_with_signals.append(f"{r.symbol} ({strategy})")
        
        input_mode = st.radio("Input Mode", ["From Scanner Results", "Manual Entry"], horizontal=True)
        
        if input_mode == "From Scanner Results" and symbols_with_signals:
            selected = st.selectbox("Select Signal", symbols_with_signals)
            # Parse selection
            sym = selected.split(" (")[0]
            strat = selected.split("(")[1].rstrip(")")
            
            result = None
            for r in st.session_state.scan_results.get(strat, []):
                if r.symbol == sym:
                    result = r
                    break
            
            if result:
                entry = result.entry
                stop_loss = result.stop_loss
                is_short = result.signal == "SHORT"
                st.info(f"**{result.symbol}** — {result.signal} | Confidence: {result.confidence}%")
            else:
                entry = 100.0
                stop_loss = 95.0
                is_short = False
        else:
            entry = st.number_input("Entry Price (₹)", value=100.0, step=1.0, min_value=0.1)
            stop_loss = st.number_input("Stop Loss (₹)", value=95.0, step=1.0, min_value=0.1)
            is_short = st.checkbox("Short Trade")
    
    with col2:
        st.markdown("### Position Sizing Results")
        
        # Market regime adjustment
        position_mult = 1.0
        if st.session_state.market_health:
            position_mult = st.session_state.market_health.get("position_multiplier", 1.0)
            if position_mult < 1.0:
                st.warning(f"⚠️ Market regime adjustment: positions scaled to {position_mult*100:.0f}%")
        
        if entry > 0 and stop_loss > 0 and entry != stop_loss:
            pos = RiskManager.calculate_position(capital, risk_pct, entry, stop_loss, position_mult)
            targets = RiskManager.calculate_targets(entry, stop_loss, is_short)
            
            st.metric("Shares to Buy", f"{pos.shares:,}")
            st.metric("Position Value", f"₹{pos.position_value:,.2f}")
            st.metric("Risk Amount", f"₹{pos.risk_amount:,.2f}")
            st.metric("% of Portfolio", f"{pos.pct_of_portfolio:.1f}%")
            
            for w in pos.warnings:
                st.warning(w)
            
            st.markdown("### 🎯 Targets")
            tc1, tc2, tc3 = st.columns(3)
            with tc1:
                st.metric(f"T1 ({targets.t1_rr}R)", f"₹{targets.t1:,.2f}")
            with tc2:
                st.metric(f"T2 ({targets.t2_rr}R)", f"₹{targets.t2:,.2f}")
            with tc3:
                st.metric(f"T3 ({targets.t3_rr}R)", f"₹{targets.t3:,.2f}")
            
            st.markdown(f"**Trail SL trigger:** ₹{targets.trailing_trigger:,.2f} — move SL to breakeven at this level")
            
            # P&L scenarios
            st.markdown("### 💰 P&L Scenarios")
            risk_per = targets.risk_per_share
            scenarios = {
                "Full Stop Loss": -pos.risk_amount,
                "Target 1 (1.5R)": pos.shares * 1.5 * risk_per,
                "Target 2 (2.5R)": pos.shares * 2.5 * risk_per,
                "Target 3 (4R)": pos.shares * 4 * risk_per,
            }
            
            for scenario, pnl in scenarios.items():
                color = "🟢" if pnl > 0 else "🔴"
                st.markdown(f"{color} **{scenario}:** ₹{pnl:+,.2f}")
        else:
            st.info("Enter valid entry and stop loss prices to calculate.")
    
    # Portfolio heat section
    st.markdown("---")
    st.markdown("### 🌡️ Portfolio Heat Monitor")
    st.markdown("Track total risk across all open positions.")
    
    if st.session_state.watchlist:
        positions_for_heat = []
        for item in st.session_state.watchlist:
            if "shares" in item:
                positions_for_heat.append({
                    "symbol": item["symbol"],
                    "entry": item["entry"],
                    "stop": item["stop"],
                    "shares": item.get("shares", 0),
                    "sector": get_sector(item["symbol"]),
                })
        
        if positions_for_heat:
            heat = RiskManager.portfolio_heat(positions_for_heat, capital)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Portfolio Heat", f"{heat['heat_pct']}%", 
                         help=f"Max allowed: {heat['max_heat']}%")
            with c2:
                st.metric("Status", heat["status"])
            with c3:
                st.metric("Open Positions", heat["position_count"])
            
            for w in heat.get("warnings", []):
                st.warning(w)


# ============================================================================
# PAGE: WATCHLIST
# ============================================================================
def page_watchlist():
    st.markdown("# ⭐ Watchlist")
    
    if not st.session_state.watchlist:
        st.info("Your watchlist is empty. Add stocks from the Scanner Hub.")
        return
    
    # Display watchlist
    rows = []
    for i, item in enumerate(st.session_state.watchlist):
        risk = abs(item["entry"] - item["stop"])
        risk_pct = (risk / item["entry"]) * 100 if item["entry"] > 0 else 0
        rows.append({
            "#": i + 1,
            "Symbol": item["symbol"],
            "Strategy": item["strategy"],
            "Entry ₹": f"{item['entry']:,.2f}",
            "Stop ₹": f"{item['stop']:,.2f}",
            "Target 1 ₹": f"{item['target1']:,.2f}",
            "Risk %": f"{risk_pct:.1f}%",
            "Conf": f"{item['confidence']}%",
            "Date": item.get("date", ""),
        })
    
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    
    # Remove items
    st.markdown("### Manage Watchlist")
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.watchlist:
            symbols = [f"{w['symbol']} ({w['strategy']})" for w in st.session_state.watchlist]
            to_remove = st.selectbox("Remove", symbols)
            if st.button("🗑️ Remove from Watchlist"):
                idx = symbols.index(to_remove)
                st.session_state.watchlist.pop(idx)
                st.rerun()
    with col2:
        if st.button("🗑️ Clear Entire Watchlist"):
            st.session_state.watchlist = []
            st.rerun()
    
    # Current prices (if data loaded)
    if st.session_state.data_loaded and st.session_state.stock_data:
        st.markdown("### 📊 Current Status")
        for item in st.session_state.watchlist:
            sym = item["symbol"]
            if sym in st.session_state.stock_data:
                df = st.session_state.stock_data[sym]
                latest_close = df["close"].iloc[-1]
                entry = item["entry"]
                stop = item["stop"]
                
                pnl_pct = (latest_close / entry - 1) * 100
                color = "metric-green" if pnl_pct > 0 else "metric-red"
                
                hit_stop = latest_close <= stop if entry > stop else latest_close >= stop
                
                status = "🔴 STOP HIT" if hit_stop else ("🟢 In Profit" if pnl_pct > 0 else "🟡 In Loss")
                
                st.markdown(f"""
                **{sym}** ({item['strategy']}) — {status} | 
                Entry: ₹{entry:,.2f} → Current: ₹{latest_close:,.2f} | 
                <span class="{color}">P&L: {pnl_pct:+.1f}%</span>
                """, unsafe_allow_html=True)


# ============================================================================
# PAGE: DAILY WORKFLOW
# ============================================================================
def page_daily_workflow():
    st.markdown("# 📋 Daily Trading Workflow")
    st.markdown("Follow this checklist every trading day for consistent results.")
    
    now = datetime.now()
    today_key = now.strftime("%Y-%m-%d")
    
    if today_key not in st.session_state.workflow_checks:
        st.session_state.workflow_checks[today_key] = {}
    
    checks = st.session_state.workflow_checks[today_key]
    
    workflow = [
        {
            "time": "8:30 AM",
            "title": "Pre-Market Prep",
            "tasks": [
                ("market_health", "Run Market Health Check on Dashboard"),
                ("global_cues", "Check global cues: US futures, SGX Nifty, Asia markets"),
                ("fii_dii", "Check FII/DII data (moneycontrol.com)"),
                ("news", "Review major news/events for the day"),
            ]
        },
        {
            "time": "9:15 AM",
            "title": "Market Open",
            "tasks": [
                ("watch_open", "Watch first 15 minutes — DON'T trade, just observe"),
                ("gap_check", "Note gap-up/gap-down stocks from watchlist"),
            ]
        },
        {
            "time": "9:30 AM",
            "title": "First Scan Window",
            "tasks": [
                ("orb_scan", "Run ORB scanner (if Breeze connected) or check gap-ups"),
                ("orb_execute", "Execute ORB trades if signals found"),
            ]
        },
        {
            "time": "10:00 AM",
            "title": "Mid-Morning Scan",
            "tasks": [
                ("vwap_scan", "Run VWAP Reclaim scanner"),
                ("vwap_execute", "Execute VWAP reclaim trades"),
            ]
        },
        {
            "time": "12:30 PM",
            "title": "Lunch Session",
            "tasks": [
                ("lunch_scan", "Run Lunch Low Buy scanner"),
                ("review_morning", "Review morning trades — trail stops to breakeven"),
            ]
        },
        {
            "time": "3:00 PM",
            "title": "Power Hour",
            "tasks": [
                ("ath_scan", "Run Last 30 Min ATH scanner — the best overnight play"),
                ("ath_execute", "Buy top signals in last 5 minutes (3:25 PM)"),
            ]
        },
        {
            "time": "3:30 PM",
            "title": "Post-Market Swing Scan",
            "tasks": [
                ("vcp_scan", "Run VCP scanner"),
                ("ema_scan", "Run 21 EMA Bounce scanner"),
                ("breakout_scan", "Run 52-Week High Breakout scanner"),
                ("short_scan", "Run Failed Breakout Short scanner"),
                ("watchlist_update", "Update watchlist with tomorrow's candidates"),
            ]
        },
        {
            "time": "Weekend",
            "title": "Weekend Review",
            "tasks": [
                ("weekly_review", "Review all trades from the week"),
                ("sector_rotation", "Check sector rotation — which sectors are leading?"),
                ("portfolio_heat", "Check portfolio heat — below 10%?"),
                ("journal", "Update trading journal with lessons learned"),
            ]
        },
    ]
    
    total_tasks = sum(len(step["tasks"]) for step in workflow)
    completed = sum(1 for v in checks.values() if v)
    
    st.progress(completed / total_tasks if total_tasks > 0 else 0)
    st.markdown(f"**Progress:** {completed}/{total_tasks} tasks completed today")
    
    for step in workflow:
        st.markdown(f"""
        <div class="workflow-step">
            <strong>⏰ {step['time']}</strong> — {step['title']}
        </div>
        """, unsafe_allow_html=True)
        
        for task_id, task_label in step["tasks"]:
            checked = checks.get(task_id, False)
            new_val = st.checkbox(task_label, value=checked, key=f"wf_{today_key}_{task_id}")
            checks[task_id] = new_val
    
    st.session_state.workflow_checks[today_key] = checks
    
    # Risk rules reminder
    st.markdown("---")
    st.markdown("### 🛡️ Risk Rules (NON-NEGOTIABLE)")
    rules = [
        ("Max risk per trade", "2% of capital"),
        ("Max positions", "5-10"),
        ("Max sector exposure", "30%"),
        ("Portfolio heat limit", "< 10% of capital"),
        ("Max single position", "20% of capital"),
        ("Time stop", "Exit if no progress in 10 days"),
        ("Max loss per trade", "-7% absolute"),
    ]
    for rule, value in rules:
        st.markdown(f"  **{rule}:** {value}")


# ============================================================================
# PAGE: SETTINGS
# ============================================================================
def page_settings():
    st.markdown("# ⚙️ Settings")
    
    # Capital
    st.markdown("### 💰 Trading Capital")
    new_capital = st.number_input("Capital (₹)", value=st.session_state.capital, step=50000)
    st.session_state.capital = new_capital
    
    # Breeze API
    st.markdown("### 🔌 ICICI Breeze API Connection")
    st.markdown("""
    Connect your ICICI Direct account for real-time intraday data. 
    This enables live ORB, VWAP Reclaim, and Lunch Low scanners.
    
    **How to get credentials:**
    1. Log into [ICICI Direct API Portal](https://api.icicidirect.com)
    2. Create an app → Get API Key and Secret
    3. Generate a session token daily (auto-expires)
    """)
    
    with st.form("breeze_form"):
        api_key = st.text_input("API Key", type="password")
        api_secret = st.text_input("API Secret", type="password")
        session_token = st.text_input("Session Token", type="password",
                                      help="Generate daily from ICICI Direct portal")
        
        if st.form_submit_button("Connect to Breeze"):
            if api_key and api_secret and session_token:
                engine = BreezeEngine()
                success, msg = engine.connect(api_key, api_secret, session_token)
                if success:
                    st.success(msg)
                    st.session_state.breeze_connected = True
                    st.session_state.breeze_engine = engine
                else:
                    st.error(msg)
            else:
                st.warning("Please fill all three fields.")
    
    st.markdown("### 📊 Stock Universe")
    universe = st.selectbox("Scan Universe", ["Nifty 100 (50 + Next 50)", "Nifty 50 only"])
    
    st.markdown("### 🎨 Strategy Preferences")
    st.markdown("Enable/disable specific scanners:")
    for key, profile in STRATEGY_PROFILES.items():
        st.checkbox(f"{profile['icon']} {profile['name']}", value=True, key=f"enable_{key}")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **NSE Scanner Pro v1.0**  
    8 battle-tested strategies from Minervini, O'Neil, Raschke & Williams  
    Built for the Indian NSE market  
    
    **Data Sources:**
    - Daily data: yfinance (free, no API key needed)
    - Intraday data: ICICI Breeze API (optional)
    
    **Strategies by Expectancy:**
    1. 🏆 VCP (Minervini): +5.12%
    2. 🚀 52-Week High Breakout: +5.82%  
    3. 📉 Failed Breakout Short: +3.12%
    4. 🔄 21 EMA Bounce: +2.14%
    5. ⭐ Last 30 Min ATH: +0.89%
    6. 🔓 Opening Range Breakout: +0.47%
    7. 📈 VWAP Reclaim: +0.39%
    8. 🍽️ Lunch Low Buy: +0.28%
    """)


# ============================================================================
# MAIN ROUTER
# ============================================================================
if page == "📊 Dashboard":
    page_dashboard()
elif page == "🔍 Scanner Hub":
    page_scanner_hub()
elif page == "📐 Trade Planner":
    page_trade_planner()
elif page == "⭐ Watchlist":
    page_watchlist()
elif page == "📋 Daily Workflow":
    page_daily_workflow()
elif page == "⚙️ Settings":
    page_settings()
