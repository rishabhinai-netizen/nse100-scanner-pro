# 🎯 NSE Scanner Pro v2.0

**8 Strategies | Charts | RS Rankings | Sector Heatmap | Trade Journal | Telegram Alerts**

## 🚀 Quick Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🔌 Breeze API (Streamlit Cloud → Settings → Secrets)

Paste ONLY these 3 lines (**no backticks, no \`\`\`toml header**):

```
BREEZE_API_KEY = "your_key"
BREEZE_API_SECRET = "your_secret"
BREEZE_SESSION_TOKEN = "daily_token"
```

⚠️ Session token expires daily. Regenerate from ICICI Direct each morning.

## 📱 Telegram Alerts

```
TELEGRAM_BOT_TOKEN = "123456:ABCdef..."
TELEGRAM_CHAT_ID = "987654321"
```

## v2.0 Features

| Feature | Description |
|---------|-------------|
| 📊 8 Scanners | VCP, EMA21, 52WH, ORB, VWAP, Lunch Low, ATH, Short |
| 📈 Candlestick Charts | Entry/SL/Target overlay, EMA, Volume, RSI |
| 💪 RS Rankings | Relative Strength vs Nifty with scatter plot |
| 🗺️ Sector Heatmap | 1W/1M/3M performance rotation |
| 📊 Market Breadth | A/D ratio, >200 SMA %, 52W high/low counts |
| 🔄 Multi-Timeframe | Weekly alignment confirmation for each signal |
| 📓 Trade Journal | Full P&L tracking, equity curve, strategy analytics |
| 🔔 Telegram Alerts | Auto-alerts on scan, per-signal manual alerts |
| 📐 Position Sizing | 2% rule with regime-adjusted sizing |
| 📋 Daily Workflow | IST-timed checklist with progress tracking |
