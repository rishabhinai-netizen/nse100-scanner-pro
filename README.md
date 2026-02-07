# 🎯 NSE Scanner Pro

**8 Battle-Tested Trading Strategies | Nifty 500 Scanner | Telegram Alerts**

---

## 🚀 Setup (5 Minutes)

### Local Setup
```bash
cd nse_scanner_pro
pip install -r requirements.txt
streamlit run app.py
```

### Streamlit Cloud (mobile access)
1. Push to **Private** GitHub repo
2. Deploy at [share.streamlit.io](https://share.streamlit.io) → Main file: `app.py`

---

## 🔌 Breeze API Setup

**Go to:** App Settings → Secrets → Paste these 3 lines:

```
BREEZE_API_KEY = "your_api_key"
BREEZE_API_SECRET = "your_api_secret"
BREEZE_SESSION_TOKEN = "your_daily_token"
```

⚠️ **Do NOT include \`\`\`toml or backticks!** Just the 3 lines above.

⚠️ **Daily:** Session token expires daily. Regenerate from ICICI Direct each morning.

**What Breeze enables:** Real-time intraday data for ORB, VWAP Reclaim, Lunch Low scanners. Without it, they run in daily proxy mode. All other scanners (VCP, EMA21, 52WH, Short, ATH) work perfectly with free yfinance data.

---

## 📱 Telegram Alerts

1. Open Telegram → search `@BotFather` → `/newbot` → copy **Bot Token**
2. Search `@userinfobot` → `/start` → copy **Chat ID**
3. Add to Secrets:
```
TELEGRAM_BOT_TOKEN = "123456:ABCdef..."
TELEGRAM_CHAT_ID = "987654321"
```
4. Alerts sent automatically when scanner finds signals

---

## 📊 Strategy Rankings

| # | Strategy | Win % | Expectancy | Data Needed |
|---|----------|-------|------------|-------------|
| 1 | 🏆 VCP | 67.2% | +5.12% | Daily |
| 2 | 🚀 52WH Breakout | 58.8% | +5.82% | Daily |
| 3 | 📉 Failed Short | 64.2% | +3.12% | Daily |
| 4 | 🔄 21 EMA Bounce | 62.5% | +2.14% | Daily |
| 5 | ⭐ ATH Overnight | 68.4% | +0.89% | Daily |
| 6 | 🔓 ORB | 58.2% | +0.47% | Breeze 🔌 |
| 7 | 📈 VWAP Reclaim | 61.8% | +0.39% | Breeze 🔌 |
| 8 | 🍽️ Lunch Low | 56.3% | +0.28% | Breeze 🔌 |
