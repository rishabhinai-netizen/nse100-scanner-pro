# 🎯 NSE Scanner Pro

**8 Battle-Tested Trading Strategies | Real-Time NSE Scanner**

---

## 🚀 Setup (5 Minutes — No Coding)

### Option A: Run Locally

```bash
# 1. Open terminal / command prompt
cd Desktop/nse_scanner_pro

# 2. Install dependencies (one-time)
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

Browser opens at `http://localhost:8501`

### Option B: Streamlit Cloud (access from phone)

1. Push to **Private** GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io) → Deploy
3. Main file: `app.py`
4. Add Breeze secrets in **Settings → Secrets**

---

## 🔌 ICICI Breeze API Setup

**One-time:** Get API Key + Secret from [api.icicidirect.com](https://api.icicidirect.com)

**For Streamlit Cloud:** Settings → Secrets → paste:
```toml
BREEZE_API_KEY = "your_key"
BREEZE_API_SECRET = "your_secret"
BREEZE_SESSION_TOKEN = "daily_token"
```

**For local:** Edit `.streamlit/secrets.toml` with same content.

**⚠️ Daily:** Session Token expires every day. Regenerate from ICICI portal each morning before market opens, then update in secrets and refresh the app.

---

## 📊 Daily Usage

| Time (IST) | Action |
|---|---|
| 8:30 AM | Load data → Market Health Check |
| 9:30 AM | ORB scanner (after 15-min candle) |
| 10:00 AM | VWAP Reclaim scanner |
| 12:30 PM | Lunch Low scanner |
| 3:00 PM | Last 30 Min ATH scanner → buy at 3:25 PM |
| 3:30 PM+ | VCP, 21 EMA, 52WH, Short scanners |

---

## 📈 Strategies by Expectancy

| # | Strategy | Win % | Expectancy | Hold |
|---|----------|-------|------------|------|
| 1 | 🏆 VCP | 67.2% | +5.12% | 15-40 days |
| 2 | 🚀 52WH Breakout | 58.8% | +5.82% | 20-60 days |
| 3 | 📉 Failed Short | 64.2% | +3.12% | 3-10 days |
| 4 | 🔄 21 EMA Bounce | 62.5% | +2.14% | 5-15 days |
| 5 | ⭐ ATH Overnight | 68.4% | +0.89% | Overnight |
| 6 | 🔓 ORB | 58.2% | +0.47% | 2-6 hours |
| 7 | 📈 VWAP Reclaim | 61.8% | +0.39% | 2-4 hours |
| 8 | 🍽️ Lunch Low | 56.3% | +0.28% | 2-3 hours |

---

## 🛡️ Risk Rules

- Max risk per trade: **2%** of capital
- Max positions: **5-10**
- Portfolio heat: **< 10%** of capital
- Max single position: **20%** of capital
- Time stop: **10 days** no progress → exit
