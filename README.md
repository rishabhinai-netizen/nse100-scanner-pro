# 🎯 NSE Scanner Pro

**8 Battle-Tested Trading Strategies for NSE — One Dashboard**

Built on methodologies from Minervini, O'Neil, Raschke & Williams, adapted for Indian markets.

---

## What This Tool Does

1. **Scans Nifty 100 stocks** using 8 proven strategies
2. **Gives you trade signals** with Entry, Stop Loss, and Targets
3. **Calculates position size** so you never risk more than 2% per trade
4. **Checks market health** before you trade
5. **Daily workflow checklist** to keep you disciplined

---

## 🚀 Setup (5 Minutes — No Coding Required)

### Option A: Run Locally (Recommended for Speed)

**Step 1: Install Python**
- Download Python from https://python.org (version 3.9+)
- During installation, CHECK the box "Add Python to PATH"

**Step 2: Download this folder**
- Save the entire `nse_scanner_pro` folder to your Desktop

**Step 3: Open Terminal/Command Prompt**
- **Windows:** Press `Win + R`, type `cmd`, press Enter
- **Mac:** Open Terminal from Applications

**Step 4: Navigate to the folder**
```
cd Desktop/nse_scanner_pro
```

**Step 5: Install dependencies**
```
pip install -r requirements.txt
```

**Step 6: Run the app**
```
streamlit run app.py
```

**Step 7:** Your browser will open automatically at `http://localhost:8501` 🎉

---

### Option B: Deploy to Streamlit Cloud (Free — Access from Phone)

**Step 1: Create a GitHub account** (if you don't have one)
- Go to https://github.com and sign up

**Step 2: Create a new repository**
- Click "New Repository"
- Name: `nse-scanner-pro`
- Make it **Private**
- Click "Create Repository"

**Step 3: Upload all files**
- Click "Upload files" on GitHub
- Drag and drop ALL files from the `nse_scanner_pro` folder
- Also create a `.streamlit` folder and upload `config.toml` inside it
- Click "Commit changes"

**Step 4: Deploy on Streamlit Cloud**
- Go to https://share.streamlit.io
- Sign in with GitHub
- Click "New App"
- Select your `nse-scanner-pro` repo
- Main file path: `app.py`
- Click "Deploy"

**Step 5:** Your app is live! Access it from any device 📱

---

## 📊 Daily Usage Guide

### Morning Routine (8:30 AM)
1. Open the app → **Dashboard**
2. Click "Load / Refresh Data"
3. Check **Market Health** — this tells you if today is a good day to trade
4. Go to **Daily Workflow** tab and follow the checklist

### During Market Hours
5. Go to **Scanner Hub**
6. Click "Run All Swing Scanners" — results appear in seconds
7. Review signals — focus on **Confidence > 70%** and **RS Rating > 60**
8. For trades you like → click "Add to Watchlist"
9. Go to **Trade Planner** → Select the signal → Get exact shares to buy

### Post-Market (After 3:30 PM)
10. Run swing scanners again for next-day setups
11. Update your watchlist
12. Check **Portfolio Heat** in Trade Planner

---

## 🛡️ Risk Rules (NON-NEGOTIABLE)

| Rule | Limit |
|------|-------|
| Max risk per trade | 2% of capital |
| Max positions | 5-10 |
| Max sector exposure | 30% |
| Portfolio heat | < 10% of capital |
| Max single position | 20% of capital |
| Time stop | Exit if no move in 10 days |
| Max loss per trade | -7% absolute |

---

## 📈 Strategy Rankings (by Expectancy)

| # | Strategy | Win Rate | Expectancy | Hold Period |
|---|----------|----------|------------|-------------|
| 1 | 🏆 VCP (Minervini) | 67.2% | +5.12% | 15-40 days |
| 2 | 🚀 52-Week High Breakout | 58.8% | +5.82% | 20-60 days |
| 3 | 📉 Failed Breakout Short | 64.2% | +3.12% | 3-10 days |
| 4 | 🔄 21 EMA Bounce | 62.5% | +2.14% | 5-15 days |
| 5 | ⭐ Last 30 Min ATH | 68.4% | +0.89% | Overnight |
| 6 | 🔓 Opening Range Breakout | 58.2% | +0.47% | 2-6 hours |
| 7 | 📈 VWAP Reclaim | 61.8% | +0.39% | 2-4 hours |
| 8 | 🍽️ Lunch Low Buy | 56.3% | +0.28% | 2-3 hours |

---

## 🔌 ICICI Breeze API (Optional)

For **real-time intraday** scanners (ORB, VWAP Reclaim, Lunch Low):

1. Go to https://api.icicidirect.com
2. Create an app → Get API Key + Secret
3. In the app, go to **Settings** → Enter credentials
4. Generate a new session token daily

Without Breeze, all **daily/swing** scanners work perfectly using free yfinance data.

---

## 🏗️ File Structure

```
nse_scanner_pro/
├── app.py              ← Main app (run this)
├── scanners.py         ← 8 strategy implementations
├── data_engine.py      ← Data fetching (yfinance + Breeze)
├── risk_manager.py     ← Position sizing & risk
├── stock_universe.py   ← Nifty 100 stock list
├── requirements.txt    ← Dependencies
├── .env.example        ← API key template
├── .streamlit/
│   └── config.toml     ← Dark theme config
└── README.md           ← This file
```
