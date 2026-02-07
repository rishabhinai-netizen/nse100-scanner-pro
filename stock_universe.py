"""
NSE Stock Universe — Nifty 50 + Next 50 = Nifty 100
Updated: Feb 2026
"""

NIFTY_50 = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
    "LT", "AXISBANK", "BAJFINANCE", "ASIANPAINT", "MARUTI",
    "HCLTECH", "SUNPHARMA", "TITAN", "ULTRACEMCO", "NTPC",
    "WIPRO", "NESTLEIND", "BAJAJFINSV", "POWERGRID", "TATAMOTORS",
    "M&M", "ONGC", "JSWSTEEL", "ADANIPORTS", "TATASTEEL",
    "COALINDIA", "HDFCLIFE", "SBILIFE", "TECHM", "GRASIM",
    "DRREDDY", "CIPLA", "BPCL", "APOLLOHOSP", "DIVISLAB",
    "EICHERMOT", "HEROMOTOCO", "INDUSINDBK", "TATACONSUM", "BRITANNIA",
    "BAJAJ-AUTO", "HINDALCO", "ADANIENT", "SHRIRAMFIN", "LTIM",
]

NIFTY_NEXT_50 = [
    "ADANIGREEN", "AMBUJACEM", "BANKBARODA", "BERGEPAINT", "BOSCHLTD",
    "CANBK", "CHOLAFIN", "COLPAL", "DLF", "DABUR",
    "GAIL", "GODREJCP", "HAVELLS", "ICICIPRULI", "IIFL",
    "INDIGO", "IOC", "IRCTC", "JINDALSTEL", "JIOFIN",
    "LICI", "LUPIN", "MARICO", "MOTHERSON", "NAUKRI",
    "PEL", "PERSISTENT", "PIDILITIND", "PNB", "POLYCAB",
    "RECLTD", "SBICARD", "SIEMENS", "SRF", "TATAPOWER",
    "TORNTPHARM", "TRENT", "UNIONBANK", "UNITDSPR", "VEDL",
    "VOLTAS", "ZOMATO", "ZYDUSLIFE", "ABB", "ATGL",
    "DMART", "HAL", "NHPC", "PIIND", "LODHA",
]

NIFTY_100 = NIFTY_50 + NIFTY_NEXT_50

SECTOR_MAP = {
    "IT": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM", "PERSISTENT", "NAUKRI"],
    "Banking": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "INDUSINDBK", "BANKBARODA", "CANBK", "PNB", "UNIONBANK"],
    "Finance": ["BAJFINANCE", "BAJAJFINSV", "HDFCLIFE", "SBILIFE", "CHOLAFIN", "ICICIPRULI", "SHRIRAMFIN", "SBICARD", "JIOFIN", "RECLTD", "PEL", "LICI"],
    "Pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "LUPIN", "APOLLOHOSP", "TORNTPHARM", "ZYDUSLIFE"],
    "Auto": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "EICHERMOT", "HEROMOTOCO", "MOTHERSON"],
    "FMCG": ["HINDUNILVR", "ITC", "NESTLEIND", "TATACONSUM", "BRITANNIA", "COLPAL", "DABUR", "GODREJCP", "MARICO", "UNITDSPR"],
    "Metals": ["JSWSTEEL", "TATASTEEL", "HINDALCO", "VEDL", "JINDALSTEL"],
    "Energy": ["RELIANCE", "NTPC", "POWERGRID", "ONGC", "BPCL", "COALINDIA", "GAIL", "IOC", "TATAPOWER", "ADANIGREEN", "NHPC", "ATGL"],
    "Infra": ["LT", "ADANIPORTS", "ADANIENT", "ULTRACEMCO", "GRASIM", "AMBUJACEM", "DLF", "LODHA", "IRCTC", "INDIGO", "SIEMENS", "ABB", "HAL", "PIIND"],
    "Consumer": ["TITAN", "ASIANPAINT", "BERGEPAINT", "HAVELLS", "PIDILITIND", "VOLTAS", "POLYCAB", "SRF", "TRENT", "DMART", "ZOMATO", "BOSCHLTD"],
}

def get_sector(symbol: str) -> str:
    for sector, stocks in SECTOR_MAP.items():
        if symbol in stocks:
            return sector
    return "Other"

def get_yfinance_symbol(symbol: str) -> str:
    """Convert NSE symbol to yfinance format. Some symbols need special handling."""
    # yfinance quirks for certain NSE stocks
    YF_OVERRIDES = {
        "M&M": "M%26M.NS",
        "BAJAJ-AUTO": "BAJAJ-AUTO.NS",
        "TATAMOTORS": "TATAMTRDVR.NS",  # Sometimes delisted on yfinance, use DVR
    }
    return YF_OVERRIDES.get(symbol, f"{symbol}.NS")
