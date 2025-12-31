import streamlit as st

# --- ASSET CONFIGURATION ---
ASSETS = {
    "Gold (Comex)": {"ticker": "GC=F", "opt_ticker": "GLD", "news_query": "Gold Price", "cg_id": None},
    "S&P 500": {"ticker": "^GSPC", "opt_ticker": "SPY", "news_query": "S&P 500", "cg_id": None},
    "NASDAQ": {"ticker": "^IXIC", "opt_ticker": "QQQ", "news_query": "Nasdaq", "cg_id": None},
    "EUR/USD": {"ticker": "EURUSD=X", "opt_ticker": None, "news_query": "EURUSD", "cg_id": None},
    "Bitcoin": {"ticker": "BTC-USD", "opt_ticker": "BITO", "news_query": "Bitcoin", "cg_id": "bitcoin"},
    "Ethereum": {"ticker": "ETH-USD", "opt_ticker": "ETHE", "news_query": "Ethereum", "cg_id": "ethereum"},
    "Solana": {"ticker": "SOL-USD", "opt_ticker": None, "news_query": "Solana Crypto", "cg_id": "solana"},
    "XRP": {"ticker": "XRP-USD", "opt_ticker": None, "news_query": "Ripple XRP", "cg_id": "ripple"},
    "BNB": {"ticker": "BNB-USD", "opt_ticker": None, "news_query": "Binance Coin", "cg_id": "binancecoin"},
    "Cardano": {"ticker": "ADA-USD", "opt_ticker": None, "news_query": "Cardano ADA", "cg_id": "cardano"},
    "Dogecoin": {"ticker": "DOGE-USD", "opt_ticker": None, "news_query": "Dogecoin", "cg_id": "dogecoin"},
    "Shiba Inu": {"ticker": "SHIB-USD", "opt_ticker": None, "news_query": "Shiba Inu Coin", "cg_id": "shiba-inu"},
    "Pepe": {"ticker": "PEPE-USD", "opt_ticker": None, "news_query": "Pepe Coin", "cg_id": "pepe"},
    "Chainlink": {"ticker": "LINK-USD", "opt_ticker": None, "news_query": "Chainlink", "cg_id": "chainlink"},
    "Polygon": {"ticker": "MATIC-USD", "opt_ticker": None, "news_query": "Polygon MATIC", "cg_id": "matic-network"},
}

# --- COT MAPPING ---
COT_MAPPING = {
    "Gold (Comex)": {"keywords": ["GOLD", "COMMODITY EXCHANGE"], "report_type": "disaggregated_fut", "labels": ("Managed Money", "Producers/Merchants")},
    "S&P 500": {"keywords": ["E-MINI S&P 500", "CHICAGO MERCANTILE EXCHANGE"], "report_type": "traders_in_financial_futures_fut", "labels": ("Leveraged Funds", "Asset Managers")},
    "NASDAQ": {"keywords": ["E-MINI NASDAQ-100", "CHICAGO MERCANTILE EXCHANGE"], "report_type": "traders_in_financial_futures_fut", "labels": ("Leveraged Funds", "Asset Managers")},
    "EUR/USD": {"keywords": ["EURO FX", "CHICAGO MERCANTILE EXCHANGE"], "report_type": "traders_in_financial_futures_fut", "labels": ("Leveraged Funds", "Asset Managers")},
    "Bitcoin": {"keywords": ["BITCOIN", "CHICAGO MERCANTILE EXCHANGE"], "report_type": "traders_in_financial_futures_fut", "labels": ("Leveraged Funds", "Asset Managers")},
    "Ethereum": {"keywords": ["ETHER", "CHICAGO MERCANTILE EXCHANGE"], "report_type": "traders_in_financial_futures_fut", "labels": ("Leveraged Funds", "Asset Managers")}
}

# --- CSS STYLES (MONOCHROMATIC NAVY/CYAN) ---
CSS_STYLE = """
<style>
    /* 1. Main Background - Deep Navy/Near Black */
    .stApp { background-color: #0A0F1E; font-family: 'Courier New', Courier, monospace; }
    
    /* 2. Sidebar - Slightly Lighter Navy */
    section[data-testid="stSidebar"] { background-color: #12161F; border-right: 1px solid #1E252F; }
    
    /* 3. Text Colors - White Primary, Light Gray Secondary */
    h1, h2, h3, h4 { color: #FFFFFF !important; font-family: 'Arial', sans-serif; text-transform: uppercase; letter-spacing: 1px; }
    p, div, span, label { color: #CCCCCC; }
    
    /* 4. Metric Styling - Cyan Values */
    div[data-testid="stMetricValue"] { 
        color: #00FFFF !important; font-family: 'Courier New', monospace;
        font-weight: bold;
    }
    div[data-testid="stMetricLabel"] { color: #AAAAAA !important; font-size: 0.8rem; }
    
    /* 5. Custom Boxes (Chart/Table Backgrounds) */
    .terminal-box { 
        border: 1px solid #1E252F; background-color: #12161F; 
        padding: 15px; 
        margin-bottom: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* 6. Signal Badges - Monochromatic Blue/Cyan Theme */
    /* Bullish = Cyan (Active), Bearish = Muted Blue/Gray (Passive) */
    .bullish { color: #0A0F1E; background-color: #00FFFF; padding: 2px 6px; font-weight: bold; border-radius: 4px; }
    .bearish { color: #FFFFFF; background-color: #405060; padding: 2px 6px; font-weight: bold; border-radius: 4px; border: 1px solid #8080FF; }
    .neutral { color: #CCCCCC; background-color: #1E252F; padding: 2px 6px; font-weight: bold; border-radius: 4px; border: 1px solid #333; }
    
    /* 7. News Link */
    .news-link { color: #40E0FF; text-decoration: none; font-size: 0.9em; }
    .news-link:hover { text-decoration: underline; color: #FFFFFF; }
    
    /* 8. UI Elements */
    .stSelectbox > div > div { border-radius: 0px; background-color: #12161F; color: white; border: 1px solid #333; }
    .stTextInput > div > div > input { color: white; background-color: #12161F; border: 1px solid #333; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] { height: 30px; white-space: pre-wrap; background-color: #12161F; color: #AAAAAA; border-radius: 0px; border: 1px solid #1E252F;}
    .stTabs [aria-selected="true"] { background-color: #1E252F; color: #00FFFF !important; font-weight: bold; border-bottom: 2px solid #00FFFF;}
    
    /* Buttons */
    button { border-radius: 0px !important; border: 1px solid #00FFFF !important; color: #00FFFF !important; background: #0A0F1E !important; }
    button:hover { background-color: #00FFFF !important; color: #0A0F1E !important; }
    
    hr { margin: 1em 0; border: 0; border-top: 1px solid #333333; }
    
    /* Progress Bars - Cyan */
    .stProgress > div > div > div > div { background-color: #00FFFF; }
    
    /* Dataframes */
    .stDataFrame { border: 1px solid #1E252F; }
</style>
"""
