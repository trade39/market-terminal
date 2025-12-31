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

# --- CSS STYLES ---
CSS_STYLE = """
<style>
    /* Main Background - True Black */
    .stApp { background-color: #000000; font-family: 'Courier New', Courier, monospace; }
    
    /* Sidebar Background */
    section[data-testid="stSidebar"] { background-color: #111111; border-right: 1px solid #333; }
    
    /* Text Colors */
    h1, h2, h3, h4 { color: #ff9900 !important; font-family: 'Arial', sans-serif; text-transform: uppercase; letter-spacing: 1px; }
    p, div, span { color: #e0e0e0; }
    
    /* Metric Styling */
    div[data-testid="stMetricValue"] { 
        color: #00e6ff !important; font-family: 'Courier New', monospace;
        font-weight: bold;
    }
    div[data-testid="stMetricLabel"] { color: #ff9900 !important; font-size: 0.8rem; }
    
    /* Tables/Dataframes */
    .stDataFrame { border: 1px solid #333; }
    
    /* Custom Boxes */
    .terminal-box { 
        border: 1px solid #333; background-color: #0a0a0a; 
        padding: 15px; 
        margin-bottom: 10px;
    }
    
    /* Signal badges */
    .bullish { color: #000000; background-color: #00ff00; padding: 2px 6px; font-weight: bold; border-radius: 4px; }
    .bearish { color: #000000; background-color: #ff3333; padding: 2px 6px; font-weight: bold; border-radius: 4px; }
    .neutral { color: #000000; background-color: #cccccc; padding: 2px 6px; font-weight: bold; border-radius: 4px; }
    
    /* News Link */
    .news-link { color: #00e6ff; text-decoration: none; font-size: 0.9em; }
    .news-link:hover { text-decoration: underline; color: #ff9900; }
    
    /* UI Elements */
    .stSelectbox > div > div { border-radius: 0px; background-color: #111; color: white; border: 1px solid #444; }
    .stTextInput > div > div > input { color: white; background-color: #111; border: 1px solid #444; }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] { height: 30px; white-space: pre-wrap; background-color: #111; color: white; border-radius: 0px; border: 1px solid #333;}
    .stTabs [aria-selected="true"] { background-color: #ff9900; color: black !important; font-weight: bold;}
    button { border-radius: 0px !important; border: 1px solid #ff9900 !important; color: #ff9900 !important; background: black !important; }
    hr { margin: 1em 0; border: 0; border-top: 1px solid #333; }
    
    /* API Limit Progress Bars */
    .stProgress > div > div > div > div { background-color: #ff9900; }
</style>
