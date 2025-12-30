import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import google.generativeai as genai
from newsapi import NewsApiClient
from datetime import datetime, timedelta
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
import os
import time
import io
import zipfile

# --- SAFE IMPORT SYSTEM (Prevents Crashes) ---
# 1. TextBlob (Sentiment)
try:
    from textblob import TextBlob
    HAS_NLP = True
except ImportError:
    HAS_NLP = False

# 2. COT Reports (Institutional Data)
try:
    import cot_reports as cot
    HAS_COT_LIB = True
except ImportError:
    HAS_COT_LIB = False

# --- APP CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Bloomberg Terminal Pro V5.18", page_icon="⚡")

# --- BLOOMBERG TERMINAL STYLING (CSS) ---
st.markdown("""
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
        color: #00e6ff !important; 
        font-family: 'Courier New', monospace;
        font-weight: bold;
    }
    div[data-testid="stMetricLabel"] { color: #ff9900 !important; font-size: 0.8rem; }
    
    /* Tables/Dataframes */
    .stDataFrame { border: 1px solid #333; }
    
    /* Custom Boxes */
    .terminal-box { 
        border: 1px solid #333; 
        background-color: #0a0a0a; 
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
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'gemini_calls' not in st.session_state: st.session_state['gemini_calls'] = 0
if 'news_calls' not in st.session_state: st.session_state['news_calls'] = 0
if 'rapid_calls' not in st.session_state: st.session_state['rapid_calls'] = 0
if 'coingecko_calls' not in st.session_state: st.session_state['coingecko_calls'] = 0
if 'fred_calls' not in st.session_state: st.session_state['fred_calls'] = 0
if 'narrative_cache' not in st.session_state: st.session_state['narrative_cache'] = None
if 'thesis_cache' not in st.session_state: st.session_state['thesis_cache'] = None

# --- CONSTANTS & MAPPINGS ---
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

# ENHANCED COT MAPPING (Matched to new logic)
COT_MAPPING = {
    "Gold (Comex)": {
        "keywords": ["GOLD", "COMMODITY EXCHANGE"], 
        "report_type": "disaggregated_fut", 
        "labels": ("Managed Money", "Producers/Merchants")
    },
    "S&P 500": {
        "keywords": ["E-MINI S&P 500", "CHICAGO MERCANTILE EXCHANGE"], 
        "report_type": "traders_in_financial_futures_fut", 
        "labels": ("Leveraged Funds", "Asset Managers")
    },
    "NASDAQ": {
        "keywords": ["E-MINI NASDAQ-100", "CHICAGO MERCANTILE EXCHANGE"], 
        "report_type": "traders_in_financial_futures_fut", 
        "labels": ("Leveraged Funds", "Asset Managers")
    },
    "EUR/USD": {
        "keywords": ["EURO FX", "CHICAGO MERCANTILE EXCHANGE"], 
        "report_type": "traders_in_financial_futures_fut", 
        "labels": ("Leveraged Funds", "Asset Managers")
    },
    "Bitcoin": {
        "keywords": ["BITCOIN", "CHICAGO MERCANTILE EXCHANGE"], 
        "report_type": "traders_in_financial_futures_fut", 
        "labels": ("Leveraged Funds", "Asset Managers")
    },
    "Ethereum": {
        "keywords": ["ETHER", "CHICAGO MERCANTILE EXCHANGE"], 
        "report_type": "traders_in_financial_futures_fut", 
        "labels": ("Leveraged Funds", "Asset Managers")
    }
}

# --- HELPER FUNCTIONS ---
def get_api_key(key_name):
    if "api_keys" in st.secrets and key_name in st.secrets["api_keys"]:
        return st.secrets["api_keys"][key_name]
    if key_name in st.secrets:
        return st.secrets[key_name]
    if key_name == "gemini_api_key":
        if "GOOGLE_API_KEY" in st.secrets: return st.secrets["GOOGLE_API_KEY"]
    if key_name in os.environ:
        return os.environ[key_name]
    return None

def flatten_dataframe(df):
    if df.empty: return df
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def safe_yf_download(tickers, period, interval, retries=3):
    for i in range(retries):
        try:
            time.sleep(0.1) 
            df = yf.download(tickers, period=period, interval=interval, progress=False)
            if not df.empty:
                return flatten_dataframe(df)
        except Exception as e:
            if i == retries - 1: return pd.DataFrame()
            time.sleep(2 ** i)
    return pd.DataFrame()

# --- FRED API ENGINE ---
@st.cache_data(ttl=86400)
def get_fred_series(series_id, api_key, observation_start=None):
    if not api_key: return pd.DataFrame()
    st.session_state['fred_calls'] += 1
    base_url = "https://api.stlouisfed.org/fred/series/observations"
    if not observation_start:
        observation_start = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json", "observation_start": observation_start}
    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        if "observations" in data:
            df = pd.DataFrame(data["observations"])
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            return df[['date', 'value']].set_index('date').dropna()
    except: return pd.DataFrame()
    return pd.DataFrame()

# --- COINGECKO API ENGINE ---
@st.cache_data(ttl=300) 
def get_coingecko_stats(cg_id, api_key):
    if not cg_id or not api_key: return None
    st.session_state['coingecko_calls'] += 1
    url = f"https://api.coingecko.com/api/v3/coins/{cg_id}"
    params = {
        "localization": "false", "tickers": "false", "market_data": "true",
        "community_data": "true", "developer_data": "true", "sparkline": "false"
    }
    headers = {"x-cg-demo-api-key": api_key}
    try:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return {
                "rank": data.get('market_cap_rank', 'N/A'),
                "sentiment": data.get('sentiment_votes_up_percentage', 50),
                "hashing": data.get('hashing_algorithm', 'N/A'),
                "ath": data['market_data']['ath']['usd'],
                "ath_change": data['market_data']['ath_change_percentage']['usd'],
                "desc": data.get('description', {}).get('en', '').split('.')[0] + "." 
            }
        return None
    except Exception: return None

# --- LLM ENGINE ---
def get_technical_narrative(ticker, price, daily_pct, regime, ml_signal, gex_data, cot_data, levels, macro_data, api_key):
    if not api_key: return "AI Analyst unavailable (No Key)."
    st.session_state['gemini_calls'] += 1
    
    gex_text = "N/A"
    if gex_data is not None:
        total_gex = gex_data['gex'].sum()
        gex_text = f"Net Gamma: ${total_gex/1_000_000:.1f}M ({'Long/Sticky' if total_gex>0 else 'Short/Volatile'})"
    lvl_text = "N/A"
    if levels:
        lvl_text = f"Pivot: {levels['Pivot']:.2f}, R1: {levels['R1']:.2f}, S1: {levels['S1']:.2f}"
    
    macro_str = "N/A"
    if macro_data:
        macro_str = f"YieldCurve: {macro_data.get('yield_curve', 'N/A')}, Inflation(CPI): {macro_data.get('cpi', 'N/A')}%, Rates: {macro_data.get('rates', 'N/A')}%, MacroRegime: {macro_data.get('regime', 'N/A')}"
        
    prompt = f"""
    You are a Senior Portfolio Manager. Analyze data for {ticker} and write a 3-bullet executive summary.
    
    DATA: Price: {price:,.2f} ({daily_pct:.2f}%), Regime: {regime['regime'] if regime else 'Unknown'}, 
    ML: {ml_signal}, GEX: {gex_text}, COT: {cot_data['sentiment'] if cot_data else 'N/A'}, Levels: {lvl_text}
    MACRO CONTEXT: {macro_str}
    TASK:
    1. Synthesize Technicals + Macro.
    2. Identify key trigger level.
    3. Final Execution bias ("Buy Dips", "Fade", etc).
    Keep it concise. Bloomberg Terminal style.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        if "429" in str(e): return "⚠️ API LIMIT REACHED."
        return f"AI Analyst unavailable: {str(e)}"

def generate_deep_dive_thesis(ticker, price, change, regime, ml_signal, gex_data, cot_data, levels, news_summary, macro_data, api_key):
    if not api_key: return "API Key Missing."
    st.session_state['gemini_calls'] += 1
    
    gex_text = "N/A"
    if gex_data is not None:
        total_gex = gex_data['gex'].sum()
        gex_text = f"Net Gamma: ${total_gex/1_000_000:.1f}M"
    macro_str = "N/A"
    if macro_data:
        macro_str = f"YieldCurve: {macro_data.get('yield_curve', 'N/A')}, CPI: {macro_data.get('cpi', 'N/A')}%, Rates: {macro_data.get('rates', 'N/A')}%, Regime: {macro_data.get('regime', 'N/A')}"
    
    prompt = f"""
    Write a detailed Investment Thesis for {ticker}.
    DATA: Price: {price:,.2f} ({change:.2f}%), Regime: {regime['regime'] if regime else 'Unknown'}, 
    ML: {ml_signal}, GEX: {gex_text}, COT: {cot_data['sentiment'] if cot_data else 'N/A'}
    MACRO: {macro_str}
    NEWS: {news_summary}
    OUTPUT FORMAT (Markdown):
    ### 1. THE MACRO & TECHNICAL CROSSROADS
    ### 2. CORE ARGUMENT (Long/Short/Neutral)
    ### 3. THE BEAR/BULL CASE (Risks)
    ### 4. KEY LEVELS (Invalidation)
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Thesis Generation Failed: {str(e)}"

# --- NLP SENTIMENT ENGINE (SAFE) ---
def calculate_news_sentiment(news_items):
    """Calculates cumulative sentiment score from news headlines."""
    if not HAS_NLP or not news_items: return pd.DataFrame()
    
    scores = []
    for news in news_items:
        try:
            # Simple Polarity: -1.0 (Negative) to 1.0 (Positive)
            blob = TextBlob(f"{news['title']} {news['title']}") # Weight title double
            score = blob.sentiment.polarity
            
            # Convert publishedAt to simple time order for plot
            scores.append({
                "title": news['title'],
                "score": score,
                "time": news['time']
            })
        except: continue
        
    df = pd.DataFrame(scores)
    if df.empty: return pd.DataFrame()
    
    # Create cumulative sentiment "Momentum"
    df = df.iloc[::-1].reset_index(drop=True) 
    df['cumulative'] = df['score'].cumsum()
    return df

# --- QUANT ENGINES ---
def calculate_hurst(series, lags=range(2, 20)):
    try:
        tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    except: return 0.5

@st.cache_data(ttl=3600)
def get_market_regime(ticker):
    try:
        df = safe_yf_download(ticker, period="5y", interval="1d")
        if df.empty: return None
        data = df.copy()
        data['Returns'] = data['Close'].pct_change()
        data['Volatility'] = data['Returns'].rolling(20).std()
        data = data.dropna()
        X = data[['Returns', 'Volatility']].values
        gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42)
        gmm.fit(X)
        current_state = gmm.predict(X[[-1]])[0]
        probs = gmm.predict_proba(X[[-1]])[0]
        means = gmm.means_
        state_order = np.argsort(means[:, 1]) 
        regime_map = {state_order[0]: "LOW VOL (Trend)", state_order[1]: "NEUTRAL (Chop)", state_order[2]: "HIGH VOL (Crisis)"}
        regime_desc = regime_map.get(current_state, "Unknown")
        color = "bullish" if "LOW VOL" in regime_desc else "bearish" if "HIGH VOL" in regime_desc else "neutral"
        return {"regime": regime_desc, "color": color, "confidence": max(probs)}
    except: return None

@st.cache_data(ttl=86400)
def get_macro_ml_regime(cpi_df, rate_df):
    if cpi_df.empty or rate_df.empty: return None
    try:
        df = pd.merge(cpi_df, rate_df, left_index=True, right_index=True, how='inner')
        df.columns = ['CPI', 'Rates']
        df['CPI_YoY'] = df['CPI'].pct_change(12) * 100
        df = df.dropna()
        X = df[['CPI_YoY', 'Rates']].values
        if len(X) < 12: return None
        gmm = GaussianMixture(n_components=4, random_state=42)
        gmm.fit(X)
        curr_cpi = df['CPI_YoY'].iloc[-1]
        curr_rate = df['Rates'].iloc[-1]
        regime_name = "Neutral"
        if curr_cpi > 4 and curr_rate < curr_cpi: regime_name = "INFLATIONARY (Neg Real Rates)"
        elif curr_cpi > 4 and curr_rate > curr_cpi: regime_name = "TIGHTENING (Pos Real Rates)"
        elif curr_cpi < 2: regime_name = "DEFLATIONARY / RISK OFF"
        else: regime_name = "GOLDILOCKS / STABLE"
        return {"regime": regime_name, "cpi": curr_cpi, "rate": curr_rate}
    except: return None

@st.cache_data(ttl=3600)
def get_ml_prediction(ticker):
    try:
        df = safe_yf_download(ticker, period="2y", interval="1d") 
        if df.empty: return None, 0.5
        data = df.copy()
        data['Returns'] = data['Close'].pct_change()
        data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
        data['Vol_5d'] = data['Returns'].rolling(5).std()
        data['Mom_5d'] = data['Close'].pct_change(5)
        data = data.dropna()
        if len(data) < 50: return None, 0.5
        X = data[['Vol_5d', 'Mom_5d']]
        y = data['Target']
        model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
        model.fit(X, y)
        prob_up = model.predict_proba(X.iloc[[-1]])[0][1]
        return model, prob_up
    except: return None, 0.5

# --- GAMMA EXPOSURE (IV UPDATE) ---
def calculate_black_scholes_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

@st.cache_data(ttl=3600)
def get_gex_profile(opt_ticker):
    if opt_ticker is None: return None, None, None, None
    try:
        tk = yf.Ticker(opt_ticker)
        try:
            hist = tk.history(period="1d")
            spot_price = hist['Close'].iloc[-1] if not hist.empty else tk.fast_info.last_price
        except: return None, None, None, None
        
        if spot_price is None: return None, None, None, None
        exps = tk.options
        if not exps: return None, None, None, None
        
        target_exp = exps[1] if len(exps) > 1 else exps[0]
        chain = tk.option_chain(target_exp)
        calls, puts = chain.calls, chain.puts
        
        if calls.empty or puts.empty: return None, None, None, None
        
        # Calculate ATM IV
        atm_mask = (calls['strike'] > spot_price * 0.95) & (calls['strike'] < spot_price * 1.05)
        atm_calls = calls[atm_mask]
        avg_iv = atm_calls['impliedVolatility'].mean() * 100 if not atm_calls.empty else 0
        
        exp_date = datetime.strptime(target_exp, "%Y-%m-%d")
        days_to_exp = (exp_date - datetime.now()).days
        T = 0.001 if days_to_exp <= 0 else days_to_exp / 365.0
        gex_data = []
        strikes = sorted(list(set(calls['strike'].tolist() + puts['strike'].tolist())))
        for K in strikes:
            if K < spot_price * 0.75 or K > spot_price * 1.25: continue
            c_row = calls[calls['strike'] == K]
            p_row = puts[puts['strike'] == K]
            c_oi = c_row['openInterest'].iloc[0] if not c_row.empty else 0
            p_oi = p_row['openInterest'].iloc[0] if not p_row.empty else 0
            c_iv = c_row['impliedVolatility'].iloc[0] if not c_row.empty and 'impliedVolatility' in c_row.columns else 0.2
            p_iv = p_row['impliedVolatility'].iloc[0] if not p_row.empty and 'impliedVolatility' in p_row.columns else 0.2
            c_gamma = calculate_black_scholes_gamma(spot_price, K, T, 0.045, c_iv)
            p_gamma = calculate_black_scholes_gamma(spot_price, K, T, 0.045, p_iv)
            net_gex = (c_gamma * c_oi - p_gamma * p_oi) * spot_price * 100
            gex_data.append({"strike": K, "gex": net_gex})
            
        df = pd.DataFrame(gex_data, columns=['strike', 'gex'])
        return df, target_exp, spot_price, avg_iv
    except: return None, None, None, None

def calculate_volume_profile(df, bins=50):
    if df.empty: return None, None
    price_range = df['High'].max() - df['Low'].min()
    if price_range == 0: return None, None
    bin_size = price_range / bins
    df['PriceBin'] = ((df['Close'] - df['Low'].min()) // bin_size).astype(int)
    vol_profile = df.groupby('PriceBin')['Volume'].sum().reset_index()
    vol_profile['PriceLevel'] = df['Low'].min() + (vol_profile['PriceBin'] * bin_size)
    poc_idx = vol_profile['Volume'].idxmax()
    return vol_profile, vol_profile.loc[poc_idx, 'PriceLevel']

@st.cache_data(ttl=3600)
def get_seasonality_stats(daily_data, ticker_name):
    stats = {}
    try:
        df = daily_data.copy()
        df['Week_Num'] = df.index.to_period('W')
        high_days = df.groupby('Week_Num')['High'].idxmax().apply(lambda x: df.loc[x].name.day_name())
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        stats['day_high'] = high_days.value_counts().reindex(days_order, fill_value=0) / len(high_days) * 100
        df['Day'] = df.index.day
        df['Month_Week'] = np.ceil(df['Day'] / 7).astype(int)
        df['Returns'] = df['Close'].pct_change()
        stats['week_returns'] = df.groupby('Month_Week')['Returns'].mean() * 100
        try:
            intra = safe_yf_download(ticker_name, period="60d", interval="1h")
            if not intra.empty:
                if intra.index.tz is None: intra.index = intra.index.tz_localize('UTC')
                intra.index = intra.index.tz_convert('America/New_York')
                intra['Hour'] = intra.index.hour
                intra['Return'] = intra['Close'].pct_change()
                target_hours = [2,3,4,5,6, 8,9,10,11, 14,15,16,17,18, 20,21,22,23]
                stats['hourly_perf'] = intra[intra['Hour'].isin(target_hours)].groupby('Hour')['Return'].mean() * 100
        except: stats['hourly_perf'] = None
        return stats
    except: return None

@st.cache_data(ttl=3600)
def generate_monte_carlo(stock_data, days=126, simulations=1000):
    if stock_data is None or stock_data.empty or len(stock_data) < 2: return None, None
    try:
        close = stock_data['Close']
        log_returns = np.log(1 + close.pct_change())
        u, var = log_returns.mean(), log_returns.var()
        drift = u - (0.5 * var)
        stdev = log_returns.std()
        price_paths = np.zeros((days + 1, simulations))
        price_paths[0] = close.iloc[-1]
        daily_returns = np.exp(drift + stdev * np.random.normal(0, 1, (days, simulations)))
        for t in range(1, days + 1): price_paths[t] = price_paths[t - 1] * daily_returns[t - 1]
        return pd.date_range(start=close.index[-1], periods=days + 1, freq='B'), price_paths
    except: return None, None

def calculate_technical_radar(df):
    if df.empty or len(df) < 30: return None
    data = df.copy()
    close = data['Close']
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    k = close.ewm(span=12, adjust=False, min_periods=12).mean()
    d = close.ewm(span=26, adjust=False, min_periods=26).mean()
    data['MACD'] = k - d
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False, min_periods=9).mean()
    data['EMA_20'] = close.ewm(span=20, adjust=False).mean()
    data['EMA_50'] = close.ewm(span=50, adjust=False).mean()
    last = data.iloc[-1]
    signals = {}
    
    if last['RSI'] < 30: signals['RSI'] = {"val": f"{last['RSI']:.0f}", "bias": "OVERSOLD (Bull)", "col": "bullish"}
    elif last['RSI'] > 70: signals['RSI'] = {"val": f"{last['RSI']:.0f}", "bias": "OVERBOUGHT (Bear)", "col": "bearish"}
    else: signals['RSI'] = {"val": f"{last['RSI']:.0f}", "bias": "NEUTRAL", "col": "neutral"}
    
    macd_hist = last['MACD'] - last['MACD_Signal']
    if macd_hist > 0 and last['MACD'] > 0: signals['MACD'] = {"val": f"{macd_hist:.2f}", "bias": "BULLISH", "col": "bullish"}
    elif macd_hist < 0 and last['MACD'] < 0: signals['MACD'] = {"val": f"{macd_hist:.2f}", "bias": "BEARISH", "col": "bearish"}
    else: signals['MACD'] = {"val": f"{macd_hist:.2f}", "bias": "NEUTRAL", "col": "neutral"}
    
    if last['Close'] > last['EMA_20'] and last['EMA_20'] > last['EMA_50']:
        signals['Trend'] = {"val": "Uptrend", "bias": "STRONG BULL", "col": "bullish"}
    elif last['Close'] < last['EMA_20'] and last['EMA_20'] < last['EMA_50']:
        signals['Trend'] = {"val": "Downtrend", "bias": "STRONG BEAR", "col": "bearish"}
    else: signals['Trend'] = {"val": "Chop", "bias": "WEAK/MIXED", "col": "neutral"}
    return signals

# --- NEWS & ECO ---
def parse_eco_value(val_str):
    if not isinstance(val_str, str) or val_str == '': return None
    clean = val_str.replace('%', '').replace(',', '')
    multiplier = 1.0
    if 'K' in clean.upper(): multiplier = 1000.0; clean = clean.upper().replace('K', '')
    elif 'M' in clean.upper(): multiplier = 1000000.0; clean = clean.upper().replace('M', '')
    elif 'B' in clean.upper(): multiplier = 1000000000.0; clean = clean.upper().replace('B', '')
    try: return float(clean) * multiplier
    except: return None

def analyze_eco_context(actual_str, forecast_str, previous_str):
    is_happened = actual_str is not None and actual_str != ""
    val_actual = parse_eco_value(actual_str)
    val_forecast = parse_eco_value(forecast_str)
    val_prev = parse_eco_value(previous_str)
    context_str = ""
    bias = "Neutral"
    if is_happened:
        if val_actual is not None and val_forecast is not None:
            context_str = f"Actual ({actual_str}) vs Forecast ({forecast_str})"
            delta = val_actual - val_forecast
            pct_dev = abs(delta / val_forecast) if val_forecast != 0 else 1.0
            if pct_dev < 0.02: bias = "Mean Reverting"
            elif delta > 0: bias = "Bullish"
            else: bias = "Bearish"
        else: context_str = f"Actual: {actual_str}"
    else:
        if val_forecast is not None and val_prev is not None:
            context_str = f"Forecast ({forecast_str}) vs Prev ({previous_str})"
            delta = val_forecast - val_prev
            pct_dev = abs(delta / val_prev) if val_prev != 0 else 1.0
            if pct_dev < 0.02: bias = "Mean Reverting"
            elif delta > 0: bias = "Bullish Exp."
            else: bias = "Bearish Exp."
        else: context_str = "Waiting for Data..."
    return context_str, bias

@st.cache_data(ttl=14400) 
def get_financial_news_general(api_key, query="Finance"):
    if not api_key: return []
    st.session_state['news_calls'] += 1
    try:
        newsapi = NewsApiClient(api_key=api_key)
        all_articles = newsapi.get_everything(q=query, language='en', sort_by='publishedAt')
        articles = []
        if all_articles['status'] == 'ok':
            for art in all_articles['articles'][:6]:
                articles.append({"title": art['title'], "source": art['source']['name'], "url": art['url'], "time": art['publishedAt']})
        return articles
    except: return []

@st.cache_data(ttl=14400)
def get_forex_factory_news(api_key, news_type='breaking'):
    if not api_key: return []
    st.session_state['rapid_calls'] += 1
    base_url = "https://forex-factory-scraper1.p.rapidapi.com/"
    endpoints = {'breaking': "latest_breaking_news", 'fundamental': "latest_fundamental_analysis_news"}
    url = base_url + endpoints.get(news_type, "latest_breaking_news")
    headers = {"x-rapidapi-key": api_key, "x-rapidapi-host": "forex-factory-scraper1.p.rapidapi.com"}
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        normalized_news = []
        if isinstance(data, list):
            for item in data[:6]:
                normalized_news.append({
                    "title": item.get('title', 'No Title'),
                    "url": item.get('link', item.get('url', '#')),
                    "source": "ForexFactory",
                    "time": item.get('date', item.get('time', 'Recent'))
                })
        return normalized_news
    except: return []

# --- NEW: ECONOMIC CALENDAR WITH MOCK/DEMO SUPPORT ---
def get_mock_calendar_data():
    """Returns mock data formatted exactly how the app expects it for seamless testing."""
    return [
        {"currency": "USD", "impact": "High", "event_name": "[DEMO] Unemployment Claims", "actual": "210K", "forecast": "215K", "previous": "208K", "time": "8:30am"},
        {"currency": "USD", "impact": "High", "event_name": "[DEMO] Philly Fed Mfg Index", "actual": "15.5", "forecast": "8.0", "previous": "12.2", "time": "8:30am"},
        {"currency": "USD", "impact": "Medium", "event_name": "[DEMO] Natural Gas Storage", "actual": "85B", "forecast": "82B", "previous": "79B", "time": "10:30am"},
        {"currency": "USD", "impact": "High", "event_name": "[DEMO] Powell Speaks", "actual": "", "forecast": "", "previous": "", "time": "2:00pm"},
    ]

@st.cache_data(ttl=21600)
def get_economic_calendar(api_key, use_demo=False):
    # 1. Check Demo Mode Flag
    if use_demo:
        return get_mock_calendar_data()

    # 2. Check API Key
    if not api_key: 
        return get_mock_calendar_data() # Fallback if no key

    st.session_state['rapid_calls'] += 1
    try:
        url = "https://forex-factory-scraper1.p.rapidapi.com/get_real_time_calendar_details"
        now = datetime.now()
        querystring = {"calendar": "Forex", "year": str(now.year), "month": str(now.month), "day": str(now.day), "currency": "ALL", "event_name": "ALL", "timezone": "GMT-04:00 Eastern Time (US & Canada)", "time_format": "12h"}
        headers = {"x-rapidapi-host": "forex-factory-scraper1.p.rapidapi.com", "x-rapidapi-key": api_key}
        
        response = requests.get(url, headers=headers, params=querystring)
        
        # 3. Handle 429 Limit Logic (Quota Saver)
        if response.status_code == 429:
            return get_mock_calendar_data() # Fallback on limit reached
            
        data = response.json()
        raw_events = data if isinstance(data, list) else data.get('data', [])
        
        filtered_events = []
        for e in raw_events:
            if e.get('currency') == 'USD' and (e.get('impact') == 'High' or e.get('impact') == 'Medium'):
                filtered_events.append(e)
                
        if filtered_events: return filtered_events
    except: 
        pass 
    
    return []

# --- TRADING LOGIC ---
@st.cache_data(ttl=300)
def run_strategy_backtest(ticker):
    try:
        df = safe_yf_download(ticker, period="2y", interval="1d")
        if df.empty: return None
        df['Returns'] = df['Close'].pct_change()
        df['Range'] = df['High'] - df['Low']
        df['TR'] = pd.concat([df['Range'], (df['High'] - df['Close'].shift(1)).abs(), (df['Low'] - df['Close'].shift(1)).abs()], axis=1).max(axis=1)
        df['Log_TR'] = np.log(df['TR'] / df['Close'])
        df['Vol_Forecast'] = df['Log_TR'].ewm(span=10).mean()
        df['Vol_Baseline'] = df['Log_TR'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['Signal'] = np.where((df['Vol_Forecast'] > df['Vol_Baseline']) & (df['Close'] > df['SMA_50']), 1, 0)
        df['Strategy_Returns'] = df['Signal'].shift(1) * df['Returns']
        df['Cum_BnH'] = (1 + df['Returns']).cumprod()
        df['Cum_Strat'] = (1 + df['Strategy_Returns']).cumprod()
        total_return = df['Cum_Strat'].iloc[-1] - 1
        sharpe = (df['Strategy_Returns'].mean() / df['Strategy_Returns'].std()) * np.sqrt(252) if df['Strategy_Returns'].std() != 0 else 0
        current_signal = "LONG" if df['Signal'].iloc[-1] == 1 else "CASH/NEUTRAL"
        return {"df": df, "signal": current_signal, "return": total_return, "sharpe": sharpe, "equity_curve": df['Cum_Strat']}
    except: return None

def calculate_vwap_bands(df):
    if df.empty: return df
    df = df.copy()
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VP'] = df['TP'] * df['Volume']
    df['Date'] = df.index.date
    df['Cum_VP'] = df.groupby('Date')['VP'].cumsum()
    df['Cum_Vol'] = df.groupby('Date')['Volume'].cumsum()
    df['VWAP'] = df['Cum_VP'] / df['Cum_Vol']
    df['Sq_Dist'] = df['Volume'] * (df['TP'] - df['VWAP'])**2
    df['Cum_Sq_Dist'] = df.groupby('Date')['Sq_Dist'].cumsum()
    df['Std_Dev'] = np.sqrt(df['Cum_Sq_Dist'] / df['Cum_Vol'])
    df['Upper_Band_1'] = df['VWAP'] + df['Std_Dev']
    df['Lower_Band_1'] = df['VWAP'] - df['Std_Dev']
    return df

@st.cache_data(ttl=300) 
def get_relative_strength(asset_ticker, benchmark_ticker="SPY"):
    try:
        asset = safe_yf_download(asset_ticker, period="5d", interval="15m")
        bench = safe_yf_download(benchmark_ticker, period="5d", interval="15m")
        if asset.empty or bench.empty: return pd.DataFrame()
        df = pd.DataFrame(index=asset.index)
        df['Asset_Close'] = asset['Close']
        df['Bench_Close'] = bench['Close']
        df = df.dropna()
        current_date = df.index[-1].date()
        session_data = df[df.index.date == current_date].copy()
        if session_data.empty: return pd.DataFrame()
        session_data['Asset_Pct'] = (session_data['Asset_Close'] / session_data['Asset_Close'].iloc[0]) - 1
        session_data['Bench_Pct'] = (session_data['Bench_Close'] / session_data['Bench_Close'].iloc[0]) - 1
        session_data['RS_Score'] = session_data['Asset_Pct'] - session_data['Bench_Pct']
        return session_data
    except: return pd.DataFrame()

def get_key_levels(daily_df):
    if daily_df.empty: return {}
    try: last_complete_day = daily_df.iloc[-2]
    except: return {}
    high, low, close = last_complete_day['High'], last_complete_day['Low'], last_complete_day['Close']
    pivot = (high + low + close) / 3
    return {
        "PDH": high, "PDL": low, "PDC": close, "Pivot": pivot, 
        "R1": (2 * pivot) - low, "S1": (2 * pivot) - high
    }

@st.cache_data(ttl=3600)
def get_correlations(base_ticker, api_key):
    try:
        tickers = {"Base": base_ticker, "VIX": "^VIX", "10Y Yield": "^TNX", "Gold": "GC=F"}
        yf_data = safe_yf_download(list(tickers.values()), period="6mo", interval="1d")
        fred_data = get_fred_series("DTWEXAFEGS", api_key) 
        if yf_data.empty: return pd.Series()
        
        if isinstance(yf_data.columns, pd.MultiIndex): yf_df = yf_data['Close'].copy()
        elif 'Close' in yf_data.columns: yf_df = yf_data['Close'].copy()
        else: yf_df = yf_data.copy()
            
        if isinstance(yf_df, pd.Series): yf_df = yf_df.to_frame(name=list(tickers.keys())[0])
        found_cols = {c: k for k, v in tickers.items() if v in yf_df.columns}
        yf_df.rename(columns=found_cols, inplace=True)
        
        combined = yf_df
        if not fred_data.empty:
            fred_data = fred_data.rename(columns={'value': 'Dollar'})
            if yf_df.index.tz is not None: yf_df.index = yf_df.index.tz_localize(None)
            combined = pd.concat([yf_df, fred_data], axis=1).dropna()
            
        if combined.empty or 'Base' not in combined.columns: return pd.Series()
        corrs = combined.pct_change().rolling(20).corr(combined['Base'].pct_change()).iloc[-1]
        return corrs.drop('Base') 
    except: return pd.Series()

# --- NEW: COT QUANT ENGINE ---
def clean_headers(df):
    if isinstance(df.columns[0], int):
        for i in range(20):
            row_str = " ".join(df.iloc[i].astype(str).tolist()).lower()
            if "market" in row_str and ("long" in row_str or "positions" in row_str):
                df.columns = df.iloc[i]
                return df.iloc[i+1:].reset_index(drop=True)
    return df

def map_columns(df, report_type):
    col_map = {}
    def get_col(keywords, exclude=None):
        for col in df.columns:
            c_str = str(col).lower()
            if all(k in c_str for k in keywords):
                if exclude and any(x in c_str for x in exclude): continue
                return col
        return None

    col_map['date'] = get_col(['report', 'date']) or get_col(['as', 'of', 'date']) or get_col(['date'])
    col_map['market'] = get_col(['market'])

    if "disaggregated" in report_type:
        col_map['spec_long'] = get_col(['money', 'long'], exclude=['lev'])
        col_map['spec_short'] = get_col(['money', 'short'], exclude=['lev'])
        col_map['hedge_long'] = get_col(['prod', 'merc', 'long'])
        col_map['hedge_short'] = get_col(['prod', 'merc', 'short'])
        
    elif "financial" in report_type:
        col_map['spec_long'] = get_col(['lev', 'money', 'long'])
        col_map['spec_short'] = get_col(['lev', 'money', 'short'])
        col_map['hedge_long'] = get_col(['asset', 'mgr', 'long'])
        col_map['hedge_short'] = get_col(['asset', 'mgr', 'short'])

    final_map = {v: k for k, v in col_map.items() if v}
    df = df.rename(columns=final_map)
    
    for c in ['spec_long', 'spec_short', 'hedge_long', 'hedge_short']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            
    return df

@st.cache_data(ttl=86400)
def fetch_cot_history(asset_name, start_year=2024):
    if asset_name not in COT_MAPPING: return None
    config = COT_MAPPING[asset_name]
    keywords = config['keywords']
    report_type = config['report_type']
    
    master_df = pd.DataFrame()
    current_year = datetime.now().year
    
    for y in range(start_year, current_year + 1):
        df_year = None
        if HAS_COT_LIB:
            try: df_year = cot.cot_year(year=y, cot_report_type=report_type)
            except: pass
        
        # Fallback Direct
        if df_year is None or df_year.empty:
            try:
                url = f"https://www.cftc.gov/files/dea/history/deahistfo{y}.zip" # Default to Future Only
                r = requests.get(url)
                if r.status_code == 200:
                    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                        filename = z.namelist()[0]
                        with z.open(filename) as f:
                            df_year = pd.read_csv(f, low_memory=False)
            except: pass
            
        if df_year is not None and not df_year.empty:
            df_year = clean_headers(df_year)
            df_year = map_columns(df_year, report_type)
            master_df = pd.concat([master_df, df_year])
            
    if master_df.empty: return None
    
    # Filter
    if 'market' not in master_df.columns: return None
    mask = master_df['market'].astype(str).apply(lambda x: all(k.lower() in x.lower() for k in keywords))
    df_asset = master_df[mask].copy()
    
    if 'date' in df_asset.columns:
        df_asset['date'] = pd.to_datetime(df_asset['date'], errors='coerce')
        df_asset = df_asset.sort_values('date')
        
    return df_asset

# --- QUANT HELPERS ---
def calculate_z_score(series, window=52):
    roll_mean = series.rolling(window=window).mean()
    roll_std = series.rolling(window=window).std()
    return (series - roll_mean) / roll_std

def get_percentile(series, current_value):
    return (series < current_value).mean() * 100

def generate_cot_analysis(spec_net, hedge_net, spec_label, hedge_label):
    spec_sent = "🟢 BULLISH" if spec_net > 0 else "🔴 BEARISH"
    hedge_sent = "🟢 BULLISH" if hedge_net > 0 else "🔴 BEARISH"
    if (spec_net > 0 and hedge_net < 0) or (spec_net < 0 and hedge_net > 0):
        structure = "✅ **Healthy Structure:** Speculators and Hedgers are on opposite sides (Risk Transfer active)."
    else:
        structure = "⚠️ **Anomaly:** Both groups are positioned in the same direction. Watch for liquidity gaps."
    return f"""
    * **{spec_label}:** {spec_sent} (Net: {int(spec_net):,})
    * **{hedge_label}:** {hedge_sent} (Net: {int(hedge_net):,})
    {structure}
    """

# --- DATA FETCHERS ---
@st.cache_data(ttl=300)
def get_daily_data(ticker):
    try: return safe_yf_download(ticker, period="10y", interval="1d")
    except: return pd.DataFrame()

@st.cache_data(ttl=300)
def get_dxy_data():
    try: return safe_yf_download("DX-Y.NYB", period="10y", interval="1d")
    except: return pd.DataFrame()

@st.cache_data(ttl=300)
def get_intraday_data(ticker):
    try: return safe_yf_download(ticker, period="5d", interval="15m")
    except: return pd.DataFrame()

def terminal_chart_layout(fig, title="", height=350):
    fig.update_layout(
        title=dict(text=title, font=dict(color="#ff9900", family="Arial")),
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        height=height,
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(showgrid=True, gridcolor="#222", zerolinecolor="#222"),
        yaxis=dict(showgrid=True, gridcolor="#222", zerolinecolor="#222"),
        font=dict(family="Courier New", color="#e0e0e0")
    )
    return fig

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h3 style='color: #ff9900;'>COMMAND LINE</h3>", unsafe_allow_html=True)
    selected_asset = st.selectbox("SEC / Ticker", list(ASSETS.keys()))
    asset_info = ASSETS[selected_asset]
    
    # --- UPDATED: QUOTA SAVER TOGGLE ---
    use_demo_data = st.checkbox("🛠️ USE DEMO DATA (Save Quota)", value=True, help="Use mock data for Calendar to save RapidAPI credits.")
    
    st.markdown("---")
    
    with st.expander("📡 API QUOTA MONITOR", expanded=True):
        st.markdown("<div style='font-size:0.7em; color:gray;'>Session Usage vs Hard Limits</div>", unsafe_allow_html=True)
        st.write(f"**NewsAPI** ({st.session_state['news_calls']} / 100)")
        st.progress(min(st.session_state['news_calls'] / 100, 1.0))
        st.write(f"**Gemini AI** ({st.session_state['gemini_calls']} / 20)")
        st.progress(min(st.session_state['gemini_calls'] / 20, 1.0))
        st.write(f"**RapidAPI** ({st.session_state['rapid_calls']} / 10)")
        st.progress(min(st.session_state['rapid_calls'] / 10, 1.0))
        st.write(f"**FRED** ({st.session_state['fred_calls']} calls)")
        st.write(f"**CoinGecko** ({st.session_state['coingecko_calls']} calls)")
        
        if use_demo_data:
            st.success("🟢 DEMO MODE ACTIVE")
        else:
            st.warning("🔴 LIVE API MODE")
        
        # Dependency Status
        if not HAS_NLP:
            st.warning("NLP Disabled: `textblob` missing")
        if not HAS_COT_LIB:
            st.info("Using Direct COT Fetch (No Lib)")
            
    st.markdown("---")
    rapid_key = get_api_key("rapidapi_key")
    news_key = get_api_key("news_api_key")
    gemini_key = get_api_key("gemini_api_key")
    cg_key = get_api_key("coingecko_key") 
    fred_key = get_api_key("fred_api_key")
    
    if st.button(">> REFRESH DATA"): 
        st.cache_data.clear()
        st.rerun()

# --- MAIN DASHBOARD ---
st.markdown(f"<h1 style='border-bottom: 2px solid #ff9900;'>{selected_asset} <span style='font-size:0.5em; color:white;'>TERMINAL PRO V5.18</span></h1>", unsafe_allow_html=True)

# Fetch Data
daily_data = get_daily_data(asset_info['ticker'])
dxy_data = get_dxy_data()
intraday_data = get_intraday_data(asset_info['ticker'])

# --- UPDATED: FETCH CALENDAR WITH TOGGLE ---
eco_events = get_economic_calendar(rapid_key, use_demo=use_demo_data)

# Fetch News from BOTH sources
news_general = get_financial_news_general(news_key, query=asset_info.get('news_query', 'Finance'))
news_ff = get_forex_factory_news(rapid_key, 'breaking')
combined_news_for_llm = news_general[:5] + news_ff[:5]

# Engines
_, ml_prob = get_ml_prediction(asset_info['ticker'])
gex_df, gex_date, gex_spot, current_iv = get_gex_profile(asset_info['opt_ticker'])
vol_profile, poc_price = calculate_volume_profile(intraday_data)
hurst = calculate_hurst(daily_data['Close'].values) if not daily_data.empty else 0.5
regime_data = get_market_regime(asset_info['ticker'])
# OLD COT FETCH REMOVED -> REPLACED BY SECTION 8 LOGIC
tech_radar = calculate_technical_radar(daily_data)
correlations = get_correlations(asset_info['ticker'], fred_key)
news_sentiment_df = calculate_news_sentiment(combined_news_for_llm)

# --- 1. OVERVIEW ---
if not daily_data.empty:
    close, high, low = daily_data['Close'], daily_data['High'], daily_data['Low']
    curr = close.iloc[-1]
    pct = ((curr - close.iloc[-2]) / close.iloc[-2]) * 100
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LAST PX", f"{curr:,.2f}", f"{pct:.2f}%")
    
    ml_bias = "BULLISH" if ml_prob > 0.55 else "BEARISH" if ml_prob < 0.45 else "NEUTRAL"
    ml_conf = abs(ml_prob - 0.5) * 200
    ml_color = "bullish" if ml_bias == "BULLISH" else "bearish" if ml_bias == "BEARISH" else "neutral"
    
    c2.markdown(f"""
    <div class='terminal-box' style="text-align:center; padding:5px;">
        <div style="font-size:0.8em; color:#ff9900;">AI PREDICTION</div>
        <span class='{ml_color}'>{ml_bias}</span>
        <div style="font-size:0.8em; margin-top:5px; color:#aaa;">CONF: {ml_conf:.0f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    hurst_type = "TRENDING" if hurst > 0.55 else "MEAN REVERT" if hurst < 0.45 else "RANDOM WALK"
    h_color = "#00ff00" if hurst > 0.55 else "#ff3333" if hurst < 0.45 else "gray"
    
    if regime_data:
        c3.markdown(f"""
        <div class='terminal-box' style="padding:10px;">
            <div style="font-size:0.8em; color:#ff9900;">QUANT REGIME</div>
            <div style="font-size:1.1em; font-weight:bold;" class='{regime_data['color']}'>{regime_data['regime']}</div>
            <div style="font-size:0.7em; display:flex; justify-content:space-between; margin-top:5px;">
                <span>FRACTAL:</span>
                <span style="color:{h_color}">{hurst_type}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        c3.info("Calculating Regime...")
    
    c4.metric("HIGH/LOW", f"{high.max():,.2f} / {low.min():,.2f}")
    
    # --- CHART: DXY OVERLAY ---
    fig = go.Figure()
    
    # Trace 1: Asset Candlesticks
    fig.add_trace(go.Candlestick(
        x=daily_data.index, 
        open=daily_data['Open'], 
        high=high, 
        low=low, 
        close=close, 
        name="Price"
    ))
    
    if poc_price:
        fig.add_hline(y=poc_price, line_dash="dash", line_color="yellow", annotation_text="POC", annotation_position="bottom right")

    # Trace 3: DXY Overlay
    if not dxy_data.empty:
        dxy_aligned = dxy_data['Close'].reindex(daily_data.index, method='ffill')
        fig.add_trace(go.Scatter(
            x=dxy_aligned.index, 
            y=dxy_aligned.values, 
            name="DXY (Dollar)", 
            line=dict(color='orange', width=2),
            opacity=0.7,
            yaxis="y2"
        ))

    # Layout Updates for Dual Axis
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(showgrid=True, gridcolor="#222", zerolinecolor="#222"),
        yaxis=dict(showgrid=True, gridcolor="#222", zerolinecolor="#222", title="Asset Price"),
        yaxis2=dict(
            title="DXY Index",
            overlaying="y",
            side="right",
            showgrid=False,
            title_font=dict(color="orange"),
            tickfont=dict(color="orange")
        ),
        font=dict(family="Courier New", color="#e0e0e0"),
        legend=dict(orientation="h", y=1.02, x=0, bgcolor="rgba(0,0,0,0)")
    )
    
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# --- 1B. TECHNICAL RADAR ---
st.markdown("---")
st.markdown("### 📡 TECHNICAL RADAR (TRIANGULATION)")
if tech_radar:
    tr1, tr2, tr3 = st.columns(3)
    
    with tr1:
        st.markdown(f"""
        <div class='terminal-box'>
            <div style='color:gray; font-size:0.8em;'>MOMENTUM (RSI)</div>
            <div style='font-size:1.5em;'>{tech_radar['RSI']['val']}</div>
            <span class='{tech_radar['RSI']['col']}'>{tech_radar['RSI']['bias']}</span>
        </div>
        """, unsafe_allow_html=True)
        
    with tr2:
        st.markdown(f"""
        <div class='terminal-box'>
            <div style='color:gray; font-size:0.8em;'>TREND (EMA)</div>
            <div style='font-size:1.5em;'>{tech_radar['Trend']['val']}</div>
            <span class='{tech_radar['Trend']['col']}'>{tech_radar['Trend']['bias']}</span>
        </div>
        """, unsafe_allow_html=True)
    with tr3:
        st.markdown(f"""
        <div class='terminal-box'>
            <div style='color:gray; font-size:0.8em;'>MACD (MOMENTUM)</div>
            <div style='font-size:1.5em;'>{tech_radar['MACD']['val']}</div>
            <span class='{tech_radar['MACD']['col']}'>{tech_radar['MACD']['bias']}</span>
        </div>
        """, unsafe_allow_html=True)

# --- 1C. RESTORED: COINGECKO INTEGRATION ---
cg_id = asset_info.get('cg_id')
if cg_id and cg_key:
    st.markdown("---")
    st.markdown("### 🦎 COINGECKO FUNDAMENTALS")
    
    with st.spinner("Fetching CoinGecko Data..."):
        cg_data = get_coingecko_stats(cg_id, cg_key)
    
    if cg_data:
        c_cg1, c_cg2, c_cg3, c_cg4 = st.columns(4)
        c_cg1.metric("Market Rank", f"#{cg_data['rank']}")
        
        ath_color = "red" if cg_data['ath_change'] < -20 else "orange"
        c_cg2.markdown(f"""
        <div class='terminal-box'>
            <div style='font-size:0.8em; color:gray;'>ATH DRAWDOWN</div>
            <div style='color:{ath_color}; font-size:1.2em;'>{cg_data['ath_change']:.2f}%</div>
            <div style='font-size:0.7em;'>High: ${cg_data['ath']:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
        sent_val = cg_data['sentiment']
        sent_color = "#00ff00" if sent_val > 60 else "#ff3333" if sent_val < 40 else "gray"
        c_cg3.markdown(f"""
        <div class='terminal-box'>
            <div style='font-size:0.8em; color:gray;'>COMMUNITY SENTIMENT</div>
            <div style='color:{sent_color}; font-size:1.2em;'>{sent_val}% Bullish</div>
            <progress value="{sent_val}" max="100" style="width:100%; height:5px;"></progress>
        </div>
        """, unsafe_allow_html=True)
        
        c_cg4.markdown(f"""
        <div class='terminal-box'>
            <div style='font-size:0.8em; color:gray;'>ALGORITHM</div>
            <div style='color:white; font-size:1em;'>{cg_data['hashing']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("Asset Description"):
            st.write(cg_data['desc'])
    else:
        st.warning("CoinGecko API Limit Reached (30 calls/min). Please wait.")

# --- 2. INTRADAY TACTICAL FEED ---
st.markdown("---")
st.markdown("### 🔭 INTRADAY TACTICAL FEED")
rs_data = get_relative_strength(asset_info['ticker'])
key_levels = get_key_levels(daily_data)
col_intra_1, col_intra_2 = st.columns([2, 1])
with col_intra_1:
    if not rs_data.empty:
        curr_rs = rs_data['RS_Score'].iloc[-1]
        rs_color = "#00ff00" if curr_rs > 0 else "#ff3333"
        rs_text = "OUTPERFORMING SPY" if curr_rs > 0 else "UNDERPERFORMING SPY"
        st.markdown(f"**RELATIVE STRENGTH (vs SPY)**: <span style='color:{rs_color}'>{rs_text}</span>", unsafe_allow_html=True)
        fig_rs = go.Figure()
        fig_rs.add_hline(y=0, line_color="#333", line_dash="dash")
        fig_rs.add_trace(go.Scatter(x=rs_data.index, y=rs_data['RS_Score'], mode='lines', name='Alpha', line=dict(color=rs_color, width=2), fill='tozeroy'))
        terminal_chart_layout(fig_rs, title="INTRADAY ALPHA (Real-Time)", height=250)
        st.plotly_chart(fig_rs, use_container_width=True)
with col_intra_2:
    st.markdown("**🔑 KEY ALGO LEVELS**")
    if key_levels:
        cur_price = intraday_data['Close'].iloc[-1] if not intraday_data.empty else 0
        def get_lvl_color(level, current):
            if current == 0: return "white"
            dist = abs(level - current) / current
            if dist < 0.002: return "#ffff00" # Near
            if level > current: return "#ff3333" # Resistance
            return "#00ff00" # Support
        levels_list = [("R1 (Resist)", key_levels['R1']), ("PDH (High)", key_levels['PDH']), ("PIVOT (Daily)", key_levels['Pivot']), ("PDL (Low)", key_levels['PDL']), ("S1 (Support)", key_levels['S1'])]
        for name, price in levels_list:
            c_code = get_lvl_color(price, cur_price)
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; border-bottom:1px solid #222; padding:5px;">
                <span style="color:#aaa;">{name}</span>
                <span style="color:{c_code}; font-family:monospace;">{price:,.2f}</span>
            </div>
            """, unsafe_allow_html=True)

# --- 3. EVENTS & NEWS & MACRO ---
st.markdown("---")
col_eco, col_news = st.columns([2, 1])
with col_eco:
    st.markdown("### 📅 ECONOMIC EVENTS (USD)")
    # Check if we are in demo mode for UI feedback
    if use_demo_data:
        st.caption("🟢 USING DEMO DATA (API Quota Saver Active)")
    
    if eco_events:
        cal_data = []
        for event in eco_events:
            context, bias = analyze_eco_context(event.get('actual', ''), event.get('forecast', ''), event.get('previous', ''))
            cal_data.append({"TIME": event.get('time', 'N/A'), "EVENT": event.get('event_name', 'Unknown'), "DATA CONTEXT": context, "BIAS": bias})
        df_cal = pd.DataFrame(cal_data)
        def color_bias(val):
            color = 'white'
            if 'Bullish' in val: color = '#00ff00' 
            elif 'Bearish' in val: color = '#ff3333'
            elif 'Mean' in val: color = '#cccc00'
            return f'color: {color}'
        if not df_cal.empty: 
            st.dataframe(df_cal.style.map(color_bias, subset=['BIAS']), use_container_width=True, hide_index=True)
    else: st.info("NO HIGH IMPACT USD EVENTS SCHEDULED.")
with col_news:
    # --- NEW: SENTIMENT CHART ---
    st.markdown(f"### 📰 {asset_info.get('news_query', 'LATEST')} WIRE & SENTIMENT")
    
    if HAS_NLP and not news_sentiment_df.empty:
         fig_sent = go.Figure()
         fig_sent.add_trace(go.Scatter(
             x=news_sentiment_df.index, 
             y=news_sentiment_df['cumulative'],
             mode='lines+markers',
             line=dict(color='#00e6ff', width=2, shape='spline'),
             name="Sentiment"
         ))
         fig_sent.update_layout(
             title="NLP SENTIMENT VELOCITY (Current Batch)",
             height=150,
             margin=dict(l=10, r=10, t=30, b=10),
             paper_bgcolor="#111", plot_bgcolor="#111",
             font=dict(size=10, color="white"),
             xaxis=dict(showgrid=False, visible=False),
             yaxis=dict(showgrid=True, gridcolor="#333")
         )
         st.plotly_chart(fig_sent, use_container_width=True)
    elif not HAS_NLP:
        st.warning("Install `textblob` to enable Sentiment Analysis.")
    
    tab_gen, tab_ff = st.tabs(["📰 GENERAL", "⚡ FOREX FACTORY"])
    def render_news(items):
        if items:
            for news in items:
                st.markdown(f"""
                <div style="border-bottom:1px solid #333; padding-bottom:5px; margin-bottom:5px;">
                    <a class='news-link' href='{news['url']}' target='_blank'>▶ {news['title']}</a><br>
                    <span style='font-size:0.7em; color:gray;'>{news['time']} | {news['source']}</span>
                </div>
                """, unsafe_allow_html=True)
        else: st.markdown("<div style='color:gray;'>No data.</div>", unsafe_allow_html=True)
    with tab_gen: render_news(news_general)
    with tab_ff: render_news(news_ff)

# --- NEW: FRED MACRO DASHBOARD ---
st.markdown("---")
st.markdown("### 🇺🇸 FED LIQUIDITY & MACRO (FRED)")
macro_context_data = {} 
if fred_key:
    # 1. Fetch Key Series
    df_yield = get_fred_series("T10Y2Y", fred_key)
    df_ff = get_fred_series("FEDFUNDS", fred_key)
    df_cpi = get_fred_series("CPIAUCSL", fred_key)
    df_m2 = get_fred_series("M2SL", fred_key)
    
    # 2. Macro ML Regime Engine
    macro_regime = get_macro_ml_regime(df_cpi, df_ff)
    
    # 3. Store for AI
    if not df_yield.empty: macro_context_data['yield_curve'] = f"{df_yield['value'].iloc[-1]:.2f}"
    if not df_cpi.empty: macro_context_data['cpi'] = f"{(df_cpi['value'].pct_change(12).iloc[-1]*100):.2f}"
    if not df_ff.empty: macro_context_data['rates'] = f"{df_ff['value'].iloc[-1]:.2f}"
    if macro_regime: macro_context_data['regime'] = macro_regime['regime']
    
    # --- UI LAYOUT ---
    macro_col_main, macro_col_ml = st.columns([3, 1])
    
    with macro_col_main:
        macro_tab1, macro_tab2 = st.tabs(["YIELD CURVE & RATES", "INFLATION & LIQUIDITY"])
        
        with macro_tab1:
            c_m1, c_m2 = st.columns(2)
            with c_m1:
                # Yield Curve
                if not df_yield.empty:
                    curr_yield = df_yield['value'].iloc[-1]
                    yield_color = "red" if curr_yield < 0 else "green"
                    fig_yc = go.Figure()
                    fig_yc.add_trace(go.Scatter(x=df_yield.index, y=df_yield['value'], fill='tozeroy', line=dict(color=yield_color)))
                    fig_yc.add_hline(y=0, line_dash="dash", line_color="white")
                    terminal_chart_layout(fig_yc, title=f"10Y-2Y SPREAD: {curr_yield:.2f}%", height=250)
                    st.plotly_chart(fig_yc, use_container_width=True)
                    yc_msg = "⚠️ INVERTED: RECESSION SIGNAL ALERT" if curr_yield < 0 else "NORMAL: GROWTH EXPECTATIONS"
                    st.caption(f"CONTEXT: {yc_msg}")
            
            with c_m2:
                # Fed Funds
                if not df_ff.empty:
                    fig_ff = go.Figure()
                    fig_ff.add_trace(go.Scatter(x=df_ff.index, y=df_ff['value'], line=dict(color="#00e6ff")))
                    terminal_chart_layout(fig_ff, title=f"FED FUNDS RATE: {df_ff['value'].iloc[-1]:.2f}%", height=250)
                    st.plotly_chart(fig_ff, use_container_width=True)
                    st.caption("CONTEXT: BASELINE RISK-FREE RATE (Cost of Capital)")
        with macro_tab2:
            c_m3, c_m4 = st.columns(2)
            with c_m3:
                # CPI
                if not df_cpi.empty:
                    df_cpi['YoY'] = df_cpi['value'].pct_change(12) * 100
                    fig_cpi = go.Figure()
                    fig_cpi.add_trace(go.Bar(x=df_cpi.index, y=df_cpi['YoY'], marker_color='#ff9900'))
                    terminal_chart_layout(fig_cpi, title=f"CPI INFLATION (YoY): {df_cpi['YoY'].iloc[-1]:.2f}%", height=250)
                    st.plotly_chart(fig_cpi, use_container_width=True)
                    cpi_trend = "RISING" if df_cpi['YoY'].iloc[-1] > df_cpi['YoY'].iloc[-2] else "FALLING"
                    st.caption(f"CONTEXT: INFLATION IS {cpi_trend} (Target: 2.0%)")
            
            with c_m4:
                # M2
                if not df_m2.empty:
                    fig_m2 = go.Figure()
                    fig_m2.add_trace(go.Scatter(x=df_m2.index, y=df_m2['value'], line=dict(color="#00ff00")))
                    terminal_chart_layout(fig_m2, title="M2 MONEY SUPPLY (Liquidity)", height=250)
                    st.plotly_chart(fig_m2, use_container_width=True)
                    st.caption("CONTEXT: FUEL FOR ASSET PRICES")
    
    with macro_col_ml:
        st.markdown("#### 🤖 MACRO ML ENGINE")
        if macro_regime:
            st.markdown(f"""
            <div class='terminal-box'>
                <div style='color:#aaa; font-size:0.8em;'>ECONOMIC REGIME (GMM)</div>
                <div style='color:#00e6ff; font-size:1.1em; font-weight:bold;'>{macro_regime['regime']}</div>
                <hr>
                <div style='font-size:0.8em;'>CPI: {macro_regime['cpi']:.1f}%</div>
                <div style='font-size:0.8em;'>RATES: {macro_regime['rate']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Insufficient data for Macro ML.")
        # DXY COMPARISON
        st.markdown("#### 💵 DOLLAR (DXY) COMPARE")
        if not correlations.empty and 'Dollar' in correlations:
            dxy_corr = correlations['Dollar']
            corr_color = "#00ff00" if dxy_corr > 0.5 else "#ff3333" if dxy_corr < -0.5 else "white"
            st.markdown(f"""
            <div class='terminal-box'>
                <div style='color:#aaa; font-size:0.8em;'>CORRELATION TO DXY</div>
                <div style='color:{corr_color}; font-size:1.5em;'>{dxy_corr:.2f}</div>
                <div style='font-size:0.7em; color:gray;'>1.0 = Moves with DXY<br>-1.0 = Inverse to DXY</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("DXY Correlation Unavailable")
else:
    st.info("FRED API Key not found. Add `fred_api_key` to secrets to view Fed Macro Data.")

# --- 4. RISK ANALYSIS & BACKTEST ---
st.markdown("---")
st.markdown("### ⚡ QUANTITATIVE RISK & EXECUTION")
strat_perf = run_strategy_backtest(asset_info['ticker'])
if not intraday_data.empty and strat_perf:
    q1, q2, q3 = st.columns([1, 2, 2])
    with q1:
        st.markdown("**STRATEGY SIGNAL**")
        sig_color = "#00ff00" if "LONG" in strat_perf['signal'] else "#ffff00"
        st.markdown(f"<span style='color:{sig_color}; font-size:1.8em; font-weight:bold;'>{strat_perf['signal']}</span>", unsafe_allow_html=True)
        st.markdown("---")
        st.metric("Total Return", f"{strat_perf['return']*100:.1f}%")
        st.metric("Sharpe Ratio", f"{strat_perf['sharpe']:.2f}")
    with q2:
        ec_df = pd.DataFrame({"Strategy": strat_perf['equity_curve'], "Buy & Hold": strat_perf['df']['Cum_BnH']})
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(x=ec_df.index, y=ec_df['Buy & Hold'], name="Buy & Hold", line=dict(color='gray', dash='dot')))
        fig_perf.add_trace(go.Scatter(x=ec_df.index, y=ec_df['Strategy'], name="Active Strat", line=dict(color='#00e6ff', width=2)))
        terminal_chart_layout(fig_perf, title="STRATEGY EDGE VALIDATION", height=300)
        st.plotly_chart(fig_perf, use_container_width=True)
    with q3:
        if vol_profile is not None:
            fig_vp = go.Figure()
            colors = ['#00e6ff' if x == poc_price else '#333' for x in vol_profile['PriceLevel']]
            fig_vp.add_trace(go.Bar(y=vol_profile['PriceLevel'], x=vol_profile['Volume'], orientation='h', marker_color='#ff9900', opacity=0.4))
            fig_vp.add_hline(y=poc_price, line_dash="dash", line_color="yellow", annotation_text="POC")
            terminal_chart_layout(fig_vp, title="INTRADAY VOLUME PROFILE", height=300)
            st.plotly_chart(fig_vp, use_container_width=True)

# --- 5. VWAP EXECUTION ---
st.markdown("#### 🎯 SESSION VWAP + KEY LEVELS")
vwap_df = calculate_vwap_bands(intraday_data)
if not vwap_df.empty:
    fig_vwap = go.Figure()
    fig_vwap.add_trace(go.Candlestick(x=vwap_df.index, open=vwap_df['Open'], high=vwap_df['High'], low=vwap_df['Low'], close=vwap_df['Close'], name="Price"))
    fig_vwap.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df['VWAP'], name="Session VWAP", line=dict(color='#ff9900', width=2)))
    fig_vwap.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df['Upper_Band_1'], name="+1 STD", line=dict(color='gray', width=1), opacity=0.3))
    fig_vwap.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df['Lower_Band_1'], name="-1 STD", line=dict(color='gray', width=1), opacity=0.3))
    if key_levels:
        fig_vwap.add_hline(y=key_levels['PDH'], line_dash="dot", line_color="#ff3333", annotation_text="PDH")
        fig_vwap.add_hline(y=key_levels['PDL'], line_dash="dot", line_color="#00ff00", annotation_text="PDL")
        fig_vwap.add_hline(y=key_levels['Pivot'], line_width=1, line_color="#00e6ff", annotation_text="DAILY PIVOT")
    terminal_chart_layout(fig_vwap, height=500)
    st.plotly_chart(fig_vwap, use_container_width=True)

# --- 6. GEX & VOLATILITY ---
st.markdown("---")
st.markdown("### 🏦 INSTITUTIONAL GEX & VOLATILITY")
if gex_df is not None and gex_spot is not None:
    g1, g2, g3 = st.columns([2, 1, 1])
    
    with g1:
        center_strike = gex_spot 
        gex_zoom = gex_df[(gex_df['strike'] > center_strike * 0.9) & (gex_df['strike'] < center_strike * 1.1)]
        fig_gex = go.Figure()
        colors = ['#00ff00' if x > 0 else '#ff3333' for x in gex_zoom['gex']]
        fig_gex.add_trace(go.Bar(x=gex_zoom['strike'], y=gex_zoom['gex'], marker_color=colors))
        fig_gex.add_vline(x=center_strike, line_dash="dot", line_color="white", annotation_text="ETF SPOT")
        terminal_chart_layout(fig_gex, title=f"NET GAMMA PROFILE (EXP: {gex_date})")
        st.plotly_chart(fig_gex, use_container_width=True)
        
    with g2:
        total_gex = gex_df['gex'].sum() / 1_000_000
        sentiment = "LOW VOL (Sticky)" if total_gex > 0 else "HIGH VOL (Slippery)"
        sent_color = "bullish" if total_gex > 0 else "bearish"
        st.markdown(f"""
        <div class='terminal-box'>
            <div style='color:#ff9900;'>NET GAMMA</div>
            <div style='font-size:1.5em; color:white;'>${total_gex:.1f}M</div>
            <div style='margin-top:10px;'><span class='{sent_color}'>{sentiment}</span></div>
        </div>
        """, unsafe_allow_html=True)
    
    # --- NEW: VOLATILITY DASHBOARD ---
    with g3:
        # Calculate HV (20D Annualized)
        if not daily_data.empty:
            hv_current = daily_data['Close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100
        else:
            hv_current = 0
            
        iv_display = current_iv if current_iv else 0
        vol_premium = iv_display - hv_current
        prem_color = "#ff3333" if vol_premium > 5 else "#00ff00" if vol_premium < 0 else "white"
        
        st.markdown(f"""
        <div class='terminal-box'>
            <div style='color:#aaa; font-size:0.8em;'>IMPLIED VOL (IV)</div>
            <div style='font-size:1.2em; color:#00e6ff;'>{iv_display:.1f}%</div>
            <hr style='margin:5px 0; border-color:#333;'>
            <div style='color:#aaa; font-size:0.8em;'>HISTORICAL VOL (HV)</div>
            <div style='font-size:1.2em; color:#e0e0e0;'>{hv_current:.1f}%</div>
            <hr style='margin:5px 0; border-color:#333;'>
            <div style='color:#aaa; font-size:0.8em;'>VOL PREMIUM (IV-HV)</div>
            <div style='font-size:1.2em; color:{prem_color};'>{vol_premium:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

else: st.warning("GEX Data Unavailable for this asset.")

# --- 7. MONTE CARLO & ADVANCED SEASONALITY ---
st.markdown("---")
st.markdown("### 🎲 SIMULATION & TIME ANALYSIS")
pred_dates, pred_paths = generate_monte_carlo(daily_data)
stats = get_seasonality_stats(daily_data, asset_info['ticker']) 
if stats:
    st.markdown("#### ⏳ SEASONAL TENDENCIES")
    tab_hour, tab_day, tab_week = st.tabs(["HOUR (NY)", "DAY", "WEEK"])
    with tab_hour:
        if 'hourly_perf' in stats and stats['hourly_perf'] is not None:
            hp = stats['hourly_perf']
            fig_h = go.Figure()
            colors = ['#00ff00' if v > 0 else '#ff3333' for v in hp.values]
            fig_h.add_trace(go.Bar(x=[f"{h:02d}:00" for h in hp.index], y=hp.values, marker_color=colors))
            terminal_chart_layout(fig_h, title="AVG RETURN BY HOUR (NY TIME)", height=350)
            st.plotly_chart(fig_h, use_container_width=True)
    with tab_day:
        fig_d = go.Figure()
        fig_d.add_trace(go.Bar(x=stats['day_high'].index, y=stats['day_high'].values, marker_color='#00ff00'))
        terminal_chart_layout(fig_d, title="PROBABILITY OF WEEKLY HIGH", height=350)
        st.plotly_chart(fig_d, use_container_width=True)
    with tab_week:
        if 'week_returns' in stats:
            wr = stats['week_returns']
            fig_w = go.Figure()
            colors = ['#00ff00' if v > 0 else '#ff3333' for v in wr.values]
            fig_w.add_trace(go.Bar(x=["Wk 1", "Wk 2", "Wk 3", "Wk 4", "Wk 5"], y=wr.values, marker_color=colors))
            terminal_chart_layout(fig_w, title="AVG RETURN BY WEEK OF MONTH", height=350)
            st.plotly_chart(fig_w, use_container_width=True)
st.markdown("#### 🎲 MONTE CARLO PROJECTION")
if pred_dates is not None and pred_paths is not None:
    fig_pred = go.Figure()
    hist_slice = daily_data['Close'].tail(90)
    fig_pred.add_trace(go.Scatter(x=hist_slice.index, y=hist_slice.values, name='History', line=dict(color='white')))
    fig_pred.add_trace(go.Scatter(x=pred_dates, y=np.mean(pred_paths, axis=1), name='Avg Path', line=dict(color='#ff9900', dash='dash')))
    terminal_chart_layout(fig_pred, title="MONTE CARLO PROJECTION (126 Days)", height=400)
    st.plotly_chart(fig_pred, use_container_width=True)

# --- 8. COT QUANT TERMINAL (V5.17 UPGRADE) ---
st.markdown("---")
st.markdown("### 🏛️ COT QUANT TERMINAL")

# 1. Fetch Historical Data
with st.spinner("Analyzing CFTC Data..."):
    cot_history = fetch_cot_history(selected_asset, start_year=2024)

if cot_history is not None and not cot_history.empty:
    
    # 2. Process Data for Metrics
    cot_config = COT_MAPPING[selected_asset]
    spec_label, hedge_label = cot_config['labels']
    
    if all(c in cot_history.columns for c in ['spec_long', 'spec_short', 'hedge_long', 'hedge_short']):
        
        # Calculations
        cot_history['Net Speculator'] = cot_history['spec_long'] - cot_history['spec_short']
        cot_history['Net Hedger'] = cot_history['hedge_long'] - cot_history['hedge_short']
        cot_history['Spec Z-Score'] = calculate_z_score(cot_history['Net Speculator'])
        
        latest_cot = cot_history.iloc[-1]
        prev_cot = cot_history.iloc[-2] if len(cot_history) > 1 else latest_cot
        
        # 3. METRICS ROW
        c_cot1, c_cot2, c_cot3, c_cot4 = st.columns(4)
        
        # Speculator Net
        spec_delta = latest_cot['Net Speculator'] - prev_cot['Net Speculator']
        c_cot1.metric(f"{spec_label} (Net)", f"{int(latest_cot['Net Speculator']):,}", f"{int(spec_delta):,}")
        
        # Hedger Net
        hedge_delta = latest_cot['Net Hedger'] - prev_cot['Net Hedger']
        c_cot2.metric(f"{hedge_label} (Net)", f"{int(latest_cot['Net Hedger']):,}", f"{int(hedge_delta):,}", delta_color="inverse")
        
        # Z-Score
        z_val = latest_cot['Spec Z-Score']
        c_cot3.metric("Z-Score (52wk)", f"{z_val:.2f}σ", "Extreme" if abs(z_val) > 2 else "Neutral", delta_color="off")
        
        # Date
        c_cot4.metric("Report Date", latest_cot['date'].strftime('%Y-%m-%d'))
        
        # 4. AI Interpretation Box
        st.info(generate_cot_analysis(latest_cot['Net Speculator'], latest_cot['Net Hedger'], spec_label, hedge_label))
        
        # 5. VISUALIZATION TABS
        tab_trend, tab_struct, tab_osc = st.tabs(["📈 NET TREND", "🦋 LONG/SHORT STRUCTURE", "📊 Z-SCORE OSCILLATOR"])
        
        with tab_trend:
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=cot_history['date'], y=cot_history['Net Speculator'], name=spec_label, line=dict(color='#00FF00', width=2)))
            fig_trend.add_trace(go.Scatter(x=cot_history['date'], y=cot_history['Net Hedger'], name=hedge_label, line=dict(color='#FF0000', width=2)))
            fig_trend.add_hline(y=0, line_dash="dash", line_color="gray")
            terminal_chart_layout(fig_trend, title="NET POSITIONING HISTORY (Smart Money vs Specs)", height=400)
            st.plotly_chart(fig_trend, use_container_width=True)
            
        with tab_struct:
            fig_struct = go.Figure()
            # Speculator Longs (Green)
            fig_struct.add_trace(go.Bar(
                x=cot_history['date'], 
                y=cot_history['spec_long'], 
                name=f"{spec_label} Longs", 
                marker_color='#00C805'
            ))
            # Speculator Shorts (Red) - Negative for Butterfly
            fig_struct.add_trace(go.Bar(
                x=cot_history['date'], 
                y=-cot_history['spec_short'], 
                name=f"{spec_label} Shorts", 
                marker_color='#FF4B4B'
            ))
            fig_struct.update_layout(barmode='overlay')
            terminal_chart_layout(fig_struct, title=f"{spec_label.upper()} STRUCTURE (Butterfly Chart)", height=400)
            st.caption("Green Bars = Long Contracts. Red Bars = Short Contracts (Plotted Inversely).")
            st.plotly_chart(fig_struct, use_container_width=True)
            
        with tab_osc:
            fig_z = go.Figure()
            colors = ['red' if val > 2 or val < -2 else 'gray' for val in cot_history['Spec Z-Score']]
            fig_z.add_trace(go.Bar(x=cot_history['date'], y=cot_history['Spec Z-Score'], marker_color=colors, name="Z-Score"))
            fig_z.add_hline(y=2, line_dash="dot", line_color="red", annotation_text="Overbought (+2σ)")
            fig_z.add_hline(y=-2, line_dash="dot", line_color="red", annotation_text="Oversold (-2σ)")
            terminal_chart_layout(fig_z, title="POSITIONING EXTREMES (Z-Score)", height=400)
            st.plotly_chart(fig_z, use_container_width=True)
            
    else:
        st.warning(f"COT Data retrieved but missing required columns for {selected_asset}. Try another asset.")
else:
    st.info(f"No COT Data available for {selected_asset}. This asset might not be a futures contract or mapping is missing.")

# --- 9. INTELLIGENT EXECUTIVE SUMMARY & THESIS ---
st.markdown("---")
st.markdown("### 🧠 AI QUANT ANALYST")
# Prepare Data for LLM
gex_summary = gex_df if gex_df is not None else None
ml_signal_str = "BULLISH" if ml_prob > 0.55 else "BEARISH" if ml_prob < 0.45 else "NEUTRAL"
news_text_summary = "\n".join([f"- {n['title']} ({n['source']})" for n in combined_news_for_llm])
if gemini_key:
    col_exec_btn, col_exec_info = st.columns([1, 4])
    with col_exec_btn:
        if st.button("📝 GENERATE BRIEF"):
            with st.spinner("Analyzing Technicals + Macro..."):
                narrative = get_technical_narrative(
                    ticker=selected_asset, price=curr, daily_pct=pct, regime=regime_data,
                    ml_signal=ml_signal_str, gex_data=gex_summary, cot_data=cot_data if 'cot_data' in locals() else None,
                    levels=key_levels, macro_data=macro_context_data, api_key=gemini_key
                )
                st.session_state['narrative_cache'] = narrative
                st.rerun()
    with col_exec_info:
        st.markdown("<span style='color:gray; vertical-align:middle;'>3-bullet Executive Summary (Costs 1 Gemini Call)</span>", unsafe_allow_html=True)
    if st.session_state['narrative_cache']:
        if "⚠️" in st.session_state['narrative_cache']: st.error(st.session_state['narrative_cache'])
        else:
            st.markdown(f"""
            <div class='terminal-box' style='border-left: 4px solid #00e6ff;'>
                <div style='font-family: monospace; font-size: 0.95em; white-space: pre-wrap;'>{st.session_state['narrative_cache']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    col_thesis_btn, col_thesis_info = st.columns([1, 4])
    with col_thesis_btn:
        if st.button("🔎 DEEP DIVE THESIS"):
            with st.spinner("Analyzing Macro, Gamma, and Order Flow..."):
                thesis_text = generate_deep_dive_thesis(
                    ticker=selected_asset, price=curr, change=pct, regime=regime_data,
                    ml_signal=ml_signal_str, gex_data=gex_summary, cot_data=cot_data if 'cot_data' in locals() else None,
                    levels=key_levels, news_summary=news_text_summary, macro_data=macro_context_data, api_key=gemini_key
                )
                st.session_state['thesis_cache'] = thesis_text
                st.rerun()
    with col_thesis_info:
        st.markdown("<span style='color:gray;'>Full 4-paragraph Hedge Fund style report. (Costs 1 Gemini Call)</span>", unsafe_allow_html=True)
    if st.session_state['thesis_cache']:
        if "⚠️" in st.session_state['thesis_cache']: st.error(st.session_state['thesis_cache'])
        else:
            st.markdown(f"""
            <div class='terminal-box' style='border: 1px solid #444; padding: 20px;'>
                {st.session_state['thesis_cache']}
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("Add GEMINI_API_KEY to see the AI Analyst Report.")
