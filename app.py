import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import cot_reports as cot
import google.generativeai as genai
from newsapi import NewsApiClient
from datetime import datetime, timedelta
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
import os
import time

# --- APP CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Bloomberg Terminal Pro V5.8", page_icon="💹")

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

# --- SESSION STATE INITIALIZATION (For API Tracking) ---
if 'gemini_calls' not in st.session_state: st.session_state['gemini_calls'] = 0
if 'news_calls' not in st.session_state: st.session_state['news_calls'] = 0
if 'rapid_calls' not in st.session_state: st.session_state['rapid_calls'] = 0
if 'coingecko_calls' not in st.session_state: st.session_state['coingecko_calls'] = 0
if 'fred_calls' not in st.session_state: st.session_state['fred_calls'] = 0
if 'narrative_cache' not in st.session_state: st.session_state['narrative_cache'] = None
if 'thesis_cache' not in st.session_state: st.session_state['thesis_cache'] = None

# --- CONSTANTS & MAPPINGS ---
ASSETS = {
    # --- MAJOR FOREX/INDICES ---
    "Gold (Comex)": {"ticker": "GC=F", "opt_ticker": "GLD", "news_query": "Gold Price", "cg_id": None},
    "S&P 500": {"ticker": "^GSPC", "opt_ticker": "SPY", "news_query": "S&P 500", "cg_id": None},
    "NASDAQ": {"ticker": "^IXIC", "opt_ticker": "QQQ", "news_query": "Nasdaq", "cg_id": None},
    "EUR/USD": {"ticker": "EURUSD=X", "opt_ticker": None, "news_query": "EURUSD", "cg_id": None},
    
    # --- CRYPTO L1s ---
    "Bitcoin": {"ticker": "BTC-USD", "opt_ticker": "BITO", "news_query": "Bitcoin", "cg_id": "bitcoin"},
    "Ethereum": {"ticker": "ETH-USD", "opt_ticker": "ETHE", "news_query": "Ethereum", "cg_id": "ethereum"},
    "Solana": {"ticker": "SOL-USD", "opt_ticker": None, "news_query": "Solana Crypto", "cg_id": "solana"},
    "XRP": {"ticker": "XRP-USD", "opt_ticker": None, "news_query": "Ripple XRP", "cg_id": "ripple"},
    "BNB": {"ticker": "BNB-USD", "opt_ticker": None, "news_query": "Binance Coin", "cg_id": "binancecoin"},
    "Cardano": {"ticker": "ADA-USD", "opt_ticker": None, "news_query": "Cardano ADA", "cg_id": "cardano"},
    
    # --- MEME COINS & ALTCOINS ---
    "Dogecoin": {"ticker": "DOGE-USD", "opt_ticker": None, "news_query": "Dogecoin", "cg_id": "dogecoin"},
    "Shiba Inu": {"ticker": "SHIB-USD", "opt_ticker": None, "news_query": "Shiba Inu Coin", "cg_id": "shiba-inu"},
    "Pepe": {"ticker": "PEPE-USD", "opt_ticker": None, "news_query": "Pepe Coin", "cg_id": "pepe"},
    "Chainlink": {"ticker": "LINK-USD", "opt_ticker": None, "news_query": "Chainlink", "cg_id": "chainlink"},
    "Polygon": {"ticker": "MATIC-USD", "opt_ticker": None, "news_query": "Polygon MATIC", "cg_id": "matic-network"},
}

# Mapping for cot_reports library (Legacy Futures Names)
COT_MAPPING = {
    "Gold (Comex)": {"name": "GOLD - COMMODITY EXCHANGE INC."},
    "S&P 500": {"name": "E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE"},
    "NASDAQ": {"name": "NASDAQ-100 CONSOLIDATED - CHICAGO MERCANTILE EXCHANGE"},
    "EUR/USD": {"name": "EURO FX - CHICAGO MERCANTILE EXCHANGE"},
    "Bitcoin": {"name": "BITCOIN - CHICAGO MERCANTILE EXCHANGE"},
    "Ethereum": {"name": "ETHER - CHICAGO MERCANTILE EXCHANGE"}
}

# --- HELPER FUNCTIONS ---
def get_api_key(key_name):
    # 1. Check Streamlit Secrets (Nested & Root)
    if "api_keys" in st.secrets and key_name in st.secrets["api_keys"]:
        return st.secrets["api_keys"][key_name]
    if key_name in st.secrets:
        return st.secrets[key_name]
    
    # 2. Check for common aliases
    if key_name == "gemini_api_key":
        if "GOOGLE_API_KEY" in st.secrets: return st.secrets["GOOGLE_API_KEY"]
        if "google_api_key" in st.secrets: return st.secrets["google_api_key"]
            
    # 3. Check OS Environment (Local dev fallback)
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

# --- SAFE YFINANCE WRAPPER (RATE LIMIT PROTECTION) ---
def safe_yf_download(tickers, period, interval, retries=3):
    """
    Wraps yf.download with auto-retry and exponential backoff 
    to prevent 'Too Many Requests' errors.
    """
    for i in range(retries):
        try:
            # We add a small delay to be kind to the API
            time.sleep(0.1) 
            df = yf.download(tickers, period=period, interval=interval, progress=False)
            if not df.empty:
                return flatten_dataframe(df)
        except Exception as e:
            if i == retries - 1: # Last try
                print(f"Failed to fetch {tickers}: {e}")
                return pd.DataFrame()
            time.sleep(2 ** i) # Exponential backoff: 1s, 2s, 4s...
    return pd.DataFrame()

# --- FRED API ENGINE (ZERO DEPENDENCY) ---
@st.cache_data(ttl=86400) # Cache for 24 hours (Macro data is slow)
def get_fred_series(series_id, api_key, observation_start=None):
    """
    Fetches data from St. Louis Fed via Requests (No extra pip install needed).
    """
    if not api_key: return pd.DataFrame()
    
    st.session_state['fred_calls'] += 1
    base_url = "https://api.stlouisfed.org/fred/series/observations"
    
    # Default lookback if not specified (5 years)
    if not observation_start:
        observation_start = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')

    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": observation_start
    }
    
    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        
        if "observations" in data:
            df = pd.DataFrame(data["observations"])
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            return df[['date', 'value']].set_index('date').dropna()
            
    except Exception as e:
        return pd.DataFrame()
    
    return pd.DataFrame()

# --- COINGECKO API ENGINE ---
@st.cache_data(ttl=300) 
def get_coingecko_stats(cg_id, api_key):
    """Fetches fundamental data from CoinGecko Demo API."""
    if not cg_id or not api_key: return None
    
    # Track usage
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
    except Exception:
        return None

# --- LLM ENGINE (SAFE MODE) ---
def get_technical_narrative(ticker, price, daily_pct, regime, ml_signal, gex_data, cot_data, levels, api_key):
    """
    Standard View: 3 Bullet Executive Summary.
    """
    if not api_key: return "AI Analyst unavailable (No Key)."
    
    # Track Usage
    st.session_state['gemini_calls'] += 1
    
    # Format data for Prompt
    gex_text = "N/A"
    if gex_data is not None:
        total_gex = gex_data['gex'].sum()
        gex_text = f"Net Gamma: ${total_gex/1_000_000:.1f}M ({'Long/Sticky' if total_gex>0 else 'Short/Volatile'})"

    lvl_text = "N/A"
    if levels:
        lvl_text = f"Pivot: {levels['Pivot']:.2f}, R1: {levels['R1']:.2f}, S1: {levels['S1']:.2f}"

    prompt = f"""
    You are a Senior Portfolio Manager. Analyze technical data for {ticker} and write a 3-bullet executive summary.
    
    DATA: Price: {price:,.2f} ({daily_pct:.2f}%), Regime: {regime['regime'] if regime else 'Unknown'}, 
    ML: {ml_signal}, GEX: {gex_text}, COT: {cot_data['sentiment'] if cot_data else 'N/A'}, Levels: {lvl_text}

    TASK:
    1. Synthesize Regime (Trend), ML (Prob), and GEX (Vol).
    2. Identify key trigger level.
    3. Final Execution bias ("Buy Dips", "Fade", etc).
    Keep it concise. Bloomberg Terminal style.
    """

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash") # Use Flash for speed/cost
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "ResourceExhausted" in error_msg:
            return "⚠️ API LIMIT REACHED: The AI Analyst is taking a nap. (Quota Exceeded)"
        return f"AI Analyst unavailable: {error_msg}"

def generate_deep_dive_thesis(ticker, price, change, regime, ml_signal, gex_data, cot_data, levels, news_summary, api_key):
    """
    On-Demand View: Full 4-Paragraph Investment Thesis.
    """
    if not api_key: return "API Key Missing."
    
    # Track Usage
    st.session_state['gemini_calls'] += 1
    
    gex_text = "N/A"
    if gex_data is not None:
        total_gex = gex_data['gex'].sum()
        gex_text = f"Net Gamma: ${total_gex/1_000_000:.1f}M"

    prompt = f"""
    Write a detailed Investment Thesis for {ticker}.
    
    DATA: Price: {price:,.2f} ({change:.2f}%), Regime: {regime['regime'] if regime else 'Unknown'}, 
    ML: {ml_signal}, GEX: {gex_text}, COT: {cot_data['sentiment'] if cot_data else 'N/A'}
    NEWS: {news_summary}

    OUTPUT FORMAT (Markdown):
    ### 1. THE CORE ARGUMENT (Long/Short/Neutral opinion)
    ### 2. SUPPORTING DATA (Cite specific numbers)
    ### 3. THE BEAR CASE (Risks)
    ### 4. INVALIDATION LEVEL (Price failure point)
    """
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "ResourceExhausted" in error_msg:
            return "⚠️ CRITICAL: API LIMIT REACHED. Thesis generation paused to prevent overage."
        return f"Thesis Generation Failed: {error_msg}"

# --- 1. QUANT ENGINE (GMM + HURST) ---
def calculate_hurst(series, lags=range(2, 20)):
    try:
        tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    except: return 0.5

@st.cache_data(ttl=3600)
def get_market_regime(ticker):
    try:
        # UPDATED: Use safe wrapper
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
        
        regime_map = {
            state_order[0]: "LOW VOL (Trend)", 
            state_order[1]: "NEUTRAL (Chop)", 
            state_order[2]: "HIGH VOL (Crisis)"
        }
        regime_desc = regime_map.get(current_state, "Unknown")
        
        if "LOW VOL" in regime_desc: color = "bullish"
        elif "HIGH VOL" in regime_desc: color = "bearish"
        else: color = "neutral"
        
        return {"regime": regime_desc, "color": color, "confidence": max(probs)}
    except: return None

# --- 2. MACHINE LEARNING ENGINE ---
@st.cache_data(ttl=3600)
def get_ml_prediction(ticker):
    try:
        # UPDATED: Use safe wrapper
        df = safe_yf_download(ticker, period="2y", interval="1d") 
        if df.empty: return None, 0.5
        
        data = df.copy()
        data['Returns'] = data['Close'].pct_change()
        data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
        
        data['Vol_5d'] = data['Returns'].rolling(5).std()
        data['Mom_5d'] = data['Close'].pct_change(5)
        
        data = data.dropna()
        if len(data) < 50: return None, 0.5
        
        features = ['Vol_5d', 'Mom_5d']
        X = data[features]
        y = data['Target']
        
        model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
        model.fit(X, y)
        
        last_row = X.iloc[[-1]]
        prob_up = model.predict_proba(last_row)[0][1]
        return model, prob_up
    except: return None, 0.5

# --- 3. GAMMA EXPOSURE ENGINE ---
def calculate_black_scholes_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

@st.cache_data(ttl=3600)
def get_gex_profile(opt_ticker):
    if opt_ticker is None: return None, None, None
    try:
        tk = yf.Ticker(opt_ticker)
        
        try:
            hist = tk.history(period="1d")
            if not hist.empty:
                spot_price = hist['Close'].iloc[-1]
            else:
                spot_price = tk.fast_info.last_price
        except:
            return None, None, None
        if spot_price is None: return None, None, None

        exps = tk.options
        if not exps: return None, None, None
        
        if len(exps) > 1:
            target_exp = exps[1] 
        else:
            target_exp = exps[0]
            
        chain = tk.option_chain(target_exp)
        calls, puts = chain.calls, chain.puts
        
        if calls.empty or puts.empty: return None, None, None
        r = 0.045
        exp_date = datetime.strptime(target_exp, "%Y-%m-%d")
        days_to_exp = (exp_date - datetime.now()).days
        T = 0.001 if days_to_exp <= 0 else days_to_exp / 365.0
        
        gex_data = []
        strikes = sorted(list(set(calls['strike'].tolist() + puts['strike'].tolist())))
        
        for K in strikes:
            if K < spot_price * 0.75 or K > spot_price * 1.25: continue
            
            c_row = calls[calls['strike'] == K]
            c_oi = c_row['openInterest'].iloc[0] if not c_row.empty else 0
            c_iv = c_row['impliedVolatility'].iloc[0] if not c_row.empty and 'impliedVolatility' in c_row.columns else 0.2
            
            p_row = puts[puts['strike'] == K]
            p_oi = p_row['openInterest'].iloc[0] if not p_row.empty else 0
            p_iv = p_row['impliedVolatility'].iloc[0] if not p_row.empty and 'impliedVolatility' in p_row.columns else 0.2
            
            c_gamma = calculate_black_scholes_gamma(spot_price, K, T, r, c_iv)
            p_gamma = calculate_black_scholes_gamma(spot_price, K, T, r, p_iv)
            
            net_gex = (c_gamma * c_oi - p_gamma * p_oi) * spot_price * 100
            gex_data.append({"strike": K, "gex": net_gex})
            
        df = pd.DataFrame(gex_data, columns=['strike', 'gex'])
        if df.empty: return None, None, None
            
        return df, target_exp, spot_price 
    except Exception as e:
        return None, None, None

# --- 4. VOLUME PROFILE ENGINE ---
def calculate_volume_profile(df, bins=50):
    if df.empty: return None, None
    price_range = df['High'].max() - df['Low'].min()
    if price_range == 0: return None, None
    bin_size = price_range / bins
    df['PriceBin'] = ((df['Close'] - df['Low'].min()) // bin_size).astype(int)
    vol_profile = df.groupby('PriceBin')['Volume'].sum().reset_index()
    vol_profile['PriceLevel'] = df['Low'].min() + (vol_profile['PriceBin'] * bin_size)
    poc_idx = vol_profile['Volume'].idxmax()
    poc_price = vol_profile.loc[poc_idx, 'PriceLevel']
    return vol_profile, poc_price

# --- 5. MONTE CARLO & SEASONALITY ---
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
        week_stats = df.groupby('Month_Week')['Returns'].mean() * 100
        stats['week_returns'] = week_stats
        try:
            # UPDATED: Use safe wrapper
            intra = safe_yf_download(ticker_name, period="60d", interval="1h")
            
            if not intra.empty:
                if intra.index.tz is None:
                    intra.index = intra.index.tz_localize('UTC')
                intra.index = intra.index.tz_convert('America/New_York')
                
                intra['Hour'] = intra.index.hour
                intra['Return'] = intra['Close'].pct_change()
                
                target_hours = [2,3,4,5,6, 8,9,10,11, 14,15,16,17,18, 20,21,22,23]
                
                hourly_perf = intra[intra['Hour'].isin(target_hours)].groupby('Hour')['Return'].mean() * 100
                stats['hourly_perf'] = hourly_perf
        except Exception:
            stats['hourly_perf'] = None
            
        return stats
    except: return None

@st.cache_data(ttl=3600)
def generate_monte_carlo(stock_data, days=126, simulations=1000):
    if stock_data is None or stock_data.empty or len(stock_data) < 2:
        return None, None
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
    except Exception as e:
        return None, None

# --- NEW: TECHNICAL RADAR ENGINE (ZERO-DEPENDENCY) ---
def calculate_technical_radar(df):
    """Calculates RSI, MACD, BB, and EMA using standard Pandas vectorization."""
    if df.empty or len(df) < 30: return None
    
    data = df.copy()
    close = data['Close']
    
    # 1. RSI (14)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # 2. MACD (12, 26, 9)
    k = close.ewm(span=12, adjust=False, min_periods=12).mean()
    d = close.ewm(span=26, adjust=False, min_periods=26).mean()
    data['MACD'] = k - d
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False, min_periods=9).mean()
    
    # 3. Bollinger Bands (20, 2)
    data['BB_Mid'] = close.rolling(window=20).mean()
    data['BB_Std'] = close.rolling(window=20).std()
    data['BB_Up'] = data['BB_Mid'] + (2 * data['BB_Std'])
    data['BB_Low'] = data['BB_Mid'] - (2 * data['BB_Std'])
    
    # 4. EMAs (Trend)
    data['EMA_20'] = close.ewm(span=20, adjust=False).mean()
    data['EMA_50'] = close.ewm(span=50, adjust=False).mean()
    
    # Extract Latest Values
    last = data.iloc[-1]
    
    # Logic Processing
    signals = {}
    
    # RSI Logic
    if last['RSI'] < 30: signals['RSI'] = {"val": f"{last['RSI']:.0f}", "bias": "OVERSOLD (Bull)", "col": "bullish"}
    elif last['RSI'] > 70: signals['RSI'] = {"val": f"{last['RSI']:.0f}", "bias": "OVERBOUGHT (Bear)", "col": "bearish"}
    else: signals['RSI'] = {"val": f"{last['RSI']:.0f}", "bias": "NEUTRAL", "col": "neutral"}
    
    # MACD Logic
    macd_hist = last['MACD'] - last['MACD_Signal']
    if macd_hist > 0 and last['MACD'] > 0: signals['MACD'] = {"val": f"{macd_hist:.2f}", "bias": "BULLISH", "col": "bullish"}
    elif macd_hist < 0 and last['MACD'] < 0: signals['MACD'] = {"val": f"{macd_hist:.2f}", "bias": "BEARISH", "col": "bearish"}
    else: signals['MACD'] = {"val": f"{macd_hist:.2f}", "bias": "NEUTRAL", "col": "neutral"}
    
    # Trend Logic (EMA)
    if last['Close'] > last['EMA_20'] and last['EMA_20'] > last['EMA_50']:
        signals['Trend'] = {"val": "Uptrend", "bias": "STRONG BULL", "col": "bullish"}
    elif last['Close'] < last['EMA_20'] and last['EMA_20'] < last['EMA_50']:
        signals['Trend'] = {"val": "Downtrend", "bias": "STRONG BEAR", "col": "bearish"}
    else:
        signals['Trend'] = {"val": "Chop", "bias": "WEAK/MIXED", "col": "neutral"}
        
    return signals

# --- 6. NEWS & ECONOMICS ---
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
            if val_forecast != 0: pct_dev = abs(delta / val_forecast)
            else: pct_dev = 1.0 if delta != 0 else 0
            
            if pct_dev < 0.02: bias = "Mean Reverting"
            elif delta > 0: bias = "Bullish"
            else: bias = "Bearish"
        else: context_str = f"Actual: {actual_str}"
    else:
        if val_forecast is not None and val_prev is not None:
            context_str = f"Forecast ({forecast_str}) vs Prev ({previous_str})"
            delta = val_forecast - val_prev
            if val_prev != 0: pct_dev = abs(delta / val_prev)
            else: pct_dev = 1.0
            
            if pct_dev < 0.02: bias = "Mean Reverting"
            elif delta > 0: bias = "Bullish Exp."
            else: bias = "Bearish Exp."
        else: context_str = "Waiting for Data..."
    return context_str, bias

@st.cache_data(ttl=14400) # CACHE: 4 Hours (Prevents hitting NewsAPI 100/day limit)
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

@st.cache_data(ttl=14400) # CACHE: 4 Hours
def get_forex_factory_news(api_key, news_type='breaking'):
    """Fetches news from Forex Factory Scraper via RapidAPI."""
    if not api_key: return []
    st.session_state['rapid_calls'] += 1
    
    base_url = "https://forex-factory-scraper1.p.rapidapi.com/"
    endpoints = {
        'breaking': "latest_breaking_news",
        'fundamental': "latest_fundamental_analysis_news",
        'hottest': "latest_hottest_news"
    }
    
    url = base_url + endpoints.get(news_type, "latest_breaking_news")
    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": "forex-factory-scraper1.p.rapidapi.com"
    }
    
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        
        normalized_news = []
        if isinstance(data, list):
            for item in data[:6]:
                title = item.get('title', 'No Title')
                link = item.get('link', item.get('url', '#'))
                date_str = item.get('date', item.get('time', 'Recent'))
                
                normalized_news.append({
                    "title": title,
                    "url": link,
                    "source": "ForexFactory",
                    "time": date_str
                })
        return normalized_news
    except Exception as e:
        return []

@st.cache_data(ttl=21600) # CACHE: 6 Hours (Economic Events rarely change intraday)
def get_economic_calendar(api_key):
    if not api_key: return None
    st.session_state['rapid_calls'] += 1
    
    # Priority 1: Forex Factory
    try:
        url = "https://forex-factory-scraper1.p.rapidapi.com/get_real_time_calendar_details"
        now = datetime.now()
        querystring = {
            "calendar": "Forex", "year": str(now.year), "month": str(now.month), "day": str(now.day),
            "currency": "ALL", "event_name": "ALL", "timezone": "GMT-04:00 Eastern Time (US & Canada)", "time_format": "12h"
        }
        headers = {"x-rapidapi-host": "forex-factory-scraper1.p.rapidapi.com", "x-rapidapi-key": api_key}
        
        response = requests.get(url, headers=headers, params=querystring)
        data = response.json()
        raw_events = data if isinstance(data, list) else data.get('data', [])
        
        filtered_events = []
        for e in raw_events:
            if e.get('currency') == 'USD' and (e.get('impact') == 'High' or e.get('impact') == 'Medium'):
                filtered_events.append(e)
        if filtered_events: return filtered_events

    except: pass 

    # Priority 2: Ultimate Economic Calendar (Backup)
    try:
        url = "https://ultimate-economic-calendar.p.rapidapi.com/economic-events/tradingview"
        now_str = datetime.now().strftime("%Y-%m-%d")
        querystring = {"from": now_str, "to": now_str, "countries": "US"}
        headers = {"x-rapidapi-key": api_key, "x-rapidapi-host": "ultimate-economic-calendar.p.rapidapi.com"}
        
        response = requests.get(url, headers=headers, params=querystring)
        data = response.json()
        backup_events = []
        if isinstance(data, list):
            for e in data:
                time_raw = e.get('date', '') 
                try:
                    dt_obj = datetime.fromisoformat(time_raw.replace('Z', '+00:00'))
                    time_display = dt_obj.strftime("%I:%M%p")
                except: time_display = time_raw
                
                backup_events.append({
                    "time": time_display, "event_name": e.get('title', 'Event'),
                    "actual": str(e.get('actual', '')), "forecast": str(e.get('forecast', '')),
                    "previous": str(e.get('previous', '')), "currency": "USD", "impact": "Medium/High"
                })
        return backup_events
    except: return []

# --- 7. INSTITUTIONAL FEATURES ---
# A. BACKTEST ENGINE
@st.cache_data(ttl=300)
def run_strategy_backtest(ticker):
    try:
        # UPDATED: Use safe wrapper
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
        
        return {
            "df": df, 
            "signal": current_signal, 
            "return": total_return, 
            "sharpe": sharpe, 
            "equity_curve": df['Cum_Strat']
        }
    except Exception as e:
        return None

# B. SESSION ANCHORED VWAP
def calculate_vwap_bands(df):
    if df.empty: return df
    df = df.copy()
    
    # Calculate Typical Price
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VP'] = df['TP'] * df['Volume']
    
    # Group by Date to reset VWAP daily
    df['Date'] = df.index.date
    
    df['Cum_VP'] = df.groupby('Date')['VP'].cumsum()
    df['Cum_Vol'] = df.groupby('Date')['Volume'].cumsum()
    
    df['VWAP'] = df['Cum_VP'] / df['Cum_Vol']
    
    df['Sq_Dist'] = df['Volume'] * (df['TP'] - df['VWAP'])**2
    df['Cum_Sq_Dist'] = df.groupby('Date')['Sq_Dist'].cumsum()
    df['Std_Dev'] = np.sqrt(df['Cum_Sq_Dist'] / df['Cum_Vol'])
    
    df['Upper_Band_1'] = df['VWAP'] + df['Std_Dev']
    df['Lower_Band_1'] = df['VWAP'] - df['Std_Dev']
    df['Upper_Band_2'] = df['VWAP'] + (df['Std_Dev'] * 2)
    df['Lower_Band_2'] = df['VWAP'] - (df['Std_Dev'] * 2)
    
    return df

# C. INTRADAY RELATIVE STRENGTH
@st.cache_data(ttl=300) # Increased TTL for safety
def get_relative_strength(asset_ticker, benchmark_ticker="SPY"):
    try:
        # UPDATED: Use safe wrapper
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

# D. FLOOR TRADER LEVELS
def get_key_levels(daily_df):
    if daily_df.empty: return {}
    try:
        last_complete_day = daily_df.iloc[-2]
    except:
        return {}
    
    high = last_complete_day['High']
    low = last_complete_day['Low']
    close = last_complete_day['Close']
    
    pivot = (high + low + close) / 3
    r1 = (2 * pivot) - low
    s1 = (2 * pivot) - high
    
    return {
        "PDH": high, "PDL": low, "PDC": close, "Pivot": pivot, "R1": r1, "S1": s1
    }

# E. MACRO CORRELATIONS
@st.cache_data(ttl=3600)
def get_correlations(base_ticker):
    try:
        tickers = {
            "Base": base_ticker,
            "VIX": "^VIX",
            "10Y Yield": "^TNX",
            "Dollar": "DX-Y.NYB",
            "Gold": "GC=F"
        }
        
        # UPDATED: Use safe wrapper
        data = safe_yf_download(list(tickers.values()), period="6mo", interval="1d")
        if data.empty: return pd.Series()
        data = data['Close']
        
        rev_tickers = {v: k for k, v in tickers.items()}
        data.rename(columns=rev_tickers, inplace=True)
        
        corrs = data.pct_change().rolling(20).corr(data[base_ticker].pct_change()).iloc[-1]
        return corrs.drop(base_ticker) 
    except: return pd.Series()

# F. CFTC COT ENGINE
@st.cache_data(ttl=86400) # Cache for 24h
def get_cot_data(asset_name):
    """Fetches CFTC Legacy Futures data for smart money tracking."""
    if asset_name not in COT_MAPPING:
        return None
    
    contract_name = COT_MAPPING[asset_name]["name"]
    
    try:
        # Load most recent year's data
        df = pd.DataFrame(cot.cot_year(datetime.now().year, cot_report_type='legacy_fut'))
        
        # Filter for contract
        asset_df = df[df['Market_and_Exchange_Names'] == contract_name].copy()
        
        if asset_df.empty:
            # Fallback to previous year
            df = pd.DataFrame(cot.cot_year(datetime.now().year - 1, cot_report_type='legacy_fut'))
            asset_df = df[df['Market_and_Exchange_Names'] == contract_name].copy()
            
        if asset_df.empty: return None
        # Sort
        asset_df['Date'] = pd.to_datetime(asset_df['As_of_Date_In_Form_YYMMDD'], format='%y%m%d')
        latest = asset_df.sort_values('Date').iloc[-1]
        
        comm_net = latest['Comm_Positions_Long_All'] - latest['Comm_Positions_Short_All']
        spec_net = latest['NonComm_Positions_Long_All'] - latest['NonComm_Positions_Short_All']
        
        sentiment = "BULLISH" if comm_net > 0 else "BEARISH"
        if asset_name in ["S&P 500", "NASDAQ", "Bitcoin"]:
            sentiment = "HEDGED (See Specs)"
            
        return {
            "date": latest['Date'].strftime('%Y-%m-%d'),
            "comm_net": comm_net,
            "spec_net": spec_net,
            "sentiment": sentiment
        }
    except Exception as e:
        return None

# --- 8. DATA FETCHERS ---
@st.cache_data(ttl=300) # Increased to 5 mins
def get_daily_data(ticker):
    try:
        # UPDATED: Use safe wrapper
        return safe_yf_download(ticker, period="10y", interval="1d")
    except: return pd.DataFrame()

@st.cache_data(ttl=300) # Increased to 5 mins
def get_intraday_data(ticker):
    try:
        # UPDATED: Use safe wrapper
        return safe_yf_download(ticker, period="5d", interval="15m")
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
    
    st.markdown("---")
    
    # --- NEW: API QUOTA MONITOR ---
    with st.expander("📡 API QUOTA MONITOR", expanded=True):
        st.markdown("<div style='font-size:0.7em; color:gray;'>Session Usage vs Hard Limits</div>", unsafe_allow_html=True)
        
        # NewsAPI
        st.write(f"**NewsAPI** ({st.session_state['news_calls']} / 100 Daily)")
        st.progress(min(st.session_state['news_calls'] / 100, 1.0))
        
        # Gemini
        st.write(f"**Gemini AI** ({st.session_state['gemini_calls']} / 20 Daily)")
        st.progress(min(st.session_state['gemini_calls'] / 20, 1.0))
        
        # RapidAPI
        st.write(f"**RapidAPI** ({st.session_state['rapid_calls']} / 10 Monthly)")
        if st.session_state['rapid_calls'] > 10:
             st.markdown("<span style='color:red; font-weight:bold;'>OVER LIMIT</span>", unsafe_allow_html=True)
        else:
             st.progress(st.session_state['rapid_calls'] / 10)
             
        # CoinGecko
        st.write(f"**CoinGecko** ({st.session_state['coingecko_calls']} / 10k Mthly)")
        st.progress(min(st.session_state['coingecko_calls'] / 10000, 1.0))

        # FRED
        st.write(f"**FRED** ({st.session_state['fred_calls']} / Calls)")
        st.progress(0.01) # Just visual

    st.markdown("---")
    
    # API KEY HANDLING
    rapid_key = get_api_key("rapidapi_key")
    news_key = get_api_key("news_api_key")
    gemini_key = get_api_key("gemini_api_key")
    cg_key = get_api_key("coingecko_key") 
    fred_key = get_api_key("fred_api_key")
    
    if st.button(">> REFRESH DATA"): 
        st.cache_data.clear()
        st.rerun()

# --- MAIN DASHBOARD ---
st.markdown(f"<h1 style='border-bottom: 2px solid #ff9900;'>{selected_asset} <span style='font-size:0.5em; color:white;'>TERMINAL PRO V5.8</span></h1>", unsafe_allow_html=True)

# Fetch Data
daily_data = get_daily_data(asset_info['ticker'])
intraday_data = get_intraday_data(asset_info['ticker'])
eco_events = get_economic_calendar(rapid_key)

# Fetch News from BOTH sources
news_general = get_financial_news_general(news_key, query=asset_info.get('news_query', 'Finance'))
news_ff = get_forex_factory_news(rapid_key, 'breaking')

# Combine for LLM
combined_news_for_llm = news_general[:5] + news_ff[:5]

# Engines
_, ml_prob = get_ml_prediction(asset_info['ticker'])
gex_df, gex_date, gex_spot = get_gex_profile(asset_info['opt_ticker'])
vol_profile, poc_price = calculate_volume_profile(intraday_data)
hurst = calculate_hurst(daily_data['Close'].values) if not daily_data.empty else 0.5
regime_data = get_market_regime(asset_info['ticker'])
cot_data = get_cot_data(selected_asset) # Fetch COT Data
tech_radar = calculate_technical_radar(daily_data) # Calculate Tech Radar

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
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=daily_data.index, open=daily_data['Open'], high=high, low=low, close=close, name="Price"))
    if poc_price:
        fig.add_hline(y=poc_price, line_dash="dash", line_color="yellow", annotation_text="POC", annotation_position="bottom right")
    terminal_chart_layout(fig, height=400)
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# --- 1B. TECHNICAL RADAR (NEW) ---
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

# --- 1C. COINGECKO INTEGRATION (DEMO SAFE) ---
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
        fig_rs.add_trace(go.Scatter(x=rs_data.index, y=rs_data['RS_Score'], mode='lines', 
                                    name='Alpha', line=dict(color=rs_color, width=2), fill='tozeroy'))
        
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
        
        levels_list = [
            ("R1 (Resist)", key_levels['R1']),
            ("PDH (High)", key_levels['PDH']),
            ("PIVOT (Daily)", key_levels['Pivot']),
            ("PDL (Low)", key_levels['PDL']),
            ("S1 (Support)", key_levels['S1'])
        ]
        
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
    if eco_events:
        cal_data = []
        for event in eco_events:
            time_str = event.get('time', 'N/A')
            name = event.get('event_name', 'Unknown')
            actual = event.get('actual', '')
            forecast = event.get('forecast', '')
            previous = event.get('previous', '')
            
            context, bias = analyze_eco_context(actual, forecast, previous)
            
            cal_data.append({"TIME": time_str, "EVENT": name, "DATA CONTEXT": context, "BIAS": bias})
            
        df_cal = pd.DataFrame(cal_data)
        
        def color_bias(val):
            color = 'white'
            if 'Bullish' in val: color = '#00ff00' 
            elif 'Bearish' in val: color = '#ff3333'
            elif 'Mean' in val: color = '#cccc00'
            return f'color: {color}'
        
        if not df_cal.empty: 
            st.dataframe(df_cal.style.map(color_bias, subset=['BIAS']), use_container_width=True, hide_index=True)
    else: 
        st.info("NO HIGH IMPACT USD EVENTS SCHEDULED.")

with col_news:
    st.markdown(f"### 📰 {asset_info.get('news_query', 'LATEST')} WIRE")
    
    # Dual Source Tabs
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
        else:
            st.markdown("<div style='color:gray;'>No data.</div>", unsafe_allow_html=True)
            
    with tab_gen:
        render_news(news_general)
    with tab_ff:
        render_news(news_ff)

# --- NEW: FRED MACRO DASHBOARD ---
st.markdown("---")
st.markdown("### 🇺🇸 FED LIQUIDITY & MACRO (FRED)")

if fred_key:
    # Use tabs for different macro categories
    macro_tab1, macro_tab2 = st.tabs(["YIELD CURVE & RATES", "INFLATION & LIQUIDITY"])
    
    with macro_tab1:
        c_m1, c_m2 = st.columns(2)
        with c_m1:
            # Yield Curve (10Y - 2Y)
            df_yield = get_fred_series("T10Y2Y", fred_key)
            if not df_yield.empty:
                curr_yield = df_yield['value'].iloc[-1]
                yield_color = "red" if curr_yield < 0 else "green"
                fig_yc = go.Figure()
                fig_yc.add_trace(go.Scatter(x=df_yield.index, y=df_yield['value'], fill='tozeroy', line=dict(color=yield_color)))
                fig_yc.add_hline(y=0, line_dash="dash", line_color="white")
                terminal_chart_layout(fig_yc, title=f"10Y-2Y SPREAD: {curr_yield:.2f}% (Recession Indicator)", height=250)
                st.plotly_chart(fig_yc, use_container_width=True)
        
        with c_m2:
            # Fed Funds Rate
            df_ff = get_fred_series("FEDFUNDS", fred_key)
            if not df_ff.empty:
                fig_ff = go.Figure()
                fig_ff.add_trace(go.Scatter(x=df_ff.index, y=df_ff['value'], line=dict(color="#00e6ff")))
                terminal_chart_layout(fig_ff, title=f"FED FUNDS RATE: {df_ff['value'].iloc[-1]:.2f}%", height=250)
                st.plotly_chart(fig_ff, use_container_width=True)

    with macro_tab2:
        c_m3, c_m4 = st.columns(2)
        with c_m3:
             # CPI (Inflation)
            df_cpi = get_fred_series("CPIAUCSL", fred_key)
            if not df_cpi.empty:
                # Calculate YoY
                df_cpi['YoY'] = df_cpi['value'].pct_change(12) * 100
                fig_cpi = go.Figure()
                fig_cpi.add_trace(go.Bar(x=df_cpi.index, y=df_cpi['YoY'], marker_color='#ff9900'))
                terminal_chart_layout(fig_cpi, title=f"CPI INFLATION (YoY): {df_cpi['YoY'].iloc[-1]:.2f}%", height=250)
                st.plotly_chart(fig_cpi, use_container_width=True)
        
        with c_m4:
            # M2 Money Supply
            df_m2 = get_fred_series("M2SL", fred_key)
            if not df_m2.empty:
                fig_m2 = go.Figure()
                fig_m2.add_trace(go.Scatter(x=df_m2.index, y=df_m2['value'], line=dict(color="#00ff00")))
                terminal_chart_layout(fig_m2, title="M2 MONEY SUPPLY (Liquidity)", height=250)
                st.plotly_chart(fig_m2, use_container_width=True)
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
        st.markdown("**BACKTEST (2Y)**")
        ret_color = "#00ff00" if strat_perf['return'] > 0 else "#ff3333"
        st.metric("Total Return", f"{strat_perf['return']*100:.1f}%")
        st.metric("Sharpe Ratio", f"{strat_perf['sharpe']:.2f}")
        
    with q2:
        # Equity Curve
        ec_df = pd.DataFrame({
            "Strategy": strat_perf['equity_curve'],
            "Buy & Hold": strat_perf['df']['Cum_BnH']
        })
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
    
    # 1. Price Candles
    fig_vwap.add_trace(go.Candlestick(x=vwap_df.index, open=vwap_df['Open'], high=vwap_df['High'], 
                                      low=vwap_df['Low'], close=vwap_df['Close'], name="Price"))
    
    # 2. Session VWAP
    fig_vwap.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df['VWAP'], name="Session VWAP", line=dict(color='#ff9900', width=2)))
    fig_vwap.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df['Upper_Band_1'], name="+1 STD", line=dict(color='gray', width=1), opacity=0.3))
    fig_vwap.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df['Lower_Band_1'], name="-1 STD", line=dict(color='gray', width=1), opacity=0.3))
    
    # 3. Key Levels
    if key_levels:
        fig_vwap.add_hline(y=key_levels['PDH'], line_dash="dot", line_color="#ff3333", annotation_text="PDH")
        fig_vwap.add_hline(y=key_levels['PDL'], line_dash="dot", line_color="#00ff00", annotation_text="PDL")
        fig_vwap.add_hline(y=key_levels['Pivot'], line_width=1, line_color="#00e6ff", annotation_text="DAILY PIVOT")
    
    terminal_chart_layout(fig_vwap, height=500)
    st.plotly_chart(fig_vwap, use_container_width=True)

# --- 6. GEX ---
st.markdown("---")
st.markdown("### 🏦 INSTITUTIONAL GAMMA EXPOSURE (GEX)")

if gex_df is not None and gex_spot is not None:
    g1, g2 = st.columns([3, 1])
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
            <p style='font-size:0.7em; margin-top:10px; color:gray;'>
            Positive = Dealers Suppress Vol<br>Negative = Dealers Amplify Vol
            </p>
        </div>
        """, unsafe_allow_html=True)
else:
    if asset_info['opt_ticker'] is None:
        st.info(f"GEX SKIPPED: OPTIONS DATA NOT AVAILABLE FOR {selected_asset} (CRYPTO/FX).")
    else:
        st.warning("GEX CALCULATION SKIPPED: NO OPTIONS CHAIN FOUND OR LIQUIDITY TOO LOW.")

# --- 7. MONTE CARLO & ADVANCED SEASONALITY ---
st.markdown("---")
st.markdown("### 🎲 SIMULATION & TIME ANALYSIS")
pred_dates, pred_paths = generate_monte_carlo(daily_data)
stats = get_seasonality_stats(daily_data, asset_info['ticker']) 

# --- SEASONALITY TABS ---
if stats:
    st.markdown("#### ⏳ SEASONAL TENDENCIES")
    tab_hour, tab_day, tab_week = st.tabs(["HOUR (NY)", "DAY", "WEEK"])
    
    with tab_hour:
        if 'hourly_perf' in stats and stats['hourly_perf'] is not None:
            hp = stats['hourly_perf']
            fig_h = go.Figure()
            hour_labels = [f"{h:02d}:00" for h in hp.index]
            colors = ['#00ff00' if v > 0 else '#ff3333' for v in hp.values]
            fig_h.add_trace(go.Bar(x=hour_labels, y=hp.values, marker_color=colors))
            terminal_chart_layout(fig_h, title="AVG RETURN BY HOUR (NY TIME)", height=350)
            st.plotly_chart(fig_h, use_container_width=True)
        else:
            st.info("Hourly data insufficient.")
            
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

st.markdown("---")

# --- MONTE CARLO ---
st.markdown("#### 🎲 MONTE CARLO PROJECTION")
if pred_dates is not None and pred_paths is not None:
    fig_pred = go.Figure()
    hist_slice = daily_data['Close'].tail(90)
    fig_pred.add_trace(go.Scatter(x=hist_slice.index, y=hist_slice.values, name='History', line=dict(color='white')))
    fig_pred.add_trace(go.Scatter(x=pred_dates, y=np.mean(pred_paths, axis=1), name='Avg Path', line=dict(color='#ff9900', dash='dash')))
    terminal_chart_layout(fig_pred, title="MONTE CARLO PROJECTION (126 Days)", height=400)
    st.plotly_chart(fig_pred, use_container_width=True)
else:
    st.info("Insufficient data for Monte Carlo simulation (Ticker may be invalid or data feed down).")

# --- 8. CFTC COT DISPLAY ---
if cot_data:
    st.markdown("---")
    st.markdown("### 🏛️ CFTC COMMITMENTS OF TRADERS")
    c_cot1, c_cot2, c_cot3 = st.columns(3)
    
    c_cot1.metric("Commercials (Smart Money)", f"{cot_data['comm_net']:,.0f}", help="Commercial Hedgers Net Position")
    c_cot2.metric("Speculators (Funds)", f"{cot_data['spec_net']:,.0f}", help="Non-Commercial/Managed Money Net Position")
    c_cot3.markdown(f"""
    <div class='terminal-box' style='text-align:center;'>
        <div style='font-size:0.8em; color:gray;'>SENTIMENT</div>
        <div style='font-size:1.2em; font-weight:bold; color:white;'>{cot_data['sentiment']}</div>
        <div style='font-size:0.7em;'>Date: {cot_data['date']}</div>
    </div>
    """, unsafe_allow_html=True)

# --- 9. INTELLIGENT EXECUTIVE SUMMARY & THESIS ---
st.markdown("---")
st.markdown("### 🧠 AI QUANT ANALYST")

# Prepare General Data for LLM
gex_summary = gex_df if gex_df is not None else None
ml_signal_str = "BULLISH" if ml_prob > 0.55 else "BEARISH" if ml_prob < 0.45 else "NEUTRAL"
news_text_summary = "\n".join([f"- {n['title']} ({n['source']})" for n in combined_news_for_llm])

if gemini_key:
    # --- 9A. EXECUTIVE SUMMARY (ON DEMAND ONLY) ---
    col_exec_btn, col_exec_info = st.columns([1, 4])
    with col_exec_btn:
        # We use a button AND check session state to keep the report visible
        if st.button("📝 GENERATE BRIEF"):
            with st.spinner("Analyzing Technicals..."):
                narrative = get_technical_narrative(
                    ticker=selected_asset, price=curr, daily_pct=pct, regime=regime_data,
                    ml_signal=ml_signal_str, gex_data=gex_summary, cot_data=cot_data,
                    levels=key_levels, api_key=gemini_key
                )
                st.session_state['narrative_cache'] = narrative
                st.rerun() # Rerun to update the session state and progress bars

    with col_exec_info:
        st.markdown("<span style='color:gray; vertical-align:middle;'>Generates a 3-bullet Executive Summary (Costs 1 Gemini Call)</span>", unsafe_allow_html=True)

    # Display Executive Summary if it exists in session state
    if st.session_state['narrative_cache']:
        # Check for error message in the output
        if "⚠️" in st.session_state['narrative_cache']:
             st.error(st.session_state['narrative_cache'])
        else:
            st.markdown(f"""
            <div class='terminal-box' style='border-left: 4px solid #00e6ff;'>
                <div style='font-family: monospace; font-size: 0.95em; white-space: pre-wrap;'>{st.session_state['narrative_cache']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # --- 9B. THESIS MODE (ON DEMAND) ---
    st.markdown("---")
    col_thesis_btn, col_thesis_info = st.columns([1, 4])
    with col_thesis_btn:
        if st.button("🔎 DEEP DIVE THESIS"):
            with st.spinner("Analyzing Macro, Gamma, and Order Flow..."):
                thesis_text = generate_deep_dive_thesis(
                    ticker=selected_asset, price=curr, change=pct, regime=regime_data,
                    ml_signal=ml_signal_str, gex_data=gex_summary, cot_data=cot_data,
                    levels=key_levels, news_summary=news_text_summary, api_key=gemini_key
                )
                st.session_state['thesis_cache'] = thesis_text
                st.rerun()

    with col_thesis_info:
        st.markdown("<span style='color:gray;'>Generates a full 4-paragraph Hedge Fund style report. (Costs 1 Gemini Call)</span>", unsafe_allow_html=True)

    if st.session_state['thesis_cache']:
        # Check for error message in the output
        if "⚠️" in st.session_state['thesis_cache']:
             st.error(st.session_state['thesis_cache'])
        else:
            st.markdown(f"""
            <div class='terminal-box' style='border: 1px solid #444; padding: 20px;'>
                {st.session_state['thesis_cache']}
            </div>
            """, unsafe_allow_html=True)

else:
    st.info("Add GEMINI_API_KEY to see the AI Analyst Report.")
