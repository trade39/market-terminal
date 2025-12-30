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
st.set_page_config(layout="wide", page_title="Bloomberg Terminal Pro V5.1", page_icon="💹")

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
    .bullish { color: #000000; background-color: #00ff00; padding: 2px 6px; font-weight: bold; }
    .bearish { color: #000000; background-color: #ff3333; padding: 2px 6px; font-weight: bold; }
    .neutral { color: #000000; background-color: #cccccc; padding: 2px 6px; font-weight: bold; }
    
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
</style>
""", unsafe_allow_html=True)

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
    if "api_keys" in st.secrets and key_name in st.secrets["api_keys"]:
        return st.secrets["api_keys"][key_name]
    if key_name in st.secrets:
        return st.secrets[key_name]
    if key_name == "gemini_api_key":
        if "GOOGLE_API_KEY" in st.secrets: return st.secrets["GOOGLE_API_KEY"]
        if "google_api_key" in st.secrets: return st.secrets["google_api_key"]
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

# --- UNIFIED YFINANCE CACHING (Optimization) ---
@st.cache_data(ttl=300) # 5 Minutes cache
def safe_yf_download(ticker, period, interval):
    """
    Wrapper for YFinance that handles errors gracefully and enables threading.
    Returns empty DataFrame on failure so app doesn't crash.
    """
    try:
        # threads=True is cleaner and faster for downloads
        df = yf.download(ticker, period=period, interval=interval, progress=False, threads=True)
        return flatten_dataframe(df)
    except Exception as e:
        # Log error in console if needed, return empty DF for UI safety
        print(f"YF Error {ticker}: {e}")
        return pd.DataFrame()

# --- COINGECKO API ENGINE ---
@st.cache_data(ttl=300)
def get_coingecko_stats(cg_id, api_key):
    if not cg_id or not api_key: return None
    url = f"https://api.coingecko.com/api/v3/coins/{cg_id}"
    params = {"localization": "false", "tickers": "false", "market_data": "true", "community_data": "true", "developer_data": "true", "sparkline": "false"}
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
    except: return None

# --- LLM ENGINE (UPDATED FOR DEEP DIVE) ---
def get_gemini_model(api_key):
    genai.configure(api_key=api_key)
    try:
        available_models = [m for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        available_models.sort(key=lambda x: 'flash' not in x.name) # Prefer Flash
        return genai.GenerativeModel(available_models[0].name) if available_models else None
    except: return None

@st.cache_data(ttl=900)
def get_executive_summary(ticker, price, daily_pct, regime, ml_signal, gex_data, cot_data, levels, api_key):
    """Standard 3-Bullet View (Fast)"""
    if not api_key: return "AI Analyst unavailable."
    
    gex_text = "N/A"
    if gex_data is not None:
        total_gex = gex_data['gex'].sum()
        gex_text = f"Net Gamma: ${total_gex/1_000_000:.1f}M"
        
    prompt = f"""
    You are a Senior Portfolio Manager. Write a 3-bullet executive summary for {ticker}.
    DATA: Price {price:.2f} ({daily_pct:.2f}%), Regime: {regime['regime'] if regime else 'N/A'}, ML: {ml_signal}, GEX: {gex_text}, COT: {cot_data['sentiment'] if cot_data else 'N/A'}.
    TASK: Synthesize conflict/confluence. Identify key trigger level. Give execution bias (e.g. "Buy Dips").
    STYLE: Professional, jargon-heavy (Bloomberg). No markdown bolding.
    """
    try:
        model = get_gemini_model(api_key)
        return model.generate_content(prompt).text if model else "Model Error"
    except Exception as e: return f"AI Error: {str(e)}"

# No Cache on this one - generated on demand
def get_deep_dive_thesis(ticker, price, daily_pct, regime, ml_signal, gex_data, cot_data, levels, tech_radar, api_key):
    """Deep Dive Thesis (Slow, On-Demand)"""
    if not api_key: return "AI Analyst unavailable."
    
    # Format Technical Radar for Prompt
    radar_str = ", ".join([f"{k}: {v['Signal']}" for k, v in tech_radar.items()])
    
    prompt = f"""
    You are a Hedge Fund Analyst. Generate a DEEP DIVE INVESTMENT THESIS for {ticker}.
    
    ### MARKET DATA
    - Price: {price:,.2f} ({daily_pct:.2f}%)
    - Market Regime: {regime['regime'] if regime else 'Unknown'}
    - ML Probability: {ml_signal}
    - Technical Radar: {radar_str}
    - Smart Money (COT): {cot_data['sentiment'] if cot_data else 'N/A'}
    
    ### OUTPUT FORMAT (Markdown)
    **The Core Argument**: (One sentence thesis, e.g., "Long {ticker} targeting X due to...")
    **Supporting Data**: (Reference specific numbers from ML, Radar, or Regime that support the view.)
    **The Bear Case (Risks)**: (What invalidates this? e.g., Regulatory headwinds, specific level breaks.)
    **Invalidation Level**: (Specific price point where this thesis is wrong.)
    
    Keep it strictly 4 paragraphs corresponding to these sections.
    """
    try:
        model = get_gemini_model(api_key)
        return model.generate_content(prompt).text if model else "Model Error"
    except Exception as e: return f"AI Error: {str(e)}"

# --- 1. QUANT ENGINE ---
def calculate_hurst(series, lags=range(2, 20)):
    try:
        tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    except: return 0.5

@st.cache_data(ttl=3600)
def get_market_regime(ticker):
    try:
        data = safe_yf_download(ticker, "5y", "1d")
        if data.empty: return None
        
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

# --- 2. MACHINE LEARNING ENGINE ---
@st.cache_data(ttl=3600)
def get_ml_prediction(ticker):
    try:
        data = safe_yf_download(ticker, "2y", "1d")
        if data.empty: return None, 0.5
        
        data['Returns'] = data['Close'].pct_change()
        data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
        data['Vol_5d'] = data['Returns'].rolling(5).std()
        data['Mom_5d'] = data['Close'].pct_change(5)
        data = data.dropna()
        
        if len(data) < 50: return None, 0.5
        
        model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
        model.fit(data[['Vol_5d', 'Mom_5d']], data['Target'])
        return model, model.predict_proba(data[['Vol_5d', 'Mom_5d']].iloc[[-1]])[0][1]
    except: return None, 0.5

# --- 3. TECHNICAL RADAR (PURE PANDAS - OPTIMIZED) ---
def calculate_technical_radar(df):
    """
    Calculates 4 key technical indicators using optimized Pandas (No TA-Lib/Pandas-TA).
    Returns a dictionary of signals.
    """
    if df.empty or len(df) < 30: return {}
    
    close = df['Close']
    radar = {}
    
    # 1. RSI (14)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    curr_rsi = rsi.iloc[-1]
    
    if curr_rsi > 70: radar['RSI'] = {'Val': f"{curr_rsi:.0f}", 'Signal': 'OVERBOUGHT', 'Color': 'bearish'}
    elif curr_rsi < 30: radar['RSI'] = {'Val': f"{curr_rsi:.0f}", 'Signal': 'OVERSOLD', 'Color': 'bullish'}
    else: radar['RSI'] = {'Val': f"{curr_rsi:.0f}", 'Signal': 'NEUTRAL', 'Color': 'neutral'}

    # 2. MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    
    if macd.iloc[-1] > signal.iloc[-1]: radar['MACD'] = {'Val': 'CROSS UP', 'Signal': 'BULLISH', 'Color': 'bullish'}
    else: radar['MACD'] = {'Val': 'CROSS DOWN', 'Signal': 'BEARISH', 'Color': 'bearish'}

    # 3. Bollinger Bands (20, 2)
    sma20 = close.rolling(window=20).mean()
    std20 = close.rolling(window=20).std()
    upper = sma20 + (std20 * 2)
    lower = sma20 - (std20 * 2)
    curr_px = close.iloc[-1]
    
    if curr_px > upper.iloc[-1]: radar['BBands'] = {'Val': 'UPPER BREAK', 'Signal': 'VOLATILE UP', 'Color': 'bullish'}
    elif curr_px < lower.iloc[-1]: radar['BBands'] = {'Val': 'LOWER BREAK', 'Signal': 'VOLATILE DOWN', 'Color': 'bearish'}
    else: radar['BBands'] = {'Val': 'INSIDE', 'Signal': 'RANGE', 'Color': 'neutral'}

    # 4. EMA Trend (50)
    ema50 = close.ewm(span=50, adjust=False).mean()
    if curr_px > ema50.iloc[-1]: radar['EMA Trend'] = {'Val': '> EMA50', 'Signal': 'UPTREND', 'Color': 'bullish'}
    else: radar['EMA Trend'] = {'Val': '< EMA50', 'Signal': 'DOWNTREND', 'Color': 'bearish'}

    return radar

# --- 4. GAMMA EXPOSURE ENGINE ---
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
            spot_price = hist['Close'].iloc[-1] if not hist.empty else tk.fast_info.last_price
        except: return None, None, None
        
        if spot_price is None: return None, None, None
        exps = tk.options
        if not exps: return None, None, None
        target_exp = exps[1] if len(exps) > 1 else exps[0]
        
        chain = tk.option_chain(target_exp)
        calls, puts = chain.calls, chain.puts
        if calls.empty or puts.empty: return None, None, None
        
        exp_date = datetime.strptime(target_exp, "%Y-%m-%d")
        days_to_exp = (exp_date - datetime.now()).days
        T = max(days_to_exp / 365.0, 0.001)
        r = 0.045
        
        gex_data = []
        strikes = sorted(list(set(calls['strike'].tolist() + puts['strike'].tolist())))
        
        for K in strikes:
            if K < spot_price * 0.75 or K > spot_price * 1.25: continue
            c_row = calls[calls['strike'] == K]
            p_row = puts[puts['strike'] == K]
            
            c_oi = c_row['openInterest'].iloc[0] if not c_row.empty else 0
            p_oi = p_row['openInterest'].iloc[0] if not p_row.empty else 0
            
            c_iv = c_row['impliedVolatility'].iloc[0] if not c_row.empty else 0.2
            p_iv = p_row['impliedVolatility'].iloc[0] if not p_row.empty else 0.2
            
            c_gamma = calculate_black_scholes_gamma(spot_price, K, T, r, c_iv)
            p_gamma = calculate_black_scholes_gamma(spot_price, K, T, r, p_iv)
            
            net_gex = (c_gamma * c_oi - p_gamma * p_oi) * spot_price * 100
            gex_data.append({"strike": K, "gex": net_gex})
            
        df = pd.DataFrame(gex_data, columns=['strike', 'gex'])
        return (None, None, None) if df.empty else (df, target_exp, spot_price)
    except: return None, None, None

# --- 5. DATA FETCHERS & HELPERS ---
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
    try:
        stats = {}
        df = daily_data.copy()
        df['Week_Num'] = df.index.to_period('W')
        high_days = df.groupby('Week_Num')['High'].idxmax().apply(lambda x: df.loc[x].name.day_name())
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        stats['day_high'] = high_days.value_counts().reindex(days_order, fill_value=0) / len(high_days) * 100
        
        df['Day'] = df.index.day
        df['Month_Week'] = np.ceil(df['Day'] / 7).astype(int)
        df['Returns'] = df['Close'].pct_change()
        stats['week_returns'] = df.groupby('Month_Week')['Returns'].mean() * 100
        
        intra = safe_yf_download(ticker_name, "60d", "1h")
        if not intra.empty:
            if intra.index.tz is None: intra.index = intra.index.tz_localize('UTC')
            intra.index = intra.index.tz_convert('America/New_York')
            intra['Hour'] = intra.index.hour
            intra['Return'] = intra['Close'].pct_change()
            target_hours = [2,3,4,5,6, 8,9,10,11, 14,15,16,17,18, 20,21,22,23]
            stats['hourly_perf'] = intra[intra['Hour'].isin(target_hours)].groupby('Hour')['Return'].mean() * 100
        else: stats['hourly_perf'] = None
        return stats
    except: return None

@st.cache_data(ttl=3600)
def generate_monte_carlo(stock_data, days=126, simulations=1000):
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
    val_actual, val_forecast, val_prev = parse_eco_value(actual_str), parse_eco_value(forecast_str), parse_eco_value(previous_str)
    bias = "Neutral"
    if actual_str:
        if val_actual is not None and val_forecast is not None:
            context_str = f"Actual ({actual_str}) vs Forecast ({forecast_str})"
            delta = val_actual - val_forecast
            pct_dev = abs(delta / val_forecast) if val_forecast != 0 else (1.0 if delta!=0 else 0)
            if pct_dev < 0.02: bias = "Mean Reverting"
            elif delta > 0: bias = "Bullish"
            else: bias = "Bearish"
        else: context_str = f"Actual: {actual_str}"
    else:
        context_str = f"Forecast ({forecast_str}) vs Prev ({previous_str})" if val_forecast and val_prev else "Waiting for Data..."
    return context_str, bias

@st.cache_data(ttl=14400)
def get_financial_news_general(api_key, query="Finance"):
    if not api_key: return []
    try:
        newsapi = NewsApiClient(api_key=api_key)
        all_articles = newsapi.get_everything(q=query, language='en', sort_by='publishedAt')
        return [{"title": a['title'], "source": a['source']['name'], "url": a['url'], "time": a['publishedAt']} for a in all_articles['articles'][:6]]
    except: return []

@st.cache_data(ttl=14400)
def get_forex_factory_news(api_key, news_type='breaking'):
    if not api_key: return []
    url = f"https://forex-factory-scraper1.p.rapidapi.com/{'latest_breaking_news' if news_type=='breaking' else 'latest_fundamental_analysis_news'}"
    headers = {"x-rapidapi-key": api_key, "x-rapidapi-host": "forex-factory-scraper1.p.rapidapi.com"}
    try:
        data = requests.get(url, headers=headers).json()
        return [{"title": i.get('title'), "url": i.get('link', '#'), "source": "ForexFactory", "time": i.get('date', 'Recent')} for i in data[:6]] if isinstance(data, list) else []
    except: return []

@st.cache_data(ttl=21600)
def get_economic_calendar(api_key):
    if not api_key: return None
    # Priority 1: Forex Factory
    try:
        url = "https://forex-factory-scraper1.p.rapidapi.com/get_real_time_calendar_details"
        now = datetime.now()
        params = {"calendar": "Forex", "year": str(now.year), "month": str(now.month), "day": str(now.day), "currency": "ALL", "event_name": "ALL", "timezone": "GMT-04:00 Eastern Time (US & Canada)", "time_format": "12h"}
        headers = {"x-rapidapi-host": "forex-factory-scraper1.p.rapidapi.com", "x-rapidapi-key": api_key}
        data = requests.get(url, headers=headers, params=params).json()
        raw = data if isinstance(data, list) else data.get('data', [])
        filtered = [e for e in raw if e.get('currency') == 'USD' and e.get('impact') in ['High', 'Medium']]
        if filtered: return filtered
    except: pass
    # Priority 2: Backup
    try:
        url = "https://ultimate-economic-calendar.p.rapidapi.com/economic-events/tradingview"
        now_str = datetime.now().strftime("%Y-%m-%d")
        headers = {"x-rapidapi-key": api_key, "x-rapidapi-host": "ultimate-economic-calendar.p.rapidapi.com"}
        data = requests.get(url, headers=headers, params={"from": now_str, "to": now_str, "countries": "US"}).json()
        return [{"time": e.get('date'), "event_name": e.get('title'), "actual": str(e.get('actual','')), "forecast": str(e.get('forecast','')), "previous": str(e.get('previous','')), "currency": "USD", "impact": "Medium/High"} for e in data] if isinstance(data, list) else []
    except: return []

# --- 7. INSTITUTIONAL FEATURES ---
@st.cache_data(ttl=300)
def run_strategy_backtest(ticker):
    try:
        df = safe_yf_download(ticker, "2y", "1d")
        if df.empty: return None
        df['Returns'] = df['Close'].pct_change()
        df['Range'] = df['High'] - df['Low']
        df['TR'] = pd.concat([df['Range'], (df['High'] - df['Close'].shift(1)).abs(), (df['Low'] - df['Close'].shift(1)).abs()], axis=1).max(axis=1)
        df['Vol_Forecast'] = np.log(df['TR'] / df['Close']).ewm(span=10).mean()
        df['Vol_Baseline'] = np.log(df['TR'] / df['Close']).rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['Signal'] = np.where((df['Vol_Forecast'] > df['Vol_Baseline']) & (df['Close'] > df['SMA_50']), 1, 0)
        df['Strategy_Returns'] = df['Signal'].shift(1) * df['Returns']
        df['Cum_BnH'] = (1 + df['Returns']).cumprod()
        df['Cum_Strat'] = (1 + df['Strategy_Returns']).cumprod()
        return {"df": df, "signal": "LONG" if df['Signal'].iloc[-1] == 1 else "CASH/NEUTRAL", "return": df['Cum_Strat'].iloc[-1] - 1, "sharpe": (df['Strategy_Returns'].mean() / df['Strategy_Returns'].std()) * np.sqrt(252) if df['Strategy_Returns'].std() != 0 else 0, "equity_curve": df['Cum_Strat']}
    except: return None

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
    
    # Calculate Standard Deviation Bands (Step-by-Step Vectorized to avoid KeyError)
    df['Sq_Dist'] = df['Volume'] * (df['TP'] - df['VWAP'])**2
    df['Cum_Sq_Dist'] = df.groupby('Date')['Sq_Dist'].cumsum()
    df['Std_Dev'] = np.sqrt(df['Cum_Sq_Dist'] / df['Cum_Vol'])
    
    df['Upper_Band_1'] = df['VWAP'] + df['Std_Dev']
    df['Lower_Band_1'] = df['VWAP'] - df['Std_Dev']
    
    return df

@st.cache_data(ttl=300)
def get_relative_strength(asset_ticker, benchmark_ticker="SPY"):
    try:
        asset = safe_yf_download(asset_ticker, "5d", "15m")
        bench = safe_yf_download(benchmark_ticker, "5d", "15m")
        if asset.empty or bench.empty: return pd.DataFrame()
        df = pd.DataFrame(index=asset.index)
        df['Asset_Close'] = asset['Close']
        df['Bench_Close'] = bench['Close']
        df = df.dropna()
        curr_date = df.index[-1].date()
        sess = df[df.index.date == curr_date].copy()
        if sess.empty: return pd.DataFrame()
        sess['RS_Score'] = ((sess['Asset_Close'] / sess['Asset_Close'].iloc[0]) - 1) - ((sess['Bench_Close'] / sess['Bench_Close'].iloc[0]) - 1)
        return sess
    except: return pd.DataFrame()

def get_key_levels(daily_df):
    if daily_df.empty: return {}
    try:
        last = daily_df.iloc[-2]
        pivot = (last['High'] + last['Low'] + last['Close']) / 3
        return {"PDH": last['High'], "PDL": last['Low'], "PDC": last['Close'], "Pivot": pivot, "R1": (2 * pivot) - last['Low'], "S1": (2 * pivot) - last['High']}
    except: return {}

@st.cache_data(ttl=3600)
def get_correlations(base_ticker):
    try:
        tickers = {"Base": base_ticker, "VIX": "^VIX", "10Y Yield": "^TNX", "Dollar": "DX-Y.NYB", "Gold": "GC=F"}
        data = safe_yf_download(list(tickers.values()), "6mo", "1d")['Close']
        data.columns = [k for k,v in tickers.items() if v in data.columns]
        return data.pct_change().rolling(20).corr(data['Base'].pct_change()).iloc[-1].drop('Base')
    except: return pd.Series()

@st.cache_data(ttl=86400)
def get_cot_data(asset_name):
    if asset_name not in COT_MAPPING: return None
    try:
        name = COT_MAPPING[asset_name]["name"]
        df = pd.DataFrame(cot.cot_year(datetime.now().year, cot_report_type='legacy_fut'))
        adf = df[df['Market_and_Exchange_Names'] == name].copy()
        if adf.empty:
            df = pd.DataFrame(cot.cot_year(datetime.now().year - 1, cot_report_type='legacy_fut'))
            adf = df[df['Market_and_Exchange_Names'] == name].copy()
        if adf.empty: return None
        adf['Date'] = pd.to_datetime(adf['As_of_Date_In_Form_YYMMDD'], format='%y%m%d')
        latest = adf.sort_values('Date').iloc[-1]
        comm_net = latest['Comm_Positions_Long_All'] - latest['Comm_Positions_Short_All']
        spec_net = latest['NonComm_Positions_Long_All'] - latest['NonComm_Positions_Short_All']
        return {"date": latest['Date'].strftime('%Y-%m-%d'), "comm_net": comm_net, "spec_net": spec_net, "sentiment": "BULLISH" if comm_net > 0 else "BEARISH"}
    except: return None

def terminal_chart_layout(fig, title="", height=350):
    fig.update_layout(title=dict(text=title, font=dict(color="#ff9900", family="Arial")), template="plotly_dark", paper_bgcolor="#000000", plot_bgcolor="#000000", height=height, margin=dict(l=40, r=40, t=40, b=40), xaxis=dict(showgrid=True, gridcolor="#222"), yaxis=dict(showgrid=True, gridcolor="#222"), font=dict(family="Courier New", color="#e0e0e0"))
    return fig

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h3 style='color: #ff9900;'>COMMAND LINE</h3>", unsafe_allow_html=True)
    selected_asset = st.selectbox("SEC / Ticker", list(ASSETS.keys()))
    asset_info = ASSETS[selected_asset]
    st.markdown("---")
    st.markdown("**MACRO CORRELATIONS (20D)**")
    corrs = get_correlations(asset_info['ticker'])
    if not corrs.empty:
        for idx, val in corrs.items():
            c_color = "#ff3333" if val > 0.5 else "#00ff00" if val < -0.5 else "gray" 
            st.markdown(f"{idx}: <span style='color:{c_color}'>{val:.2f}</span>", unsafe_allow_html=True)
    st.markdown("---")
    rapid_key = get_api_key("rapidapi_key")
    news_key = get_api_key("news_api_key")
    gemini_key = get_api_key("gemini_api_key")
    cg_key = get_api_key("coingecko_key") 
    st.markdown(f"<div style='font-size:0.8em; color:gray;'>API STATUS:<br>RAPID: {'[OK]' if rapid_key else '[FAIL]'}<br>NEWS: {'[OK]' if news_key else '[FAIL]'}<br>GEMINI: {'[OK]' if gemini_key else '[FAIL]'}<br>GECKO: {'[OK]' if cg_key else '[FAIL]'}</div>", unsafe_allow_html=True)
    if st.button(">> REFRESH DATA"): st.cache_data.clear()

# --- MAIN DASHBOARD ---
st.markdown(f"<h1 style='border-bottom: 2px solid #ff9900;'>{selected_asset} <span style='font-size:0.5em; color:white;'>TERMINAL PRO V5.1</span></h1>", unsafe_allow_html=True)

# Fetch Data
daily_data = safe_yf_download(asset_info['ticker'], "10y", "1d")
intraday_data = safe_yf_download(asset_info['ticker'], "5d", "15m")
eco_events = get_economic_calendar(rapid_key)
news_general = get_financial_news_general(news_key, query=asset_info.get('news_query', 'Finance'))
news_ff = get_forex_factory_news(rapid_key, 'breaking')

# Engines
_, ml_prob = get_ml_prediction(asset_info['ticker'])
gex_df, gex_date, gex_spot = get_gex_profile(asset_info['opt_ticker'])
vol_profile, poc_price = calculate_volume_profile(intraday_data)
hurst = calculate_hurst(daily_data['Close'].values) if not daily_data.empty else 0.5
regime_data = get_market_regime(asset_info['ticker'])
cot_data = get_cot_data(selected_asset)
tech_radar = calculate_technical_radar(daily_data) # New Technical Radar

# --- 1. OVERVIEW ---
if not daily_data.empty:
    close, high, low = daily_data['Close'], daily_data['High'], daily_data['Low']
    curr = close.iloc[-1]
    pct = ((curr - close.iloc[-2]) / close.iloc[-2]) * 100
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LAST PX", f"{curr:,.2f}", f"{pct:.2f}%")
    
    ml_bias = "BULLISH" if ml_prob > 0.55 else "BEARISH" if ml_prob < 0.45 else "NEUTRAL"
    c2.markdown(f"<div class='terminal-box' style='text-align:center;'><div style='color:#ff9900;'>AI PREDICTION</div><span class='{ml_bias.lower()}'>{ml_bias}</span></div>", unsafe_allow_html=True)
    
    if regime_data:
        c3.markdown(f"<div class='terminal-box'><div style='color:#ff9900;'>QUANT REGIME</div><span class='{regime_data['color']}'>{regime_data['regime']}</span></div>", unsafe_allow_html=True)
    
    c4.metric("HIGH/LOW", f"{high.max():,.2f} / {low.min():,.2f}")
    
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=daily_data.index, open=daily_data['Open'], high=high, low=low, close=close, name="Price"))
    if poc_price: fig.add_hline(y=poc_price, line_dash="dash", line_color="yellow", annotation_text="POC")
    terminal_chart_layout(fig, height=400)
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# --- 1B. COINGECKO ---
cg_id = asset_info.get('cg_id')
if cg_id and cg_key:
    st.markdown("---")
    st.markdown("### 🦎 COINGECKO FUNDAMENTALS")
    cg_data = get_coingecko_stats(cg_id, cg_key)
    if cg_data:
        c_cg1, c_cg2, c_cg3 = st.columns(3)
        c_cg1.metric("Rank", f"#{cg_data['rank']}")
        c_cg2.markdown(f"<div class='terminal-box'>ATH Drawdown: <span style='color:red'>{cg_data['ath_change']:.2f}%</span></div>", unsafe_allow_html=True)
        c_cg3.markdown(f"<div class='terminal-box'>Sentiment: {cg_data['sentiment']}% Bullish</div>", unsafe_allow_html=True)

# --- 2. INTRADAY TACTICAL & TECHNICAL RADAR ---
st.markdown("---")
st.markdown("### 🔭 TACTICAL FEED & RADAR")
rs_data = get_relative_strength(asset_info['ticker'])
key_levels = get_key_levels(daily_data)

col_intra, col_radar = st.columns([2, 1])

with col_intra:
    if not rs_data.empty:
        curr_rs = rs_data['RS_Score'].iloc[-1]
        rs_color = "#00ff00" if curr_rs > 0 else "#ff3333"
        st.markdown(f"**RELATIVE STRENGTH (vs SPY)**: <span style='color:{rs_color}'>{'OUTPERFORMING' if curr_rs > 0 else 'UNDERPERFORMING'}</span>", unsafe_allow_html=True)
        fig_rs = go.Figure()
        fig_rs.add_hline(y=0, line_color="#333", line_dash="dash")
        fig_rs.add_trace(go.Scatter(x=rs_data.index, y=rs_data['RS_Score'], mode='lines', name='Alpha', line=dict(color=rs_color, width=2), fill='tozeroy'))
        terminal_chart_layout(fig_rs, height=250)
        st.plotly_chart(fig_rs, use_container_width=True)

with col_radar:
    st.markdown("**📡 TECHNICAL RADAR**")
    if tech_radar:
        for ind, data in tech_radar.items():
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; border-bottom:1px solid #222; padding:5px;">
                <span style="color:#aaa;">{ind}</span>
                <div style="text-align:right;">
                    <span style="font-size:0.8em; color:white;">{data['Val']}</span><br>
                    <span class="{data['Color']}" style="font-size:0.7em;">{data['Signal']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Insufficient data for Radar.")

# --- 3. EVENTS & NEWS (Condensed) ---
st.markdown("---")
col_eco, col_news = st.columns([2, 1])
with col_eco:
    st.markdown("### 📅 EVENTS (USD)")
    if eco_events:
        df_cal = pd.DataFrame([{"TIME": e.get('time'), "EVENT": e.get('event_name'), "ACT": e.get('actual'), "FCST": e.get('forecast')} for e in eco_events])
        st.dataframe(df_cal, use_container_width=True, hide_index=True)
    else: st.info("NO HIGH IMPACT EVENTS.")
with col_news:
    st.markdown(f"### 📰 NEWS")
    tab_gen, tab_ff = st.tabs(["GENERAL", "FX FACTORY"])
    with tab_gen:
        for n in news_general: st.markdown(f"<a class='news-link' href='{n['url']}'>{n['title']}</a>", unsafe_allow_html=True)
    with tab_ff:
        for n in news_ff: st.markdown(f"<a class='news-link' href='{n['url']}'>{n['title']}</a>", unsafe_allow_html=True)

# --- 4. RISK, VWAP, GEX ---
st.markdown("---")
strat_perf = run_strategy_backtest(asset_info['ticker'])
if not intraday_data.empty and strat_perf:
    st.markdown("### ⚡ QUANT EXECUTION & GEX")
    q1, q2, q3 = st.columns(3)
    with q1:
        st.markdown(f"SIGNAL: <span style='color:{'#00ff00' if 'LONG' in strat_perf['signal'] else '#ffff00'}; font-size:1.5em;'>{strat_perf['signal']}</span>", unsafe_allow_html=True)
        st.metric("Strategy Return (2Y)", f"{strat_perf['return']*100:.1f}%")
    with q2:
        vwap_df = calculate_vwap_bands(intraday_data)
        if not vwap_df.empty:
            fig_vwap = go.Figure()
            fig_vwap.add_trace(go.Candlestick(x=vwap_df.index, open=vwap_df['Open'], close=vwap_df['Close'], high=vwap_df['High'], low=vwap_df['Low'], name='Price'))
            fig_vwap.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df['VWAP'], line=dict(color='#ff9900')))
            terminal_chart_layout(fig_vwap, title="SESSION VWAP", height=250)
            st.plotly_chart(fig_vwap, use_container_width=True)
    with q3:
        if gex_df is not None:
            total_gex = gex_df['gex'].sum() / 1_000_000
            st.metric("Net Gamma", f"${total_gex:.1f}M", "Sticky" if total_gex > 0 else "Volatile")
            st.bar_chart(gex_df.set_index('strike')['gex'])
        else: st.info("No Options Data")

# --- 5. AI ANALYST (UPDATED) ---
st.markdown("---")
st.markdown("### 🧠 AI QUANT ANALYST")

if gemini_key:
    # Tab 1: Fast Executive Summary (Cached)
    # Tab 2: Deep Dive Thesis (On Demand)
    tab_exec, tab_thesis = st.tabs(["⚡ EXECUTIVE SUMMARY", "🕵️ DEEP DIVE THESIS"])
    
    with tab_exec:
        with st.spinner("Synthesizing Summary..."):
            summary = get_executive_summary(
                ticker=selected_asset, price=curr, daily_pct=pct, regime=regime_data,
                ml_signal="BULLISH" if ml_prob > 0.55 else "BEARISH",
                gex_data=gex_df, cot_data=cot_data, levels=key_levels, api_key=gemini_key
            )
        st.markdown(f"<div class='terminal-box'>{summary}</div>", unsafe_allow_html=True)
        
    with tab_thesis:
        st.write("Generate a structured 4-paragraph report covering Macro, Technicals, and Risks.")
        if st.button("GENERATE INVESTMENT THESIS (Takes ~10s)"):
            with st.spinner("Analyzing Market Structure, Order Flow, and Macro..."):
                thesis = get_deep_dive_thesis(
                    ticker=selected_asset, price=curr, daily_pct=pct, regime=regime_data,
                    ml_signal="BULLISH" if ml_prob > 0.55 else "BEARISH",
                    gex_data=gex_df, cot_data=cot_data, levels=key_levels, 
                    tech_radar=tech_radar, api_key=gemini_key
                )
            st.markdown(thesis)
else:
    st.warning("GEMINI_API_KEY required for AI Analysis.")
