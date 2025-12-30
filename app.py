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
st.set_page_config(layout="wide", page_title="Bloomberg Terminal Pro V5.6", page_icon="💹")

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
    
    /* API Limit Progress Bars */
    .stProgress > div > div > div > div { background-color: #ff9900; }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'gemini_calls' not in st.session_state: st.session_state['gemini_calls'] = 0
if 'news_calls' not in st.session_state: st.session_state['news_calls'] = 0
if 'rapid_calls' not in st.session_state: st.session_state['rapid_calls'] = 0
if 'coingecko_calls' not in st.session_state: st.session_state['coingecko_calls'] = 0
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
    if df is None or df.empty: return pd.DataFrame()
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    return df

# --- UNIFIED DATA FETCHERS (OPTIMIZED) ---
@st.cache_data(ttl=300)
def get_daily_data(ticker):
    """Fetches daily data with threads enabled."""
    try:
        data = yf.download(ticker, period="10y", interval="1d", progress=False, threads=True)
        return flatten_dataframe(data)
    except Exception: return pd.DataFrame()

@st.cache_data(ttl=300)
def get_intraday_data(ticker):
    """Fetches intraday data with threads enabled."""
    try:
        data = yf.download(ticker, period="5d", interval="15m", progress=False, threads=True)
        return flatten_dataframe(data)
    except Exception: return pd.DataFrame()

# --- COINGECKO API ENGINE ---
@st.cache_data(ttl=300) 
def get_coingecko_stats(cg_id, api_key):
    if not cg_id or not api_key: return None
    st.session_state['coingecko_calls'] += 1
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
    except Exception: return None

# --- LLM ENGINE (SAFE MODE) ---
def get_technical_narrative(ticker, price, daily_pct, regime, ml_signal, gex_data, cot_data, levels, api_key):
    if not api_key: return "AI Analyst unavailable (No Key)."
    st.session_state['gemini_calls'] += 1
    gex_text = f"Net Gamma: ${gex_data['gex'].sum()/1_000_000:.1f}M" if gex_data is not None else "N/A"
    lvl_text = f"Pivot: {levels['Pivot']:.2f}, R1: {levels['R1']:.2f}, S1: {levels['S1']:.2f}" if levels else "N/A"
    prompt = f"""
    You are a Senior Portfolio Manager. Analyze technical data for {ticker} and write a 3-bullet executive summary.
    DATA: Price: {price:,.2f} ({daily_pct:.2f}%), Regime: {regime['regime'] if regime else 'Unknown'}, 
    ML: {ml_signal}, GEX: {gex_text}, COT: {cot_data['sentiment'] if cot_data else 'N/A'}, Levels: {lvl_text}
    TASK: 1. Synthesize Regime (Trend), ML (Prob), and GEX (Vol). 2. Identify key trigger level. 3. Final Execution bias.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "ResourceExhausted" in error_msg: return "⚠️ API LIMIT REACHED: The AI Analyst is taking a nap. (Quota Exceeded)"
        return f"AI Analyst unavailable: {error_msg}"

def generate_deep_dive_thesis(ticker, price, change, regime, ml_signal, gex_data, cot_data, levels, news_summary, api_key):
    if not api_key: return "API Key Missing."
    st.session_state['gemini_calls'] += 1
    gex_text = f"Net Gamma: ${gex_data['gex'].sum()/1_000_000:.1f}M" if gex_data is not None else "N/A"
    prompt = f"""
    Write a detailed Investment Thesis for {ticker}.
    DATA: Price: {price:,.2f} ({change:.2f}%), Regime: {regime['regime'] if regime else 'Unknown'}, ML: {ml_signal}, GEX: {gex_text}, COT: {cot_data['sentiment'] if cot_data else 'N/A'}
    NEWS: {news_summary}
    OUTPUT FORMAT (Markdown): ### 1. THE CORE ARGUMENT ### 2. SUPPORTING DATA ### 3. THE BEAR CASE ### 4. INVALIDATION LEVEL
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "ResourceExhausted" in error_msg: return "⚠️ CRITICAL: API LIMIT REACHED. Thesis generation paused."
        return f"Thesis Generation Failed: {error_msg}"

# --- ANALYTICS ENGINES (WITH THREADED DOWNLOADS) ---
def calculate_hurst(series, lags=range(2, 20)):
    try:
        tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    except: return 0.5

@st.cache_data(ttl=3600)
def get_market_regime(ticker):
    try:
        # Optimization: Threads=True
        df = yf.download(ticker, period="5y", interval="1d", progress=False, threads=True)
        df = flatten_dataframe(df)
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

@st.cache_data(ttl=3600)
def get_ml_prediction(ticker):
    try:
        # Optimization: Threads=True
        df = yf.download(ticker, period="2y", interval="1d", progress=False, threads=True)
        df = flatten_dataframe(df) 
        if df.empty: return None, 0.5
        
        data = df.copy()
        data['Returns'] = data['Close'].pct_change()
        data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
        data['Vol_5d'] = data['Returns'].rolling(5).std()
        data['Mom_5d'] = data['Close'].pct_change(5)
        data = data.dropna()
        if len(data) < 50: return None, 0.5
        
        model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
        model.fit(data[['Vol_5d', 'Mom_5d']], data['Target'])
        prob_up = model.predict_proba(data[['Vol_5d', 'Mom_5d']].iloc[[-1]])[0][1]
        return model, prob_up
    except: return None, 0.5

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
            # Not a download call, but needs protection
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
        
        r = 0.045
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
            c_gamma = calculate_black_scholes_gamma(spot_price, K, T, r, c_iv)
            p_gamma = calculate_black_scholes_gamma(spot_price, K, T, r, p_iv)
            gex_data.append({"strike": K, "gex": (c_gamma * c_oi - p_gamma * p_oi) * spot_price * 100})
            
        df = pd.DataFrame(gex_data, columns=['strike', 'gex'])
        return (None, None, None) if df.empty else (df, target_exp, spot_price)
    except Exception: return None, None, None

def calculate_volume_profile(df, bins=50):
    if df is None or df.empty: return None, None
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
    if daily_data is None or daily_data.empty: return None
    stats = {}
    try:
        df = daily_data.copy()
        df['Week_Num'] = df.index.to_period('W')
        high_days = df.groupby('Week_Num')['High'].idxmax().apply(lambda x: df.loc[x].name.day_name())
        stats['day_high'] = high_days.value_counts().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], fill_value=0) / len(high_days) * 100
        df['Day'] = df.index.day
        df['Month_Week'] = np.ceil(df['Day'] / 7).astype(int)
        stats['week_returns'] = df.groupby('Month_Week')['Close'].pct_change().mean() * 100
        
        # Optimization: Threads=True for hourly fetch
        intra = yf.download(ticker_name, period="60d", interval="1h", progress=False, threads=True)
        intra = flatten_dataframe(intra)
        if not intra.empty:
            if intra.index.tz is None: intra.index = intra.index.tz_localize('UTC')
            intra.index = intra.index.tz_convert('America/New_York')
            intra['Hour'] = intra.index.hour
            intra['Return'] = intra['Close'].pct_change()
            stats['hourly_perf'] = intra[intra['Hour'].isin([2,3,4,5,6,8,9,10,11,14,15,16,17,18,20,21,22,23])].groupby('Hour')['Return'].mean() * 100
        else: stats['hourly_perf'] = None
        return stats
    except: return None

@st.cache_data(ttl=3600)
def generate_monte_carlo(stock_data, days=126, simulations=1000):
    if stock_data is None or stock_data.empty or len(stock_data) < 2: return None, None
    try:
        close = stock_data['Close']
        log_returns = np.log(1 + close.pct_change())
        drift = log_returns.mean() - (0.5 * log_returns.var())
        stdev = log_returns.std()
        price_paths = np.zeros((days + 1, simulations))
        price_paths[0] = close.iloc[-1]
        daily_returns = np.exp(drift + stdev * np.random.normal(0, 1, (days, simulations)))
        for t in range(1, days + 1): price_paths[t] = price_paths[t - 1] * daily_returns[t - 1]
        return pd.date_range(start=close.index[-1], periods=days + 1, freq='B'), price_paths
    except Exception: return None, None

def parse_eco_value(val_str):
    if not isinstance(val_str, str) or val_str == '': return None
    clean = val_str.replace('%', '').replace(',', '')
    multiplier = 1.0
    if 'K' in clean.upper(): multiplier = 1000.0
    elif 'M' in clean.upper(): multiplier = 1000000.0
    elif 'B' in clean.upper(): multiplier = 1000000000.0
    try: return float(clean.upper().replace('K','').replace('M','').replace('B','')) * multiplier
    except: return None

def analyze_eco_context(actual_str, forecast_str, previous_str):
    val_actual = parse_eco_value(actual_str)
    val_forecast = parse_eco_value(forecast_str)
    val_prev = parse_eco_value(previous_str)
    context_str = ""
    bias = "Neutral"
    if actual_str and actual_str != "":
        if val_actual is not None and val_forecast is not None:
            context_str = f"Actual ({actual_str}) vs Forecast ({forecast_str})"
            delta = val_actual - val_forecast
            pct_dev = abs(delta / val_forecast) if val_forecast != 0 else 1.0 if delta != 0 else 0
            if pct_dev < 0.02: bias = "Mean Reverting"
            elif delta > 0: bias = "Bullish"
            else: bias = "Bearish"
        else: context_str = f"Actual: {actual_str}"
    else:
        if val_forecast is not None and val_prev is not None:
            context_str = f"Forecast ({forecast_str}) vs Prev ({previous_str})"
            delta = val_forecast - val_prev
            bias = "Bullish Exp." if delta > 0 else "Bearish Exp."
        else: context_str = "Waiting for Data..."
    return context_str, bias

@st.cache_data(ttl=14400)
def get_financial_news_general(api_key, query="Finance"):
    if not api_key: return []
    st.session_state['news_calls'] += 1
    try:
        newsapi = NewsApiClient(api_key=api_key)
        all_articles = newsapi.get_everything(q=query, language='en', sort_by='publishedAt')
        return [{"title": art['title'], "source": art['source']['name'], "url": art['url'], "time": art['publishedAt']} for art in all_articles['articles'][:6]] if all_articles['status'] == 'ok' else []
    except: return []

@st.cache_data(ttl=14400)
def get_forex_factory_news(api_key, news_type='breaking'):
    if not api_key: return []
    st.session_state['rapid_calls'] += 1
    try:
        url = f"https://forex-factory-scraper1.p.rapidapi.com/{'latest_breaking_news'}"
        headers = {"x-rapidapi-key": api_key, "x-rapidapi-host": "forex-factory-scraper1.p.rapidapi.com"}
        data = requests.get(url, headers=headers).json()
        return [{"title": i.get('title', 'No Title'), "url": i.get('link', i.get('url', '#')), "source": "ForexFactory", "time": i.get('date', i.get('time', 'Recent'))} for i in data[:6]] if isinstance(data, list) else []
    except: return []

@st.cache_data(ttl=21600)
def get_economic_calendar(api_key):
    if not api_key: return None
    st.session_state['rapid_calls'] += 1
    try:
        url = "https://forex-factory-scraper1.p.rapidapi.com/get_real_time_calendar_details"
        now = datetime.now()
        params = {"calendar": "Forex", "year": str(now.year), "month": str(now.month), "day": str(now.day), "currency": "ALL", "event_name": "ALL", "timezone": "GMT-04:00 Eastern Time (US & Canada)", "time_format": "12h"}
        headers = {"x-rapidapi-host": "forex-factory-scraper1.p.rapidapi.com", "x-rapidapi-key": api_key}
        data = requests.get(url, headers=headers, params=params).json()
        events = data if isinstance(data, list) else data.get('data', [])
        return [e for e in events if e.get('currency') == 'USD' and e.get('impact') in ['High', 'Medium']]
    except: return []

@st.cache_data(ttl=300)
def run_strategy_backtest(ticker):
    try:
        # Optimization: Threads=True
        df = yf.download(ticker, period="2y", interval="1d", progress=False, threads=True)
        df = flatten_dataframe(df)
        if df.empty: return None
        df['Returns'] = df['Close'].pct_change()
        df['Range'] = df['High'] - df['Low']
        df['TR'] = pd.concat([df['Range'], (df['High'] - df['Close'].shift(1)).abs(), (df['Low'] - df['Close'].shift(1)).abs()], axis=1).max(axis=1)
        df['Log_TR'] = np.log(df['TR'] / df['Close'])
        df['Vol_Forecast'] = df['Log_TR'].ewm(span=10).mean()
        df['Vol_Baseline'] = df['Log_TR'].rolling(20).mean()
        df['Signal'] = np.where((df['Vol_Forecast'] > df['Vol_Baseline']) & (df['Close'] > df['Close'].rolling(50).mean()), 1, 0)
        df['Strategy_Returns'] = df['Signal'].shift(1) * df['Returns']
        df['Cum_BnH'] = (1 + df['Returns']).cumprod()
        df['Cum_Strat'] = (1 + df['Strategy_Returns']).cumprod()
        sharpe = (df['Strategy_Returns'].mean() / df['Strategy_Returns'].std()) * np.sqrt(252) if df['Strategy_Returns'].std() != 0 else 0
        return {"df": df, "signal": "LONG" if df['Signal'].iloc[-1] == 1 else "CASH/NEUTRAL", "return": df['Cum_Strat'].iloc[-1] - 1, "sharpe": sharpe, "equity_curve": df['Cum_Strat']}
    except: return None

def calculate_vwap_bands(df):
    if df is None or df.empty: return df
    df = df.copy()
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VP'] = df['TP'] * df['Volume']
    df['Date'] = df.index.date
    df['Cum_VP'] = df.groupby('Date')['VP'].cumsum()
    df['Cum_Vol'] = df.groupby('Date')['Volume'].cumsum()
    df['VWAP'] = df['Cum_VP'] / df['Cum_Vol']
    df['Std_Dev'] = np.sqrt(df.groupby('Date')['Volume'].transform(lambda x: x * (df['TP'] - df['VWAP'])**2).groupby(df['Date']).cumsum() / df['Cum_Vol'])
    df['Upper_Band_1'] = df['VWAP'] + df['Std_Dev']
    df['Lower_Band_1'] = df['VWAP'] - df['Std_Dev']
    return df

@st.cache_data(ttl=300)
def get_relative_strength(asset_ticker, benchmark_ticker="SPY"):
    try:
        # Optimization: Threads=True
        asset = yf.download(asset_ticker, period="5d", interval="15m", progress=False, threads=True)
        bench = yf.download(benchmark_ticker, period="5d", interval="15m", progress=False, threads=True)
        asset, bench = flatten_dataframe(asset), flatten_dataframe(bench)
        if asset.empty or bench.empty: return pd.DataFrame()
        df = pd.DataFrame(index=asset.index)
        df['Asset_Close'] = asset['Close']
        df['Bench_Close'] = bench['Close']
        df = df.dropna()
        session_data = df[df.index.date == df.index[-1].date()].copy()
        if session_data.empty: return pd.DataFrame()
        session_data['RS_Score'] = ((session_data['Asset_Close'] / session_data['Asset_Close'].iloc[0]) - 1) - ((session_data['Bench_Close'] / session_data['Bench_Close'].iloc[0]) - 1)
        return session_data
    except: return pd.DataFrame()

def get_key_levels(daily_df):
    if daily_df is None or daily_df.empty: return {}
    try:
        last = daily_df.iloc[-2]
        pivot = (last['High'] + last['Low'] + last['Close']) / 3
        return {"PDH": last['High'], "PDL": last['Low'], "PDC": last['Close'], "Pivot": pivot, "R1": (2 * pivot) - last['Low'], "S1": (2 * pivot) - last['High']}
    except: return {}

@st.cache_data(ttl=3600)
def get_correlations(base_ticker):
    try:
        tickers = {"Base": base_ticker, "VIX": "^VIX", "10Y Yield": "^TNX", "Dollar": "DX-Y.NYB", "Gold": "GC=F"}
        # Optimization: Threads=True (Critical for multiple tickers)
        data = yf.download(list(tickers.values()), period="6mo", progress=False, threads=True)['Close']
        data = flatten_dataframe(data)
        data.rename(columns={v: k for k, v in tickers.items()}, inplace=True)
        return data.pct_change().rolling(20).corr(data[base_ticker].pct_change()).iloc[-1].drop(base_ticker)
    except: return pd.Series()

@st.cache_data(ttl=86400)
def get_cot_data(asset_name):
    if asset_name not in COT_MAPPING: return None
    try:
        df = pd.DataFrame(cot.cot_year(datetime.now().year, cot_report_type='legacy_fut'))
        asset_df = df[df['Market_and_Exchange_Names'] == COT_MAPPING[asset_name]["name"]].copy()
        if asset_df.empty:
            df = pd.DataFrame(cot.cot_year(datetime.now().year - 1, cot_report_type='legacy_fut'))
            asset_df = df[df['Market_and_Exchange_Names'] == COT_MAPPING[asset_name]["name"]].copy()
        if asset_df.empty: return None
        latest = asset_df.sort_values('As_of_Date_In_Form_YYMMDD').iloc[-1]
        comm_net = latest['Comm_Positions_Long_All'] - latest['Comm_Positions_Short_All']
        return {"date": pd.to_datetime(latest['As_of_Date_In_Form_YYMMDD'], format='%y%m%d').strftime('%Y-%m-%d'), "comm_net": comm_net, "spec_net": latest['NonComm_Positions_Long_All'] - latest['NonComm_Positions_Short_All'], "sentiment": "BULLISH" if comm_net > 0 else "BEARISH"}
    except: return None

def terminal_chart_layout(fig, title="", height=350):
    fig.update_layout(title=dict(text=title, font=dict(color="#ff9900", family="Arial")), template="plotly_dark", paper_bgcolor="#000000", plot_bgcolor="#000000", height=height, margin=dict(l=40, r=40, t=40, b=40), xaxis=dict(showgrid=True, gridcolor="#222"), yaxis=dict(showgrid=True, gridcolor="#222"), font=dict(family="Courier New", color="#e0e0e0"))
    return fig

# --- SIDEBAR & MAIN ---
with st.sidebar:
    st.markdown("<h3 style='color: #ff9900;'>COMMAND LINE</h3>", unsafe_allow_html=True)
    selected_asset = st.selectbox("SEC / Ticker", list(ASSETS.keys()))
    asset_info = ASSETS[selected_asset]
    
    st.markdown("---")
    with st.expander("📡 API QUOTA MONITOR", expanded=True):
        st.write(f"**NewsAPI** ({st.session_state['news_calls']}/100)")
        st.progress(min(st.session_state['news_calls']/100, 1.0))
        st.write(f"**Gemini** ({st.session_state['gemini_calls']}/20)")
        st.progress(min(st.session_state['gemini_calls']/20, 1.0))
        st.write(f"**RapidAPI** ({st.session_state['rapid_calls']}/10)")
        st.progress(min(st.session_state['rapid_calls']/10, 1.0)) if st.session_state['rapid_calls'] <= 10 else st.warning("OVER LIMIT")
        st.write(f"**CoinGecko** ({st.session_state['coingecko_calls']}/10k)")
        st.progress(min(st.session_state['coingecko_calls']/10000, 1.0))

    rapid_key = get_api_key("rapidapi_key")
    news_key = get_api_key("news_api_key")
    gemini_key = get_api_key("gemini_api_key")
    cg_key = get_api_key("coingecko_key") 
    
    if st.button(">> REFRESH DATA"): st.cache_data.clear(); st.rerun()

st.markdown(f"<h1 style='border-bottom: 2px solid #ff9900;'>{selected_asset} <span style='font-size:0.5em; color:white;'>TERMINAL PRO V5.6</span></h1>", unsafe_allow_html=True)

# LOAD DATA
daily_data = get_daily_data(asset_info['ticker'])
intraday_data = get_intraday_data(asset_info['ticker'])
eco_events = get_economic_calendar(rapid_key)
news_gen = get_financial_news_general(news_key, query=asset_info.get('news_query', 'Finance'))
news_ff = get_forex_factory_news(rapid_key, 'breaking')
combined_news = news_gen[:5] + news_ff[:5]

# CALCULATE
_, ml_prob = get_ml_prediction(asset_info['ticker'])
gex_df, gex_date, gex_spot = get_gex_profile(asset_info['opt_ticker'])
vol_profile, poc_price = calculate_volume_profile(intraday_data)
hurst = calculate_hurst(daily_data['Close'].values) if not daily_data.empty else 0.5
regime_data = get_market_regime(asset_info['ticker'])
cot_data = get_cot_data(selected_asset)

# DASHBOARD
if not daily_data.empty:
    curr = daily_data['Close'].iloc[-1]
    pct = ((curr - daily_data['Close'].iloc[-2]) / daily_data['Close'].iloc[-2]) * 100
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LAST PX", f"{curr:,.2f}", f"{pct:.2f}%")
    
    ml_bias = "BULLISH" if ml_prob > 0.55 else "BEARISH" if ml_prob < 0.45 else "NEUTRAL"
    c2.markdown(f"<div class='terminal-box' style='text-align:center'><div style='color:#ff9900'>AI PRED</div><span class='{ml_bias.lower() if ml_bias != 'NEUTRAL' else 'neutral'}'>{ml_bias}</span><div style='color:#aaa'>CONF: {abs(ml_prob-0.5)*200:.0f}%</div></div>", unsafe_allow_html=True)
    
    if regime_data:
        c3.markdown(f"<div class='terminal-box'><div style='color:#ff9900'>REGIME</div><div class='{regime_data['color']}'>{regime_data['regime']}</div><div>Hurst: {hurst:.2f}</div></div>", unsafe_allow_html=True)
    else: c3.info("Calculating...")
    
    c4.metric("RANGE", f"{daily_data['High'].max():,.2f} / {daily_data['Low'].min():,.2f}")
    
    fig = go.Figure(go.Candlestick(x=daily_data.index, open=daily_data['Open'], high=daily_data['High'], low=daily_data['Low'], close=daily_data['Close'], name="Price"))
    if poc_price: fig.add_hline(y=poc_price, line_dash="dash", line_color="yellow", annotation_text="POC")
    st.plotly_chart(terminal_chart_layout(fig, height=400), use_container_width=True)

# COINGECKO
if asset_info.get('cg_id') and cg_key:
    cg_data = get_coingecko_stats(asset_info['cg_id'], cg_key)
    if cg_data:
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rank", f"#{cg_data['rank']}")
        c2.metric("ATH Drawdown", f"{cg_data['ath_change']:.2f}%")
        c3.metric("Sentiment", f"{cg_data['sentiment']}% Bullish")
        c4.write(cg_data['hashing'])
    else: st.warning("CoinGecko Limit Hit")

# INTRADAY
st.markdown("---")
st.markdown("### 🔭 INTRADAY")
rs_data = get_relative_strength(asset_info['ticker'])
key_levels = get_key_levels(daily_data)
c1, c2 = st.columns([2, 1])
with c1:
    if not rs_data.empty:
        curr_rs = rs_data['RS_Score'].iloc[-1]
        st.markdown(f"**ALPHA (vs SPY)**: <span style='color:{'#00ff00' if curr_rs > 0 else '#ff3333'}'>{'OUTPERFORM' if curr_rs > 0 else 'UNDERPERFORM'}</span>", unsafe_allow_html=True)
        fig_rs = go.Figure(go.Scatter(x=rs_data.index, y=rs_data['RS_Score'], mode='lines', fill='tozeroy', line=dict(color='#00ff00' if curr_rs > 0 else '#ff3333')))
        st.plotly_chart(terminal_chart_layout(fig_rs, height=250), use_container_width=True)
with c2:
    if key_levels:
        cur_p = intraday_data['Close'].iloc[-1] if not intraday_data.empty else 0
        for k, v in key_levels.items():
            color = "#ff3333" if v > cur_p else "#00ff00"
            st.markdown(f"<div style='display:flex;justify-content:space-between;border-bottom:1px solid #333'><span>{k}</span><span style='color:{color}'>{v:,.2f}</span></div>", unsafe_allow_html=True)

# NEWS & ECO
st.markdown("---")
c1, c2 = st.columns([2, 1])
with c1:
    st.markdown("### 📅 ECO CALENDAR")
    if eco_events:
        df_eco = pd.DataFrame([{"Time": e.get('time'), "Event": e.get('event_name'), "Actual": e.get('actual'), "Forecast": e.get('forecast')} for e in eco_events])
        st.dataframe(df_eco, use_container_width=True, hide_index=True)
    else: st.info("No High Impact Events")
with c2:
    st.markdown("### 📰 NEWS")
    for n in combined_news:
        st.markdown(f"<div style='border-bottom:1px solid #333;font-size:0.8em'><a href='{n['url']}' class='news-link'>{n['title']}</a><br><span style='color:gray'>{n['source']}</span></div>", unsafe_allow_html=True)

# RISK & STRATEGY
st.markdown("---")
strat = run_strategy_backtest(asset_info['ticker'])
if strat:
    c1, c2, c3 = st.columns([1, 2, 2])
    c1.markdown(f"### SIGNAL: <span style='color:{'#00ff00' if 'LONG' in strat['signal'] else 'yellow'}'>{strat['signal']}</span>", unsafe_allow_html=True)
    c1.metric("Sharpe", f"{strat['sharpe']:.2f}")
    
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(y=strat['equity_curve'], name="Strategy", line=dict(color='#00e6ff')))
    c2.plotly_chart(terminal_chart_layout(fig_eq, title="Equity Curve", height=300), use_container_width=True)
    
    if vol_profile is not None:
        fig_vp = go.Figure(go.Bar(y=vol_profile['PriceLevel'], x=vol_profile['Volume'], orientation='h', marker_color='#ff9900', opacity=0.4))
        fig_vp.add_hline(y=poc_price, line_dash="dash", line_color="yellow")
        c3.plotly_chart(terminal_chart_layout(fig_vp, title="Volume Profile", height=300), use_container_width=True)

# VWAP
vwap_df = calculate_vwap_bands(intraday_data)
if not vwap_df.empty:
    fig_v = go.Figure(go.Candlestick(x=vwap_df.index, open=vwap_df['Open'], high=vwap_df['High'], low=vwap_df['Low'], close=vwap_df['Close'], name="Price"))
    fig_v.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df['VWAP'], name="VWAP", line=dict(color='#ff9900')))
    fig_v.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df['Upper_Band_1'], name="+1 STD", line=dict(color='gray'), opacity=0.3))
    fig_v.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df['Lower_Band_1'], name="-1 STD", line=dict(color='gray'), opacity=0.3))
    st.plotly_chart(terminal_chart_layout(fig_v, height=500), use_container_width=True)

# GEX
if gex_df is not None:
    st.markdown("---")
    st.markdown(f"### 🏦 GEX PROFILE: ${gex_df['gex'].sum()/1_000_000:.1f}M")
    fig_g = go.Figure(go.Bar(x=gex_df['strike'], y=gex_df['gex'], marker_color=['#00ff00' if x>0 else '#ff3333' for x in gex_df['gex']]))
    fig_g.add_vline(x=gex_spot, line_dash="dot", line_color="white")
    st.plotly_chart(terminal_chart_layout(fig_g, height=350), use_container_width=True)

# MONTE CARLO
dates, paths = generate_monte_carlo(daily_data)
if dates is not None:
    st.markdown("---")
    st.markdown("### 🎲 MONTE CARLO")
    fig_mc = go.Figure()
    fig_mc.add_trace(go.Scatter(x=daily_data.index[-90:], y=daily_data['Close'][-90:], name='History', line=dict(color='white')))
    fig_mc.add_trace(go.Scatter(x=dates, y=np.mean(paths, axis=1), name='Forecast', line=dict(color='#ff9900', dash='dash')))
    st.plotly_chart(terminal_chart_layout(fig_mc, height=400), use_container_width=True)

# COT
if cot_data:
    st.markdown("---")
    c1, c2 = st.columns(2)
    c1.metric("Commercial Net", f"{cot_data['comm_net']:,.0f}")
    c2.metric("Speculator Net", f"{cot_data['spec_net']:,.0f}")

# AI ANALYST
if gemini_key:
    st.markdown("---")
    st.markdown("### 🧠 AI ANALYST")
    c1, c2 = st.columns([1, 4])
    with c1:
        if st.button("📝 BRIEF"):
            with st.spinner("Thinking..."):
                st.session_state['narrative_cache'] = get_technical_narrative(selected_asset, curr, pct, regime_data, ml_bias, gex_df, cot_data, key_levels, gemini_key)
                st.rerun()
        if st.button("🔎 THESIS"):
            with st.spinner("Deep Dive..."):
                news_txt = "\n".join([f"- {n['title']}" for n in combined_news])
                st.session_state['thesis_cache'] = generate_deep_dive_thesis(selected_asset, curr, pct, regime_data, ml_bias, gex_df, cot_data, key_levels, news_txt, gemini_key)
                st.rerun()
    
    if st.session_state['narrative_cache']: st.info(st.session_state['narrative_cache'])
    if st.session_state['thesis_cache']: st.markdown(f"<div class='terminal-box'>{st.session_state['thesis_cache']}</div>", unsafe_allow_html=True)
else: st.info("Enter Gemini Key")
