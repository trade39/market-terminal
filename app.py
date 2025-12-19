import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from newsapi import NewsApiClient
from datetime import datetime, timedelta
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture

# --- APP CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Bloomberg Terminal Pro V3.2", page_icon="💹")

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

    /* Volatility Badges */
    .vol-go { border: 1px solid #00ff00; color: #00ff00; padding: 2px 8px; font-weight: bold; letter-spacing: 1px; }
    .vol-stop { border: 1px solid #ff3333; color: #ff3333; padding: 2px 8px; font-weight: bold; letter-spacing: 1px; }

    /* UI Elements */
    .stSelectbox > div > div { border-radius: 0px; background-color: #111; color: white; border: 1px solid #444; }
    button { border-radius: 0px !important; border: 1px solid #ff9900 !important; color: #ff9900 !important; background: black !important; }
    hr { margin: 1em 0; border: 0; border-top: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTS & MAPPINGS (UPDATED FOR NEWS) ---
ASSETS = {
    "Gold (Comex)": {"ticker": "GC=F", "opt_ticker": "GLD", "news_query": "Gold Price"},
    "S&P 500": {"ticker": "^GSPC", "opt_ticker": "SPY", "news_query": "S&P 500"},
    "NASDAQ": {"ticker": "^IXIC", "opt_ticker": "QQQ", "news_query": "Nasdaq"},
    "EUR/USD": {"ticker": "EURUSD=X", "opt_ticker": None, "news_query": "EURUSD"}, 
    "NVIDIA": {"ticker": "NVDA", "opt_ticker": "NVDA", "news_query": "NVDA"},
    "Bitcoin": {"ticker": "BTC-USD", "opt_ticker": None, "news_query": "Bitcoin"}
}

# --- HELPER FUNCTIONS ---

def get_api_key(key_name):
    if "api_keys" in st.secrets and key_name in st.secrets["api_keys"]:
        return st.secrets["api_keys"][key_name]
    if key_name in st.secrets:
        return st.secrets[key_name]
    return None

def flatten_dataframe(df):
    if df.empty: return df
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    return df

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
        df = yf.download(ticker, period="5y", interval="1d", progress=False)
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
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        df = flatten_dataframe(df) 
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

# --- 3. UPDATED GAMMA EXPOSURE ENGINE ---

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
        hist = tk.history(period="1d")
        if hist.empty: return None, None, None
        spot_price = hist['Close'].iloc[-1]
        exps = tk.options
        if not exps or len(exps) < 2: return None, None, None
        target_exp = exps[1]
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
            if K < spot_price * 0.7 or K > spot_price * 1.3: continue
            
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
    except: return None, None, None

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
def get_seasonality_stats(daily_data):
    try:
        df = daily_data.copy()
        df['Week_Num'] = df.index.to_period('W')
        high_days = df.groupby('Week_Num')['High'].idxmax().apply(lambda x: df.loc[x].name.day_name())
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        return {'day_high': high_days.value_counts().reindex(days_order, fill_value=0) / len(high_days) * 100}
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

# --- UPDATED NEWS FUNCTION ---
@st.cache_data(ttl=3600)
def get_financial_news(api_key, query="Finance"):
    if not api_key: return []
    try:
        newsapi = NewsApiClient(api_key=api_key)
        # UPDATED: Uses get_everything with specific query and sort by time
        all_articles = newsapi.get_everything(q=query, language='en', sort_by='publishedAt')
        articles = []
        if all_articles['status'] == 'ok':
            for art in all_articles['articles'][:5]:
                articles.append({"title": art['title'], "source": art['source']['name'], "url": art['url'], "time": art['publishedAt']})
        return articles
    except: return []

@st.cache_data(ttl=3600)
def get_economic_calendar(api_key):
    if not api_key: return None
    url = "https://forex-factory-scraper1.p.rapidapi.com/get_calendar_details"
    now = datetime.now()
    querystring = {"year": str(now.year), "month": str(now.month), "day": str(now.day)}
    headers = {"x-rapidapi-host": "forex-factory-scraper1.p.rapidapi.com", "x-rapidapi-key": api_key}
    
    try:
        response = requests.get(url, headers=headers, params=querystring)
        data = response.json()
        raw_events = data if isinstance(data, list) else data.get('data', [])
        
        filtered_events = []
        for e in raw_events:
            if e.get('currency') == 'USD' and e.get('impact') == 'High':
                filtered_events.append(e)
        return filtered_events
    except: return []

# --- 7. NEW INSTITUTIONAL FEATURES ---

# A. BACKTEST ENGINE (Replaces Vol Permission)
@st.cache_data(ttl=300)
def run_strategy_backtest(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        df = flatten_dataframe(df)
        if df.empty: return None

        # 1. Calculate Indicators
        df['Returns'] = df['Close'].pct_change()
        df['Range'] = df['High'] - df['Low']
        df['TR'] = pd.concat([df['Range'], (df['High'] - df['Close'].shift(1)).abs(), (df['Low'] - df['Close'].shift(1)).abs()], axis=1).max(axis=1)
        
        # Volatility Filter Logic
        df['Log_TR'] = np.log(df['TR'] / df['Close'])
        df['Vol_Forecast'] = df['Log_TR'].ewm(span=10).mean()
        df['Vol_Baseline'] = df['Log_TR'].rolling(20).mean()
        
        # 2. Vectorized Backtest
        # Signal: Long only if Vol Forecast > Baseline (expansion) AND Trend is up (SMA 50)
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['Signal'] = np.where((df['Vol_Forecast'] > df['Vol_Baseline']) & (df['Close'] > df['SMA_50']), 1, 0)
        
        # Shift signal to prevent lookahead bias
        df['Strategy_Returns'] = df['Signal'].shift(1) * df['Returns']
        
        # 3. Performance Metrics
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

# B. ANCHORED VWAP
def calculate_vwap_bands(df):
    if df.empty: return df
    df = df.copy()
    
    # Calculate typical price
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VP'] = df['TP'] * df['Volume']
    
    # Anchor VWAP to the start of the data loaded
    df['Cum_VP'] = df['VP'].cumsum()
    df['Cum_Vol'] = df['Volume'].cumsum()
    df['VWAP'] = df['Cum_VP'] / df['Cum_Vol']
    
    # Calculate Variance for Bands
    df['Sq_Dist'] = df['Volume'] * (df['TP'] - df['VWAP'])**2
    df['Cum_Sq_Dist'] = df['Sq_Dist'].cumsum()
    df['Std_Dev'] = np.sqrt(df['Cum_Sq_Dist'] / df['Cum_Vol'])
    
    df['Upper_Band_1'] = df['VWAP'] + df['Std_Dev']
    df['Lower_Band_1'] = df['VWAP'] - df['Std_Dev']
    df['Upper_Band_2'] = df['VWAP'] + (df['Std_Dev'] * 2)
    df['Lower_Band_2'] = df['VWAP'] - (df['Std_Dev'] * 2)
    
    return df

# C. MACRO CORRELATIONS
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
        
        data = yf.download(list(tickers.values()), period="6mo", progress=False)['Close']
        data = flatten_dataframe(data)
        
        rev_tickers = {v: k for k, v in tickers.items()}
        data.rename(columns=rev_tickers, inplace=True)
        
        corrs = data.pct_change().rolling(20).corr(data[base_ticker].pct_change()).iloc[-1]
        return corrs.drop(base_ticker) 
    except: return pd.Series()

# --- 8. DATA FETCHERS ---
@st.cache_data(ttl=60)
def get_daily_data(ticker):
    try:
        data = yf.download(ticker, period="10y", interval="1d", progress=False)
        return flatten_dataframe(data)
    except: return pd.DataFrame()

@st.cache_data(ttl=60)
def get_intraday_data(ticker):
    try:
        data = yf.download(ticker, period="5d", interval="15m", progress=False)
        return flatten_dataframe(data)
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
    st.markdown("**MACRO CORRELATIONS (20D)**")
    corrs = get_correlations(asset_info['ticker'])
    if not corrs.empty:
        for idx, val in corrs.items():
            c_color = "#ff3333" if val > 0.5 else "#00ff00" if val < -0.5 else "gray" 
            st.markdown(f"{idx}: <span style='color:{c_color}'>{val:.2f}</span>", unsafe_allow_html=True)
    
    st.markdown("---")
    fred_key = get_api_key("fred_api_key")
    rapid_key = get_api_key("rapidapi_key")
    news_key = get_api_key("news_api_key")
    
    st.markdown(f"""
    <div style='font-size:0.8em; color:gray; font-family:Courier New;'>
    API STATUS:<br>
    FRED: {'[OK]' if fred_key else '[FAIL]'}<br>
    RAPID: {'[OK]' if rapid_key else '[FAIL]'}<br>
    NEWS: {'[OK]' if news_key else '[FAIL]'}
    </div>
    """, unsafe_allow_html=True)
    
    if st.button(">> REFRESH DATA"): st.cache_data.clear()

# --- MAIN DASHBOARD ---
st.markdown(f"<h1 style='border-bottom: 2px solid #ff9900;'>{selected_asset} <span style='font-size:0.5em; color:white;'>TERMINAL PRO V3.2</span></h1>", unsafe_allow_html=True)

# Fetch Data
daily_data = get_daily_data(asset_info['ticker'])
intraday_data = get_intraday_data(asset_info['ticker'])
eco_events = get_economic_calendar(rapid_key)

# UPDATED: FETCH NEWS USING SPECIFIC QUERY
news_items = get_financial_news(news_key, query=asset_info.get('news_query', 'Finance'))

# Engines
_, ml_prob = get_ml_prediction(asset_info['ticker'])
gex_df, gex_date, gex_spot = get_gex_profile(asset_info['opt_ticker'])
vol_profile, poc_price = calculate_volume_profile(intraday_data)
hurst = calculate_hurst(daily_data['Close'].values) if not daily_data.empty else 0.5
regime_data = get_market_regime(asset_info['ticker'])

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

# --- 2. EVENTS & NEWS ---
st.markdown("---")
col_eco, col_news = st.columns([2, 1])

with col_eco:
    st.markdown("### 📅 ECONOMIC EVENTS (USD | HIGH)")
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
    if news_items:
        for news in news_items:
            st.markdown(f"""
            <div style="border-bottom:1px solid #333; padding-bottom:10px; margin-bottom:10px;">
                <a class='news-link' href='{news['url']}' target='_blank'>▶ {news['title']}</a><br>
                <span style='font-size:0.7em; color:gray;'>{news['source']} | {news['time'][:10]}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("<div style='color:gray;'>NO NEWS FEED AVAILABLE</div>", unsafe_allow_html=True)

# --- 3. RISK ANALYSIS & BACKTEST ---
st.markdown("---")
st.markdown("### ⚡ QUANTITATIVE RISK & EXECUTION")

# Run the Backtest
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
        # Plot Equity Curve vs Buy & Hold
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

# --- 4. VWAP EXECUTION ---
st.markdown("#### 🎯 VWAP LIQUIDITY BANDS")
vwap_df = calculate_vwap_bands(intraday_data)

if not vwap_df.empty:
    fig_vwap = go.Figure()
    # Candlestick
    fig_vwap.add_trace(go.Candlestick(x=vwap_df.index, open=vwap_df['Open'], high=vwap_df['High'], low=vwap_df['Low'], close=vwap_df['Close'], name="Price"))
    # VWAP & Bands
    fig_vwap.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df['VWAP'], name="VWAP", line=dict(color='#ff9900', width=2)))
    fig_vwap.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df['Upper_Band_1'], name="+1 STD", line=dict(color='gray', width=1), opacity=0.5))
    fig_vwap.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df['Lower_Band_1'], name="-1 STD", line=dict(color='gray', width=1), opacity=0.5))
    
    terminal_chart_layout(fig_vwap, height=400)
    st.plotly_chart(fig_vwap, use_container_width=True)


# --- 5. GEX ---
st.markdown("---")
st.markdown("### 🏦 INSTITUTIONAL GAMMA EXPOSURE (GEX)")

if gex_df is not None and gex_spot is not None:
    g1, g2 = st.columns([3, 1])
    with g1:
        # Use the Internal Spot Price for chart centering
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

# --- 6. MONTE CARLO & SEASONALITY ---
st.markdown("---")
st.markdown("### 🎲 SIMULATION & SEASONALITY")
s1, s2 = st.columns(2)

with s1:
    pred_dates, pred_paths = generate_monte_carlo(daily_data)
    fig_pred = go.Figure()
    hist_slice = daily_data['Close'].tail(90)
    fig_pred.add_trace(go.Scatter(x=hist_slice.index, y=hist_slice.values, name='History', line=dict(color='white')))
    fig_pred.add_trace(go.Scatter(x=pred_dates, y=np.mean(pred_paths, axis=1), name='Avg Path', line=dict(color='#ff9900', dash='dash')))
    terminal_chart_layout(fig_pred, title="MONTE CARLO PROJECTION")
    st.plotly_chart(fig_pred, use_container_width=True)

with s2:
    stats = get_seasonality_stats(daily_data)
    if stats:
        fig_d = go.Figure()
        fig_d.add_trace(go.Bar(x=stats['day_high'].index, y=stats['day_high'].values, marker_color='#00ff00'))
        terminal_chart_layout(fig_d, title="PROBABILITY OF WEEKLY HIGH")
        st.plotly_chart(fig_d, use_container_width=True)

# --- 7. CONCLUSION ---
st.markdown("---")
st.markdown("### 🏁 EXECUTIVE SUMMARY")

bias_score = 0
reasons = []

if ml_prob > 0.55: bias_score += 1; reasons.append(f"AI: Model predicts UP ({ml_prob:.0%})")
elif ml_prob < 0.45: bias_score -= 1; reasons.append(f"AI: Model predicts DOWN ({ml_prob:.0%})")

if regime_data:
    if "LOW VOL" in regime_data['regime']: bias_score += 1; reasons.append(f"REGIME: {regime_data['regime']}")
    elif "HIGH VOL" in regime_data['regime']: bias_score -= 1; reasons.append(f"REGIME: {regime_data['regime']}")

if gex_df is not None:
    if gex_df['gex'].sum() > 0: reasons.append("GEX: Dealers Long Gamma (Expect Range).")
    else: reasons.append("GEX: Dealers Short Gamma (Expect Breakouts).")

final_text = "BULLISH BIAS" if bias_score > 0 else "BEARISH BIAS" if bias_score < 0 else "NEUTRAL"
final_color = "bullish" if bias_score > 0 else "bearish" if bias_score < 0 else "neutral"

st.markdown(f"""
<div class='terminal-box'>
    <h2 style="text-align:center; margin-top:0; color: #ff9900;">{selected_asset} FINAL OUTLOOK</h2>
    <div style='text-align:center; margin-bottom:15px;'><span class='{final_color}' style='font-size:1.5em;'>{final_text}</span></div>
    <hr>
    <ul style='font-family:Courier New; color:#e0e0e0;'>
        {''.join([f'<li>{r}</li>' for r in reasons])}
    </ul>
</div>
""", unsafe_allow_html=True)
