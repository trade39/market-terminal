import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
from fredapi import Fred
from newsapi import NewsApiClient
from datetime import datetime, timedelta
from scipy.interpolate import UnivariateSpline
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture

# --- APP CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Bloomberg Terminal Pro V4", page_icon="💹")

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
    
    /* Metric Styling (The "Terminal" Data Look) */
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
    
    /* Signal badges - Neon Style */
    .bullish { color: #000000; background-color: #00ff00; padding: 2px 6px; font-weight: bold; }
    .bearish { color: #000000; background-color: #ff3333; padding: 2px 6px; font-weight: bold; }
    .neutral { color: #000000; background-color: #cccccc; padding: 2px 6px; font-weight: bold; }
    
    /* Volatility Badges */
    .vol-go { border: 1px solid #00ff00; color: #00ff00; padding: 2px 8px; font-weight: bold; letter-spacing: 1px; }
    .vol-stop { border: 1px solid #ff3333; color: #ff3333; padding: 2px 8px; font-weight: bold; letter-spacing: 1px; }

    /* Remove rounded corners from buttons/inputs for industrial look */
    .stSelectbox > div > div { border-radius: 0px; background-color: #111; color: white; border: 1px solid #444; }
    button { border-radius: 0px !important; border: 1px solid #ff9900 !important; color: #ff9900 !important; background: black !important; }
    
    /* Divider */
    hr { margin: 1em 0; border: 0; border-top: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTS & MAPPINGS ---
ASSETS = {
    "Gold (Comex)": {"ticker": "GC=F", "opt_ticker": "GLD", "news_query": "Gold Price"},
    "S&P 500": {"ticker": "^GSPC", "opt_ticker": "SPY", "news_query": "S&P 500"},
    "NASDAQ": {"ticker": "^IXIC", "opt_ticker": "QQQ", "news_query": "Nasdaq"},
    "EUR/USD": {"ticker": "EURUSD=X", "opt_ticker": "FXE", "news_query": "EURUSD"},
    "NVIDIA": {"ticker": "NVDA", "opt_ticker": "NVDA", "news_query": "Nvidia Stock"},
    "Bitcoin": {"ticker": "BTC-USD", "opt_ticker": "BITO", "news_query": "Bitcoin"}
}

DXY_TICKER = "DX-Y.NYB"

# --- HELPER FUNCTIONS ---

def get_api_key(key_name):
    if "api_keys" in st.secrets and key_name in st.secrets["api_keys"]:
        return st.secrets["api_keys"][key_name]
    if key_name in st.secrets:
        return st.secrets[key_name]
    return None

def flatten_dataframe(df):
    """Robust flattening of MultiIndex columns from yfinance"""
    if df.empty: return df
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    return df

# --- 1. NEW: INSTITUTIONAL QUANT ENGINE (GMM + HURST) ---

def calculate_hurst(series, lags=range(2, 20)):
    """Calculates the Hurst Exponent to determine fractal dimension."""
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
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=3600)
def get_ml_prediction(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        df = flatten_dataframe(df) 
        if df.empty: return None, 0.5
        
        data = df.copy()
        data['Returns'] = data['Close'].pct_change()
        data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
        
        data['RSI'] = calculate_rsi(data['Close'])
        data['Vol_5d'] = data['Returns'].rolling(5).std()
        data['Mom_5d'] = data['Close'].pct_change(5)
        data['DayOfWeek'] = data.index.dayofweek
        
        data = data.dropna()
        if len(data) < 50: return None, 0.5
        
        features = ['RSI', 'Vol_5d', 'Mom_5d', 'DayOfWeek']
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
def get_gex_profile(opt_ticker, spot_price):
    try:
        tk = yf.Ticker(opt_ticker)
        exps = tk.options
        if not exps or len(exps) < 2: return None, None
        target_exp = exps[1]
        
        chain = tk.option_chain(target_exp)
        calls, puts = chain.calls, chain.puts
        
        if calls.empty or puts.empty: return None, None

        r = 0.045
        exp_date = datetime.strptime(target_exp, "%Y-%m-%d")
        days_to_exp = (exp_date - datetime.now()).days
        if days_to_exp <= 0: T = 0.001 
        else: T = days_to_exp / 365.0
        
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
        
        if df.empty: return None, None
            
        return df, target_exp
    except Exception as e:
        return None, None

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

# --- 5. KELLY CRITERION ---
def calculate_kelly(prob_win, risk_reward_ratio):
    p = prob_win
    q = 1 - p
    b = risk_reward_ratio
    if b == 0: return 0
    f = p - (q / b)
    return max(0, f)

# --- ORIGINAL HELPER FUNCTIONS ---

def parse_eco_value(val_str):
    if not isinstance(val_str, str) or val_str == '': return None
    clean = val_str.replace('%', '').replace(',', '')
    multiplier = 1.0
    if 'K' in clean.upper(): multiplier = 1000.0; clean = clean.upper().replace('K', '')
    elif 'M' in clean.upper(): multiplier = 1000000.0; clean = clean.upper().replace('M', '')
    elif 'B' in clean.upper(): multiplier = 1000000000.0; clean = clean.upper().replace('B', '')
    try: return float(clean) * multiplier
    except: return None

# UPDATED EVENT ANALYSIS LOGIC
def analyze_event_impact(event_name, val_main, val_compare):
    v1 = parse_eco_value(val_main)
    v2 = parse_eco_value(val_compare)
    if v1 is None or v2 is None: return "Neutral"
    
    # Define if High Value is Bullish for USD
    usd_bullish_if_high = {
        "CPI": True, "PPI": True, "Non-Farm": True, "GDP": True, 
        "Sales": True, "Confidence": True, "Rates": True,
        "Unemployment": False, "Claims": False
    }
    
    is_direct_bullish = True 
    for key, val in usd_bullish_if_high.items():
        if key.lower() in event_name.lower():
            is_direct_bullish = val
            break
            
    delta = v1 - v2
    pct_diff = 0
    if v2 != 0: pct_diff = abs(delta / v2)
    
    # Check for mean reversion (small deviation)
    if pct_diff < 0.01: return "Mean Reverting"
    
    if delta > 0: return "USD Bullish" if is_direct_bullish else "USD Bearish"
    elif delta < 0: return "USD Bearish" if is_direct_bullish else "USD Bullish"
    
    return "Neutral"

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

@st.cache_data(ttl=3600)
def get_economic_calendar(api_key):
    if not api_key: return None
    url = "https://forex-factory-scraper1.p.rapidapi.com/get_calendar_details"
    now = datetime.now()
    querystring = {"year": str(now.year), "month": str(now.month), "day": str(now.day), "currency": "USD"}
    headers = {"x-rapidapi-host": "forex-factory-scraper1.p.rapidapi.com", "x-rapidapi-key": api_key}
    try:
        response = requests.get(url, headers=headers, params=querystring)
        data = response.json()
        if isinstance(data, list): return data
        elif 'data' in data: return data['data']
        return []
    except: return []

@st.cache_data(ttl=3600)
def get_news(api_key, query):
    if not api_key: return None
    try:
        newsapi = NewsApiClient(api_key=api_key)
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        articles = newsapi.get_everything(q=query, from_param=start_date, language='en', sort_by='relevancy', page_size=5)
        return articles['articles']
    except Exception as e:
        return []

@st.cache_data(ttl=86400)
def get_macro_regime_data(api_key):
    if not api_key: return None
    try:
        fred = Fred(api_key=api_key)
        ffr = fred.get_series('FEDFUNDS').iloc[-1]
        cpi = fred.get_series('CPIAUCSL').pct_change(12).iloc[-1] * 100
        real_rate = ffr - cpi
        bias = "BULLISH" if real_rate < 0.5 else "BEARISH"
        return {"ffr": ffr, "cpi": cpi, "real_rate": real_rate, "bias": bias}
    except: return None

@st.cache_data(ttl=300)
def calculate_volatility_permission(ticker):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        df = flatten_dataframe(df)
        if df.empty: return None
        close = df['Close']
        high, low = df['High'], df['Low']
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        tr_pct = (tr / close) * 100
        tr_pct = tr_pct.dropna()
        log_tr = np.log(tr_pct)
        forecast_tr = np.exp(log_tr.ewm(alpha=0.94).mean().iloc[-1])
        baseline = tr_pct.rolling(20).mean().iloc[-1]
        return {"forecast": forecast_tr, "baseline": baseline, "is_go": forecast_tr > baseline, "signal": "TRADE PERMITTED" if forecast_tr > baseline else "NO TRADE / CAUTION", "history": tr_pct}
    except: return None

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
st.markdown(f"<h1 style='border-bottom: 2px solid #ff9900;'>{selected_asset} <span style='font-size:0.5em; color:white;'>TERMINAL PRO V4</span></h1>", unsafe_allow_html=True)

# Fetch Data
daily_data = get_daily_data(asset_info['ticker'])
intraday_data = get_intraday_data(asset_info['ticker'])
macro_regime = get_macro_regime_data(fred_key)
vol_forecast = calculate_volatility_permission(asset_info['ticker'])
eco_events = get_economic_calendar(rapid_key)
latest_news = get_news(news_key, asset_info['news_query'])

# New Quant Engines
_, ml_prob = get_ml_prediction(asset_info['ticker'])
gex_df, gex_date = get_gex_profile(asset_info['opt_ticker'], daily_data['Close'].iloc[-1] if not daily_data.empty else 0)
vol_profile, poc_price = calculate_volume_profile(intraday_data)

# Institutional Engines
hurst = calculate_hurst(daily_data['Close'].values) if not daily_data.empty else 0.5
regime_data = get_market_regime(asset_info['ticker'])

# --- 1. OVERVIEW & INSTITUTIONAL DASHBOARD ---
if not daily_data.empty:
    close, high, low = daily_data['Close'], daily_data['High'], daily_data['Low']
    curr = close.iloc[-1]
    pct = ((curr - close.iloc[-2]) / close.iloc[-2]) * 100
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LAST PX", f"{curr:,.2f}", f"{pct:.2f}%")
    
    # ML WIDGET
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
    
    # INSTITUTIONAL REGIME WIDGET
    hurst_type = "TRENDING" if hurst > 0.55 else "MEAN REVERT" if hurst < 0.45 else "RANDOM WALK"
    h_color = "#00ff00" if hurst > 0.55 else "#ff3333" if hurst < 0.45 else "gray"
    
    if regime_data:
        c3.markdown(f"""
        <div class='terminal-box' style="padding:10px;">
            <div style="font-size:0.8em; color:#ff9900;">QUANT REGIME</div>
            <div style="font-size:1.1em; font-weight:bold;" class='{regime_data['color']}'>{regime_data['regime']}</div>
            <hr style='margin:5px 0; border-top:1px solid #333;'>
            <div style="font-size:0.7em; display:flex; justify-content:space-between;">
                <span>FRACTAL:</span>
                <span style="color:{h_color}">{hurst_type}</span>
            </div>
             <div style="font-size:0.7em; display:flex; justify-content:space-between;">
                <span>MACRO:</span>
                <span>{macro_regime['bias'] if macro_regime else 'N/A'}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        c3.info("Calculating Regime...")
    
    c4.metric("HIGH/LOW", f"{high.max():,.2f} / {low.min():,.2f}")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=daily_data.index, open=daily_data['Open'], high=high, low=low, close=close, name="Price"))
    # Add POC Line if available
    if poc_price:
        fig.add_hline(y=poc_price, line_dash="dash", line_color="yellow", annotation_text="POC", annotation_position="bottom right")
        
    terminal_chart_layout(fig, height=400)
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# --- 2. ECONOMIC CALENDAR (UPDATED) ---
st.markdown("---")
st.markdown("### 📅 HIGH IMPACT ECONOMIC EVENTS (USD)")

if eco_events:
    cal_data = []
    for event in eco_events:
        impact = event.get('impact', 'Low')
        
        # FILTER: Only show High impact events
        if impact != 'High': continue
        
        name = event.get('event_name', 'Unknown')
        actual = event.get('actual', '')
        forecast = event.get('forecast', '')
        previous = event.get('previous', '')
        
        # LOGIC: 
        # If Happened: Actual vs Forecast
        # If Upcoming: Forecast vs Previous
        
        context_msg = ""
        bias = "Neutral"
        
        if actual and actual != '':
            # Event Happened
            bias = analyze_event_impact(name, actual, forecast)
            context_msg = f"Actual ({actual}) vs Forecast ({forecast})"
        elif forecast and forecast != '':
            # Event Upcoming
            bias = analyze_event_impact(name, forecast, previous)
            context_msg = f"Forecast ({forecast}) vs Previous ({previous})"
        else:
            context_msg = "No Data"
            
        cal_data.append({"TIME": event.get('time', 'N/A'), "EVENT": name, "ANALYSIS": context_msg, "BIAS": bias})
        
    df_cal = pd.DataFrame(cal_data)
    
    # Highlight function for dataframe
    def highlight_bias(val):
        color = ''
        if 'Bullish' in val: color = '#00ff00'
        elif 'Bearish' in val: color = '#ff3333'
        elif 'Mean' in val: color = '#cccccc'
        return f'color: {color}'

    if not df_cal.empty: 
        st.dataframe(df_cal.style.map(highlight_bias, subset=['BIAS']), use_container_width=True, hide_index=True)
    else: 
        st.info("NO HIGH IMPACT USD EVENTS SCHEDULED.")
else: 
    st.info("CALENDAR DATA UNAVAILABLE.")

# --- 3. VOLATILITY, KELLY & VOLUME PROFILE ---
st.markdown("---")
st.markdown("### ⚡ QUANTITATIVE RISK ANALYSIS")

if not intraday_data.empty and vol_forecast:
    q1, q2, q3 = st.columns([1, 2, 2])
    
    with q1:
        st.markdown("**VOL FILTER**")
        badge_class = "vol-go" if vol_forecast['is_go'] else "vol-stop"
        st.markdown(f"<div style='margin:10px 0;'><span class='{badge_class}'>{vol_forecast['signal']}</span></div>", unsafe_allow_html=True)
        
        rr_ratio = 1.5 if vol_forecast['is_go'] else 1.0
        kelly_pct = calculate_kelly(ml_prob, rr_ratio) * 100
        safe_kelly = kelly_pct * 0.5 
        
        st.markdown("**KELLY SIZING**")
        k_color = "#00ff00" if safe_kelly > 0 else "#ff3333"
        st.markdown(f"<span style='color:{k_color}; font-size:1.5em; font-weight:bold;'>{safe_kelly:.1f}%</span>", unsafe_allow_html=True)
        st.caption(f"Based on AI Prob: {ml_prob:.0%}")

    with q2:
        hist_tr = vol_forecast['history'].tail(40)
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=hist_tr.index, y=hist_tr.values, name="Realized", marker_color='#333333'))
        next_day = hist_tr.index[-1] + timedelta(days=1)
        f_color = '#00ff00' if vol_forecast['is_go'] else '#ff3333'
        fig_vol.add_trace(go.Bar(x=[next_day], y=[vol_forecast['forecast']], name="Forecast", marker_color=f_color))
        terminal_chart_layout(fig_vol, title="VOLATILITY REGIME", height=250)
        st.plotly_chart(fig_vol, use_container_width=True)

    with q3:
        if vol_profile is not None:
            fig_vp = go.Figure()
            colors = ['#00e6ff' if x == poc_price else '#333' for x in vol_profile['PriceLevel']]
            fig_vp.add_trace(go.Bar(y=vol_profile['PriceLevel'], x=vol_profile['Volume'], orientation='h', marker_color='#ff9900', opacity=0.4))
            fig_vp.add_hline(y=poc_price, line_dash="dash", line_color="yellow", annotation_text="POC")
            terminal_chart_layout(fig_vp, title="INTRADAY VOLUME PROFILE", height=250)
            st.plotly_chart(fig_vp, use_container_width=True)

# --- 4. INSTITUTIONAL GEX ---
st.markdown("---")
st.markdown("### 🏦 INSTITUTIONAL GAMMA EXPOSURE (GEX)")

if gex_df is not None:
    g1, g2 = st.columns([3, 1])
    with g1:
        center_strike = curr
        gex_zoom = gex_df[(gex_df['strike'] > center_strike * 0.9) & (gex_df['strike'] < center_strike * 1.1)]
        
        fig_gex = go.Figure()
        colors = ['#00ff00' if x > 0 else '#ff3333' for x in gex_zoom['gex']]
        fig_gex.add_trace(go.Bar(x=gex_zoom['strike'], y=gex_zoom['gex'], marker_color=colors))
        fig_gex.add_vline(x=curr, line_dash="dot", line_color="white", annotation_text="SPOT")
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
            Positive Gamma = Dealers suppress vol.<br>
            Negative Gamma = Dealers amplify vol.
            </p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("NO OPTIONS DATA AVAILABLE OR GEX CALCULATION FAILED.")

# --- 5. LATEST NEWS WIRE (NEW) ---
st.markdown("---")
st.markdown("### 📰 LATEST WIRE")
if latest_news:
    for article in latest_news:
        st.markdown(f"""
        <div style='border-bottom: 1px solid #333; padding: 10px 0;'>
            <a href='{article['url']}' target='_blank' style='text-decoration: none; color: #00e6ff; font-weight: bold;'>{article['title']}</a>
            <div style='font-size: 0.8em; color: gray; margin-top: 5px;'>{article['source']['name']} - {article['publishedAt'][:10]}</div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("NO NEWS FOUND OR API KEY MISSING.")

# --- 6. CONCLUSION ---
st.markdown("---")
st.markdown("### 🏁 EXECUTIVE SUMMARY")

bias_score = 0
reasons = []

if ml_prob > 0.55: bias_score += 1; reasons.append(f"AI: Model predicts UP ({ml_prob:.0%})")
elif ml_prob < 0.45: bias_score -= 1; reasons.append(f"AI: Model predicts DOWN ({ml_prob:.0%})")

if regime_data:
    if "LOW VOL" in regime_data['regime']: bias_score += 1; reasons.append(f"REGIME: Market is in {regime_data['regime']} (Bullish).")
    elif "HIGH VOL" in regime_data['regime']: bias_score -= 1; reasons.append(f"REGIME: Market is in {regime_data['regime']} (Bearish/Crisis).")
    else: reasons.append(f"REGIME: Market is in {regime_data['regime']} (Neutral/Chop).")

if gex_df is not None:
    if gex_df['gex'].sum() > 0: reasons.append("GEX: Dealers Long Gamma (Expect Range/Low Vol).")
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
    <div style='text-align:center; margin-top:10px;'>
        SUGGESTED KELLY SIZE: <span style='color:#ff9900; font-weight:bold'>{safe_kelly:.1f}%</span>
    </div>
</div>
""", unsafe_allow_html=True)
