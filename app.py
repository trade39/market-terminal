import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from fredapi import Fred
from newsapi import NewsApiClient
import google.generativeai as genai
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier

# --- APP CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Market Terminal Pro v2", page_icon="⚡")

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .metric-container {
        background-color: #1e1e1e;
        border: 1px solid #333;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    .sentiment-box {
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        border-left: 5px solid #d4af37;
        background-color: #262730;
    }
    /* Signal badges */
    .bullish { color: #00ff00; font-weight: bold; background-color: rgba(0, 255, 0, 0.1); padding: 2px 8px; border-radius: 4px; }
    .bearish { color: #ff4b4b; font-weight: bold; background-color: rgba(255, 75, 75, 0.1); padding: 2px 8px; border-radius: 4px; }
    .neutral { color: #cccccc; font-weight: bold; background-color: rgba(200, 200, 200, 0.1); padding: 2px 8px; border-radius: 4px; }
    
    /* Z-Score Badge */
    .extreme { color: #ffeb3b; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTS & MAPPINGS ---
ASSETS = {
    "Gold (Comex)": {"ticker": "GC=F", "news_query": "Gold Price", "fred_series": "DGS10", "fred_label": "10Y Yield", "correlation": "inverse"}, 
    "S&P 500": {"ticker": "^GSPC", "news_query": "S&P 500", "fred_series": "WALCL", "fred_label": "Fed Bal Sheet", "correlation": "direct"},
    "NASDAQ": {"ticker": "^IXIC", "news_query": "Nasdaq", "fred_series": "FEDFUNDS", "fred_label": "Fed Funds Rate", "correlation": "inverse"},
    "EUR/USD": {"ticker": "EURUSD=X", "news_query": "EURUSD", "fred_series": "DEXUSEU", "fred_label": "Exch Rate", "correlation": "direct"},
    "GBP/USD": {"ticker": "GBPUSD=X", "news_query": "GBPUSD", "fred_series": "DEXUSUK", "fred_label": "Exch Rate", "correlation": "direct"},
    "Bitcoin": {"ticker": "BTC-USD", "news_query": "Bitcoin", "fred_series": "DGS10", "fred_label": "10Y Yield", "correlation": "inverse"}
}

DXY_TICKER = "DX-Y.NYB"

# --- HELPER FUNCTIONS ---

def get_api_key(key_name):
    if "api_keys" in st.secrets and key_name in st.secrets["api_keys"]:
        return st.secrets["api_keys"][key_name]
    if key_name in st.secrets:
        return st.secrets[key_name]
    return None

@st.cache_data(ttl=60)
def get_daily_data(ticker):
    """Fetches Daily Data (2 Years)"""
    try:
        data = yf.download(ticker, period="2y", interval="1d", progress=False)
        return data
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60)
def get_intraday_data(ticker):
    """Fetches 15m Intraday Data (Last 5 Days)"""
    try:
        data = yf.download(ticker, period="5d", interval="15m", progress=False)
        return data
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_news(api_key, query):
    if not api_key: return None
    try:
        newsapi = NewsApiClient(api_key=api_key)
        start_date = (datetime.now() - timedelta(days=28)).strftime('%Y-%m-%d')
        articles = newsapi.get_everything(q=query, from_param=start_date, language='en', sort_by='relevancy', page_size=10)
        return articles['articles']
    except Exception as e:
        return f"Error: {str(e)}"

@st.cache_data(ttl=86400)
def get_fred_data(api_key, series_id):
    if not api_key: return None
    try:
        fred = Fred(api_key=api_key)
        data = fred.get_series(series_id)
        df = pd.DataFrame(data, columns=['Value'])
        df.index.name = 'Date'
        start_date = datetime.now() - timedelta(days=730)
        return df[df.index > start_date]
    except Exception:
        return None

# --- ANALYTICS CALCULATIONS ---

def calculate_vwap(df):
    if df.empty: return df
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['TPV'] = df['TP'] * df['Volume']
    df['CumTPV'] = df['TPV'].cumsum()
    df['CumVol'] = df['Volume'].cumsum()
    df['VWAP'] = df['CumTPV'] / df['CumVol']
    return df

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_z_score(series, window=20):
    """Calculates Statistical Z-Score (Standard Deviations from Mean)"""
    mean = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    z_score = (series - mean) / std
    return z_score

def calculate_volume_profile(df, bins=50):
    """Calculates Volume Profile by Price Levels"""
    if df.empty: return pd.DataFrame()
    
    # Handle MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        high = df['High'].iloc[:, 0]
        low = df['Low'].iloc[:, 0]
        close = df['Close'].iloc[:, 0]
        volume = df['Volume'].iloc[:, 0]
    else:
        high = df['High']
        low = df['Low']
        close = df['Close']
        volume = df['Volume']

    price_range = high.max() - low.min()
    bin_size = price_range / bins
    
    # Create bins based on Close price
    # We use a copy to avoid SettingWithCopy warnings on cached data
    temp_df = pd.DataFrame({'Close': close, 'Volume': volume})
    temp_df['Price_Bin'] = ((temp_df['Close'] - low.min()) // bin_size).astype(int)
    
    # Group volume by price bin
    vp = temp_df.groupby('Price_Bin')['Volume'].sum().reset_index()
    
    # Calculate price levels for plotting
    vp['Price_Level'] = low.min() + (vp['Price_Bin'] * bin_size)
    return vp

@st.cache_data(ttl=3600)
def get_ml_signal(df):
    """
    Trains a Random Forest Classifier to predict if the NEXT day will be Green (1) or Red (0).
    Returns the prediction for tomorrow and the confidence probability.
    """
    if df.empty: return None, 0.5
    
    data = df.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Feature Engineering
    data['Returns'] = data['Close'].pct_change()
    data['RSI'] = calculate_rsi(data['Close'])
    data['Vol_Change'] = data['Volume'].pct_change()
    data['Range'] = (data['High'] - data['Low']) / data['Close']
    
    # Target: 1 if Next Close > Current Close
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    data = data.dropna()
    
    features = ['Returns', 'RSI', 'Vol_Change', 'Range']
    X = data[features]
    y = data['Target']
    
    # Train on all available data except the last row (which has no target yet)
    if len(X) < 50: return None, 0.5 # Not enough data
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X.iloc[:-1], y.iloc[:-1])
    
    # Predict for the most recent candle (to forecast tomorrow)
    current_features = X.iloc[[-1]]
    prediction = model.predict(current_features)[0]
    probability = model.predict_proba(current_features)[0][1] # Probability of Class 1 (Bullish)
    
    return prediction, probability

# --- EXISTING FEATURES (Monte Carlo, Macro, Correlation) ---
@st.cache_data(ttl=3600)
def get_correlation_data():
    tickers = {v['ticker']: k for k, v in ASSETS.items()}
    tickers[DXY_TICKER] = "US Dollar Index (DXY)"
    ticker_list = list(tickers.keys())
    try:
        data = yf.download(ticker_list, period="1y", interval="1d", progress=False)['Close']
        data = data.rename(columns=tickers)
        return data
    except: return pd.DataFrame()

@st.cache_data(ttl=3600)
def generate_monte_carlo(stock_data, days=90, simulations=1000):
    if isinstance(stock_data.columns, pd.MultiIndex):
        close = stock_data['Close'].iloc[:, 0]
    else:
        close = stock_data['Close']
    log_returns = np.log(1 + close.pct_change())
    u = log_returns.mean()
    var = log_returns.var()
    drift = u - (0.5 * var)
    stdev = log_returns.std()
    last_price = close.iloc[-1]
    prediction_dates = pd.date_range(start=close.index[-1], periods=days + 1, freq='B')
    daily_returns = np.exp(drift + stdev * np.random.normal(0, 1, (days, simulations)))
    price_paths = np.zeros((days + 1, simulations))
    price_paths[0] = last_price
    for t in range(1, days + 1):
        price_paths[t] = price_paths[t - 1] * daily_returns[t - 1]
    return prediction_dates, price_paths

def analyze_macro_signal(macro_df, correlation_type):
    if macro_df.empty: return "Neutral", "No Data"
    current_val = macro_df['Value'].iloc[-1]
    try: past_val = macro_df['Value'].iloc[-22]
    except: past_val = macro_df['Value'].iloc[0]
    delta = current_val - past_val
    trend_up = delta > 0
    if correlation_type == "inverse":
        return ("BEARISH", "Rising (Inverse)") if trend_up else ("BULLISH", "Falling (Inverse)")
    else:
        return ("BULLISH", "Rising (Direct)") if trend_up else ("BEARISH", "Falling (Direct)")

@st.cache_data(ttl=900)
def get_ai_sentiment(api_key, asset_name, news_items):
    if not api_key or not news_items or isinstance(news_items, str): return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        headlines = [f"- {item['title']}" for item in news_items[:8]]
        prompt = f"Analyze these headlines for {asset_name}. concise sentiment summary (Bullish/Bearish/Neutral) max 40 words:\n" + "\n".join(headlines)
        response = model.generate_content(prompt)
        return response.text
    except: return None

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    selected_asset = st.selectbox("Select Asset", list(ASSETS.keys()))
    asset_info = ASSETS[selected_asset]
    st.markdown("---")
    news_key = get_api_key("news_api_key")
    fred_key = get_api_key("fred_api_key")
    google_key = get_api_key("google_api_key")
    st.caption(f"Keys Loaded: NewsAPI {'✅' if news_key else '❌'} | FRED {'✅' if fred_key else '❌'}")
    if st.button("Refresh Data"): st.cache_data.clear()

# --- MAIN DASHBOARD ---
st.title(f"⚡ {selected_asset} Pro Terminal")

# Fetch Data
daily_data = get_daily_data(asset_info['ticker'])
intraday_data = get_intraday_data(asset_info['ticker'])
macro_df = get_fred_data(fred_key, asset_info['fred_series'])

# --- 1. OVERVIEW & CHART WITH VOLUME PROFILE ---
if not daily_data.empty:
    if isinstance(daily_data.columns, pd.MultiIndex):
        close = daily_data['Close'].iloc[:, 0]
        high = daily_data['High'].iloc[:, 0]
        low = daily_data['Low'].iloc[:, 0]
        open_p = daily_data['Open'].iloc[:, 0]
        vol = daily_data['Volume'].iloc[:, 0]
    else:
        close = daily_data['Close']
        high = daily_data['High']
        low = daily_data['Low']
        open_p = daily_data['Open']
        vol = daily_data['Volume']

    curr = close.iloc[-1]
    pct = ((curr - close.iloc[-2]) / close.iloc[-2]) * 100
    
    # Top Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"{curr:,.2f}", f"{pct:.2f}%")
    c2.metric("High", f"{high.max():,.2f}")
    c3.metric("Low", f"{low.min():,.2f}")
    c4.metric("Vol", f"{(close.pct_change().std()* (252**0.5)*100):.2f}%")

    # --- MAIN CHART (Daily + Volume Profile) ---
    fig = go.Figure()
    
    # 1. Candlestick
    fig.add_trace(go.Candlestick(x=daily_data.index, open=open_p, high=high, low=low, close=close, name="Price"))
    
    # 2. Macro Overlay
    if isinstance(macro_df, pd.DataFrame):
        macro_aligned = macro_df.reindex(daily_data.index, method='ffill')
        fig.add_trace(go.Scatter(x=macro_df.index, y=macro_df['Value'], name=asset_info['fred_label'], line=dict(color='#d4af37', width=2), yaxis="y2"))

    # 3. Volume Profile (Horizontal Bars)
    vp_data = calculate_volume_profile(daily_data)
    if not vp_data.empty:
        # Scale volume for visualization
        max_vol_p = vp_data['Volume'].max()
        # Create a dummy x-axis scaler or use secondary x-axis
        fig.add_trace(go.Bar(
            y=vp_data['Price_Level'], 
            x=vp_data['Volume'],
            orientation='h',
            name="Vol Profile",
            marker=dict(color='rgba(255, 255, 255, 0.15)', line=dict(color='rgba(255,255,255,0.3)', width=1)),
            xaxis="x2",
            hoverinfo="skip"
        ))

    # Layout Updates for Dual Axis & VP
    fig.update_layout(
        height=500, 
        template="plotly_dark", 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        xaxis_rangeslider_visible=False, 
        yaxis2=dict(overlaying="y", side="right", showgrid=False, title=asset_info['fred_label']),
        xaxis2=dict(overlaying="x", side="top", showgrid=False, visible=False), # VP axis hidden
        title="Price Action & Liquidity Profile"
    )
    st.plotly_chart(fig, use_container_width=True)

# --- 2. ⚡ INTRADAY PRO DASHBOARD (With Z-Score) ---
st.markdown("---")
st.subheader("⚡ Intraday & Stat Arbitrage")

if not intraday_data.empty and not daily_data.empty:
    if isinstance(intraday_data.columns, pd.MultiIndex):
        i_close = intraday_data['Close'].iloc[:, 0]
        i_high = intraday_data['High'].iloc[:, 0]
        i_low = intraday_data['Low'].iloc[:, 0]
        i_vol = intraday_data['Volume'].iloc[:, 0]
    else:
        i_close, i_high, i_low, i_vol = intraday_data['Close'], intraday_data['High'], intraday_data['Low'], intraday_data['Volume']

    df_vwap = pd.DataFrame({'High': i_high, 'Low': i_low, 'Close': i_close, 'Volume': i_vol})
    df_vwap = calculate_vwap(df_vwap)
    current_vwap = df_vwap['VWAP'].iloc[-1]
    current_price = i_close.iloc[-1]
    rsi_val = calculate_rsi(i_close).iloc[-1]
    
    # Calculate Z-Score on Daily Data for robustness
    z_score = calculate_z_score(close).iloc[-1]
    
    col_dash1, col_dash2, col_dash3 = st.columns([1.5, 1, 1.5])
    
    with col_dash1:
        st.markdown("**1. Bias Meter & Z-Score**")
        bias = "NEUTRAL"
        if current_price > current_vwap and rsi_val > 50: bias = "BULLISH"
        elif current_price < current_vwap and rsi_val < 50: bias = "BEARISH"
        
        bias_color = "bullish" if bias == "BULLISH" else "bearish" if bias == "BEARISH" else "neutral"
        st.markdown(f"### <span class='{bias_color}'>{bias}</span>", unsafe_allow_html=True)
        
        # Z-Score Display
        z_color = "extreme" if abs(z_score) > 2.0 else "neutral"
        st.markdown(f"Stat Deviation (Z-Score): <span class='{z_color}'>{z_score:.2f}</span>", unsafe_allow_html=True)
        st.caption("Z > 2.0: Overbought | Z < -2.0: Oversold")
        
    with col_dash2:
        yest_close = close.iloc[-2]
        today_open = open_p.iloc[-1]
        gap_pct = ((today_open - yest_close) / yest_close) * 100
        st.markdown("**2. Opening Gap**")
        st.metric("Overnight Gap", f"{gap_pct:.2f}%")
        st.caption(f"VWAP: {current_vwap:.2f}")
        
    with col_dash3:
        st.markdown("**3. Momentum**")
        trend_15m = "Bullish" if current_price > i_close.rolling(20).mean().iloc[-1] else "Bearish"
        trend_4h = "Bullish" if close.iloc[-1] > close.rolling(5).mean().iloc[-1] else "Bearish" # approx 4h on daily
        st.metric("15m Trend", trend_15m, f"RSI: {rsi_val:.0f}")
else:
    st.warning("Intraday data unavailable.")

# --- 3. PREDICTION LAB (ML + Monte Carlo) ---
st.markdown("---")
st.subheader("🔮 Prediction Lab (AI + Stats)")

pred_col1, pred_col2 = st.columns([1, 2])

with pred_col1:
    st.markdown("#### 🤖 ML Signal (Random Forest)")
    if not daily_data.empty:
        pred, prob = get_ml_signal(daily_data)
        if pred is not None:
            direction = "BULLISH" if pred == 1 else "BEARISH"
            color = "bullish" if pred == 1 else "bearish"
            confidence = prob if pred == 1 else (1 - prob)
            
            st.markdown(f"### <span class='{color}'>{direction}</span>", unsafe_allow_html=True)
            st.metric("Model Confidence", f"{confidence*100:.1f}%")
            st.caption("Training: Returns, RSI, Vol, Range")
            st.progress(confidence)
        else:
            st.info("Not enough data for ML.")

with pred_col2:
    st.markdown("#### 🎲 Monte Carlo Projection (90 Days)")
    if not daily_data.empty:
        pred_dates, pred_paths = generate_monte_carlo(daily_data, simulations=2000)
        fig_pred = go.Figure()
        # Plot only average and bounds to save performance
        avg_path = np.mean(pred_paths, axis=1)
        p95 = np.percentile(pred_paths, 95, axis=1)
        p05 = np.percentile(pred_paths, 5, axis=1)
        
        fig_pred.add_trace(go.Scatter(x=pred_dates, y=avg_path, name='Avg Path', line=dict(color='#00ff00', width=2)))
        fig_pred.add_trace(go.Scatter(x=pred_dates, y=p95, name='95% Conf', line=dict(color='gray', width=1, dash='dot')))
        fig_pred.add_trace(go.Scatter(x=pred_dates, y=p05, name='5% Conf', line=dict(color='gray', width=1, dash='dot')))
        
        fig_pred.update_layout(height=300, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig_pred, use_container_width=True)

# --- 4. MACRO & SENTIMENT ---
st.markdown("---")
col_macro, col_news = st.columns(2)

with col_macro:
    st.subheader("🏛️ Macro Context")
    if isinstance(macro_df, pd.DataFrame) and not macro_df.empty:
        signal, reason = analyze_macro_signal(macro_df, asset_info['correlation'])
        color_class = "bullish" if signal == "BULLISH" else "bearish" if signal == "BEARISH" else "neutral"
        st.markdown(f"Signal: <span class='{color_class}'>{signal}</span> ({reason})", unsafe_allow_html=True)
        st.line_chart(macro_df['Value'].tail(50), color="#d4af37", height=200)

with col_news:
    st.subheader("📰 AI Analysis")
    if news_key and google_key:
        news_data = get_news(news_key, asset_info['news_query'])
        if isinstance(news_data, list):
            with st.spinner("Gemini Analyzing..."):
                s = get_ai_sentiment(google_key, selected_asset, news_data)
                if s: st.info(s)
    else:
        st.caption("Add API Keys to .streamlit/secrets.toml to unlock AI News.")
