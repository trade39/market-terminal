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

# --- APP CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Market Terminal Pro", page_icon="📈")

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
    
    /* Heatmap Grid */
    .heatmap-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; text-align: center; }
    .heatmap-item { padding: 10px; border-radius: 5px; border: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTS & MAPPINGS ---
ASSETS = {
    "Gold (Comex)": {"ticker": "GC=F", "news_query": "Gold Price", "fred_series": "DGS10", "fred_label": "10Y Yield", "correlation": "inverse"}, 
    "S&P 500": {"ticker": "^GSPC", "news_query": "S&P 500", "fred_series": "WALCL", "fred_label": "Fed Bal Sheet", "correlation": "direct"},
    "NASDAQ": {"ticker": "^IXIC", "news_query": "Nasdaq", "fred_series": "FEDFUNDS", "fred_label": "Fed Funds Rate", "correlation": "inverse"},
    "EUR/USD": {"ticker": "EURUSD=X", "news_query": "EURUSD", "fred_series": "DEXUSEU", "fred_label": "Exch Rate", "correlation": "direct"},
    "GBP/USD": {"ticker": "GBPUSD=X", "news_query": "GBPUSD", "fred_series": "DEXUSUK", "fred_label": "Exch Rate", "correlation": "direct"}
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
    """Fetches 15m Intraday Data (Last 5 Days) for VWAP & Momentum"""
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

# --- NEW: INTRADAY CALCULATIONS ---
def calculate_vwap(df):
    """Calculates Rolling VWAP for the fetched period"""
    if df.empty: return df
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['TPV'] = df['TP'] * df['Volume']
    df['CumTPV'] = df['TPV'].cumsum()
    df['CumVol'] = df['Volume'].cumsum()
    df['VWAP'] = df['CumTPV'] / df['CumVol']
    return df

def calculate_rsi(series, period=14):
    """Calculates RSI for Momentum"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

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
def generate_monte_carlo(stock_data, days=126, simulations=10000):
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
st.title(f"📊 {selected_asset} Pro Terminal")

# Fetch Data
daily_data = get_daily_data(asset_info['ticker'])
intraday_data = get_intraday_data(asset_info['ticker'])
macro_df = get_fred_data(fred_key, asset_info['fred_series'])

# --- 1. OVERVIEW SECTION ---
if not daily_data.empty:
    if isinstance(daily_data.columns, pd.MultiIndex):
        close = daily_data['Close'].iloc[:, 0]
        high = daily_data['High'].iloc[:, 0]
        low = daily_data['Low'].iloc[:, 0]
        open_p = daily_data['Open'].iloc[:, 0]
    else:
        close = daily_data['Close']
        high = daily_data['High']
        low = daily_data['Low']
        open_p = daily_data['Open']

    curr = close.iloc[-1]
    pct = ((curr - close.iloc[-2]) / close.iloc[-2]) * 100
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"{curr:,.2f}", f"{pct:.2f}%")
    c2.metric("High", f"{high.max():,.2f}")
    c3.metric("Low", f"{low.min():,.2f}")
    c4.metric("Vol", f"{(close.pct_change().std()* (252**0.5)*100):.2f}%")

    # --- MAIN CHART (Daily) ---
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=daily_data.index, open=open_p, high=high, low=low, close=close, name="Price"))
    if isinstance(macro_df, pd.DataFrame):
        macro_aligned = macro_df.reindex(daily_data.index, method='ffill')
        fig.add_trace(go.Scatter(x=macro_df.index, y=macro_df['Value'], name=asset_info['fred_label'], line=dict(color='#d4af37', width=2), yaxis="y2"))
    fig.update_layout(height=400, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_rangeslider_visible=False, yaxis2=dict(overlaying="y", side="right", showgrid=False))
    st.plotly_chart(fig, use_container_width=True)

# --- 2. ⚡ INTRADAY PRO DASHBOARD ---
st.markdown("---")
st.subheader("⚡ Intraday Pro Dashboard")

if not intraday_data.empty and not daily_data.empty:
    # Handle MultiIndex
    if isinstance(intraday_data.columns, pd.MultiIndex):
        i_close = intraday_data['Close'].iloc[:, 0]
        i_high = intraday_data['High'].iloc[:, 0]
        i_low = intraday_data['Low'].iloc[:, 0]
        i_vol = intraday_data['Volume'].iloc[:, 0]
    else:
        i_close, i_high, i_low, i_vol = intraday_data['Close'], intraday_data['High'], intraday_data['Low'], intraday_data['Volume']

    # Calculations
    df_vwap = pd.DataFrame({'High': i_high, 'Low': i_low, 'Close': i_close, 'Volume': i_vol})
    df_vwap = calculate_vwap(df_vwap)
    current_vwap = df_vwap['VWAP'].iloc[-1]
    current_price = i_close.iloc[-1]
    rsi_val = calculate_rsi(i_close).iloc[-1]
    
    yest_close = close.iloc[-2]
    today_open = open_p.iloc[-1]
    gap_pct = ((today_open - yest_close) / yest_close) * 100
    
    # Trends
    trend_15m = "Bullish" if current_price > i_close.rolling(20).mean().iloc[-1] else "Bearish"
    df_1h = df_vwap.resample('1h').agg({'Close': 'last'})
    trend_1h = "Bullish" if df_1h['Close'].iloc[-1] > df_1h['Close'].rolling(20).mean().iloc[-1] else "Bearish"
    df_4h = df_vwap.resample('4h').agg({'Close': 'last'})
    trend_4h = "Bullish" if df_4h['Close'].iloc[-1] > df_4h['Close'].rolling(20).mean().iloc[-1] else "Bearish"

    # Dashboard
    col_dash1, col_dash2, col_dash3 = st.columns([1.5, 1, 1.5])
    
    with col_dash1:
        st.markdown("**1. Intraday Bias Meter**")
        bias = "NEUTRAL"
        if current_price > current_vwap and rsi_val > 50: bias = "BULLISH"
        elif current_price < current_vwap and rsi_val < 50: bias = "BEARISH"
        bias_color = "bullish" if bias == "BULLISH" else "bearish" if bias == "BEARISH" else "neutral"
        st.markdown(f"### <span class='{bias_color}'>{bias}</span>", unsafe_allow_html=True)
        st.caption(f"Price vs VWAP: {'Above' if current_price > current_vwap else 'Below'} | RSI(14): {rsi_val:.1f}")
        
    with col_dash2:
        st.markdown("**2. Opening Gap**")
        st.metric("Overnight Gap", f"{gap_pct:.2f}%")
        
    with col_dash3:
        st.markdown("**3. Multi-Timeframe Momentum**")
        hm1, hm2, hm3 = st.columns(3)
        def get_arrow(trend): return "🟢 ⬆" if trend == "Bullish" else "🔴 ⬇"
        with hm1: 
            st.caption("15 Min")
            st.markdown(get_arrow(trend_15m))
        with hm2:
            st.caption("1 Hour")
            st.markdown(get_arrow(trend_1h))
        with hm3:
            st.caption("4 Hour")
            st.markdown(get_arrow(trend_4h))
else:
    st.warning("Intraday data unavailable.")

# --- 3. MACRO CONTEXT ---
st.markdown("---")
st.subheader("🏛️ Macro Data & Signal")
if isinstance(macro_df, pd.DataFrame) and not macro_df.empty:
    signal, reason = analyze_macro_signal(macro_df, asset_info['correlation'])
    color_class = "bullish" if signal == "BULLISH" else "bearish" if signal == "BEARISH" else "neutral"
    t1, t2, t3 = st.columns([1, 2, 1])
    with t1: st.markdown(f"#### Signal: <span class='{color_class}'>{signal}</span>", unsafe_allow_html=True)
    with t2: 
        st.markdown(f"**Driver:** {asset_info['fred_label']}")
        st.caption(f"{reason[1]}")
    with t3: st.metric("Latest Macro", f"{macro_df['Value'].iloc[-1]:.2f}")
    st.line_chart(macro_df['Value'].tail(100), color="#d4af37")

# --- 4. PREDICTION ---
st.markdown("---")
st.subheader("🔮 6-Month Volatility Drift")
if not daily_data.empty:
    pred_dates, pred_paths = generate_monte_carlo(daily_data, simulations=10000)
    fig_pred = go.Figure()
    hist_slice = close.tail(90)
    fig_pred.add_trace(go.Scatter(x=hist_slice.index, y=hist_slice.values, name='Historical', line=dict(color='white', width=2)))
    for i in range(100): fig_pred.add_trace(go.Scatter(x=pred_dates, y=pred_paths[:, i], mode='lines', line=dict(color='cyan', width=1), opacity=0.05, showlegend=False, hoverinfo='skip'))
    fig_pred.add_trace(go.Scatter(x=pred_dates, y=np.mean(pred_paths, axis=1), name='Avg Proj', line=dict(color='#00ff00', width=3, dash='dash')))
    fig_pred.update_layout(height=400, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_pred, use_container_width=True)

# --- 5. CORRELATIONS & SENTIMENT (RESTORED) ---
st.markdown("---")
st.subheader("🌐 DXY Correlation Matrix & Sentiment")
corr_data = get_correlation_data()
if not corr_data.empty:
    cm1, cm2 = st.columns(2)
    corr_matrix = corr_data.corr()
    
    with cm1:
        fig_heat = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
        fig_heat.update_layout(template="plotly_dark", height=400, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_heat, use_container_width=True)
        
    with cm2:
        norm_data = (corr_data / corr_data.iloc[0] - 1) * 100
        fig_s = go.Figure()
        for c in norm_data.columns: 
            width = 3 if "DXY" in c else 1.5
            fig_s.add_trace(go.Scatter(x=norm_data.index, y=norm_data[c], name=c, line=dict(width=width)))
        fig_s.update_layout(template="plotly_dark", height=400, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_s, use_container_width=True)

    # --- RESTORED TEXT CONTEXT LOGIC ---
    st.markdown("#### 🧠 Correlation Context")
    dxy_name = "US Dollar Index (DXY)"
    if dxy_name in corr_matrix.columns:
        dxy_corrs = corr_matrix[dxy_name].drop(dxy_name)
        strongest_inv = dxy_corrs.idxmin()
        val_inv = dxy_corrs.min()
        strongest_dir = dxy_corrs.idxmax()
        val_dir = dxy_corrs.max()
        
        sentiment_text = f"""
        **Market Structure:**
        * **Strongest Inverse:** **{strongest_inv}** (Corr: `{val_inv:.2f}`). {'Classic "Safe Haven" dynamic.' if 'Gold' in strongest_inv else 'Capital rotation active.'}
        * **Strongest Direct:** **{strongest_dir}** (Corr: `{val_dir:.2f}`).
        
        **Interpretation:**
        """
        if val_inv < -0.7: sentiment_text += " **Strong Dollar Flows.** Risk assets are reacting heavily to DXY moves."
        elif abs(val_inv) < 0.3: sentiment_text += " **Decoupled Market.** Assets are moving on their own news, ignoring the Dollar."
        else: sentiment_text += " Mixed regime. Watch DXY levels for direction."
        
        st.info(sentiment_text)

# --- 6. NEWS & AI ---
st.markdown("---")
st.subheader("📰 AI Market Intelligence")
if news_key:
    news_data = get_news(news_key, asset_info['news_query'])
    if google_key and isinstance(news_data, list):
        with st.spinner("Gemini reading news..."):
            s = get_ai_sentiment(google_key, selected_asset, news_data)
            if s: st.markdown(f"<div class='sentiment-box'><h4>🤖 AI Summary</h4><p>{s}</p></div>", unsafe_allow_html=True)
    if isinstance(news_data, list):
        nc = st.columns(3)
        for i, a in enumerate(news_data[:3]):
            with nc[i]:
                st.markdown(f"**[{a['title']}]({a['url']})**")
                st.caption(f"{a['source']['name']} • {a['publishedAt'][:10]}")
