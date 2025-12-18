import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from fredapi import Fred
from newsapi import NewsApiClient
import google.generativeai as genai
from datetime import datetime, timedelta

# --- APP CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Market Terminal", page_icon="📈")

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
</style>
""", unsafe_allow_html=True)

# --- CONSTANTS & MAPPINGS ---
ASSETS = {
    "Gold (Comex)": {
        "ticker": "GC=F", 
        "news_query": "Gold Price market", 
        "fred_series": "DGS10", 
        "fred_label": "10-Year Treasury Yield",
        "correlation": "inverse" # Yields UP = Gold DOWN
    }, 
    "S&P 500": {
        "ticker": "^GSPC", 
        "news_query": "S&P 500 market", 
        "fred_series": "WALCL", 
        "fred_label": "Fed Balance Sheet (Liquidity)",
        "correlation": "direct" # Liquidity UP = Stocks UP
    },
    "NASDAQ": {
        "ticker": "^IXIC", 
        "news_query": "Nasdaq tech stocks", 
        "fred_series": "FEDFUNDS", 
        "fred_label": "Fed Funds Rate",
        "correlation": "inverse" # Rates UP = Tech DOWN
    },
    "EUR/USD": {
        "ticker": "EURUSD=X", 
        "news_query": "EURUSD forex", 
        "fred_series": "DEXUSEU", 
        "fred_label": "USD/EUR Rate",
        "correlation": "direct" 
    },
    "GBP/USD": {
        "ticker": "GBPUSD=X", 
        "news_query": "GBPUSD forex", 
        "fred_series": "DEXUSUK", 
        "fred_label": "USD/GBP Rate",
        "correlation": "direct"
    }
}

# --- HELPER FUNCTIONS ---

def get_api_key(key_name):
    if "api_keys" in st.secrets and key_name in st.secrets["api_keys"]:
        return st.secrets["api_keys"][key_name]
    if key_name in st.secrets:
        return st.secrets[key_name]
    return None

@st.cache_data(ttl=60)
def get_market_data(ticker, period="2y", interval="1d"): # Increased history for better drift calc
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
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

# --- ANALYTICS FUNCTIONS ---

def get_macro_signal(macro_df, correlation_type):
    """
    Analyzes the last 30 days of Macro data to determine trend
    and maps it to Bullish/Bearish based on asset correlation.
    """
    if macro_df is None or macro_df.empty:
        return "No Data", "gray"
    
    # Compare current value to 30 days ago (smooth out noise)
    current_val = macro_df['Value'].iloc[-1]
    past_val = macro_df['Value'].iloc[-20] # approx 1 month of trading days
    
    delta = current_val - past_val
    
    # 1. Determine Macro Trend
    macro_trend_up = delta > 0
    
    # 2. Map to Asset Sentiment
    if correlation_type == "inverse":
        if macro_trend_up:
            return "BEARISH (Macro Headwind)", "red"  # Yields Up -> Gold Down
        else:
            return "BULLISH (Macro Tailwind)", "#00ff00" # Yields Down -> Gold Up
            
    elif correlation_type == "direct":
        if macro_trend_up:
            return "BULLISH (Macro Tailwind)", "#00ff00"
        else:
            return "BEARISH (Macro Headwind)", "red"
            
    return "NEUTRAL", "gray"

def calculate_projection(stock_data, days_forward=126):
    """
    Calculates Volatility Drift (Geometric Brownian Motion parameters)
    Projects 6 months (approx 126 trading days)
    """
    if stock_data.empty: return None
    
    # 1. Calculate Returns
    # Handle MultiIndex if present
    if isinstance(stock_data.columns, pd.MultiIndex):
        close = stock_data['Close'].iloc[:, 0]
    else:
        close = stock_data['Close']
        
    returns = close.pct_change().dropna()
    
    # 2. Calculate Drift (mu) and Volatility (sigma)
    # Annualized values
    mu = returns.mean() * 252 
    sigma = returns.std() * np.sqrt(252)
    
    last_price = close.iloc[-1]
    
    # 3. Generate Projection Arrays
    # Time steps (in years)
    dt = 1/252
    t = np.linspace(dt, days_forward * dt, days_forward)
    
    # Expected Path (Drift only)
    expected_path = last_price * np.exp(mu * t)
    
    # Bull Case (+1 Sigma)
    bull_path = last_price * np.exp((mu + sigma) * t)
    
    # Bear Case (-1 Sigma)
    bear_path = last_price * np.exp((mu - sigma) * t)
    
    dates = [close.index[-1] + timedelta(days=i) for i in range(1, days_forward + 1)]
    
    return dates, expected_path, bull_path, bear_path, mu, sigma

# --- AI SENTIMENT ---
@st.cache_data(ttl=900)
def get_ai_sentiment(api_key, asset_name, news_items):
    if not api_key or not news_items or isinstance(news_items, str):
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        headlines = [f"- {item['title']}" for item in news_items[:8]]
        headlines_text = "\n".join(headlines)
        prompt = f"""
        You are a financial analyst. Analyze these headlines for {asset_name}:
        {headlines_text}
        Summarize sentiment (BULLISH/BEARISH/NEUTRAL) in max 50 words.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Analysis Unavailable: {str(e)}"

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    selected_asset = st.selectbox("Select Asset", list(ASSETS.keys()))
    asset_info = ASSETS[selected_asset]
    
    st.markdown("---")
    news_key = get_api_key("news_api_key")
    fred_key = get_api_key("fred_api_key")
    google_key = get_api_key("google_api_key")

    if not fred_key: st.warning("Need FRED Key for Macro Logic")
    if not google_key: st.info("Add Gemini Key for AI Summary")
    
    if st.button("Refresh Data"):
        st.cache_data.clear()

# --- MAIN DASHBOARD ---
st.title(f"📊 {selected_asset} Professional Terminal")

stock_data = get_market_data(asset_info['ticker'])
macro_df = get_fred_data(fred_key, asset_info['fred_series'])

# --- 1. MACRO CONTEXT LOGIC ---
macro_signal, signal_color = "Neutral", "gray"
if isinstance(macro_df, pd.DataFrame) and not macro_df.empty:
    macro_signal, signal_color = get_macro_signal(macro_df, asset_info['correlation'])

st.markdown(f"""
    <div style="padding: 10px; border-radius: 5px; background-color: #1e1e1e; border: 1px solid #333; margin-bottom: 20px;">
        <span style="font-size: 1.2em; font-weight: bold;">Macro Context: </span>
        <span style="font-size: 1.2em; font-weight: bold; color: {signal_color};">{macro_signal}</span>
        <br>
        <span style="font-size: 0.9em; color: #aaa;">Based on correlation with {asset_info['fred_label']} (Last 30 days trend)</span>
    </div>
""", unsafe_allow_html=True)

# --- 2. MARKET DATA ---
if not stock_data.empty:
    if isinstance(stock_data.columns, pd.MultiIndex):
        close = stock_data['Close'].iloc[:, 0]
        high = stock_data['High'].iloc[:, 0]
        low = stock_data['Low'].iloc[:, 0]
        open_p = stock_data['Open'].iloc[:, 0]
    else:
        close = stock_data['Close']
        high = stock_data['High']
        low = stock_data['Low']
        open_p = stock_data['Open']

    curr_price = close.iloc[-1]
    change = curr_price - close.iloc[-2]
    pct = (change / close.iloc[-2]) * 100
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"{curr_price:,.2f}", f"{change:,.2f} ({pct:.2f}%)")
    c2.metric("High", f"{high.max():,.2f}")
    c3.metric("Low", f"{low.min():,.2f}")
    vol = close.pct_change().std() * (252**0.5) * 100
    c4.metric("Volatility", f"{vol:.2f}%")

    # TABS FOR CHARTING
    tab1, tab2 = st.tabs(["📈 Historical & Macro", "🔮 6-Month Forecast"])

    # TAB 1: HISTORICAL
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=stock_data.index, open=open_p, high=high, low=low, close=close, name="Price"))
        if isinstance(macro_df, pd.DataFrame):
            macro_aligned = macro_df.reindex(stock_data.index, method='ffill')
            fig.add_trace(go.Scatter(x=macro_df.index, y=macro_df['Value'], name=asset_info['fred_label'], line=dict(color='#d4af37', width=2), yaxis="y2"))
        
        fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False, yaxis2=dict(overlaying="y", side="right", showgrid=False))
        st.plotly_chart(fig, use_container_width=True)

    # TAB 2: FORECAST
    with tab2:
        proj_dates, exp_path, bull_path, bear_path, mu, sigma = calculate_projection(stock_data)
        
        f_fig = go.Figure()
        
        # Fan Chart
        f_fig.add_trace(go.Scatter(x=proj_dates, y=bull_path, mode='lines', line=dict(width=0), name='Upper Bound'))
        f_fig.add_trace(go.Scatter(x=proj_dates, y=bear_path, mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(100, 100, 100, 0.2)', name='Volatility Cone'))
        f_fig.add_trace(go.Scatter(x=proj_dates, y=exp_path, mode='lines', line=dict(color='#00ff00', dash='dash'), name='Drift (Expected)'))
        f_fig.add_trace(go.Scatter(x=[stock_data.index[-1]], y=[curr_price], mode='markers', marker=dict(color='white', size=5), name='Current Price'))

        f_fig.update_layout(
            title=f"Volatility Drift Projection (6 Months)",
            height=500, 
            template="plotly_dark",
            yaxis_title="Projected Price",
            showlegend=True
        )
        
        st.plotly_chart(f_fig, use_container_width=True)
        
        # Stats
        fc1, fc2 = st.columns(2)
        fc1.info(f"**Annualized Drift (Trend):** {mu*100:.2f}%")
        fc2.info(f"**Implied Volatility (Risk):** {sigma*100:.2f}%")
        st.caption("Based on Geometric Brownian Motion (GBM) using last 2 years of data. Not financial advice.")

# --- 3. NEWS & AI ---
st.markdown("---")
st.subheader("📰 Market Intelligence")

if news_key:
    news_data = get_news(news_key, asset_info['news_query'])
    
    if google_key and isinstance(news_data, list):
        with st.spinner("AI analyzing sentiment..."):
            sentiment = get_ai_sentiment(google_key, selected_asset, news_data)
            if sentiment:
                st.markdown(f"<div class='sentiment-box'><h4>🤖 Gemini AI Sentiment</h4><p>{sentiment}</p></div>", unsafe_allow_html=True)
    
    if isinstance(news_data, list):
        news_cols = st.columns(3)
        for i, article in enumerate(news_data[:6]):
            with news_cols[i % 3]:
                with st.container(border=True):
                    st.markdown(f"**[{article['title']}]({article['url']})**")
                    st.caption(f"{article['source']['name']} • {article['publishedAt'][:10]}")
    else:
        st.info("No news found.")
else:
    st.info("Enter NewsAPI Key.")
