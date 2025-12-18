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
st.set_page_config(layout="wide", page_title="Market Terminal Pro", page_icon="⚡")

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
    .bullish { color: #00ff00; font-weight: bold; border: 1px solid #00ff00; padding: 2px 8px; border-radius: 4px; }
    .bearish { color: #ff4b4b; font-weight: bold; border: 1px solid #ff4b4b; padding: 2px 8px; border-radius: 4px; }
    .neutral { color: #cccccc; font-weight: bold; border: 1px solid #cccccc; padding: 2px 8px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTS ---
ASSETS = {
    "Gold (Comex)": {"ticker": "GC=F", "news_query": "Gold Price", "fred_series": "DGS10", "fred_label": "10Y Yield", "correlation": "inverse"}, 
    "S&P 500": {"ticker": "^GSPC", "news_query": "S&P 500", "fred_series": "WALCL", "fred_label": "Fed Balance Sheet", "correlation": "direct"},
    "NASDAQ": {"ticker": "^IXIC", "news_query": "Nasdaq", "fred_series": "FEDFUNDS", "fred_label": "Fed Funds Rate", "correlation": "inverse"},
    "EUR/USD": {"ticker": "EURUSD=X", "news_query": "EURUSD", "fred_series": "DEXUSEU", "fred_label": "FX Trend", "correlation": "direct"},
    "GBP/USD": {"ticker": "GBPUSD=X", "news_query": "GBPUSD", "fred_series": "DEXUSUK", "fred_label": "FX Trend", "correlation": "direct"}
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
def get_market_data(ticker, period="1y", interval="1d"):
    """
    Dynamic fetcher: Handles both Daily (Macro) and Intraday (15m/5m) requests.
    """
    try:
        # yfinance requires specific period/interval combos
        # For 15m data, max period is usually 60d. For 5m, it's 5d.
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        return data
    except Exception:
        return pd.DataFrame()

# --- TECHNICAL INDICATORS (NEW) ---
def calculate_technicals(df):
    """Adds VWAP and RSI to the dataframe"""
    if df.empty: return df
    
    # 1. RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 2. VWAP (Intraday Anchor)
    # VWAP = Cumulative(Price * Volume) / Cumulative(Volume)
    # We reset VWAP calculation daily ideally, but for this rolling view, a simple calculation suffices for trends
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['cum_vol_price'] = (df['TP'] * df['Volume']).cumsum()
    df['cum_vol'] = df['Volume'].cumsum()
    df['VWAP'] = df['cum_vol_price'] / df['cum_vol']
    
    return df

@st.cache_data(ttl=300)
def get_news(api_key, query):
    if not api_key: return None
    try:
        newsapi = NewsApiClient(api_key=api_key)
        start_date = (datetime.now() - timedelta(days=28)).strftime('%Y-%m-%d')
        articles = newsapi.get_everything(q=query, from_param=start_date, language='en', sort_by='relevancy', page_size=10)
        return articles['articles']
    except Exception: return None

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
    except Exception: return None

@st.cache_data(ttl=3600)
def get_correlation_data():
    tickers = {v['ticker']: k for k, v in ASSETS.items()}
    tickers[DXY_TICKER] = "US Dollar Index (DXY)"
    try:
        data = yf.download(list(tickers.keys()), period="1y", interval="1d", progress=False)['Close']
        data = data.rename(columns=tickers)
        return data
    except Exception: return pd.DataFrame()

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("⚙️ Settings")
    selected_asset = st.selectbox("Select Asset", list(ASSETS.keys()))
    asset_info = ASSETS[selected_asset]
    
    st.markdown("---")
    st.subheader("⏱️ Timeframe")
    # NEW: Timeframe Selector
    timeframe_mode = st.radio("Chart Mode:", ["Intraday (15m)", "Hourly (1h)", "Daily (1y)"])
    
    # Map selection to yfinance parameters
    if timeframe_mode == "Intraday (15m)":
        yf_period, yf_interval = "5d", "15m"
    elif timeframe_mode == "Hourly (1h)":
        yf_period, yf_interval = "1mo", "1h"
    else:
        yf_period, yf_interval = "1y", "1d"

    st.markdown("---")
    news_key = get_api_key("news_api_key")
    fred_key = get_api_key("fred_api_key")
    google_key = get_api_key("google_api_key")
    if st.button("Refresh Data"): st.cache_data.clear()

# --- MAIN DASHBOARD ---
st.title(f"⚡ {selected_asset} | {timeframe_mode}")

# 1. FETCH DATA
stock_data = get_market_data(asset_info['ticker'], period=yf_period, interval=yf_interval)
macro_df = get_fred_data(fred_key, asset_info['fred_series'])

if not stock_data.empty:
    # Prepare Data
    if isinstance(stock_data.columns, pd.MultiIndex):
        # Flatten MultiIndex if present
        stock_data_flat = stock_data.copy()
        stock_data_flat.columns = stock_data_flat.columns.get_level_values(0)
        stock_data = stock_data_flat

    # Calculate Intraday Technicals
    stock_data = calculate_technicals(stock_data)
    
    # Metrics
    curr = stock_data['Close'].iloc[-1]
    prev = stock_data['Close'].iloc[-2]
    pct = ((curr - prev) / prev) * 100
    
    # RSI Status
    rsi_val = stock_data['RSI'].iloc[-1]
    rsi_status = "OVERBOUGHT 🔴" if rsi_val > 70 else "OVERSOLD 🟢" if rsi_val < 30 else "NEUTRAL ⚪"
    
    # Display Top Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"{curr:,.2f}", f"{pct:.2f}%")
    c2.metric("RSI (14)", f"{rsi_val:.1f}", rsi_status)
    c3.metric("High", f"{stock_data['High'].max():,.2f}")
    c4.metric("Low", f"{stock_data['Low'].min():,.2f}")

    # --- 2. ADVANCED CHART ---
    fig = go.Figure()

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=stock_data.index,
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        name="Price"
    ))
    
    # VWAP (Only for Intraday/Hourly)
    if timeframe_mode in ["Intraday (15m)", "Hourly (1h)"]:
        fig.add_trace(go.Scatter(
            x=stock_data.index, 
            y=stock_data['VWAP'], 
            name="VWAP", 
            line=dict(color='#ff9f00', width=1.5, dash='dot')
        ))
        
    # Macro Overlay (Only for Daily View)
    if timeframe_mode == "Daily (1y)" and isinstance(macro_df, pd.DataFrame):
        macro_aligned = macro_df.reindex(stock_data.index, method='ffill')
        fig.add_trace(go.Scatter(
            x=macro_df.index, 
            y=macro_df['Value'], 
            name=asset_info['fred_label'], 
            line=dict(color='#d4af37', width=2), 
            yaxis="y2"
        ))

    fig.update_layout(
        height=600, 
        template="plotly_dark", 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        xaxis_rangeslider_visible=False, 
        yaxis2=dict(overlaying="y", side="right", showgrid=False),
        title=f"Price Action vs {'VWAP' if timeframe_mode != 'Daily (1y)' else 'Macro'}"
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("Data unavailable. Market might be closed or ticker changed.")

# --- 3. INTRADAY ANALYSIS (CONDITIONAL) ---
if timeframe_mode != "Daily (1y)":
    st.markdown("### 🎯 Intraday Pulse")
    col_pulse1, col_pulse2 = st.columns(2)
    
    with col_pulse1:
        st.info(f"""
        **VWAP Signal:**
        The Volume Weighted Average Price (VWAP) is the institutional benchmark.
        * **Current Price:** {curr:,.2f}
        * **VWAP:** {stock_data['VWAP'].iloc[-1]:,.2f}
        * **Trend:** {'BULLISH (Price > VWAP)' if curr > stock_data['VWAP'].iloc[-1] else 'BEARISH (Price < VWAP)'}
        """)
        
    with col_pulse2:
        # Mini RSI Chart
        fig_rsi = px.line(stock_data, x=stock_data.index, y="RSI", title="Momentum (RSI)")
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig_rsi.update_layout(height=200, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_rsi, use_container_width=True)

# --- 4. MACRO & CORRELATION (Keep existing sections below) ---
if timeframe_mode == "Daily (1y)":
    # (Only show Macro/Monte Carlo in Daily mode to reduce clutter)
    st.markdown("---")
    st.subheader("🌐 Inter-Market Correlation (DXY Focus)")
    corr_data = get_correlation_data()
    if not corr_data.empty:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_heat = px.imshow(corr_data.corr(), text_auto=True, color_continuous_scale='RdBu_r', title="Correlation Matrix")
            fig_heat.update_layout(height=400, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_heat, use_container_width=True)
        with col_c2:
            normalized = (corr_data / corr_data.iloc[0] - 1) * 100
            st.line_chart(normalized)

# --- NEWS (Always Visible) ---
st.markdown("---")
st.subheader("📰 Market Intelligence")
if news_key:
    news_data = get_news(news_key, asset_info['news_query'])
    if isinstance(news_data, list):
        cols = st.columns(3)
        for i, art in enumerate(news_data[:3]):
            with cols[i]:
                st.markdown(f"**[{art['title']}]({art['url']})**")
                st.caption(f"{art['source']['name']} • {art['publishedAt'][:10]}")
