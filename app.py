import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from fredapi import Fred
from newsapi import NewsApiClient
import google.generativeai as genai
from datetime import datetime, timedelta

# --- APP CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Market Terminal", page_icon="📈")

# Custom CSS for "Terminal" feel
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
    /* AI Sentiment Box Styling */
    .sentiment-box {
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
        border-left: 5px solid #d4af37; /* Gold accent */
        background-color: #262730;
    }
    .sentiment-box h4 { margin-top: 0; color: #d4af37; }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTS & MAPPINGS ---
ASSETS = {
    "Gold (Comex)": {
        "ticker": "GC=F", 
        "news_query": "Gold Price market", 
        "fred_series": "DGS10", 
        "fred_label": "10-Year Treasury Yield (Inverse proxy)"
    }, 
    "S&P 500": {
        "ticker": "^GSPC", 
        "news_query": "S&P 500 market", 
        "fred_series": "WALCL", 
        "fred_label": "Fed Balance Sheet (Liquidity)"
    },
    "NASDAQ": {
        "ticker": "^IXIC", 
        "news_query": "Nasdaq tech stocks", 
        "fred_series": "FEDFUNDS", 
        "fred_label": "Effective Federal Funds Rate"
    },
    "EUR/USD": {
        "ticker": "EURUSD=X", 
        "news_query": "EURUSD forex", 
        "fred_series": "DEXUSEU", 
        "fred_label": "U.S. / Euro Foreign Exchange Rate"
    },
    "GBP/USD": {
        "ticker": "GBPUSD=X", 
        "news_query": "GBPUSD forex", 
        "fred_series": "DEXUSUK", 
        "fred_label": "U.S. / UK Foreign Exchange Rate"
    }
}

# --- HELPER FUNCTIONS ---

def get_api_key(key_name):
    """
    Robustly retrieve API keys from secrets.
    Checks both nested [api_keys] and top-level secrets.
    """
    # Check 1: Inside [api_keys] section
    if "api_keys" in st.secrets and key_name in st.secrets["api_keys"]:
        return st.secrets["api_keys"][key_name]
    
    # Check 2: Top level (fallback)
    if key_name in st.secrets:
        return st.secrets[key_name]
        
    return None

@st.cache_data(ttl=60)
def get_market_data(ticker, period="1y", interval="1d"):
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
        # Limit search to last 28 days for Free Tier compliance
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

# --- AI SENTIMENT ANALYSIS ---
@st.cache_data(ttl=900) # Cache for 15 mins
def get_ai_sentiment(api_key, asset_name, news_items):
    if not api_key or not news_items or isinstance(news_items, str):
        return None
    
    try:
        genai.configure(api_key=api_key)
        # UPDATED: Using 'gemini-pro' which is more widely available on free tier than 'flash'
        model = genai.GenerativeModel('gemini-pro')
        
        # Prepare context
        headlines = [f"- {item['title']}" for item in news_items[:8]]
        headlines_text = "\n".join(headlines)
        
        prompt = f"""
        You are a financial analyst. Analyze the following recent news headlines for {asset_name}:
        
        {headlines_text}
        
        Provide a brief sentiment summary (max 50 words). 
        Start with exactly one of these labels: **BULLISH**, **BEARISH**, or **NEUTRAL**, followed by your reasoning.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Analysis Unavailable: {str(e)}"

# --- SIDEBAR & SETUP ---

with st.sidebar:
    st.header("⚙️ Terminal Settings")
    selected_asset = st.selectbox("Select Asset", list(ASSETS.keys()))
    asset_info = ASSETS[selected_asset]
    
    st.markdown("---")
    st.subheader("API Status")
    
    # Retrieve Keys
    news_key = get_api_key("news_api_key")
    fred_key = get_api_key("fred_api_key")
    google_key = get_api_key("google_api_key")

    # Status Indicators
    if news_key: st.success("NewsAPI: Connected")
    else: st.warning("NewsAPI: Missing")
        
    if fred_key: st.success("FRED: Connected")
    else: st.warning("FRED: Missing")
        
    if google_key: st.success("Gemini AI: Connected")
    else: st.info("Gemini AI: Not Set (Optional)")

    st.markdown("---")
    if st.button("Refresh Data"):
        st.cache_data.clear()

# --- MAIN DASHBOARD ---

st.title(f"📊 {selected_asset} Professional Terminal")

# 1. Fetch Data
stock_data = get_market_data(asset_info['ticker'])
macro_df = get_fred_data(fred_key, asset_info['fred_series'])

# 2. Main Visuals
if not stock_data.empty:
    # Handle MultiIndex columns (common with yfinance updates)
    if isinstance(stock_data.columns, pd.MultiIndex):
        close = stock_data['Close'].iloc[:, 0]
        high = stock_data['High'].iloc[:, 0]
        low = stock_data['Low'].iloc[:, 0]
        open_p = stock_data['Open'].iloc[:, 0]
    else:
        close, high, low, open_p = stock_data['Close'], stock_data['High'], stock_data['Low'], stock_data['Open']

    # KPI Metrics
    curr_price = close.iloc[-1]
    change = curr_price - close.iloc[-2]
    pct = (change / close.iloc[-2]) * 100
    volatility = close.pct_change().std() * (252**0.5) * 100
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"{curr_price:,.2f}", f"{change:,.2f} ({pct:.2f}%)")
    c2.metric("52W High", f"{high.max():,.2f}")
    c3.metric("52W Low", f"{low.min():,.2f}")
    c4.metric("Volatility (Ann.)", f"{volatility:.2f}%")

    # Charting
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=stock_data.index, open=open_p, high=high, low=low, close=close, name="Price"))
    
    # Add Macro overlay if available
    if isinstance(macro_df, pd.DataFrame) and not macro_df.empty:
        # Align macro data to stock dates
        macro_aligned = macro_df.reindex(stock_data.index, method='ffill')
        fig.add_trace(go.Scatter(
            x=macro_df.index, 
            y=macro_df['Value'], 
            name=asset_info['fred_label'], 
            line=dict(color='#d4af37', width=2), 
            yaxis="y2"
        ))

    fig.update_layout(
        height=500, 
        template="plotly_dark", 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)', 
        xaxis_rangeslider_visible=False, 
        yaxis2=dict(overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=1.02, x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("Market data unavailable. Please check the ticker or try again later.")

# 3. Macro Data Section (Placed Above News)
st.markdown("---")
st.subheader("🏛️ Macro Data Context")

if isinstance(macro_df, pd.DataFrame) and not macro_df.empty:
    m1, m2 = st.columns([1, 3])
    
    with m1:
        curr_macro = macro_df['Value'].iloc[-1]
        prev_macro = macro_df['Value'].iloc[-2]
        st.metric("Latest Reading", f"{curr_macro:,.2f}", f"{curr_macro - prev_macro:,.2f}")
        st.caption(f"**Indicator:** {asset_info['fred_label']}")
        
        if "Gold" in selected_asset:
            st.info("Gold often moves inversely to real yields.")
        elif "S&P" in selected_asset:
            st.info("Tracks Central Bank Liquidity.")
            
    with m2:
        st.line_chart(macro_df['Value'].tail(100))
elif fred_key:
    st.warning("Macro data unavailable for this asset/timeframe.")
else:
    st.info("Add FRED API Key to see macro charts.")

# 4. News & AI Sentiment Section
st.markdown("---")
st.subheader("📰 Market Intelligence & AI Analysis")

if news_key:
    news_data = get_news(news_key, asset_info['news_query'])
    
    # --- AI SENTIMENT BLOCK ---
    if google_key and isinstance(news_data, list):
        # We only run AI if we have valid news data
        with st.spinner(f"Gemini is analyzing {len(news_data)} news items..."):
            sentiment = get_ai_sentiment(google_key, selected_asset, news_data)
            if sentiment:
                st.markdown(f"""
                <div class="sentiment-box">
                    <h4>🤖 Gemini AI Sentiment Analysis</h4>
                    <p style="font-size: 1.1em; color: #e0e0e0;">{sentiment}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # --- NEWS GRID ---
    if isinstance(news_data, list):
        news_cols = st.columns(3)
        for i, article in enumerate(news_data[:6]): # Show top 6
            with news_cols[i % 3]:
                with st.container(border=True):
                    st.markdown(f"**[{article['title']}]({article['url']})**")
                    st.caption(f"{article['source']['name']} • {article['publishedAt'][:10]}")
    elif isinstance(news_data, str):
        st.error(f"News Error: {news_data}")
    else:
        st.info("No recent news found.")
else:
    st.info("Enter NewsAPI Key in secrets to view live headlines.")
