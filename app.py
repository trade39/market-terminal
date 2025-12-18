import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from fredapi import Fred
from newsapi import NewsApiClient
from datetime import datetime, timedelta

# --- APP CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Market Terminal", page_icon="📈")

# Custom CSS for "Terminal" feel
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .metric-container {
        background-color: #1e1e1e;
        border: 1px solid #333;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    /* Make metrics stand out against dark background */
    [data-testid="stMetricValue"] {
        color: #e0e0e0;
    }
    [data-testid="stMetricLabel"] {
        color: #a0a0a0;
    }
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
    Tries to get API key from Streamlit secrets.
    Checks both [api_keys] section and top-level for robustness.
    """
    # Check 1: Inside [api_keys] section (Best Practice)
    if "api_keys" in st.secrets and key_name in st.secrets["api_keys"]:
        return st.secrets["api_keys"][key_name]
    
    # Check 2: Top level (Fallback)
    if key_name in st.secrets:
        return st.secrets[key_name]
        
    return None

@st.cache_data(ttl=60)  # Cache market data for 1 minute
def get_market_data(ticker, period="1y", interval="1d"):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        return data
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(ttl=300) # Cache news for 5 minutes
def get_news(api_key, query):
    if not api_key:
        return None
    try:
        newsapi = NewsApiClient(api_key=api_key)
        # NewsAPI free tier limits: usually articles from last 30 days
        start_date = (datetime.now() - timedelta(days=28)).strftime('%Y-%m-%d')
        
        articles = newsapi.get_everything(
            q=query, 
            from_param=start_date, 
            language='en', 
            sort_by='publishedAt', 
            page_size=6
        )
        return articles['articles']
    except Exception as e:
        return f"Error: {str(e)}"

@st.cache_data(ttl=86400) # Cache FRED data for 24 hours (macro data is slow)
def get_fred_data(api_key, series_id):
    if not api_key:
        return None
    try:
        fred = Fred(api_key=api_key)
        data = fred.get_series(series_id)
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=['Value'])
        df.index.name = 'Date'
        # Filter to last 2 years to ensure context
        start_date = datetime.now() - timedelta(days=730)
        return df[df.index > start_date]
    except Exception as e:
        return f"Error fetching FRED data: {str(e)}"

# --- SIDEBAR & SETUP ---

with st.sidebar:
    st.header("⚙️ Terminal Settings")
    
    selected_asset = st.selectbox("Select Asset", list(ASSETS.keys()))
    asset_info = ASSETS[selected_asset]
    
    st.markdown("---")
    st.subheader("API Configuration")
    
    # 1. Try to load keys from Secrets
    news_key = get_api_key("news_api_key")
    fred_key = get_api_key("fred_api_key")

    # 2. Fallback to manual input if secrets are missing
    if not news_key:
        st.warning("NewsAPI Key not found in Secrets.")
        news_key = st.text_input("Enter NewsAPI.org Key", type="password")
    else:
        st.success("NewsAPI Key loaded from Secrets ✅")
    
    if not fred_key:
        st.warning("FRED API Key not found in Secrets.")
        fred_key = st.text_input("Enter FRED API Key", type="password")
    else:
        st.success("FRED API Key loaded from Secrets ✅")
    
    st.markdown("---")
    if st.button("Refresh Data / Clear Cache"):
        st.cache_data.clear()

# --- MAIN DASHBOARD ---

st.title(f"📊 {selected_asset} Professional Terminal")
st.markdown(f"**Ticker:** `{asset_info['ticker']}` | **Macro Indicator:** `{asset_info['fred_label']}`")

# 1. Fetch Market Data
stock_data = get_market_data(asset_info['ticker'])
macro_df = get_fred_data(fred_key, asset_info['fred_series'])

if not stock_data.empty:
    # 2. Calculate Top Level Metrics
    try:
        # Handle cases where yfinance returns MultiIndex columns
        if isinstance(stock_data.columns, pd.MultiIndex):
            close_series = stock_data['Close'].iloc[:, 0]
            high_val = stock_data['High'].iloc[:, 0].max()
            low_val = stock_data['Low'].iloc[:, 0].min()
            open_series = stock_data['Open'].iloc[:, 0]
            high_series = stock_data['High'].iloc[:, 0]
            low_series = stock_data['Low'].iloc[:, 0]
        else:
            close_series = stock_data['Close']
            high_val = stock_data['High'].max()
            low_val = stock_data['Low'].min()
            open_series = stock_data['Open']
            high_series = stock_data['High']
            low_series = stock_data['Low']

        current_price = close_series.iloc[-1]
        prev_close = close_series.iloc[-2]
        delta = current_price - prev_close
        pct_change = (delta / prev_close) * 100
        
        # Display Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Price", f"{current_price:,.2f}", f"{delta:,.2f} ({pct_change:.2f}%)")
        col2.metric("52W High", f"{high_val:,.2f}")
        col3.metric("52W Low", f"{low_val:,.2f}")
        
        # Calculate Volatility (Standard Deviation of last 30 days)
        volatility = close_series.pct_change().tail(30).std() * (252**0.5) * 100
        col4.metric("30D Volatility (Ann.)", f"{volatility:.2f}%")

        # 3. Main Chart Construction
        fig = go.Figure()

        # Candlestick Trace
        fig.add_trace(go.Candlestick(
            x=stock_data.index,
            open=open_series,
            high=high_series,
            low=low_series,
            close=close_series,
            name="Price"
        ))

        # Macro Data Overlay (Secondary Y-Axis)
        if isinstance(macro_df, pd.DataFrame) and not macro_df.empty:
            # Reindex macro data to match stock data range
            aligned_macro = macro_df.reindex(stock_data.index, method='ffill')
            
            fig.add_trace(go.Scatter(
                x=macro_df.index,
                y=macro_df['Value'],
                name=asset_info['fred_label'],
                line=dict(color='#d4af37', width=2), # Gold color for macro
                opacity=0.7,
                yaxis="y2"
            ))

        # Chart Layout
        fig.update_layout(
            height=600,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            yaxis=dict(
                title="Asset Price",
                showgrid=True, 
                gridcolor='#333'
            ),
            yaxis2=dict(
                title="Macro Data",
                overlaying="y",
                side="right",
                showgrid=False
            ),
            legend=dict(
                orientation="h", 
                yanchor="bottom", 
                y=1.02, 
                xanchor="right", 
                x=1
            )
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error processing market data: {e}")

else:
    st.error("Unable to load market data. The API might be down or the ticker is invalid.")

# --- NEW LAYOUT: MACRO DATA (FULL WIDTH) ---
st.markdown("---")
st.subheader("🏛️ Macro Data Context")

if fred_key:
    if isinstance(macro_df, pd.DataFrame) and not macro_df.empty:
        # Layout for Macro: Metric on Left, Chart on Right
        m_col1, m_col2 = st.columns([1, 3])
        
        with m_col1:
            curr_macro = macro_df['Value'].iloc[-1]
            prev_macro = macro_df['Value'].iloc[-2]
            st.metric(
                label="Latest Reading", 
                value=f"{curr_macro:,.2f}", 
                delta=f"{curr_macro - prev_macro:,.2f}"
            )
            st.markdown(f"**Indicator:** {asset_info['fred_label']}")
            
            # Context helper text
            if "Gold" in selected_asset:
                st.info("💡 Gold often moves inversely to real yields.")
            elif "S&P" in selected_asset:
                st.info("💡 Tracks Central Bank Liquidity.")
            elif "EUR" in selected_asset or "GBP" in selected_asset:
                st.info("💡 Driven by interest rate differentials.")
                
        with m_col2:
            st.line_chart(macro_df['Value'].tail(100))
    else:
        st.warning("Macro data unavailable for this selection or API key is invalid.")
else:
    st.info("⚠️ Enter FRED API Key to view macro-economic context.")

# --- NEW LAYOUT: MARKET INTELLIGENCE (FULL WIDTH) ---
st.markdown("---")
st.subheader("📰 Market Intelligence")

if news_key:
    news_data = get_news(news_key, asset_info['news_query'])
    if isinstance(news_data, list) and len(news_data) > 0:
        # Create a grid for news items (3 columns)
        news_cols = st.columns(3)
        for i, article in enumerate(news_data):
            # cycle through columns 0, 1, 2
            with news_cols[i % 3]: 
                with st.container(border=True):
                    st.markdown(f"**[{article['title']}]({article['url']})**")
                    st.caption(f"{article['source']['name']} • {article['publishedAt'][:10]}")
                    if article['description']:
                        st.write(article['description'][:100] + "...")
    elif isinstance(news_data, str):
        st.error(news_data)
    else:
        st.info("No recent news found for this asset.")
else:
    st.info("⚠️ Enter NewsAPI Key to view live headlines.")
