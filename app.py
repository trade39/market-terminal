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
    .bullish { color: #00ff00; font-weight: bold; border: 1px solid #00ff00; padding: 2px 8px; border-radius: 4px; }
    .bearish { color: #ff4b4b; font-weight: bold; border: 1px solid #ff4b4b; padding: 2px 8px; border-radius: 4px; }
    .neutral { color: #cccccc; font-weight: bold; border: 1px solid #cccccc; padding: 2px 8px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTS & MAPPINGS ---
ASSETS = {
    "Gold (Comex)": {
        "ticker": "GC=F", 
        "news_query": "Gold Price market", 
        "fred_series": "DGS10", 
        "fred_label": "10-Year Treasury Yield",
        "correlation": "inverse" 
    }, 
    "S&P 500": {
        "ticker": "^GSPC", 
        "news_query": "S&P 500 market", 
        "fred_series": "WALCL", 
        "fred_label": "Fed Balance Sheet",
        "correlation": "direct" 
    },
    "NASDAQ": {
        "ticker": "^IXIC", 
        "news_query": "Nasdaq tech stocks", 
        "fred_series": "FEDFUNDS", 
        "fred_label": "Fed Funds Rate",
        "correlation": "inverse"
    },
    "EUR/USD": {
        "ticker": "EURUSD=X", 
        "news_query": "EURUSD forex", 
        "fred_series": "DEXUSEU", 
        "fred_label": "Exchange Rate Trend",
        "correlation": "direct"
    },
    "GBP/USD": {
        "ticker": "GBPUSD=X", 
        "news_query": "GBPUSD forex", 
        "fred_series": "DEXUSUK", 
        "fred_label": "Exchange Rate Trend",
        "correlation": "direct"
    }
}

DXY_TICKER = "DX-Y.NYB" # US Dollar Index

# --- HELPER FUNCTIONS ---

def get_api_key(key_name):
    if "api_keys" in st.secrets and key_name in st.secrets["api_keys"]:
        return st.secrets["api_keys"][key_name]
    if key_name in st.secrets:
        return st.secrets[key_name]
    return None

@st.cache_data(ttl=60)
def get_market_data(ticker):
    try:
        data = yf.download(ticker, period="2y", interval="1d", progress=False)
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

# --- NEW: CORRELATION DATA FETCH ---
@st.cache_data(ttl=3600)
def get_correlation_data():
    """Fetches 1y close data for ALL assets + DXY for correlation analysis"""
    tickers = {v['ticker']: k for k, v in ASSETS.items()}
    tickers[DXY_TICKER] = "US Dollar Index (DXY)"
    
    ticker_list = list(tickers.keys())
    
    try:
        data = yf.download(ticker_list, period="1y", interval="1d", progress=False)['Close']
        # Rename columns to friendly names
        data = data.rename(columns=tickers)
        return data
    except Exception as e:
        return pd.DataFrame()

# --- MONTE CARLO ---
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

# --- MACRO SIGNAL ANALYZER ---
def analyze_macro_signal(macro_df, correlation_type):
    if macro_df.empty: return "Neutral", "No Data"
    
    current_val = macro_df['Value'].iloc[-1]
    try:
        past_val = macro_df['Value'].iloc[-22]
    except IndexError:
        past_val = macro_df['Value'].iloc[0]
        
    delta = current_val - past_val
    trend_up = delta > 0
    
    signal = "Neutral"
    reason = ""
    
    if correlation_type == "inverse":
        if trend_up:
            signal = "BEARISH"
            reason = "Macro indicator is Rising (Inverse Correlation)"
        else:
            signal = "BULLISH"
            reason = "Macro indicator is Falling (Inverse Correlation)"
    else:
        if trend_up:
            signal = "BULLISH"
            reason = "Macro indicator is Rising (Direct Correlation)"
        else:
            signal = "BEARISH"
            reason = "Macro indicator is Falling (Direct Correlation)"
            
    return signal, f"{reason} | 30D Change: {delta:+.2f}"

# --- AI SENTIMENT ---
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
    except Exception: return None

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

stock_data = get_market_data(asset_info['ticker'])
macro_df = get_fred_data(fred_key, asset_info['fred_series'])

if not stock_data.empty:
    # --- 1. TOP METRICS & CHART ---
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

    curr = close.iloc[-1]
    pct = ((curr - close.iloc[-2]) / close.iloc[-2]) * 100
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"{curr:,.2f}", f"{pct:.2f}%")
    c2.metric("High", f"{high.max():,.2f}")
    c3.metric("Low", f"{low.min():,.2f}")
    c4.metric("Vol", f"{(close.pct_change().std()* (252**0.5)*100):.2f}%")

    # Main Chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=stock_data.index, open=open_p, high=high, low=low, close=close, name="Price"))
    
    if isinstance(macro_df, pd.DataFrame):
        macro_aligned = macro_df.reindex(stock_data.index, method='ffill')
        fig.add_trace(go.Scatter(x=macro_df.index, y=macro_df['Value'], name=asset_info['fred_label'], line=dict(color='#d4af37', width=2), yaxis="y2"))

    fig.update_layout(height=500, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_rangeslider_visible=False, yaxis2=dict(overlaying="y", side="right", showgrid=False))
    st.plotly_chart(fig, use_container_width=True)

# --- 2. MACRO CONTEXT & SIGNAL ---
st.markdown("---")
st.subheader("🏛️ Macro Data & Signal")

if isinstance(macro_df, pd.DataFrame) and not macro_df.empty:
    signal, reason = analyze_macro_signal(macro_df, asset_info['correlation'])
    color_class = "bullish" if signal == "BULLISH" else "bearish" if signal == "BEARISH" else "neutral"

    text_col1, text_col2, text_col3 = st.columns([1, 2, 1])
    with text_col1: st.markdown(f"#### Signal: <span class='{color_class}'>{signal}</span>", unsafe_allow_html=True)
    with text_col2: 
        st.markdown(f"**Driver:** {asset_info['fred_label']}")
        st.caption(f"{reason}")
    with text_col3: st.metric("Latest Macro Reading", f"{macro_df['Value'].iloc[-1]:.2f}")

    st.line_chart(macro_df['Value'].tail(100), color="#d4af37")
else:
    st.warning("Macro data unavailable (Check API Key).")

# --- 3. PRICE PREDICTION ---
st.markdown("---")
st.subheader("🔮 6-Month Volatility Drift Prediction")
st.caption("Monte Carlo Simulation - 10,000 Scenarios (100 Sample Paths Visualized)")

if not stock_data.empty:
    pred_dates, pred_paths = generate_monte_carlo(stock_data, simulations=10000)
    
    fig_pred = go.Figure()
    hist_slice = close.tail(90)
    fig_pred.add_trace(go.Scatter(x=hist_slice.index, y=hist_slice.values, name='Historical', line=dict(color='white', width=2)))
    
    for i in range(100): 
        fig_pred.add_trace(go.Scatter(x=pred_dates, y=pred_paths[:, i], mode='lines', line=dict(color='cyan', width=1), opacity=0.05, showlegend=False, hoverinfo='skip'))
        
    mean_path = np.mean(pred_paths, axis=1)
    fig_pred.add_trace(go.Scatter(x=pred_dates, y=mean_path, name='Avg Projection', line=dict(color='#00ff00', width=3, dash='dash')))
    fig_pred.update_layout(height=400, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=True)
    st.plotly_chart(fig_pred, use_container_width=True)

# --- NEW: 4. DXY CORRELATION ANALYSIS ---
st.markdown("---")
st.subheader("🌐 Inter-Market Correlation (DXY Focus)")
st.caption("Analyzing relationships between the US Dollar Index (DXY) and all tracked assets over the last 1 year.")

corr_data = get_correlation_data()

if not corr_data.empty:
    col_corr1, col_corr2 = st.columns([1, 1])
    
    # 1. Correlation Matrix Heatmap
    with col_corr1:
        corr_matrix = corr_data.corr()
        fig_heat = px.imshow(
            corr_matrix, 
            text_auto=True, 
            aspect="auto",
            color_continuous_scale='RdBu_r', # Red = Positive, Blue = Inverse
            title="Correlation Matrix (1 Year)"
        )
        fig_heat.update_layout(template="plotly_dark", height=400, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_heat, use_container_width=True)
        
    # 2. Normalized Time Series Chart
    with col_corr2:
        # Normalize to start at 0%
        normalized_data = (corr_data / corr_data.iloc[0] - 1) * 100
        fig_series = go.Figure()
        
        for col in normalized_data.columns:
            # Highlight DXY, fade others slightly
            width = 3 if "DXY" in col else 1.5
            opacity = 1 if "DXY" in col else 0.7
            
            fig_series.add_trace(go.Scatter(
                x=normalized_data.index,
                y=normalized_data[col],
                name=col,
                line=dict(width=width),
                opacity=opacity
            ))
            
        fig_series.update_layout(
            title="Relative Performance (1 Year % Change)",
            template="plotly_dark",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(title="% Change"),
            legend=dict(orientation="h", y=-0.2)
        )
        st.plotly_chart(fig_series, use_container_width=True)

    # 3. Dynamic Correlation Sentiment
    st.markdown("#### 🧠 Correlation Sentiment Analysis")
    
    dxy_name = "US Dollar Index (DXY)"
    if dxy_name in corr_matrix.columns:
        dxy_corrs = corr_matrix[dxy_name].drop(dxy_name)
        
        # Find strongest inverse and direct correlations
        strongest_inv = dxy_corrs.idxmin()
        val_inv = dxy_corrs.min()
        strongest_dir = dxy_corrs.idxmax()
        val_dir = dxy_corrs.max()
        
        # Generate Text Logic
        sentiment_text = f"""
        **Market Structure:**
        * **Strongest Inverse:** **{strongest_inv}** is trading against the Dollar (Corr: `{val_inv:.2f}`). 
            {'This is a classic "Safe Haven" vs "Cash" dynamic.' if 'Gold' in strongest_inv else 'This indicates capital flow rotation.'}
        * **Strongest Direct:** **{strongest_dir}** is moving with the Dollar (Corr: `{val_dir:.2f}`).
        
        **General Interpretation:**
        """
        
        if val_inv < -0.7:
            sentiment_text += " The market is currently driven by **strong Dollar flows**. Risk assets and currencies are reacting heavily to DXY strength/weakness."
        elif abs(val_inv) < 0.3 and abs(val_dir) < 0.3:
            sentiment_text += " Correlations are currently **weak (Decoupled)**. Assets are moving based on their own idiosyncratic news rather than broad macro flows."
        else:
            sentiment_text += " A mixed market regime. Watch the DXY key levels for broader direction."
            
        st.info(sentiment_text)

# --- 5. NEWS & AI ---
st.markdown("---")
st.subheader("📰 AI Market Intelligence")

if news_key:
    news_data = get_news(news_key, asset_info['news_query'])
    
    if google_key and isinstance(news_data, list):
        with st.spinner("Gemini is reading the news..."):
            sentiment = get_ai_sentiment(google_key, selected_asset, news_data)
            if sentiment:
                st.markdown(f"<div class='sentiment-box'><h4>🤖 AI Summary</h4><p>{sentiment}</p></div>", unsafe_allow_html=True)
    
    if isinstance(news_data, list):
        cols = st.columns(3)
        for i, art in enumerate(news_data[:3]):
            with cols[i]:
                st.markdown(f"**[{art['title']}]({art['url']})**")
                st.caption(f"{art['source']['name']} • {art['publishedAt'][:10]}")
else:
    st.info("Add NewsAPI Key for headlines.")
