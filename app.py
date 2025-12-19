import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from newsapi import NewsApiClient
from datetime import datetime, timedelta
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture

# --- SAFE IMPORT FOR NLTK ---
# This prevents the app from crashing if NLTK is not installed
try:
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# --- APP CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Bloomberg Terminal Pro V4 (Inst. Grade)", page_icon="💹")

# --- BLOOMBERG TERMINAL STYLING (CSS) ---
st.markdown("""
<style>
    /* Main Background - True Black */
    .stApp { background-color: #000000; font-family: 'Courier New', Courier, monospace; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #111111; border-right: 1px solid #333; }
    
    /* Typography */
    h1, h2, h3, h4 { color: #ff9900 !important; font-family: 'Arial', sans-serif; text-transform: uppercase; letter-spacing: 1px; }
    p, div, span { color: #e0e0e0; }
    
    /* Metrics */
    div[data-testid="stMetricValue"] { color: #00e6ff !important; font-family: 'Courier New', monospace; font-weight: bold; }
    div[data-testid="stMetricLabel"] { color: #ff9900 !important; font-size: 0.8rem; }
    
    /* Components */
    .stDataFrame { border: 1px solid #333; }
    .terminal-box { border: 1px solid #333; background-color: #0a0a0a; padding: 15px; margin-bottom: 10px; }
    
    /* Badges */
    .bullish { color: #000; background-color: #00ff00; padding: 2px 6px; font-weight: bold; }
    .bearish { color: #000; background-color: #ff3333; padding: 2px 6px; font-weight: bold; }
    .neutral { color: #000; background-color: #cccccc; padding: 2px 6px; font-weight: bold; }
    .whale-alert { color: #fff; background-color: #9900ff; padding: 2px 6px; font-weight: bold; border: 1px solid #fff; }
    
    /* UI Elements */
    button { border-radius: 0px !important; border: 1px solid #ff9900 !important; color: #ff9900 !important; background: black !important; }
    .stSelectbox > div > div { border-radius: 0px; background-color: #111; color: white; border: 1px solid #444; }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTS ---
ASSETS = {
    "S&P 500": {"ticker": "^GSPC", "opt_ticker": "SPY"},
    "NASDAQ": {"ticker": "^IXIC", "opt_ticker": "QQQ"},
    "Gold": {"ticker": "GC=F", "opt_ticker": "GLD"},
    "NVIDIA": {"ticker": "NVDA", "opt_ticker": "NVDA"},
    "Bitcoin": {"ticker": "BTC-USD", "opt_ticker": None},
    "EUR/USD": {"ticker": "EURUSD=X", "opt_ticker": None}
}

# --- NLTK SETUP (Resilient) ---
@st.cache_resource
def download_nltk_data():
    if NLTK_AVAILABLE:
        try:
            nltk.download('vader_lexicon', quiet=True)
        except: pass

download_nltk_data()

# --- HELPER FUNCTIONS ---

def get_api_key(key_name):
    if "api_keys" in st.secrets and key_name in st.secrets["api_keys"]:
        return st.secrets["api_keys"][key_name]
    return st.secrets.get(key_name, None)

def flatten_dataframe(df):
    if df.empty: return df
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    return df

# --- 1. CORE ENGINES (ML, REGIME, ETC) ---

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
        
        model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
        model.fit(data[['Vol_5d', 'Mom_5d']], data['Target'])
        prob_up = model.predict_proba(data[['Vol_5d', 'Mom_5d']].iloc[[-1]])[0][1]
        return model, prob_up
    except: return None, 0.5

@st.cache_data(ttl=3600)
def get_market_regime(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        df = flatten_dataframe(df)
        if df.empty: return None
        
        data = df.dropna().copy()
        data['Returns'] = data['Close'].pct_change()
        data['Vol'] = data['Returns'].rolling(20).std()
        data = data.dropna()
        
        X = data[['Returns', 'Vol']].values
        gmm = GaussianMixture(n_components=3, random_state=42).fit(X)
        state = gmm.predict(X[[-1]])[0]
        
        # Sort states by volatility to label them correctly
        means = gmm.means_
        sorted_idx = np.argsort(means[:, 1])
        labels = {sorted_idx[0]: "LOW VOL (Trend)", sorted_idx[1]: "NEUTRAL (Chop)", sorted_idx[2]: "HIGH VOL (Crisis)"}
        
        desc = labels.get(state, "Unknown")
        color = "bullish" if "LOW VOL" in desc else "bearish" if "HIGH VOL" in desc else "neutral"
        return {"regime": desc, "color": color}
    except: return None

# --- 2. INSTITUTIONAL GREEKS (GEX + VANNA) ---

def calculate_greeks(S, K, T, r, sigma, opt_type):
    """Calculates Gamma and Vanna"""
    if T <= 0 or sigma <= 0: return 0, 0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Gamma
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Vanna
    vanna = -norm.pdf(d1) * d2 / sigma
    
    return gamma, vanna

@st.cache_data(ttl=300)
def get_institutional_greeks(opt_ticker):
    if not opt_ticker: return None, None, None
    try:
        tk = yf.Ticker(opt_ticker)
        
        # Safe history check
        hist = tk.history(period="1d")
        if hist.empty: return None, None, None
        spot = hist['Close'].iloc[-1]
        
        exps = tk.options
        if len(exps) < 2: return None, None, None
        target_exp = exps[1] # Avoid 0DTE
        
        chain = tk.option_chain(target_exp)
        calls, puts = chain.calls, chain.puts
        
        exp_date = datetime.strptime(target_exp, "%Y-%m-%d")
        T = max((exp_date - datetime.now()).days / 365.0, 0.001)
        r = 0.045
        
        greeks_data = []
        strikes = set(calls['strike']).union(set(puts['strike']))
        
        for K in strikes:
            if K < spot*0.8 or K > spot*1.2: continue # Focus on 20% OTM/ITM
            
            # Get Call/Put Data
            c_row = calls[calls['strike'] == K]
            p_row = puts[puts['strike'] == K]
            
            c_oi = c_row['openInterest'].iloc[0] if not c_row.empty else 0
            p_oi = p_row['openInterest'].iloc[0] if not p_row.empty else 0
            
            c_iv = c_row['impliedVolatility'].iloc[0] if not c_row.empty else 0.2
            p_iv = p_row['impliedVolatility'].iloc[0] if not p_row.empty else 0.2
            
            # Calc Greeks
            c_gamma, c_vanna = calculate_greeks(spot, K, T, r, c_iv, 'call')
            p_gamma, p_vanna = calculate_greeks(spot, K, T, r, p_iv, 'put')
            
            # Net Exposure
            net_gamma = (c_gamma * c_oi - p_gamma * p_oi) * spot * 100
            net_vanna = (c_vanna * c_oi - p_vanna * p_oi) * 1000 
            
            greeks_data.append({"strike": K, "gamma": net_gamma, "vanna": net_vanna})
            
        return pd.DataFrame(greeks_data), target_exp, spot
    except: return None, None, None

# --- 3. VRP & DARK POOLS ---

def calculate_vrp(daily_df):
    """Calculates Variance Risk Premium"""
    if daily_df.empty: return None
    try:
        # Realized Vol (20d)
        df = daily_df.copy()
        df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['rv'] = df['log_ret'].rolling(20).std() * np.sqrt(252) * 100
        
        # Implied Vol Proxy (Range Volatility)
        df['range_vol'] = (np.log(df['High'] / df['Low'])).rolling(20).mean() * np.sqrt(252) * 100 * 1.5
        
        curr_rv = df['rv'].iloc[-1]
        curr_iv_proxy = df['range_vol'].iloc[-1]
        
        vrp = curr_iv_proxy - curr_rv
        
        status = "EXPENSIVE (Sell Prem)" if vrp > 5 else "CHEAP (Buy Prem)" if vrp < 0 else "FAIR VALUE"
        color = "bearish" if vrp > 5 else "bullish" if vrp < 0 else "neutral"
        
        return {"rv": curr_rv, "iv_proxy": curr_iv_proxy, "vrp": vrp, "status": status, "color": color}
    except: return None

def detect_dark_pools(intraday_df):
    """Detects 15m volume spikes > 3x average"""
    if intraday_df.empty: return intraday_df
    try:
        df = intraday_df.copy()
        df['Vol_MA'] = df['Volume'].rolling(20).mean()
        df['Is_Whale'] = df['Volume'] > (df['Vol_MA'] * 3.0)
        return df
    except: return intraday_df

# --- 4. NEWS SENTIMENT ---

@st.cache_data(ttl=3600)
def get_smart_news(api_key, query="Finance"):
    if not api_key: return []
    # If NLTK failed to import, return basic news without sentiment scores
    if not NLTK_AVAILABLE:
        try:
            newsapi = NewsApiClient(api_key=api_key)
            articles = newsapi.get_top_headlines(category='business', language='en', country='us')['articles'][:5]
            return [{**art, "sentiment": "N/A", "score": 0, "color": "gray"} for art in articles]
        except: return []

    # If NLTK works, run sentiment analysis
    try:
        newsapi = NewsApiClient(api_key=api_key)
        articles = newsapi.get_top_headlines(category='business', language='en', country='us')['articles'][:5]
        
        sia = SentimentIntensityAnalyzer()
        results = []
        for art in articles:
            score = sia.polarity_scores(art['title'])['compound']
            sentiment = "POSITIVE" if score > 0.2 else "NEGATIVE" if score < -0.2 else "NEUTRAL"
            color = "#00ff00" if score > 0.2 else "#ff3333" if score < -0.2 else "gray"
            results.append({**art, "sentiment": sentiment, "score": score, "color": color})
            
        return results
    except: return []

# --- 5. DATA FETCHERS ---
@st.cache_data(ttl=60)
def get_data(ticker):
    try:
        daily = flatten_dataframe(yf.download(ticker, period="1y", interval="1d", progress=False))
        intraday = flatten_dataframe(yf.download(ticker, period="5d", interval="15m", progress=False))
        return daily, intraday
    except: return pd.DataFrame(), pd.DataFrame()

def terminal_chart_layout(fig, title="", height=350):
    fig.update_layout(
        title=dict(text=title, font=dict(color="#ff9900", family="Arial")),
        template="plotly_dark", paper_bgcolor="#000000", plot_bgcolor="#000000", height=height,
        margin=dict(l=40, r=40, t=40, b=40),
        font=dict(family="Courier New", color="#e0e0e0")
    )
    return fig

# --- SIDEBAR ---
with st.sidebar:
    st.title("COMMAND LINE")
    selected_asset = st.selectbox("TICKER", list(ASSETS.keys()))
    asset_info = ASSETS[selected_asset]
    
    # Keys
    news_key = get_api_key("news_api_key")
    rapid_key = get_api_key("rapidapi_key")
    
    st.info("Institutional Modules Loaded:\n- GEX/Vanna Surface\n- Dark Pool Scanner\n- VRP Calculator\n- NLP Sentiment")
    if not NLTK_AVAILABLE:
        st.warning("⚠️ NLTK not found. Sentiment disabled. Add 'nltk' to requirements.txt")
        
    if st.button("HARD REFRESH"): st.cache_data.clear()

# --- MAIN DASHBOARD ---
st.markdown(f"<h1 style='border-bottom: 2px solid #ff9900;'>{selected_asset} <span style='font-size:0.5em; color:white;'>INSTITUTIONAL V4.0</span></h1>", unsafe_allow_html=True)

daily_df, intraday_df = get_data(asset_info['ticker'])
ml_model, ml_prob = get_ml_prediction(asset_info['ticker'])
regime = get_market_regime(asset_info['ticker'])
vrp_data = calculate_vrp(daily_df)
intraday_df = detect_dark_pools(intraday_df)
greeks_df, greeks_exp, greeks_spot = get_institutional_greeks(asset_info['opt_ticker'])
news_data = get_smart_news(news_key)

# --- 1. HEADS UP DISPLAY (HUD) ---
if not daily_df.empty:
    curr = daily_df['Close'].iloc[-1]
    prev = daily_df['Close'].iloc[-2]
    pct = (curr / prev - 1) * 100
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("PRICE", f"{curr:,.2f}", f"{pct:.2f}%")
    
    # ML
    ml_bias = "BULLISH" if ml_prob > 0.55 else "BEARISH" if ml_prob < 0.45 else "NEUTRAL"
    c2.metric("AI SIGNAL", ml_bias, f"{abs(ml_prob-0.5)*200:.0f}% Conf")
    
    # REGIME
    if regime: c3.metric("REGIME", regime['regime'], delta=None)
    
    # VRP
    if vrp_data: 
        c4.metric("VOL PREMIUM", vrp_data['status'], f"Sprd: {vrp_data['vrp']:.1f}")
        
    # DARK POOL COUNT
    whale_count = intraday_df['Is_Whale'].sum() if not intraday_df.empty else 0
    c5.metric("DARK POOL BARS", f"{whale_count} Detected", "Last 5d")

# --- 2. INSTITUTIONAL CHARTING (DARK POOLS) ---
if not daily_df.empty:
    st.markdown("### 👁️ INSTITUTIONAL ORDER FLOW")
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    
    # Candle Chart (Intraday)
    fig.add_trace(go.Candlestick(x=intraday_df.index, open=intraday_df['Open'], high=intraday_df['High'],
                                 low=intraday_df['Low'], close=intraday_df['Close'], name='Price'), row=1, col=1)
    
    # Volume with Dark Pool Highlight
    colors = ['#9900ff' if x else '#333' for x in intraday_df['Is_Whale']]
    fig.add_trace(go.Bar(x=intraday_df.index, y=intraday_df['Volume'], marker_color=colors, name='Volume'), row=2, col=1)
    
    fig.update_layout(title="5-DAY MICROSTRUCTURE (Purple = >3x Vol / Whale Activity)", xaxis_rangeslider_visible=False)
    terminal_chart_layout(fig, height=500)
    st.plotly_chart(fig, use_container_width=True)

# --- 3. GREEKS SURFACE (GEX + VANNA) ---
st.markdown("---")
st.markdown("### 🏦 DEALER POSITIONING (GEX & VANNA)")

if greeks_df is not None:
    g1, g2 = st.columns(2)
    
    center = greeks_spot
    zoom_df = greeks_df[(greeks_df['strike'] > center * 0.9) & (greeks_df['strike'] < center * 1.1)]
    
    with g1:
        # GEX CHART
        fig_gex = go.Figure()
        colors = ['#00ff00' if x > 0 else '#ff3333' for x in zoom_df['gamma']]
        fig_gex.add_trace(go.Bar(x=zoom_df['strike'], y=zoom_df['gamma'], marker_color=colors))
        fig_gex.add_vline(x=center, line_dash="dot", line_color="white", annotation_text="SPOT")
        terminal_chart_layout(fig_gex, title=f"GAMMA EXPOSURE (Sticky vs Slippery)")
        st.plotly_chart(fig_gex, use_container_width=True)
        
    with g2:
        # VANNA CHART
        fig_van = go.Figure()
        colors_v = ['#00e6ff' if x > 0 else '#ff00ff' for x in zoom_df['vanna']]
        fig_van.add_trace(go.Bar(x=zoom_df['strike'], y=zoom_df['vanna'], marker_color=colors_v))
        fig_van.add_vline(x=center, line_dash="dot", line_color="white", annotation_text="SPOT")
        terminal_chart_layout(fig_van, title=f"VANNA EXPOSURE (Delta sensitivity to Vol)")
        st.plotly_chart(fig_van, use_container_width=True)
        
    st.caption(f"**Analysis:** Net Gamma: ${greeks_df['gamma'].sum()/1e6:.1f}M | Net Vanna: {greeks_df['vanna'].sum()/1e3:.1f}k")
    st.caption("*Positive Vanna (Blue) implies dealers buy Delta as Vol rises. Negative Vanna (Pink) implies dealers sell Delta as Vol rises.*")

else:
    if asset_info['opt_ticker'] is None:
         st.info("Options data unavailable for this asset class (Crypto/FX).")
    else:
         st.warning("Options Data Unavailable or Market Closed.")

# --- 4. NEWS & SENTIMENT ---
st.markdown("---")
n1, n2 = st.columns([2, 1])

with n1:
    st.markdown("### 📰 NLP SENTIMENT FEED")
    if news_data:
        for item in news_data:
            st.markdown(f"""
            <div style='border-left: 3px solid {item['color']}; padding-left: 10px; margin-bottom: 10px;'>
                <a href='{item['url']}' target='_blank' style='color: white; text-decoration: none; font-weight: bold;'>{item['title']}</a>
                <div style='font-size: 0.8em; color: gray; display: flex; justify-content: space-between;'>
                    <span>{item['source']}</span>
                    <span style='color:{item['color']}'>{item['sentiment']} ({item['score']})</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No news feed active or API Key missing.")

with n2:
    st.markdown("### ⚡ RISK SUMMARY")
    if vrp_data:
        st.markdown(f"**VRP STATUS:** <span class='{vrp_data['color']}'>{vrp_data['status']}</span>", unsafe_allow_html=True)
        st.progress(max(0, min(100, int(50 + vrp_data['vrp']*10))))
        st.caption("Lower = Buy Options (Long Vol) | Higher = Sell Options (Short Vol)")
        
    st.markdown("---")
    st.markdown("**WHALE ACTIVITY**")
    if whale_count > 5: st.markdown("<span class='whale-alert'>HIGH INST. ACTIVITY</span>", unsafe_allow_html=True)
    else: st.markdown("<span class='neutral'>NORMAL ACTIVITY</span>", unsafe_allow_html=True)

# --- 5. FOOTER ---
st.markdown("---")
st.markdown("<div style='text-align:center; color:gray; font-size:0.8em;'>INSTITUTIONAL GRADE TERMINAL V4.0 | POWERED BY PYTHON</div>", unsafe_allow_html=True)
