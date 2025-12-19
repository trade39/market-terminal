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
from scipy.interpolate import UnivariateSpline
from scipy.stats import norm

# --- APP CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Market Terminal Pro", page_icon="📈")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .metric-container { background-color: #1e1e1e; border: 1px solid #333; padding: 10px; border-radius: 5px; text-align: center; }
    .sentiment-box { padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 5px solid #d4af37; background-color: #262730; }
    
    /* Signal badges */
    .bullish { color: #00ff00; font-weight: bold; background-color: rgba(0, 255, 0, 0.1); padding: 2px 8px; border-radius: 4px; }
    .bearish { color: #ff4b4b; font-weight: bold; background-color: rgba(255, 75, 75, 0.1); padding: 2px 8px; border-radius: 4px; }
    .neutral { color: #cccccc; font-weight: bold; background-color: rgba(200, 200, 200, 0.1); padding: 2px 8px; border-radius: 4px; }
    
    /* Volatility Badges */
    .vol-go { background-color: rgba(0, 255, 0, 0.2); color: #00ff00; padding: 4px 10px; border-radius: 4px; font-weight: bold; border: 1px solid #00ff00; }
    .vol-stop { background-color: rgba(255, 75, 75, 0.2); color: #ff4b4b; padding: 4px 10px; border-radius: 4px; font-weight: bold; border: 1px solid #ff4b4b; }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTS ---
# Restored 'fred_series' and 'correlation' for the specific USD comparison
ASSETS = {
    "S&P 500": {
        "ticker": "^GSPC", "opt_ticker": "SPY", "news_query": "S&P 500", 
        "fred_series": "WALCL", "fred_label": "Fed Balance Sheet", "correlation": "direct"
    },
    "NASDAQ": {
        "ticker": "^IXIC", "opt_ticker": "QQQ", "news_query": "Nasdaq", 
        "fred_series": "T10Y2Y", "fred_label": "10Y-2Y Yield Curve", "correlation": "direct"
    },
    "Gold (Comex)": {
        "ticker": "GC=F", "opt_ticker": "GLD", "news_query": "Gold Price", 
        "fred_series": "DGS10", "fred_label": "US 10Y Real Yield", "correlation": "inverse"
    },
    "EUR/USD": {
        "ticker": "EURUSD=X", "opt_ticker": "FXE", "news_query": "EURUSD", 
        "fred_series": "DTWEXBGS", "fred_label": "DXY (Trade Weighted)", "correlation": "inverse"
    },
     "NVIDIA": {
        "ticker": "NVDA", "opt_ticker": "NVDA", "news_query": "Nvidia", 
        "fred_series": "DGS10", "fred_label": "US 10Y Yield", "correlation": "inverse"
    }
}

DXY_TICKER = "DX-Y.NYB"

# --- HELPER FUNCTIONS ---

def get_api_key(key_name):
    if "api_keys" in st.secrets and key_name in st.secrets["api_keys"]:
        return st.secrets["api_keys"][key_name]
    if key_name in st.secrets:
        return st.secrets[key_name]
    return None

# [PRESERVED] Daily Data
@st.cache_data(ttl=60)
def get_daily_data(ticker):
    try:
        data = yf.download(ticker, period="2y", interval="1d", progress=False)
        return data
    except Exception:
        return pd.DataFrame()

# [PRESERVED] Intraday Data
@st.cache_data(ttl=60)
def get_intraday_data(ticker):
    try:
        data = yf.download(ticker, period="5d", interval="15m", progress=False)
        return data
    except Exception:
        return pd.DataFrame()

# [PRESERVED] News
@st.cache_data(ttl=300)
def get_news(api_key, query):
    if not api_key: return None
    try:
        newsapi = NewsApiClient(api_key=api_key)
        start_date = (datetime.now() - timedelta(days=28)).strftime('%Y-%m-%d')
        articles = newsapi.get_everything(q=query, from_param=start_date, language='en', sort_by='relevancy', page_size=5)
        return articles['articles']
    except Exception as e:
        return str(e)

# [NEW + RESTORED] Macro Analysis (Both Global Regime AND Specific Driver)
@st.cache_data(ttl=86400)
def get_macro_analysis(api_key, specific_series_id, correlation_type):
    """
    Returns two layers of Macro:
    1. Global Regime: Fed Funds vs CPI (Real Rates)
    2. Specific Driver: Asset vs USD Indicator (e.g. Gold vs 10Y)
    """
    if not api_key: return None
    try:
        fred = Fred(api_key=api_key)
        
        # 1. GLOBAL REGIME (Concept 2: Real Rates)
        ffr = fred.get_series('FEDFUNDS').iloc[-1]
        cpi = fred.get_series('CPIAUCSL').pct_change(12).iloc[-1] * 100
        real_rate = ffr - cpi
        regime_bias = "BULLISH" if real_rate < 0.5 else "BEARISH"
        
        # 2. SPECIFIC DRIVER (Original Feature)
        specific_data = fred.get_series(specific_series_id)
        current_val = specific_data.iloc[-1]
        try: past_val = specific_data.iloc[-22] # 1 month ago
        except: past_val = specific_data.iloc[0]
        
        delta = current_val - past_val
        trend_up = delta > 0
        
        # Determine signal based on correlation (Direct vs Inverse)
        if correlation_type == "inverse":
            driver_signal = "BEARISH" if trend_up else "BULLISH"
            driver_text = "Rising (Headwind)" if trend_up else "Falling (Tailwind)"
        else:
            driver_signal = "BULLISH" if trend_up else "BEARISH"
            driver_text = "Rising (Tailwind)" if trend_up else "Falling (Headwind)"

        return {
            "regime": {"real_rate": real_rate, "bias": regime_bias, "ffr": ffr, "cpi": cpi},
            "driver": {"series": specific_data, "current": current_val, "signal": driver_signal, "desc": driver_text}
        }
    except Exception as e:
        return None

# [NEW] GARCH-Style Volatility Permission
@st.cache_data(ttl=300)
def calculate_volatility_permission(ticker):
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if df.empty: return None
        
        if isinstance(df.columns, pd.MultiIndex):
            high, low, close = df['High'].iloc[:, 0], df['Low'].iloc[:, 0], df['Close'].iloc[:, 0]
        else:
            high, low, close = df['High'], df['Low'], df['Close']
            
        prev_close = close.shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        tr_pct = (tr / close) * 100
        tr_pct = tr_pct.dropna()
        
        log_tr = np.log(tr_pct)
        forecast_log = log_tr.ewm(alpha=0.94).mean().iloc[-1]
        forecast_tr = np.exp(forecast_log)
        baseline = tr_pct.rolling(20).mean().iloc[-1]
        
        return {
            "forecast": forecast_tr,
            "baseline": baseline,
            "signal": "TRADE PERMITTED" if forecast_tr > baseline else "NO TRADE / CAUTION",
            "is_go": forecast_tr > baseline
        }
    except: return None

# [NEW] Options Probability
@st.cache_data(ttl=3600)
def get_options_pdf(opt_ticker):
    try:
        tk = yf.Ticker(opt_ticker)
        exps = tk.options
        if len(exps) < 2: return None
        target_exp = exps[1] 
        
        chain = tk.option_chain(target_exp)
        calls = chain.calls
        calls = calls[(calls['volume'] > 5) & (calls['openInterest'] > 20)]
        if calls.empty: return None
        
        calls['mid'] = (calls['bid'] + calls['ask']) / 2
        calls['price'] = np.where((calls['bid']==0), calls['lastPrice'], calls['mid'])
        df = calls[['strike', 'price']].sort_values('strike')
        
        spline = UnivariateSpline(df['strike'], df['price'], k=4, s=len(df)*3)
        strikes_smooth = np.linspace(df['strike'].min(), df['strike'].max(), 200)
        pdf = spline.derivative(n=2)(strikes_smooth)
        pdf = np.maximum(pdf, 0)
        peak_price = strikes_smooth[np.argmax(pdf)]
        
        return {"strikes": strikes_smooth, "pdf": pdf, "peak": peak_price, "date": target_exp}
    except: return None

# [PRESERVED] Intraday Helpers
def calculate_vwap(df):
    if df.empty: return df
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['TPV'] = df['TP'] * df['Volume']
    df['VWAP'] = df['TPV'].cumsum() / df['Volume'].cumsum()
    return df

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# [PRESERVED] Monte Carlo & Correlation
@st.cache_data(ttl=3600)
def get_correlation_data():
    tickers = {v['ticker']: k for k, v in ASSETS.items()}
    tickers[DXY_TICKER] = "US Dollar Index (DXY)"
    try:
        data = yf.download(list(tickers.keys()), period="1y", interval="1d", progress=False)['Close']
        data = data.rename(columns=tickers)
        return data
    except: return pd.DataFrame()

@st.cache_data(ttl=3600)
def generate_monte_carlo(stock_data, days=126, simulations=1000):
    if isinstance(stock_data.columns, pd.MultiIndex): close = stock_data['Close'].iloc[:, 0]
    else: close = stock_data['Close']
    log_returns = np.log(1 + close.pct_change())
    u, var = log_returns.mean(), log_returns.var()
    drift = u - (0.5 * var)
    stdev = log_returns.std()
    price_paths = np.zeros((days + 1, simulations))
    price_paths[0] = close.iloc[-1]
    daily_returns = np.exp(drift + stdev * np.random.normal(0, 1, (days, simulations)))
    for t in range(1, days + 1): price_paths[t] = price_paths[t - 1] * daily_returns[t - 1]
    return pd.date_range(start=close.index[-1], periods=days + 1, freq='B'), price_paths

# [PRESERVED] AI Sentiment
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
    st.caption(f"Keys: News {'✅' if news_key else '❌'} | FRED {'✅' if fred_key else '❌'}")
    if st.button("Refresh Data"): st.cache_data.clear()

# --- MAIN DASHBOARD ---
st.title(f"📊 {selected_asset} Pro Terminal")

# Fetch Data
daily_data = get_daily_data(asset_info['ticker'])
intraday_data = get_intraday_data(asset_info['ticker'])
macro_full = get_macro_analysis(fred_key, asset_info['fred_series'], asset_info['correlation']) # Restored Specifics
vol_forecast = calculate_volatility_permission(asset_info['ticker'])
options_pdf = get_options_pdf(asset_info['opt_ticker'])

# --- 1. OVERVIEW & MACRO REGIMES ---
if not daily_data.empty:
    if isinstance(daily_data.columns, pd.MultiIndex): 
        close, high, low, open_p = daily_data['Close'].iloc[:, 0], daily_data['High'].iloc[:, 0], daily_data['Low'].iloc[:, 0], daily_data['Open'].iloc[:, 0]
    else: 
        close, high, low, open_p = daily_data['Close'], daily_data['High'], daily_data['Low'], daily_data['Open']

    curr = close.iloc[-1]
    pct = ((curr - close.iloc[-2]) / close.iloc[-2]) * 100
    
    # Header Metrics with Dual Macro
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"{curr:,.2f}", f"{pct:.2f}%")
    
    # [NEW] Dual Macro Display
    if macro_full:
        # 1. Global Regime (Concept 2)
        regime = macro_full['regime']
        regime_color = "bullish" if regime['bias'] == "BULLISH" else "bearish"
        
        # 2. Specific Driver (Original Feature)
        driver = macro_full['driver']
        driver_color = "bullish" if driver['signal'] == "BULLISH" else "bearish"
        
        c2.markdown(f"""
        <div style="font-size:0.8em; color:gray;">Global Regime (Real Rates)</div>
        <span class='{regime_color}'>{regime['bias']}</span>
        <div style="font-size:0.7em;">{regime['real_rate']:.2f}%</div>
        """, unsafe_allow_html=True)
        
        c3.markdown(f"""
        <div style="font-size:0.8em; color:gray;">vs {asset_info['fred_label']}</div>
        <span class='{driver_color}'>{driver['signal']}</span>
        <div style="font-size:0.7em;">{driver['desc']}</div>
        """, unsafe_allow_html=True)
    else:
        c2.metric("High", f"{high.max():,.2f}")
        c3.metric("Low", f"{low.min():,.2f}")

    c4.metric("Vol (Annual)", f"{(close.pct_change().std()* (252**0.5)*100):.2f}%")

    # Main Chart
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=daily_data.index, open=open_p, high=high, low=low, close=close, name="Price"))
    # [RESTORED] Overlay specific USD Driver on Chart
    if macro_full:
        driver_series = macro_full['driver']['series']
        driver_series = driver_series[driver_series.index >= daily_data.index[0]] # Align dates
        fig.add_trace(go.Scatter(x=driver_series.index, y=driver_series.values, name=asset_info['fred_label'], line=dict(color='#d4af37', width=2), yaxis="y2"))
    
    fig.update_layout(height=400, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                     xaxis_rangeslider_visible=False, yaxis2=dict(overlaying="y", side="right", showgrid=False))
    st.plotly_chart(fig, use_container_width=True)

# --- 2. VOLATILITY & INTRADAY ---
st.markdown("---")
st.subheader("⚡ Intraday & Volatility Permissions")

if not intraday_data.empty:
    if vol_forecast:
        v1, v2 = st.columns([1, 3])
        with v1:
            badge_class = "vol-go" if vol_forecast['is_go'] else "vol-stop"
            st.markdown(f"**Action:** <span class='{badge_class}'>{vol_forecast['signal']}</span>", unsafe_allow_html=True)
        with v2:
            st.caption(f"GARCH Forecast: {vol_forecast['forecast']:.2f}% (Baseline: {vol_forecast['baseline']:.2f}%)")
            
    # Intraday Dashboard
    if isinstance(intraday_data.columns, pd.MultiIndex): i_close = intraday_data['Close'].iloc[:, 0]; i_vol = intraday_data['Volume'].iloc[:, 0]
    else: i_close = intraday_data['Close']; i_vol = intraday_data['Volume']
    
    df_vwap = calculate_vwap(pd.DataFrame({'High': intraday_data['High'].iloc[:,0] if isinstance(intraday_data.columns, pd.MultiIndex) else intraday_data['High'], 
                                         'Low': intraday_data['Low'].iloc[:,0] if isinstance(intraday_data.columns, pd.MultiIndex) else intraday_data['Low'], 
                                         'Close': i_close, 'Volume': i_vol}))
    current_vwap = df_vwap['VWAP'].iloc[-1]
    
    col_dash1, col_dash2, col_dash3 = st.columns(3)
    with col_dash1:
        bias = "BULLISH" if i_close.iloc[-1] > current_vwap else "BEARISH"
        st.metric("VWAP Bias", bias)
    with col_dash2:
        st.metric("Gap %", f"{((open_p.iloc[-1] - close.iloc[-2])/close.iloc[-2]*100):.2f}%")
    with col_dash3:
        st.metric("RSI (15m)", f"{calculate_rsi(i_close).iloc[-1]:.1f}")

# --- 3. OPTIONS HEATMAP ---
st.markdown("---")
st.subheader("🏦 Institutional Expectations")
if options_pdf:
    op_col1, op_col2 = st.columns([3, 1])
    with op_col1:
        fig_opt = go.Figure()
        fig_opt.add_trace(go.Scatter(x=options_pdf['strikes'], y=options_pdf['pdf'], fill='tozeroy', name='Implied Prob', line=dict(color='#00d4ff')))
        fig_opt.add_vline(x=curr, line_dash="dot", annotation_text="Spot")
        fig_opt.add_vline(x=options_pdf['peak'], line_dash="dash", line_color="#d4af37", annotation_text="Mkt Expectation")
        fig_opt.update_layout(template="plotly_dark", height=350, title=f"Probability Distribution (Exp: {options_pdf['date']})", paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_opt, use_container_width=True)
    with op_col2:
        skew = "Bullish" if options_pdf['peak'] > curr else "Bearish"
        st.markdown(f"**Target:** `${options_pdf['peak']:.2f}`\n\n**Skew:** `{skew}`")
else:
    st.warning("Options data unavailable.")

# --- 4. PREDICTION & CORRELATION ---
st.markdown("---")
c_pred, c_corr = st.columns(2)

with c_pred:
    st.subheader("🔮 Monte Carlo Projection")
    if not daily_data.empty:
        pred_dates, pred_paths = generate_monte_carlo(daily_data)
        fig_pred = go.Figure()
        hist_slice = close.tail(60)
        fig_pred.add_trace(go.Scatter(x=hist_slice.index, y=hist_slice.values, name='History', line=dict(color='white')))
        fig_pred.add_trace(go.Scatter(x=pred_dates, y=np.mean(pred_paths, axis=1), name='Avg Path', line=dict(color='#00ff00', dash='dash')))
        for i in range(20): fig_pred.add_trace(go.Scatter(x=pred_dates, y=pred_paths[:, i], mode='lines', line=dict(color='cyan', width=1), opacity=0.1, showlegend=False))
        fig_pred.update_layout(height=350, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pred, use_container_width=True)

with c_corr:
    st.subheader("🌐 USD Correlations")
    corr_data = get_correlation_data()
    if not corr_data.empty:
        fig_heat = px.imshow(corr_data.corr(), text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
        fig_heat.update_layout(template="plotly_dark", height=350, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_heat, use_container_width=True)

# --- 5. AI NEWS ---
if news_key:
    st.markdown("---")
    news_data = get_news(news_key, asset_info['news_query'])
    if google_key and isinstance(news_data, list):
        s = get_ai_sentiment(google_key, selected_asset, news_data)
        if s: st.markdown(f"<div class='sentiment-box'><strong>🤖 AI Sentiment:</strong> {s}</div>", unsafe_allow_html=True)
