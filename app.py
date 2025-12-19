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

# Custom CSS
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

# --- CONSTANTS & MAPPINGS ---
ASSETS = {
    "S&P 500": {"ticker": "^GSPC", "opt_ticker": "SPY", "news_query": "S&P 500"},
    "NASDAQ": {"ticker": "^IXIC", "opt_ticker": "QQQ", "news_query": "Nasdaq"},
    "Gold (Comex)": {"ticker": "GC=F", "opt_ticker": "GLD", "news_query": "Gold Price"},
    "EUR/USD": {"ticker": "EURUSD=X", "opt_ticker": "FXE", "news_query": "EURUSD"},
    "NVIDIA": {"ticker": "NVDA", "opt_ticker": "NVDA", "news_query": "Nvidia Stock"}
}

DXY_TICKER = "DX-Y.NYB"

# --- HELPER FUNCTIONS ---

def get_api_key(key_name):
    """Retrieve API keys from secrets.toml"""
    if "api_keys" in st.secrets and key_name in st.secrets["api_keys"]:
        return st.secrets["api_keys"][key_name]
    if key_name in st.secrets:
        return st.secrets[key_name]
    return None

# --- DATA FETCHING ---

@st.cache_data(ttl=60)
def get_daily_data(ticker):
    try:
        data = yf.download(ticker, period="2y", interval="1d", progress=False)
        return data
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60)
def get_intraday_data(ticker):
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

# --- INSTITUTIONAL ALGORITHMS ---

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
    except Exception:
        return None

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
        
        baseline_series = tr_pct.rolling(20).mean()
        baseline = baseline_series.iloc[-1]
        
        return {
            "forecast": forecast_tr,
            "baseline": baseline,
            "signal": "TRADE PERMITTED" if forecast_tr > baseline else "NO TRADE / CAUTION",
            "is_go": forecast_tr > baseline,
            "history": tr_pct,
            "baseline_history": baseline_series
        }
    except: return None

@st.cache_data(ttl=3600)
def get_options_pdf(opt_ticker):
    try:
        tk = yf.Ticker(opt_ticker)
        exps = tk.options
        if len(exps) < 2: return None
        target_exp = exps[1] 
        
        chain = tk.option_chain(target_exp)
        calls = chain.calls
        
        calls = calls[(calls['volume'] > 10) & (calls['openInterest'] > 50)]
        if calls.empty: return None
        
        calls['mid'] = (calls['bid'] + calls['ask']) / 2
        calls['price'] = np.where((calls['bid']==0), calls['lastPrice'], calls['mid'])
        
        df = calls[['strike', 'price']].sort_values('strike')
        
        spline = UnivariateSpline(df['strike'], df['price'], k=4, s=len(df)*2)
        strikes_smooth = np.linspace(df['strike'].min(), df['strike'].max(), 200)
        
        pdf = spline.derivative(n=2)(strikes_smooth)
        pdf = np.maximum(pdf, 0)
        
        peak_price = strikes_smooth[np.argmax(pdf)]
        
        return {"strikes": strikes_smooth, "pdf": pdf, "peak": peak_price, "date": target_exp}
    except: return None

# --- NEW: SEASONALITY FUNCTION ---
@st.cache_data(ttl=3600)
def get_day_of_week_stats(daily_data):
    """Calculates frequency of Weekly Highs/Lows by Day of Week"""
    try:
        df = daily_data.copy()
        
        # Handle MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(1, axis=1) # Flatten if needed for simple access
            
        # Add Week Identifier and Day Name
        df['Week_Num'] = df.index.to_period('W')
        df['Day_Name'] = df.index.day_name()
        
        # Filter incomplete weeks (counts must be > 1 to be a valid week)
        valid_weeks = df['Week_Num'].value_counts()
        valid_weeks = valid_weeks[valid_weeks >= 2].index
        df = df[df['Week_Num'].isin(valid_weeks)]
        
        # Find Day of Weekly High and Low
        weekly_groups = df.groupby('Week_Num')
        
        # Get the Day Name where the Max High occurred for each week
        high_days = df.loc[weekly_groups['High'].idxmax()]['Day_Name']
        
        # Get the Day Name where the Min Low occurred for each week
        low_days = df.loc[weekly_groups['Low'].idxmin()]['Day_Name']
        
        # Count frequencies
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        high_counts = high_days.value_counts().reindex(days_order, fill_value=0)
        low_counts = low_days.value_counts().reindex(days_order, fill_value=0)
        
        # Normalize to Percentages
        total_weeks = len(high_days)
        high_pct = (high_counts / total_weeks) * 100
        low_pct = (low_counts / total_weeks) * 100
        
        return high_pct, low_pct, total_weeks
    except Exception as e:
        return None, None, 0

# --- TECHNICAL CALCULATIONS ---

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
macro_regime = get_macro_regime_data(fred_key)
vol_forecast = calculate_volatility_permission(asset_info['ticker'])
options_pdf = get_options_pdf(asset_info['opt_ticker'])

# --- 1. OVERVIEW & MACRO REGIME ---
if not daily_data.empty:
    if isinstance(daily_data.columns, pd.MultiIndex): close, high, low, open_p = daily_data['Close'].iloc[:, 0], daily_data['High'].iloc[:, 0], daily_data['Low'].iloc[:, 0], daily_data['Open'].iloc[:, 0]
    else: close, high, low, open_p = daily_data['Close'], daily_data['High'], daily_data['Low'], daily_data['Open']

    curr = close.iloc[-1]
    pct = ((curr - close.iloc[-2]) / close.iloc[-2]) * 100
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"{curr:,.2f}", f"{pct:.2f}%")
    c2.metric("High", f"{high.max():,.2f}")
    c3.metric("Low", f"{low.min():,.2f}")
    
    if macro_regime:
        bias_color = "bullish" if macro_regime['bias'] == "BULLISH" else "bearish"
        c4.markdown(f"""
        <div style="text-align:center; padding:5px;">
            <div style="font-size:0.8em; color:gray;">Macro Regime</div>
            <span class='{bias_color}'>{macro_regime['bias']}</span>
            <div style="font-size:0.8em;">{macro_regime['real_rate']:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        c4.metric("Vol", f"{(close.pct_change().std()* (252**0.5)*100):.2f}%")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=daily_data.index, open=open_p, high=high, low=low, close=close, name="Price"))
    fig.update_layout(height=400, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# --- 2. VOLATILITY & INTRADAY ---
st.markdown("---")
st.subheader("⚡ Intraday & Volatility Permissions")

if not intraday_data.empty and vol_forecast:
    v1, v2 = st.columns([1, 3])
    with v1:
        st.markdown("**Daily Volatility Filter**")
        badge_class = "vol-go" if vol_forecast['is_go'] else "vol-stop"
        st.markdown(f"<span class='{badge_class}'>{vol_forecast['signal']}</span>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.9em; margin-top:5px;'>Expected: {vol_forecast['forecast']:.2f}%<br><span style='color:gray'>Base: {vol_forecast['baseline']:.2f}%</span></div>", unsafe_allow_html=True)

    with v2:
        hist_tr = vol_forecast['history'].tail(40)
        hist_base = vol_forecast['baseline_history'].tail(40)
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=hist_tr.index, y=hist_tr.values, name="Realized TR%", marker_color='#333333'))
        fig_vol.add_trace(go.Scatter(x=hist_base.index, y=hist_base.values, name="Baseline", line=dict(color='gray', dash='dot')))
        
        next_day = hist_tr.index[-1] + timedelta(days=1)
        f_color = '#00ff00' if vol_forecast['is_go'] else '#ff4b4b'
        fig_vol.add_trace(go.Bar(x=[next_day], y=[vol_forecast['forecast']], name="Forecast", marker_color=f_color))

        fig_vol.update_layout(title="Volatility Regime", yaxis_title="True Range %", template="plotly_dark", height=250, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_vol, use_container_width=True)

    # Intraday Metrics
    if isinstance(intraday_data.columns, pd.MultiIndex): i_close = intraday_data['Close'].iloc[:, 0]; i_vol = intraday_data['Volume'].iloc[:, 0]
    else: i_close = intraday_data['Close']; i_vol = intraday_data['Volume']
    
    df_vwap = calculate_vwap(pd.DataFrame({'High': intraday_data['High'].iloc[:,0] if isinstance(intraday_data.columns, pd.MultiIndex) else intraday_data['High'], 
                                         'Low': intraday_data['Low'].iloc[:,0] if isinstance(intraday_data.columns, pd.MultiIndex) else intraday_data['Low'], 
                                         'Close': i_close, 'Volume': i_vol}))
    current_vwap = df_vwap['VWAP'].iloc[-1]
    
    col_dash1, col_dash2, col_dash3 = st.columns(3)
    with col_dash1: st.metric("VWAP Bias", "BULLISH" if i_close.iloc[-1] > current_vwap else "BEARISH")
    with col_dash2: st.metric("Volume Trend", "Rising" if i_vol.tail(3).mean() > i_vol.mean() else "Falling")
    with col_dash3: st.metric("Gap %", f"{((open_p.iloc[-1] - close.iloc[-2])/close.iloc[-2]*100):.2f}%")

# --- 3. [NEW] SEASONALITY SECTION ---
st.markdown("---")
st.subheader("📅 Day-of-Week Seasonality Stats")

high_pct, low_pct, total_weeks = get_day_of_week_stats(daily_data)

if high_pct is not None:
    s_col1, s_col2 = st.columns([3, 1])
    
    with s_col1:
        # Clustered Bar Chart
        fig_season = go.Figure()
        
        # Series 1: Highs (Green)
        fig_season.add_trace(go.Bar(
            x=high_pct.index, 
            y=high_pct.values, 
            name='Weekly High Occurs',
            marker_color='#00ff00',
            opacity=0.7
        ))
        
        # Series 2: Lows (Red)
        fig_season.add_trace(go.Bar(
            x=low_pct.index, 
            y=low_pct.values, 
            name='Weekly Low Occurs',
            marker_color='#ff4b4b',
            opacity=0.7
        ))

        fig_season.update_layout(
            title=f"Distribution of Weekly Extremes (Last {total_weeks} Weeks)",
            yaxis_title="Frequency (%)",
            barmode='group', # Cluster the bars
            template="plotly_dark",
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_season, use_container_width=True)
        
    with s_col2:
        st.info("**What this shows:**\n\nThis analyzes on which Day of the Week the **High** or **Low** of the entire week typically lands.")
        
        # Identify Key patterns
        most_common_high = high_pct.idxmax()
        most_common_low = low_pct.idxmax()
        
        st.markdown(f"**Dominant High Day:**\n\n`{most_common_high}` ({high_pct.max():.1f}%)")
        st.markdown(f"**Dominant Low Day:**\n\n`{most_common_low}` ({low_pct.max():.1f}%)")
        
        if most_common_low in ['Monday', 'Tuesday'] and most_common_high in ['Thursday', 'Friday']:
            st.markdown("---")
            st.success("Pattern: **Classic Trend Week** (Low early, High late)")

# --- 4. OPTIONS HEATMAP ---
st.markdown("---")
st.subheader("🏦 Institutional Expectations")
if options_pdf:
    op_col1, op_col2 = st.columns([3, 1])
    with op_col1:
        fig_opt = go.Figure()
        fig_opt.add_trace(go.Scatter(x=options_pdf['strikes'], y=options_pdf['pdf'], fill='tozeroy', name='Implied Prob', line=dict(color='#00d4ff')))
        fig_opt.add_vline(x=curr, line_dash="dot", annotation_text="Spot")
        fig_opt.add_vline(x=options_pdf['peak'], line_dash="dash", line_color="#d4af37", annotation_text="Expected")
        fig_opt.update_layout(template="plotly_dark", height=350, title=f"Probability Distribution (Exp: {options_pdf['date']})", paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_opt, use_container_width=True)
    with op_col2:
        st.markdown(f"**Market Center:** `${options_pdf['peak']:.2f}`")
        st.markdown(f"**Skew:** `{'Bullish' if options_pdf['peak'] > curr else 'Bearish'}`")

# --- 5. PREDICTION & CONTEXT ---
st.markdown("---")
st.subheader("🔮 6-Month Projection & Global Context")
if not daily_data.empty:
    pred_dates, pred_paths = generate_monte_carlo(daily_data)
    
    pc1, pc2 = st.columns(2)
    with pc1:
        fig_pred = go.Figure()
        hist_slice = close.tail(90)
        fig_pred.add_trace(go.Scatter(x=hist_slice.index, y=hist_slice.values, name='History', line=dict(color='white')))
        fig_pred.add_trace(go.Scatter(x=pred_dates, y=np.mean(pred_paths, axis=1), name='Avg Path', line=dict(color='#00ff00', dash='dash')))
        fig_pred.update_layout(height=350, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', title="Monte Carlo Drift")
        st.plotly_chart(fig_pred, use_container_width=True)
    
    with pc2:
        corr_data = get_correlation_data()
        if not corr_data.empty:
            fig_heat = px.imshow(corr_data.corr(), text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
            fig_heat.update_layout(template="plotly_dark", height=350, paper_bgcolor='rgba(0,0,0,0)', title="Asset Correlations")
            st.plotly_chart(fig_heat, use_container_width=True)
