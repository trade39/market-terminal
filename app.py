import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
import re
from fredapi import Fred
from newsapi import NewsApiClient
import google.generativeai as genai
from datetime import datetime, timedelta
from scipy.interpolate import UnivariateSpline
from scipy.stats import norm

# --- APP CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Bloomberg Terminal Pro", page_icon="💹")

# --- BLOOMBERG TERMINAL STYLING (CSS) ---
st.markdown("""
<style>
    /* Main Background - True Black */
    .stApp { background-color: #000000; font-family: 'Courier New', Courier, monospace; }
    
    /* Sidebar Background */
    section[data-testid="stSidebar"] { background-color: #111111; border-right: 1px solid #333; }
    
    /* Text Colors */
    h1, h2, h3, h4 { color: #ff9900 !important; font-family: 'Arial', sans-serif; text-transform: uppercase; letter-spacing: 1px; }
    p, div, span { color: #e0e0e0; }
    
    /* Metric Styling (The "Terminal" Data Look) */
    div[data-testid="stMetricValue"] { 
        color: #00e6ff !important; 
        font-family: 'Courier New', monospace;
        font-weight: bold;
    }
    div[data-testid="stMetricLabel"] { color: #ff9900 !important; font-size: 0.8rem; }
    
    /* Tables/Dataframes */
    .stDataFrame { border: 1px solid #333; }
    
    /* Custom Boxes */
    .terminal-box { 
        border: 1px solid #333; 
        background-color: #0a0a0a; 
        padding: 15px; 
        margin-bottom: 10px;
    }
    
    /* Signal badges - Neon Style */
    .bullish { color: #000000; background-color: #00ff00; padding: 2px 6px; font-weight: bold; border-radius: 0px; }
    .bearish { color: #000000; background-color: #ff3333; padding: 2px 6px; font-weight: bold; border-radius: 0px; }
    .neutral { color: #000000; background-color: #cccccc; padding: 2px 6px; font-weight: bold; border-radius: 0px; }
    
    /* Volatility Badges */
    .vol-go { border: 1px solid #00ff00; color: #00ff00; padding: 2px 8px; font-weight: bold; letter-spacing: 1px; }
    .vol-stop { border: 1px solid #ff3333; color: #ff3333; padding: 2px 8px; font-weight: bold; letter-spacing: 1px; }

    /* Remove rounded corners from buttons/inputs for industrial look */
    .stSelectbox > div > div { border-radius: 0px; background-color: #111; color: white; border: 1px solid #444; }
    button { border-radius: 0px !important; border: 1px solid #ff9900 !important; color: #ff9900 !important; background: black !important; }
    
    /* Divider */
    hr { margin: 1em 0; border: 0; border-top: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTS & MAPPINGS (GOLD MOVED TO TOP) ---
ASSETS = {
    "Gold (Comex)": {"ticker": "GC=F", "opt_ticker": "GLD", "news_query": "Gold Price"},
    "S&P 500": {"ticker": "^GSPC", "opt_ticker": "SPY", "news_query": "S&P 500"},
    "NASDAQ": {"ticker": "^IXIC", "opt_ticker": "QQQ", "news_query": "Nasdaq"},
    "EUR/USD": {"ticker": "EURUSD=X", "opt_ticker": "FXE", "news_query": "EURUSD"},
    "NVIDIA": {"ticker": "NVDA", "opt_ticker": "NVDA", "news_query": "Nvidia Stock"},
    "Bitcoin": {"ticker": "BTC-USD", "opt_ticker": "BITO", "news_query": "Bitcoin"}
}

DXY_TICKER = "DX-Y.NYB"

# --- HELPER FUNCTIONS ---

def get_api_key(key_name):
    if "api_keys" in st.secrets and key_name in st.secrets["api_keys"]:
        return st.secrets["api_keys"][key_name]
    if key_name in st.secrets:
        return st.secrets[key_name]
    return None

# --- SENTIMENT LOGIC ENGINE ---
def parse_eco_value(val_str):
    if not isinstance(val_str, str) or val_str == '': return None
    clean = val_str.replace('%', '').replace(',', '')
    multiplier = 1.0
    if 'K' in clean.upper():
        multiplier = 1000.0
        clean = clean.upper().replace('K', '')
    elif 'M' in clean.upper():
        multiplier = 1000000.0
        clean = clean.upper().replace('M', '')
    elif 'B' in clean.upper():
        multiplier = 1000000000.0
        clean = clean.upper().replace('B', '')
    try:
        return float(clean) * multiplier
    except:
        return None

def analyze_event_impact(event_name, val_main, val_compare, is_actual):
    v1 = parse_eco_value(val_main)
    v2 = parse_eco_value(val_compare)
    if v1 is None or v2 is None: return "Neutral"
    usd_logic = {
        "CPI": True, "PPI": True, "Non-Farm": True, "GDP": True, 
        "Sales": True, "Confidence": True, "Rates": True,
        "Unemployment": False, "Claims": False
    }
    is_direct = True 
    for key, val in usd_logic.items():
        if key.lower() in event_name.lower():
            is_direct = val
            break
    delta = v1 - v2
    pct_diff = 0
    if v2 != 0: pct_diff = abs(delta / v2)
    is_percentage_data = "%" in str(val_main)
    is_mean_reverting = False
    if is_percentage_data and abs(delta) < 0.05: is_mean_reverting = True 
    elif not is_percentage_data and pct_diff < 0.01: is_mean_reverting = True 
    if is_mean_reverting: return "Mean Reverting (Neutral)"
    if delta > 0: return "USD Bullish" if is_direct else "USD Bearish"
    elif delta < 0: return "USD Bearish" if is_direct else "USD Bullish"
    return "Mean Reverting"

# --- DATA FETCHING ---

@st.cache_data(ttl=60)
def get_daily_data(ticker):
    try:
        data = yf.download(ticker, period="10y", interval="1d", progress=False)
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

@st.cache_data(ttl=3600)
def get_economic_calendar(api_key):
    if not api_key: return None
    url = "https://forex-factory-scraper1.p.rapidapi.com/get_calendar_details"
    now = datetime.now()
    querystring = {
        "year": str(now.year), "month": str(now.month), "day": str(now.day),
        "currency": "USD", "event_name": "ALL", "timezone": "GMT-05:00 Eastern Time (US & Canada)", "time_format": "12h"
    }
    headers = {"x-rapidapi-host": "forex-factory-scraper1.p.rapidapi.com", "x-rapidapi-key": api_key}
    try:
        response = requests.get(url, headers=headers, params=querystring)
        data = response.json()
        if isinstance(data, list): return data
        elif 'data' in data: return data['data']
        return []
    except Exception as e:
        return []

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
            "forecast": forecast_tr, "baseline": baseline,
            "signal": "TRADE PERMITTED" if forecast_tr > baseline else "NO TRADE / CAUTION",
            "is_go": forecast_tr > baseline, "history": tr_pct, "baseline_history": baseline_series
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

# --- COMPREHENSIVE SEASONALITY STATS ---
@st.cache_data(ttl=3600)
def get_seasonality_stats(daily_data):
    try:
        df = daily_data.copy()
        if isinstance(df.columns, pd.MultiIndex): df = df.droplevel(1, axis=1)
        
        df['Year'] = df.index.year
        df['Month'] = df.index.month
        df['Week_Num'] = df.index.to_period('W')
        df['Day'] = df.index.day
        df['Day_Name'] = df.index.day_name()
        df['Week_of_Month'] = (df['Day'] - 1) // 7 + 1
        
        stats = {}
        valid_weeks = df['Week_Num'].value_counts()
        valid_weeks = valid_weeks[valid_weeks >= 2].index
        df_weeks = df[df['Week_Num'].isin(valid_weeks)]
        
        weekly_groups = df_weeks.groupby('Week_Num')
        high_days = df_weeks.loc[weekly_groups['High'].idxmax()]['Day_Name']
        low_days = df_weeks.loc[weekly_groups['Low'].idxmin()]['Day_Name']
        
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        stats['day_high'] = high_days.value_counts().reindex(days_order, fill_value=0) / len(high_days) * 100
        stats['day_low'] = low_days.value_counts().reindex(days_order, fill_value=0) / len(low_days) * 100
        
        monthly_groups = df.groupby(['Year', 'Month'])
        m_high_idx = monthly_groups['High'].idxmax()
        m_low_idx = monthly_groups['Low'].idxmin()
        week_highs = df.loc[m_high_idx]['Week_of_Month'].value_counts().sort_index()
        week_lows = df.loc[m_low_idx]['Week_of_Month'].value_counts().sort_index()
        stats['week_high'] = week_highs / week_highs.sum() * 100
        stats['week_low'] = week_lows / week_lows.sum() * 100
        
        yearly_groups = df.groupby(['Year'])
        y_high_idx = yearly_groups['High'].idxmax()
        y_low_idx = yearly_groups['Low'].idxmin()
        month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        m_high_counts = df.loc[y_high_idx].index.month_name().value_counts().reindex(month_names, fill_value=0)
        m_low_counts = df.loc[y_low_idx].index.month_name().value_counts().reindex(month_names, fill_value=0)
        stats['month_high'] = m_high_counts
        stats['month_low'] = m_low_counts
        return stats
    except Exception as e:
        return None

# --- TECHNICAL CALCULATIONS ---

def calculate_vwap(df):
    if df.empty: return df
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['TPV'] = df['TP'] * df['Volume']
    df['VWAP'] = df['TPV'].cumsum() / df['Volume'].cumsum()
    return df

@st.cache_data(ttl=3600)
def get_correlation_data():
    tickers = {v['ticker']: k for k, v in ASSETS.items()}
    tickers[DXY_TICKER] = "US Dollar (DXY)"
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

# --- PLOTLY THEME HELPERS ---
def terminal_chart_layout(fig, title="", height=350):
    fig.update_layout(
        title=dict(text=title, font=dict(color="#ff9900", family="Arial")),
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        height=height,
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(showgrid=True, gridcolor="#222", zerolinecolor="#222"),
        yaxis=dict(showgrid=True, gridcolor="#222", zerolinecolor="#222"),
        font=dict(family="Courier New", color="#e0e0e0")
    )
    return fig

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h3 style='color: #ff9900;'>COMMAND LINE</h3>", unsafe_allow_html=True)
    selected_asset = st.selectbox("SEC / Ticker", list(ASSETS.keys()))
    asset_info = ASSETS[selected_asset]
    st.markdown("---")
    news_key = get_api_key("news_api_key")
    fred_key = get_api_key("fred_api_key")
    google_key = get_api_key("google_api_key")
    rapid_key = get_api_key("rapidapi_key")
    
    st.markdown(f"""
    <div style='font-size:0.8em; color:gray; font-family:Courier New;'>
    API STATUS:<br>
    NEWS: {'[OK]' if news_key else '[FAIL]'}<br>
    FRED: {'[OK]' if fred_key else '[FAIL]'}<br>
    RAPID: {'[OK]' if rapid_key else '[FAIL]'}
    </div>
    """, unsafe_allow_html=True)
    
    if st.button(">> REFRESH DATA"): st.cache_data.clear()

# --- MAIN DASHBOARD ---
st.markdown(f"<h1 style='border-bottom: 2px solid #ff9900;'>{selected_asset} <span style='font-size:0.5em; color:white;'>TERMINAL PRO</span></h1>", unsafe_allow_html=True)

# Fetch Data
daily_data = get_daily_data(asset_info['ticker'])
intraday_data = get_intraday_data(asset_info['ticker'])
macro_regime = get_macro_regime_data(fred_key)
vol_forecast = calculate_volatility_permission(asset_info['ticker'])
options_pdf = get_options_pdf(asset_info['opt_ticker'])
eco_events = get_economic_calendar(rapid_key)

# --- 1. OVERVIEW & MACRO REGIME ---
if not daily_data.empty:
    if isinstance(daily_data.columns, pd.MultiIndex): close, high, low, open_p = daily_data['Close'].iloc[:, 0], daily_data['High'].iloc[:, 0], daily_data['Low'].iloc[:, 0], daily_data['Open'].iloc[:, 0]
    else: close, high, low, open_p = daily_data['Close'], daily_data['High'], daily_data['Low'], daily_data['Open']

    curr = close.iloc[-1]
    pct = ((curr - close.iloc[-2]) / close.iloc[-2]) * 100
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LAST PX", f"{curr:,.2f}", f"{pct:.2f}%")
    c2.metric("HIGH", f"{high.max():,.2f}")
    c3.metric("LOW", f"{low.min():,.2f}")
    
    if macro_regime:
        bias_color = "bullish" if macro_regime['bias'] == "BULLISH" else "bearish"
        c4.markdown(f"""
        <div class='terminal-box' style="text-align:center; padding:5px;">
            <div style="font-size:0.8em; color:#ff9900;">MACRO REGIME</div>
            <span class='{bias_color}'>{macro_regime['bias']}</span>
            <div style="font-size:0.8em; margin-top:5px; color:#aaa;">REAL RATE: {macro_regime['real_rate']:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        c4.metric("VOLATILITY", f"{(close.pct_change().std()* (252**0.5)*100):.2f}%")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=daily_data.index, open=open_p, high=high, low=low, close=close, name="Price"))
    terminal_chart_layout(fig, height=400)
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# --- 2. ECONOMIC CALENDAR ---
st.markdown("---")
st.markdown("### 📅 ECONOMIC EVENTS (USD)")

if eco_events:
    cal_data = []
    for event in eco_events:
        impact = event.get('impact', 'Low')
        name = event.get('event_name', 'Unknown')
        actual = event.get('actual', '')
        forecast = event.get('forecast', '')
        previous = event.get('previous', '')
        
        context_msg = ""
        if actual and actual != '':
            bias = analyze_event_impact(name, actual, forecast, is_actual=True)
            if forecast and forecast != '': context_msg = f"{actual} vs {forecast} ({bias})"
            else: context_msg = f"{actual} (No Fcst)"
        elif forecast and forecast != '':
            bias = analyze_event_impact(name, forecast, previous, is_actual=False)
            if previous and previous != '': context_msg = f"Fcst: {forecast} vs Prev: {previous} ({bias})"
            else: context_msg = f"Fcst: {forecast}"
        else: context_msg = "Waiting..."

        cal_data.append({"TIME": event.get('time', 'N/A'), "EVENT": name, "IMPACT": impact, "ANALYSIS": context_msg})
    
    df_cal = pd.DataFrame(cal_data)
    if not df_cal.empty:
        def highlight_cols(val):
            if 'High' in str(val): return 'color: #ff3333; font-weight: bold;'
            if 'Bullish' in str(val): return 'color: #00ff00;'
            if 'Bearish' in str(val): return 'color: #ff3333;'
            if 'Mean Reverting' in str(val): return 'color: #cccccc;'
            return ''
        st.dataframe(df_cal.style.map(highlight_cols), use_container_width=True, hide_index=True)
    else: st.info("NO USD EVENTS SCHEDULED.")
else:
    if not rapid_key: st.error("MISSING API KEY: rapidapi_key")
    else: st.info("NO DATA FOUND.")

# --- 3. VOLATILITY & INTRADAY ---
st.markdown("---")
st.markdown("### ⚡ INTRADAY & VOLATILITY")

if not intraday_data.empty and vol_forecast:
    v1, v2 = st.columns([1, 3])
    with v1:
        st.markdown("**VOL FILTER**")
        badge_class = "vol-go" if vol_forecast['is_go'] else "vol-stop"
        st.markdown(f"<div style='margin:10px 0;'><span class='{badge_class}'>{vol_forecast['signal']}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.9em; font-family:monospace;'>EXP: {vol_forecast['forecast']:.2f}%<br><span style='color:gray'>BASE: {vol_forecast['baseline']:.2f}%</span></div>", unsafe_allow_html=True)
    with v2:
        hist_tr = vol_forecast['history'].tail(40)
        hist_base = vol_forecast['baseline_history'].tail(40)
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=hist_tr.index, y=hist_tr.values, name="Realized", marker_color='#333333'))
        fig_vol.add_trace(go.Scatter(x=hist_base.index, y=hist_base.values, name="Baseline", line=dict(color='gray', dash='dot')))
        next_day = hist_tr.index[-1] + timedelta(days=1)
        f_color = '#00ff00' if vol_forecast['is_go'] else '#ff3333'
        fig_vol.add_trace(go.Bar(x=[next_day], y=[vol_forecast['forecast']], name="Forecast", marker_color=f_color))
        terminal_chart_layout(fig_vol, title="VOLATILITY REGIME", height=250)
        st.plotly_chart(fig_vol, use_container_width=True)
    
    if isinstance(intraday_data.columns, pd.MultiIndex): i_close = intraday_data['Close'].iloc[:, 0]; i_vol = intraday_data['Volume'].iloc[:, 0]
    else: i_close = intraday_data['Close']; i_vol = intraday_data['Volume']
    df_vwap = calculate_vwap(pd.DataFrame({'High': intraday_data['High'].iloc[:,0] if isinstance(intraday_data.columns, pd.MultiIndex) else intraday_data['High'], 'Low': intraday_data['Low'].iloc[:,0] if isinstance(intraday_data.columns, pd.MultiIndex) else intraday_data['Low'], 'Close': i_close, 'Volume': i_vol}))
    current_vwap = df_vwap['VWAP'].iloc[-1]
    col_dash1, col_dash2, col_dash3 = st.columns(3)
    with col_dash1: st.metric("VWAP BIAS", "BULLISH" if i_close.iloc[-1] > current_vwap else "BEARISH")
    with col_dash2: st.metric("VOL TREND", "RISING" if i_vol.tail(3).mean() > i_vol.mean() else "FALLING")
    with col_dash3: st.metric("GAP %", f"{((open_p.iloc[-1] - close.iloc[-2])/close.iloc[-2]*100):.2f}%")

# --- 4. TIME-BASED SEASONALITY ---
st.markdown("---")
st.markdown("### 📅 SEASONALITY ANALYSIS")
season_stats = get_seasonality_stats(daily_data)

if season_stats:
    tab1, tab2, tab3 = st.tabs(["DAY OF WEEK", "WEEK OF MONTH", "MONTH OF YEAR"])
    
    with tab1:
        fig_d = go.Figure()
        fig_d.add_trace(go.Bar(x=season_stats['day_high'].index, y=season_stats['day_high'].values, name='Highs', marker_color='#00ff00'))
        fig_d.add_trace(go.Bar(x=season_stats['day_low'].index, y=season_stats['day_low'].values, name='Lows', marker_color='#ff3333'))
        terminal_chart_layout(fig_d, title="WEEKLY EXTREMES DISTRIBUTION")
        st.plotly_chart(fig_d, use_container_width=True)
        
    with tab2:
        fig_w = go.Figure()
        fig_w.add_trace(go.Bar(x=["Week 1", "Week 2", "Week 3", "Week 4", "Week 5"], y=season_stats['week_high'].values, name='Monthly High', marker_color='#00ff00'))
        fig_w.add_trace(go.Bar(x=["Week 1", "Week 2", "Week 3", "Week 4", "Week 5"], y=season_stats['week_low'].values, name='Monthly Low', marker_color='#ff3333'))
        terminal_chart_layout(fig_w, title="MONTHLY EXTREMES BY WEEK")
        st.plotly_chart(fig_w, use_container_width=True)

    with tab3:
        fig_m = go.Figure()
        fig_m.add_trace(go.Bar(x=season_stats['month_high'].index, y=season_stats['month_high'].values, name='Yearly High', marker_color='#00ff00'))
        fig_m.add_trace(go.Bar(x=season_stats['month_low'].index, y=season_stats['month_low'].values, name='Yearly Low', marker_color='#ff3333'))
        terminal_chart_layout(fig_m, title="YEARLY EXTREMES BY MONTH")
        st.plotly_chart(fig_m, use_container_width=True)

# --- 5. INSTITUTIONAL EXPECTATIONS ---
st.markdown("---")
st.markdown("### 🏦 INSTITUTIONAL FLOW & PREDICTION")
if options_pdf:
    op_col1, op_col2 = st.columns([3, 1])
    with op_col1:
        fig_opt = go.Figure()
        fig_opt.add_trace(go.Scatter(x=options_pdf['strikes'], y=options_pdf['pdf'], fill='tozeroy', name='Implied Prob', line=dict(color='#00d4ff')))
        fig_opt.add_vline(x=curr, line_dash="dot", line_color="white", annotation_text="SPOT")
        fig_opt.add_vline(x=options_pdf['peak'], line_dash="dash", line_color="#ff9900", annotation_text="EXP")
        terminal_chart_layout(fig_opt, title=f"OPTIONS GAMMA (EXP: {options_pdf['date']})")
        st.plotly_chart(fig_opt, use_container_width=True)
    with op_col2:
        st.markdown(f"<div class='terminal-box'><b>TARGET:</b><br><span style='color:#00e6ff; font-size:1.2em'>${options_pdf['peak']:.2f}</span></div>", unsafe_allow_html=True)
        skew_txt = 'BULLISH' if options_pdf['peak'] > curr else 'BEARISH'
        skew_col = '#00ff00' if skew_txt == 'BULLISH' else '#ff3333'
        st.markdown(f"<div class='terminal-box'><b>SKEW:</b><br><span style='color:{skew_col}; font-weight:bold'>{skew_txt}</span></div>", unsafe_allow_html=True)

pred_dates, pred_paths = generate_monte_carlo(daily_data)
pc1, pc2 = st.columns(2)
with pc1:
    fig_pred = go.Figure()
    hist_slice = close.tail(90)
    fig_pred.add_trace(go.Scatter(x=hist_slice.index, y=hist_slice.values, name='History', line=dict(color='white')))
    fig_pred.add_trace(go.Scatter(x=pred_dates, y=np.mean(pred_paths, axis=1), name='Avg Path', line=dict(color='#ff9900', dash='dash')))
    terminal_chart_layout(fig_pred, title="MONTE CARLO SIMULATION")
    st.plotly_chart(fig_pred, use_container_width=True)
with pc2:
    corr_data = get_correlation_data()
    if not corr_data.empty:
        fig_heat = px.imshow(corr_data.corr(), text_auto=True, color_continuous_scale='RdBu_r', aspect="auto")
        terminal_chart_layout(fig_heat, title="ASSET CORRELATION MATRIX")
        st.plotly_chart(fig_heat, use_container_width=True)

# --- 6. CONCLUSION ---
st.markdown("---")
st.markdown("### 🏁 EXECUTIVE SUMMARY")

# Determine Final Bias
bias_score = 0
reasons = []

# 1. Macro
if macro_regime:
    if macro_regime['bias'] == "BULLISH": 
        bias_score += 1
        reasons.append("MACRO: Real Rates Supportive.")
    else: 
        bias_score -= 1
        reasons.append("MACRO: Real Rates Restrictive.")

# 2. Volatility
if vol_forecast and vol_forecast['is_go']:
    reasons.append("VOLATILITY: Forecast supports expansion.")
else:
    reasons.append("VOLATILITY: Forecast suggests chop/consolidation.")

# 3. Options
if options_pdf:
    if options_pdf['peak'] > curr:
        bias_score += 1
        reasons.append(f"OPTIONS: Positioning for higher prices (${options_pdf['peak']:.0f}).")
    else:
        bias_score -= 1
        reasons.append(f"OPTIONS: Positioning for lower prices (${options_pdf['peak']:.0f}).")

# Final Output
final_color = "neutral"
final_text = "NEUTRAL / MIXED"
if bias_score > 0: 
    final_text = "BULLISH BIAS"
    final_color = "bullish"
elif bias_score < 0: 
    final_text = "BEARISH BIAS"
    final_color = "bearish"

st.markdown(f"""
<div class='terminal-box'>
    <h2 style="text-align:center; margin-top:0; color: #ff9900;">{selected_asset} OUTLOOK</h2>
    <div style='text-align:center; margin-bottom:15px;'><span class='{final_color}' style='font-size:1.5em;'>{final_text}</span></div>
    <hr>
    <ul style='font-family:Courier New; color:#e0e0e0;'>
        {''.join([f'<li>{r}</li>' for r in reasons])}
    </ul>
    <p style="text-align:center; font-size:0.8em; color:#666;">*Generated algorithmically based on Macro, Volatility, and Option Flows.</p>
</div>
""", unsafe_allow_html=True)
