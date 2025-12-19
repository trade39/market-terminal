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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# --- APP CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Market Terminal Pro", page_icon="📈")

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .metric-container { background-color: #1e1e1e; border: 1px solid #333; padding: 10px; border-radius: 5px; text-align: center; }
    .sentiment-box { padding: 15px; border-radius: 5px; margin-bottom: 20px; border-left: 5px solid #d4af37; background-color: #262730; }
    
    /* Badges */
    .bullish { color: #00ff00; font-weight: bold; background-color: rgba(0, 255, 0, 0.1); padding: 2px 8px; border-radius: 4px; }
    .bearish { color: #ff4b4b; font-weight: bold; background-color: rgba(255, 75, 75, 0.1); padding: 2px 8px; border-radius: 4px; }
    .neutral { color: #cccccc; font-weight: bold; background-color: rgba(200, 200, 200, 0.1); padding: 2px 8px; border-radius: 4px; }
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
    if "api_keys" in st.secrets and key_name in st.secrets["api_keys"]: return st.secrets["api_keys"][key_name]
    if key_name in st.secrets: return st.secrets[key_name]
    return None

def flatten_dataframe(df):
    """Fixes MultiIndex issues from yfinance"""
    if df.empty: return df
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    return df

# --- 1. NEW ENGINES (GEX, VOL PROFILE, ML, KELLY) ---

def calculate_black_scholes_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

@st.cache_data(ttl=3600)
def get_gex_profile(opt_ticker, spot_price):
    try:
        tk = yf.Ticker(opt_ticker)
        exps = tk.options
        if len(exps) < 2: return None
        target_exp = exps[1]
        chain = tk.option_chain(target_exp)
        calls, puts = chain.calls, chain.puts
        r, T = 0.045, (datetime.strptime(target_exp, "%Y-%m-%d") - datetime.now()).days / 365.0
        gex_data = []
        strikes = sorted(list(set(calls['strike'].tolist() + puts['strike'].tolist())))
        for K in strikes:
            c = calls[calls['strike'] == K]
            p = puts[puts['strike'] == K]
            c_oi, c_iv = (c['openInterest'].iloc[0], c['impliedVolatility'].iloc[0]) if not c.empty else (0, 0.2)
            p_oi, p_iv = (p['openInterest'].iloc[0], p['impliedVolatility'].iloc[0]) if not p.empty else (0, 0.2)
            c_g, p_g = calculate_black_scholes_gamma(spot_price, K, T, r, c_iv), calculate_black_scholes_gamma(spot_price, K, T, r, p_iv)
            gex_data.append({"strike": K, "gex": (c_g * c_oi - p_g * p_oi) * spot_price * 100})
        return pd.DataFrame(gex_data)
    except: return None

def calculate_volume_profile(df, bins=50):
    if df.empty: return None, None, None
    df = flatten_dataframe(df)
    price_range = df['High'].max() - df['Low'].min()
    bin_size = price_range / bins
    df['PriceBin'] = ((df['Close'] - df['Low'].min()) // bin_size).astype(int)
    vol_profile = df.groupby('PriceBin')['Volume'].sum().reset_index()
    vol_profile['PriceLevel'] = df['Low'].min() + (vol_profile['PriceBin'] * bin_size)
    poc_price = vol_profile.loc[vol_profile['Volume'].idxmax(), 'PriceLevel']
    total_vol = vol_profile['Volume'].sum()
    vol_profile = vol_profile.sort_values('Volume', ascending=False)
    vol_profile['InVA'] = vol_profile['Volume'].cumsum() <= (total_vol * 0.70)
    return vol_profile.sort_values('PriceLevel'), poc_price, bin_size

@st.cache_data(ttl=3600)
def get_ml_prediction(ticker):
    try:
        df = flatten_dataframe(yf.download(ticker, period="2y", interval="1d", progress=False))
        if df.empty: return None
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df['RSI'] = calculate_rsi(df['Close'])
        df['Vol'] = df['Close'].pct_change().rolling(5).std()
        df = df.dropna()
        X, y = df[['RSI', 'Vol']], df['Target']
        model = RandomForestClassifier(n_estimators=50, max_depth=3).fit(X, y)
        return model.predict_proba(X.iloc[[-1]])[0][1]
    except: return 0.5

def calculate_kelly(prob_win, rr_ratio):
    return max(0, prob_win - (1 - prob_win) / rr_ratio) if rr_ratio > 0 else 0

# --- 2. EXISTING ENGINES (Calendar, Seasonality, etc.) ---

def parse_eco_value(val_str):
    if not isinstance(val_str, str): return None
    clean = val_str.replace('%', '').replace(',', '')
    mult = 1000 if 'K' in clean else 1000000 if 'M' in clean else 1
    clean = re.sub(r'[KMB]', '', clean)
    try: return float(clean) * mult
    except: return None

def analyze_event_impact(event_name, val_main, val_compare):
    v1, v2 = parse_eco_value(val_main), parse_eco_value(val_compare)
    if v1 is None or v2 is None: return "Neutral"
    usd_bullish_is_high = True # Simplified default
    delta = v1 - v2
    if abs(delta / (v2 if v2 else 1)) < 0.01: return "Mean Reverting"
    if delta > 0: return "USD Bullish" if usd_bullish_is_high else "USD Bearish"
    return "USD Bearish" if usd_bullish_is_high else "USD Bullish"

@st.cache_data(ttl=60)
def get_daily_data(ticker): return yf.download(ticker, period="10y", interval="1d", progress=False)

@st.cache_data(ttl=60)
def get_intraday_data(ticker): return yf.download(ticker, period="5d", interval="15m", progress=False)

@st.cache_data(ttl=3600)
def get_economic_calendar(api_key):
    if not api_key: return None
    url = "https://forex-factory-scraper1.p.rapidapi.com/get_calendar_details"
    now = datetime.now()
    params = {"year": now.year, "month": now.month, "day": now.day, "currency": "USD", "timezone": "GMT-05:00 Eastern Time (US & Canada)"}
    try:
        data = requests.get(url, headers={"x-rapidapi-key": api_key, "x-rapidapi-host": "forex-factory-scraper1.p.rapidapi.com"}, params=params).json()
        return data if isinstance(data, list) else data.get('data', [])
    except: return []

@st.cache_data(ttl=86400)
def get_macro_regime_data(api_key):
    if not api_key: return None
    try:
        fred = Fred(api_key=api_key)
        ffr = fred.get_series('FEDFUNDS').iloc[-1]
        cpi = fred.get_series('CPIAUCSL').pct_change(12).iloc[-1] * 100
        return {"bias": "BULLISH" if ffr - cpi < 0.5 else "BEARISH", "real_rate": ffr - cpi}
    except: return None

@st.cache_data(ttl=300)
def calculate_volatility_permission(ticker):
    try:
        df = flatten_dataframe(yf.download(ticker, period="1y", interval="1d", progress=False))
        if df.empty: return None
        tr = np.maximum(df['High'] - df['Low'], (df['High'] - df['Close'].shift(1)).abs())
        tr_pct = (tr / df['Close']) * 100
        forecast = np.exp(np.log(tr_pct).ewm(alpha=0.94).mean().iloc[-1])
        baseline = tr_pct.rolling(20).mean().iloc[-1]
        return {"forecast": forecast, "baseline": baseline, "is_go": forecast > baseline, "history": tr_pct, "signal": "TRADE PERMITTED" if forecast > baseline else "CAUTION"}
    except: return None

@st.cache_data(ttl=3600)
def get_options_pdf(opt_ticker):
    try:
        tk = yf.Ticker(opt_ticker)
        if len(tk.options) < 2: return None
        chain = tk.option_chain(tk.options[1])
        calls = chain.calls[(chain.calls['volume']>10) & (chain.calls['openInterest']>50)]
        if calls.empty: return None
        calls['mid'] = (calls['bid'] + calls['ask']) / 2
        df = calls[['strike', 'mid']].sort_values('strike')
        spline = UnivariateSpline(df['strike'], df['mid'], k=4, s=len(df)*2)
        x = np.linspace(df['strike'].min(), df['strike'].max(), 200)
        pdf = np.maximum(spline.derivative(n=2)(x), 0)
        return {"strikes": x, "pdf": pdf, "peak": x[np.argmax(pdf)], "date": tk.options[1]}
    except: return None

@st.cache_data(ttl=3600)
def get_seasonality_stats(daily_data):
    try:
        df = flatten_dataframe(daily_data.copy())
        df['Year'], df['Month'], df['Week'] = df.index.year, df.index.month, df.index.isocalendar().week
        df['Day'] = df.index.day_name()
        
        # Day Stats
        day_stats = df.groupby('Day')['Close'].mean().reindex(['Monday','Tuesday','Wednesday','Thursday','Friday'])
        
        # Month Stats
        month_stats = df.groupby('Month')['Close'].mean()
        
        return {"day": day_stats, "month": month_stats}
    except: return None

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    return 100 - (100 / (1 + gain/loss))

@st.cache_data(ttl=3600)
def generate_monte_carlo(stock_data, days=126, simulations=1000):
    try:
        close = flatten_dataframe(stock_data)['Close']
        returns = np.log(1 + close.pct_change())
        drift, stdev = returns.mean() - 0.5 * returns.var(), returns.std()
        paths = np.zeros((days, simulations))
        paths[0] = close.iloc[-1]
        for t in range(1, days):
            paths[t] = paths[t-1] * np.exp(drift + stdev * np.random.normal(0, 1, simulations))
        return pd.date_range(start=close.index[-1], periods=days), paths
    except: return None, None

@st.cache_data(ttl=3600)
def get_correlation_data():
    try:
        tickers = [v['ticker'] for k,v in ASSETS.items()]
        return flatten_dataframe(yf.download(tickers, period="1y", progress=False)['Close'])
    except: return pd.DataFrame()

# --- MAIN DASHBOARD ---
with st.sidebar:
    st.header("⚙️ Settings")
    selected_asset = st.selectbox("Select Asset", list(ASSETS.keys()))
    asset_info = ASSETS[selected_asset]
    news_key = get_api_key("news_api_key")
    fred_key = get_api_key("fred_api_key")
    rapid_key = get_api_key("rapidapi_key")
    if st.button("Refresh"): st.cache_data.clear()

st.title(f"📊 {selected_asset} Quantitative Terminal")

# Data
daily_data = flatten_dataframe(get_daily_data(asset_info['ticker']))
intraday_data = flatten_dataframe(get_intraday_data(asset_info['ticker']))
macro = get_macro_regime_data(fred_key)
vol = calculate_volatility_permission(asset_info['ticker'])
opts = get_options_pdf(asset_info['opt_ticker'])
eco = get_economic_calendar(rapid_key)
ml_prob = get_ml_prediction(asset_info['ticker'])
gex = get_gex_profile(asset_info['opt_ticker'], daily_data['Close'].iloc[-1])

# 1. OVERVIEW & ML
if not daily_data.empty:
    col1, col2, col3, col4 = st.columns(4)
    curr = daily_data['Close'].iloc[-1]
    col1.metric("Price", f"{curr:,.2f}", f"{(curr/daily_data['Close'].iloc[-2]-1)*100:.2f}%")
    
    if ml_prob:
        bias = "BULLISH" if ml_prob > 0.55 else "BEARISH" if ml_prob < 0.45 else "NEUTRAL"
        col2.markdown(f"<div class='metric-container'>🤖 ML Prediction<br><span class='{bias.lower()}'>{bias}</span> ({abs(ml_prob-0.5)*200:.0f}%)</div>", unsafe_allow_html=True)
    
    if vol and ml_prob:
        kelly = calculate_kelly(ml_prob, 1.5 if vol['is_go'] else 1.0) * 50 # Half Kelly
        col3.metric("Kelly Size", f"{kelly:.1f}%")

    if macro:
        col4.markdown(f"<div class='metric-container'>Macro Regime<br><span class='{macro['bias'].lower()}'>{macro['bias']}</span></div>", unsafe_allow_html=True)

# 2. PRICE & VOLUME ARCHITECTURE
st.markdown("---")
st.subheader("Price & Volume Architecture")
c_chart, c_prof = st.columns([3, 1])
with c_chart:
    fig = go.Figure(data=[go.Candlestick(x=daily_data.index, open=daily_data['Open'], high=daily_data['High'], low=daily_data['Low'], close=daily_data['Close'])])
    fig.update_layout(height=400, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
with c_prof:
    vp, poc, _ = calculate_volume_profile(intraday_data if not intraday_data.empty else daily_data.tail(30))
    if vp is not None:
        fig_vp = go.Figure(go.Bar(y=vp['PriceLevel'], x=vp['Volume'], orientation='h', marker_color=['#00ff00' if x else '#333' for x in vp['InVA']]))
        fig_vp.add_hline(y=poc, line_dash="dash", line_color="yellow")
        fig_vp.update_layout(height=400, template="plotly_dark", showlegend=False, xaxis_visible=False, title="Volume Profile")
        st.plotly_chart(fig_vp, use_container_width=True)


# 3. ECONOMIC CALENDAR
if eco:
    st.markdown("---")
    st.subheader("📅 Economic Events")
    cal_df = pd.DataFrame([{ "Event": e['event_name'], "Impact": e['impact'], "Analysis": analyze_event_impact(e['event_name'], e.get('actual',''), e.get('forecast',''))} for e in eco])
    st.dataframe(cal_df, use_container_width=True)

# 4. GAMMA EXPOSURE
if gex is not None:
    st.markdown("---")
    st.subheader("☢️ Gamma Exposure")
    fig_gex = go.Figure(go.Bar(x=gex['strike'], y=gex['gex'], marker_color=['#00ff00' if x>0 else '#ff4b4b' for x in gex['gex']]))
    fig_gex.add_vline(x=curr, line_dash="dot")
    fig_gex.update_layout(height=300, template="plotly_dark", title="Dealer Gamma Positioning")
    st.plotly_chart(fig_gex, use_container_width=True)


# 5. VOLATILITY & SEASONALITY
st.markdown("---")
c_vol, c_seas = st.columns(2)
with c_vol:
    st.subheader("⚡ Volatility")
    if vol:
        fig_v = go.Figure([go.Bar(x=vol['history'].tail(30).index, y=vol['history'].tail(30).values), go.Bar(x=[vol['history'].index[-1]+timedelta(days=1)], y=[vol['forecast']], marker_color='red')])
        fig_v.update_layout(height=300, template="plotly_dark", showlegend=False)
        st.plotly_chart(fig_v, use_container_width=True)

with c_seas:
    st.subheader("📅 Seasonality")
    seas = get_seasonality_stats(daily_data)
    if seas is not None:
        fig_s = go.Figure(go.Bar(x=seas['day'].index, y=seas['day'].values))
        fig_s.update_layout(height=300, template="plotly_dark", title="Avg Close by Day")
        st.plotly_chart(fig_s, use_container_width=True)

# 6. MONTE CARLO & OPTIONS
st.markdown("---")
c_mc, c_opt = st.columns(2)
with c_mc:
    st.subheader("🔮 Monte Carlo")
    dates, paths = generate_monte_carlo(daily_data)
    if dates is not None:
        fig_mc = go.Figure(go.Scatter(x=dates, y=np.mean(paths, axis=1), line=dict(color='#00ff00', dash='dash')))
        fig_mc.add_trace(go.Scatter(x=daily_data.index[-50:], y=daily_data['Close'].tail(50)))
        fig_mc.update_layout(height=300, template="plotly_dark", showlegend=False)
        st.plotly_chart(fig_mc, use_container_width=True)

with c_opt:
    st.subheader("🏦 Options Expectations")
    if opts:
        fig_o = go.Figure(go.Scatter(x=opts['strikes'], y=opts['pdf'], fill='tozeroy'))
        fig_o.add_vline(x=curr, line_dash="dot")
        fig_o.update_layout(height=300, template="plotly_dark", title=f"Target: {opts['peak']:.0f}")
        st.plotly_chart(fig_o, use_container_width=True)

# 7. EXECUTIVE SUMMARY
st.markdown("---")
bias_score = 0
reasons = []
if ml_prob > 0.55: bias_score += 1; reasons.append("ML Model Bullish")
elif ml_prob < 0.45: bias_score -= 1; reasons.append("ML Model Bearish")
if macro and macro['bias']=="BULLISH": bias_score+=1; reasons.append("Macro Bullish")
if vol and vol['is_go']: reasons.append("Vol Expansion Likely")
if opts and opts['peak'] > curr: bias_score+=1; reasons.append("Options Skew Bullish")

final = "BULLISH" if bias_score > 0 else "BEARISH" if bias_score < 0 else "NEUTRAL"
st.success(f"### Final Executive Bias: {final}")
for r in reasons: st.write(f"- {r}")
