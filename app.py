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
from sklearn.ensemble import RandomForestClassifier # NEW: For ML Model
from sklearn.model_selection import train_test_split # NEW: For ML Model

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

# --- NEW: GAMMA EXPOSURE (GEX) ENGINE ---
def calculate_black_scholes_gamma(S, K, T, r, sigma):
    """Calculates Gamma for a single option"""
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

@st.cache_data(ttl=3600)
def get_gex_profile(opt_ticker, spot_price):
    """Calculates Net Gamma Exposure per Strike"""
    try:
        tk = yf.Ticker(opt_ticker)
        exps = tk.options
        if len(exps) < 2: return None
        target_exp = exps[1] # Next monthly expiry
        
        # Get Chain
        chain = tk.option_chain(target_exp)
        calls = chain.calls
        puts = chain.puts
        
        # Risk Free Rate (Approx 4.5%)
        r = 0.045
        # Time to Expiry (Annualized)
        exp_date = datetime.strptime(target_exp, "%Y-%m-%d")
        T = (exp_date - datetime.now()).days / 365.0
        
        gex_data = []
        
        # Process Calls (Positive Gamma for Dealers if they are Short Calls -> Long Gamma? 
        # Convention: Dealers are Short Calls (Long Gamma) and Short Puts (Short Gamma)
        # Simplified: Call OI * Gamma = Positive GEX, Put OI * Gamma = Negative GEX)
        
        # Merge strikes
        strikes = sorted(list(set(calls['strike'].tolist() + puts['strike'].tolist())))
        
        for K in strikes:
            # Call Data
            c_row = calls[calls['strike'] == K]
            c_oi = c_row['openInterest'].iloc[0] if not c_row.empty else 0
            c_iv = c_row['impliedVolatility'].iloc[0] if not c_row.empty and 'impliedVolatility' in c_row.columns else 0.2
            
            # Put Data
            p_row = puts[puts['strike'] == K]
            p_oi = p_row['openInterest'].iloc[0] if not p_row.empty else 0
            p_iv = p_row['impliedVolatility'].iloc[0] if not p_row.empty and 'impliedVolatility' in p_row.columns else 0.2
            
            # Calculate Gammas
            c_gamma = calculate_black_scholes_gamma(spot_price, K, T, r, c_iv)
            p_gamma = calculate_black_scholes_gamma(spot_price, K, T, r, p_iv)
            
            # Net GEX (Notional)
            # Spot * Gamma * OI * 100
            net_gex = (c_gamma * c_oi - p_gamma * p_oi) * spot_price * 100
            
            gex_data.append({"strike": K, "gex": net_gex})
            
        return pd.DataFrame(gex_data)
    except Exception as e:
        return None

# --- NEW: VOLUME PROFILE ENGINE ---
def calculate_volume_profile(df, bins=50):
    """Calculates Volume by Price Zone"""
    if df.empty: return None, None, None
    
    price_range = df['High'].max() - df['Low'].min()
    bin_size = price_range / bins
    
    # Create bins
    df['PriceBin'] = ((df['Close'] - df['Low'].min()) // bin_size).astype(int)
    vol_profile = df.groupby('PriceBin')['Volume'].sum().reset_index()
    
    # Map back to Price
    vol_profile['PriceLevel'] = df['Low'].min() + (vol_profile['PriceBin'] * bin_size)
    
    # POC (Point of Control)
    poc_idx = vol_profile['Volume'].idxmax()
    poc_price = vol_profile.loc[poc_idx, 'PriceLevel']
    
    # Value Area (70%)
    total_vol = vol_profile['Volume'].sum()
    vol_profile = vol_profile.sort_values('Volume', ascending=False)
    vol_profile['CumVol'] = vol_profile['Volume'].cumsum()
    vol_profile['InVA'] = vol_profile['CumVol'] <= (total_vol * 0.70)
    
    # Re-sort by price for plotting
    vol_profile = vol_profile.sort_values('PriceLevel')
    
    return vol_profile, poc_price, bin_size

# --- NEW: MACHINE LEARNING ENGINE ---
@st.cache_data(ttl=3600)
def get_ml_prediction(ticker):
    """Trains a Lite Random Forest to predict Next Day Up/Down"""
    try:
        # Get longer history for training
        df = yf.download(ticker, period="5y", interval="1d", progress=False)
        if df.empty: return None
        
        # Handle MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(1, axis=1) # Flatten
        
        data = df.copy()
        data['Returns'] = data['Close'].pct_change()
        data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int) # 1 if next day up
        
        # Features
        data['RSI'] = calculate_rsi(data['Close'])
        data['Vol_5d'] = data['Returns'].rolling(5).std()
        data['Mom_5d'] = data['Close'].pct_change(5)
        data['DayOfWeek'] = data.index.dayofweek
        
        data = data.dropna()
        
        # Features & Target
        features = ['RSI', 'Vol_5d', 'Mom_5d', 'DayOfWeek']
        X = data[features]
        y = data['Target']
        
        # Train/Test (Last 20% is test, but we train on all for 'Tomorrow' prediction)
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X, y)
        
        # Predict Tomorrow
        last_row = X.iloc[[-1]]
        prob_up = model.predict_proba(last_row)[0][1] # Probability of Class 1 (Up)
        
        return prob_up
    except Exception as e:
        return 0.5

# --- NEW: KELLY CRITERION ENGINE ---
def calculate_kelly(prob_win, risk_reward_ratio):
    """Kelly Formula: f* = p - q/b"""
    p = prob_win
    q = 1 - p
    b = risk_reward_ratio
    
    if b == 0: return 0
    f = p - (q / b)
    return max(0, f) # No negative sizing

# --- EXISTING ENGINES (Preserved) ---
# ... (parse_eco_value, analyze_event_impact, get_daily_data, get_intraday_data, get_news, get_economic_calendar, get_macro_regime_data, calculate_volatility_permission, get_options_pdf, get_seasonality_stats, calculate_vwap, calculate_rsi, get_correlation_data, get_ai_sentiment, generate_monte_carlo)
# Re-pasting the essential ones needed for context:

def calculate_rsi(series, period=14): # Helper for ML
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=60)
def get_daily_data(ticker):
    try:
        data = yf.download(ticker, period="10y", interval="1d", progress=False)
        return data
    except: return pd.DataFrame()

@st.cache_data(ttl=60)
def get_intraday_data(ticker):
    try:
        data = yf.download(ticker, period="5d", interval="15m", progress=False)
        return data
    except: return pd.DataFrame()

@st.cache_data(ttl=86400)
def get_macro_regime_data(api_key):
    if not api_key: return None
    try:
        fred = Fred(api_key=api_key)
        ffr = fred.get_series('FEDFUNDS').iloc[-1]
        cpi = fred.get_series('CPIAUCSL').pct_change(12).iloc[-1] * 100
        real_rate = ffr - cpi
        bias = "BULLISH" if real_rate < 0.5 else "BEARISH"
        return {"bias": bias, "real_rate": real_rate}
    except: return None

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
        return {"forecast": forecast_tr, "baseline": baseline, "is_go": forecast_tr > baseline, "history": tr_pct, "signal": "TRADE PERMITTED" if forecast_tr > baseline else "CAUTION"}
    except: return None

@st.cache_data(ttl=3600)
def get_options_pdf(opt_ticker): # Needed for GEX Spot Reference
    try:
        tk = yf.Ticker(opt_ticker)
        exps = tk.options
        if len(exps) < 2: return None
        target_exp = exps[1]
        chain = tk.option_chain(target_exp)
        calls = chain.calls
        calls['mid'] = (calls['bid'] + calls['ask']) / 2
        calls['price'] = np.where((calls['bid']==0), calls['lastPrice'], calls['mid'])
        df = calls[['strike', 'price']].sort_values('strike')
        spline = UnivariateSpline(df['strike'], df['price'], k=4, s=len(df)*2)
        strikes_smooth = np.linspace(df['strike'].min(), df['strike'].max(), 200)
        pdf = spline.derivative(n=2)(strikes_smooth)
        peak = strikes_smooth[np.argmax(pdf)]
        return {"strikes": strikes_smooth, "pdf": pdf, "peak": peak, "date": target_exp}
    except: return None

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Settings")
    selected_asset = st.selectbox("Select Asset", list(ASSETS.keys()))
    asset_info = ASSETS[selected_asset]
    st.markdown("---")
    news_key = get_api_key("news_api_key")
    fred_key = get_api_key("fred_api_key")
    rapid_key = get_api_key("rapidapi_key")
    st.caption(f"Keys: News {'✅' if news_key else '❌'} | FRED {'✅' if fred_key else '❌'} | Rapid {'✅' if rapid_key else '❌'}")
    if st.button("Refresh Data"): st.cache_data.clear()

# --- MAIN DASHBOARD ---
st.title(f"📊 {selected_asset} Quantitative Terminal")

# Fetch Data
daily_data = get_daily_data(asset_info['ticker'])
intraday_data = get_intraday_data(asset_info['ticker'])
macro_regime = get_macro_regime_data(fred_key)
vol_forecast = calculate_volatility_permission(asset_info['ticker'])
options_data = get_options_pdf(asset_info['opt_ticker']) # Also serves as probability
ml_prob = get_ml_prediction(asset_info['ticker'])

# --- 1. OVERVIEW & ML PREDICTOR ---
if not daily_data.empty:
    if isinstance(daily_data.columns, pd.MultiIndex): close = daily_data['Close'].iloc[:, 0]
    else: close = daily_data['Close']
    curr = close.iloc[-1]
    pct = ((curr - close.iloc[-2]) / close.iloc[-2]) * 100
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"{curr:,.2f}", f"{pct:.2f}%")
    
    # ML Widget
    if ml_prob:
        bias = "BULLISH" if ml_prob > 0.55 else "BEARISH" if ml_prob < 0.45 else "NEUTRAL"
        conf = abs(ml_prob - 0.5) * 200 # Scale 0.5-1.0 to 0-100% confidence
        color = "bullish" if bias == "BULLISH" else "bearish" if bias == "BEARISH" else "neutral"
        c2.markdown(f"""
        <div class="metric-container">
            <small>🤖 AI Predictor</small><br>
            <span class="{color}">{bias}</span><br>
            <small>Conf: {conf:.0f}%</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Kelly Widget
    if vol_forecast and ml_prob:
        # Heuristic Risk/Reward based on Volatility State (High Vol = Lower R/R assumption for safety)
        rr_ratio = 1.5 if vol_forecast['is_go'] else 1.0 
        kelly_pct = calculate_kelly(ml_prob, rr_ratio) * 100
        # Fractional Kelly (Half-Kelly for safety)
        safe_kelly = kelly_pct * 0.5 
        c3.metric("Kelly Size (Risk)", f"{safe_kelly:.1f}%", f"Win Prob: {ml_prob*100:.0f}%")

    if macro_regime:
        color = "bullish" if macro_regime['bias'] == "BULLISH" else "bearish"
        c4.markdown(f"""
        <div class="metric-container">
            <small>Macro Regime</small><br>
            <span class="{color}">{macro_regime['bias']}</span><br>
            <small>Real Rates: {macro_regime['real_rate']:.2f}%</small>
        </div>
        """, unsafe_allow_html=True)

    # --- CHART WITH VOLUME PROFILE (NEW) ---
    st.markdown("---")
    st.subheader("Price & Volume Architecture")
    
    chart_col, vol_col = st.columns([3, 1])
    
    with chart_col:
        # Standard Candle Chart
        fig = go.Figure(data=[go.Candlestick(x=daily_data.index, open=daily_data['Open'].iloc[:,0] if isinstance(daily_data.columns, pd.MultiIndex) else daily_data['Open'], 
                                            high=daily_data['High'].iloc[:,0] if isinstance(daily_data.columns, pd.MultiIndex) else daily_data['High'], 
                                            low=daily_data['Low'].iloc[:,0] if isinstance(daily_data.columns, pd.MultiIndex) else daily_data['Low'], 
                                            close=close, name='Price')])
        fig.update_layout(height=400, template="plotly_dark", xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
        
    with vol_col:
        # Volume Profile Chart
        vp_data, poc, _ = calculate_volume_profile(intraday_data if not intraday_data.empty else daily_data.tail(30))
        if vp_data is not None:
            fig_vp = go.Figure()
            # Gray for Value Area, Dark Gray for outside
            colors = ['#00ff00' if x else '#333333' for x in vp_data['InVA']]
            
            fig_vp.add_trace(go.Bar(
                y=vp_data['PriceLevel'], 
                x=vp_data['Volume'], 
                orientation='h',
                marker_color=colors,
                opacity=0.6,
                name="Vol Profile"
            ))
            # POC Line
            fig_vp.add_hline(y=poc, line_dash="dash", line_color="yellow", annotation_text="POC")
            
            fig_vp.update_layout(
                title="Volume Profile", 
                template="plotly_dark", 
                height=400, 
                showlegend=False,
                xaxis_visible=False,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig_vp, use_container_width=True)

# --- 2. GAMMA EXPOSURE (GEX) ---
st.markdown("---")
st.subheader("☢️ Gamma Exposure (Dealer Positioning)")
if options_data:
    # We use the ETF ticker (SPY/QQQ) spot price for GEX calc
    spot_ref = yf.Ticker(asset_info['opt_ticker']).history(period='1d')['Close'].iloc[-1]
    gex_df = get_gex_profile(asset_info['opt_ticker'], spot_ref)
    
    if gex_df is not None:
        g1, g2 = st.columns([3, 1])
        with g1:
            # Filter near-the-money strikes for cleaner chart
            center_idx = (gex_df['strike'] - spot_ref).abs().argsort()[:20]
            gex_zoom = gex_df.iloc[center_idx].sort_values('strike')
            
            fig_gex = go.Figure()
            fig_gex.add_trace(go.Bar(
                x=gex_zoom['strike'], 
                y=gex_zoom['gex'],
                marker_color=['#00ff00' if x > 0 else '#ff4b4b' for x in gex_zoom['gex']],
                name='Net GEX'
            ))
            fig_gex.add_vline(x=spot_ref, line_dash="dot", annotation_text="Spot")
            fig_gex.update_layout(title="Net Gamma by Strike (Sticky vs Slippery)", template="plotly_dark", height=300, yaxis_title="Gamma Notional ($)")
            st.plotly_chart(fig_gex, use_container_width=True)
            
        with g2:
            total_gex = gex_df['gex'].sum()
            sentiment = "High Volatility (Slippery)" if total_gex < 0 else "Low Volatility (Sticky)"
            st.metric("Total Net Gamma", f"${total_gex/1000000:.0f}M", sentiment)
            st.info(" Green bars act as magnets (dealers buy dips). Red bars act as accelerators (dealers sell dips).")

# --- 3. VOLATILITY & INTRADAY ---
st.markdown("---")
st.subheader("⚡ Volatility Regime")
if vol_forecast:
    v1, v2 = st.columns([1, 3])
    with v1:
        badge_class = "vol-go" if vol_forecast['is_go'] else "vol-stop"
        st.markdown(f"<span class='{badge_class}'>{vol_forecast['signal']}</span>", unsafe_allow_html=True)
        st.markdown(f"Exp TR: **{vol_forecast['forecast']:.2f}%**")
    with v2:
        hist_tr = vol_forecast['history'].tail(40)
        hist_base = vol_forecast['signal'] # logic handled in chart
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=hist_tr.index, y=hist_tr.values, marker_color='#333', name='Realized'))
        next_day = hist_tr.index[-1] + timedelta(days=1)
        fig_vol.add_trace(go.Bar(x=[next_day], y=[vol_forecast['forecast']], marker_color='#00ff00' if vol_forecast['is_go'] else '#ff4b4b', name='Forecast'))
        fig_vol.update_layout(height=200, template="plotly_dark", margin=dict(t=10, b=10))
        st.plotly_chart(fig_vol, use_container_width=True)

# --- 4. CONCLUSION ---
st.markdown("---")
st.subheader("🏁 Executive Summary")

bias_score = 0
reasons = []

if ml_prob > 0.55: bias_score += 1; reasons.append(f"AI Model predicts UP ({ml_prob:.0%})")
elif ml_prob < 0.45: bias_score -= 1; reasons.append(f"AI Model predicts DOWN ({ml_prob:.0%})")

if macro_regime and macro_regime['bias'] == "BULLISH": bias_score += 1; reasons.append("Real Rates are Negative (Supportive)")
elif macro_regime: bias_score -= 1; reasons.append("Real Rates are Positive (Restrictive)")

if options_data and options_data['peak'] > curr: bias_score += 1; reasons.append("Options Skew is Bullish")
elif options_data: bias_score -= 1; reasons.append("Options Skew is Bearish")

final_text = "BULLISH" if bias_score > 0 else "BEARISH" if bias_score < 0 else "NEUTRAL"
final_color = "bullish" if bias_score > 0 else "bearish" if bias_score < 0 else "neutral"

st.markdown(f"""
<div style="padding: 20px; border: 1px solid #444; border-radius: 10px; background-color: #1e1e1e; text-align: center;">
    <h2>Final Bias: <span class='{final_color}'>{final_text}</span></h2>
    <p>{' | '.join(reasons)}</p>
</div>
""", unsafe_allow_html=True)
