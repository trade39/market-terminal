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
from scipy.interpolate import UnivariateSpline  # For Options Smoothing
from scipy.stats import norm

# --- APP CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Quant Terminal Pro", page_icon="🧠")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .metric-container { background-color: #1e1e1e; border: 1px solid #333; padding: 10px; border-radius: 5px; text-align: center; }
    .regime-box { padding: 15px; border-radius: 5px; margin-bottom: 20px; background-color: #262730; border-left: 5px solid; }
    .regime-bull { border-color: #00ff00; }
    .regime-bear { border-color: #ff4b4b; }
    .regime-neutral { border-color: #d4af37; }
    .vol-badge-go { background-color: rgba(0, 255, 0, 0.2); color: #00ff00; padding: 4px 10px; border-radius: 4px; font-weight: bold; }
    .vol-badge-stop { background-color: rgba(255, 75, 75, 0.2); color: #ff4b4b; padding: 4px 10px; border-radius: 4px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- CONSTANTS ---
ASSETS = {
    "S&P 500": {"ticker": "^GSPC", "opt_ticker": "SPY", "news_query": "S&P 500"}, # Using SPY for options data availability
    "NASDAQ": {"ticker": "^IXIC", "opt_ticker": "QQQ", "news_query": "Nasdaq"},
    "Gold": {"ticker": "GC=F", "opt_ticker": "GLD", "news_query": "Gold Price"},
    "NVIDIA": {"ticker": "NVDA", "opt_ticker": "NVDA", "news_query": "Nvidia stock"},
}

# --- API HELPERS ---
def get_api_key(key_name):
    if "api_keys" in st.secrets and key_name in st.secrets["api_keys"]:
        return st.secrets["api_keys"][key_name]
    return None

# --- QUANTITATIVE ENGINE (THE NEW CONCEPTS) ---

@st.cache_data(ttl=3600)
def get_macro_regime(api_key):
    """
    CONCEPT 2: Macro Regimes
    Compares Fed Funds Rate vs CPI to determine Real Rates.
    Regime = Bullish (Real Rates < 0) or Bearish (Real Rates > 0).
    """
    if not api_key: return None
    try:
        fred = Fred(api_key=api_key)
        # Fetch Fed Funds (Interest Rates) and CPI (Inflation)
        ffr = fred.get_series('FEDFUNDS').iloc[-1]
        cpi = fred.get_series('CPIAUCSL').pct_change(12).iloc[-1] * 100 # YoY Inflation
        
        real_rate = ffr - cpi
        
        regime = {
            "rate": ffr,
            "cpi": cpi,
            "real_rate": real_rate,
            "status": "LIQUIDITY EXPANSION" if real_rate < 0 else "LIQUIDITY CONTRACTION",
            "bias": "BULLISH" if real_rate < 0.5 else "BEARISH" # Threshold tuning
        }
        return regime
    except:
        return None

@st.cache_data(ttl=300)
def calculate_volatility_forecast(ticker):
    """
    CONCEPT 3: GARCH-style Volatility Forecasting on True Range.
    Returns: Forecasted Range %, Signal (Trade/No Trade).
    """
    try:
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if df.empty: return None

        # 1. Calculate True Range (TR)
        # Handling MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            high = df['High'].iloc[:, 0]
            low = df['Low'].iloc[:, 0]
            close = df['Close'].iloc[:, 0]
        else:
            high, low, close = df['High'], df['Low'], df['Close']
            
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        # Combine into DataFrame to take max
        tr_df = pd.concat([tr1, tr2, tr3], axis=1)
        true_range = tr_df.max(axis=1)
        
        # Normalize TR as % of price
        tr_pct = (true_range / close) * 100
        tr_pct = tr_pct.dropna()
        
        # 2. Log Transform (Stabilize variance)
        log_tr = np.log(tr_pct)
        
        # 3. Simple GARCH Proxy (EWMA of Volatility)
        # We use a recursive smoothing to simulate volatility persistence
        # Forecast_t = lambda * Actual_{t-1} + (1-lambda) * Forecast_{t-1}
        ewma_vol = log_tr.ewm(alpha=0.94).mean() # Alpha similar to RiskMetrics decay
        
        # Forecast for "Tomorrow" (Last value of the EWMA)
        forecast_log = ewma_vol.iloc[-1]
        forecast_tr_pct = np.exp(forecast_log)
        
        # 4. Generate Signal
        # Baseline: 20-day moving average of realized TR
        baseline_vol = tr_pct.rolling(20).mean().iloc[-1]
        
        signal = 1 if forecast_tr_pct > baseline_vol else 0
        
        return {
            "forecast_pct": forecast_tr_pct,
            "baseline_pct": baseline_vol,
            "signal": signal,
            "tr_history": tr_pct
        }
    except Exception as e:
        st.error(f"Vol Error: {e}")
        return None

@st.cache_data(ttl=3600)
def get_options_probability(ticker_symbol):
    """
    CONCEPT 1: Options Implied Probabilities (Breeden-Litzenberger).
    Extracts PDF from Option Chain.
    """
    try:
        tk = yf.Ticker(ticker_symbol)
        
        # Get nearest monthly expiration (usually 3rd or 4th in list to avoid 0DTE noise)
        exps = tk.options
        if len(exps) < 2: return None
        target_exp = exps[1] # Look ~1-2 weeks out for better curve
        
        # Get Chain
        chain = tk.option_chain(target_exp)
        calls = chain.calls
        
        # Filter for liquidity
        calls = calls[(calls['volume'] > 10) & (calls['openInterest'] > 50)]
        
        if calls.empty: return None
        
        # Prepare Data for Calculus
        # Using Midpoint price for better accuracy
        calls['mid'] = (calls['bid'] + calls['ask']) / 2
        # Fallback to lastPrice if bid/ask is missing/wide
        calls['price'] = np.where((calls['bid']==0) | (calls['ask']==0), calls['lastPrice'], calls['mid'])
        
        df = calls[['strike', 'price']].sort_values('strike')
        
        # 1. Fit Smooth Curve (Spline) to Call Prices vs Strike
        # S parameter controls smoothing (critical for 2nd derivative stability)
        spline = UnivariateSpline(df['strike'], df['price'], k=4, s=len(df)*2) 
        
        strikes_smooth = np.linspace(df['strike'].min(), df['strike'].max(), 200)
        prices_smooth = spline(strikes_smooth)
        
        # 2. Calculate Second Derivative (PDF)
        # Breeden-Litzenberger: PDF ~ d^2C / dK^2
        deriv1 = spline.derivative(n=1)
        deriv2 = spline.derivative(n=2)
        
        pdf = deriv2(strikes_smooth)
        
        # Normalize PDF so area under curve ~ 1 (visualization only)
        pdf = np.maximum(pdf, 0) # Remove negative artifacts
        pdf = pdf / pdf.sum()
        
        return {
            "strikes": strikes_smooth,
            "pdf": pdf,
            "date": target_exp,
            "raw_strikes": df['strike'],
            "raw_prices": df['price']
        }
        
    except Exception as e:
        return None

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Quantitative Specs")
    selected_asset_name = st.selectbox("Select Asset", list(ASSETS.keys()))
    asset_info = ASSETS[selected_asset_name]
    
    st.markdown("---")
    st.caption("API Configuration")
    fred_key = get_api_key("fred_api_key")
    news_key = get_api_key("news_api_key")
    
    st.success(f"Macro Data: {'Connected' if fred_key else 'Missing Key'}")
    
    if st.button("Refresh Models"):
        st.cache_data.clear()

# --- MAIN DASHBOARD ---
st.title(f"🧠 {selected_asset_name} Quantitative Terminal")

# 1. MACRO REGIME SECTION
macro_data = get_macro_regime(fred_key)

if macro_data:
    st.markdown("### 1. Macro-Economic Regime (Context Filter)")
    
    # Dynamic CSS class based on regime
    regime_cls = "regime-bull" if macro_data['bias'] == "BULLISH" else "regime-bear"
    
    st.markdown(f"""
    <div class='regime-box {regime_cls}'>
        <div style="display:flex; justify-content: space-between; align-items:center;">
            <div>
                <h3 style="margin:0;">Regime: {macro_data['status']}</h3>
                <p style="margin:0; opacity:0.8;">Primary Bias: <strong>{macro_data['bias']}</strong></p>
            </div>
            <div style="text-align:right;">
                <h2 style="margin:0;">{macro_data['real_rate']:.2f}%</h2>
                <small>Real Rate (Fed Funds - CPI)</small>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Fed Funds Rate (Cost of Money)", f"{macro_data['rate']:.2f}%")
    m2.metric("CPI YoY (Inflation)", f"{macro_data['cpi']:.2f}%")
    m3.metric("Regime Implication", "Seek Long Exposure" if macro_data['bias']=="BULLISH" else "Cash / Short Bias")

else:
    st.warning("Please add FRED API Key to secrets.toml to unlock Macro Regime.")

st.markdown("---")

# 2. VOLATILITY & PERMISSION
st.markdown("### 2. Volatility Forecast (Permission to Trade)")
vol_data = calculate_volatility_forecast(asset_info['ticker'])

if vol_data:
    c1, c2 = st.columns([1, 2])
    
    with c1:
        signal_text = "TRADE PERMITTED" if vol_data['signal'] == 1 else "NO TRADE / CAUTION"
        badge_cls = "vol-badge-go" if vol_data['signal'] == 1 else "vol-badge-stop"
        
        st.markdown(f"""
        <div style="padding: 20px; border: 1px solid #333; border-radius: 10px; text-align: center;">
            <h4>Tomorrow's Bias</h4>
            <span class='{badge_cls}' style="font-size: 1.2em;">{signal_text}</span>
            <hr style="border-color: #444;">
            <p>Projected Range: <strong>{vol_data['forecast_pct']:.2f}%</strong></p>
            <p style="font-size: 0.8em; opacity: 0.7;">Baseline (20d): {vol_data['baseline_pct']:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("**Concept:** Using GARCH-style logic on Log-True Range. High forecasted volatility (relative to history) favors breakout/momentum strategies. Low volatility suggests chop.")
        
    with c2:
        # Plot TR History vs Forecast
        hist = vol_data['tr_history'].tail(50)
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=hist.index, y=hist.values, name="Realized True Range %", marker_color='#333'))
        
        # Add a line for the forecast
        fig_vol.add_hline(y=vol_data['baseline_pct'], line_dash="dash", line_color="gray", annotation_text="Baseline")
        fig_vol.add_hrect(y0=0, y1=vol_data['baseline_pct'], fillcolor="red", opacity=0.1, layer="below", line_width=0)
        fig_vol.add_hrect(y0=vol_data['baseline_pct'], y1=hist.max(), fillcolor="green", opacity=0.1, layer="below", line_width=0)
        
        fig_vol.update_layout(
            title="True Range Regimes (Green Zone = Actionable Volatility)",
            template="plotly_dark", 
            height=300, 
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_vol, use_container_width=True)

st.markdown("---")

# 3. INSTITUTIONAL EXPECTATIONS (OPTIONS)
st.markdown("### 3. Institutional Expectations (Market Implied Probabilities)")

opt_data = get_options_probability(asset_info['opt_ticker'])

if opt_data:
    st.markdown(f"**Target Expiration:** {opt_data['date']}")
    
    # Identify Peak Probability Price
    peak_idx = np.argmax(opt_data['pdf'])
    most_likely_price = opt_data['strikes'][peak_idx]
    
    # Get Current Price for comparison
    curr_df = yf.Ticker(asset_info['opt_ticker']).history(period='1d')
    curr_price = curr_df['Close'].iloc[-1] if not curr_df.empty else 0
    
    c_opt1, c_opt2 = st.columns([3, 1])
    
    with c_opt1:
        fig_pdf = go.Figure()
        
        # Plot the PDF Curve
        fig_pdf.add_trace(go.Scatter(
            x=opt_data['strikes'], 
            y=opt_data['pdf'], 
            mode='lines', 
            name='Implied Probability Density',
            fill='tozeroy',
            line=dict(color='#00d4ff', width=3)
        ))
        
        # Add Current Price Line
        fig_pdf.add_vline(x=curr_price, line_dash="dot", line_color="white", annotation_text=f"Current: {curr_price:.2f}")
        
        # Add Market Expected Price Line
        fig_pdf.add_vline(x=most_likely_price, line_dash="dash", line_color="#d4af37", annotation_text=f"Market Expectation: {most_likely_price:.0f}")

        fig_pdf.update_layout(
            title=f"Options-Implied Probability Distribution (Breeden-Litzenberger)",
            xaxis_title="Price Level",
            yaxis_title="Probability Density",
            template="plotly_dark",
            height=400,
            hovermode="x"
        )
        st.plotly_chart(fig_pdf, use_container_width=True)
        
    with c_opt2:
        st.info("""
        **How to read this:**
        This curve is derived from the **2nd Derivative of Call Option Prices**.
        
        It shows where Institutions (who hedge with options) are betting the price will land.
        
        * **Tall/Narrow Peak:** High confidence, low expected volatility.
        * **Wide/Flat:** High uncertainty, high expected volatility.
        * **Skewed:** Tail risk protection (if skewed left, hedging against crash).
        """)
        
        skew = "Neutral"
        if most_likely_price > curr_price * 1.01: skew = "Bullish Skew"
        elif most_likely_price < curr_price * 0.99: skew = "Bearish Skew"
        
        st.metric("Market Gravity Center", f"${most_likely_price:.2f}", skew)

else:
    st.warning("Could not build Options Probability Surface (Data Unavailable or Market Closed).")
    st.caption("Try selecting SPY or QQQ for best options data liquidity.")

# 4. TRADITIONAL METRICS (Simplified)
with st.expander("Show Traditional Metrics (News & AI)"):
    if news_key:
        news = NewsApiClient(api_key=news_key).get_everything(q=asset_info['news_query'], language='en', page_size=3)
        if news['articles']:
            st.write(f"**Latest News for {asset_info['news_query']}**")
            for art in news['articles']:
                st.markdown(f"- [{art['title']}]({art['url']})")
