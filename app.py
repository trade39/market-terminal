import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import concurrent.futures
import json
from datetime import datetime

# --- MODULES ---
import config
from utils import get_api_key, terminal_chart_layout
import data_engine as de
import quant_engine as qe
import ai_engine as ai

# --- APP CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Bloomberg Terminal Pro V6.1", page_icon="⚡")
st.markdown(config.CSS_STYLE, unsafe_allow_html=True)

# --- HELPER UI CLASS (MODULARIZATION) ---
class UI:
    @staticmethod
    def card(label, value, sub_value, sentiment="neutral", extra_content=""):
        color = "#00FFFF" if sentiment == "bullish" else "#8080FF" if sentiment == "bearish" else "#AAAAAA"
        st.markdown(f"""
        <div class='terminal-box' style="padding:10px;">
            <div style="font-size:0.8em; color:#AAAAAA;">{label}</div>
            <div style="font-size:1.5em; font-weight:bold; color:{color};">{value}</div>
            <div style="font-size:0.8em; color:#CCCCCC;">{sub_value}</div>
            {extra_content}
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def header(title, subtitle=""):
        st.markdown(f"<h3 style='color: #00FFFF; border-bottom: 1px solid #333; padding-bottom: 5px;'>{title} <span style='font-size:0.6em; color:#888;'>{subtitle}</span></h3>", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
for key in ['gemini_calls', 'news_calls', 'rapid_calls', 'coingecko_calls', 'fred_calls']:
    if key not in st.session_state: st.session_state[key] = 0
if 'messages' not in st.session_state: st.session_state['messages'] = []
if 'chat_history' not in st.session_state: st.session_state.chat_history = []

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h3 style='color: #00FFFF;'>COMMAND LINE</h3>", unsafe_allow_html=True)
    st.caption(f"Last Update: {datetime.now().strftime('%H:%M:%S')}") 
    
    selected_asset = st.selectbox("SEC / Ticker", list(config.ASSETS.keys()))
    asset_info = config.ASSETS[selected_asset]
    use_demo_data = st.checkbox("🛠️ USE DEMO DATA", value=True, help="Save RapidAPI credits.")
    
    st.markdown("---")
    with st.expander("📡 API HEALTH", expanded=False):
        st.progress(min(st.session_state['news_calls'] / 100, 1.0), text="NewsAPI")
        st.progress(min(st.session_state['gemini_calls'] / 20, 1.0), text="Gemini AI")
        if not ai.HAS_NLP: st.warning("NLP Disabled (textblob missing)")
        
    st.markdown("---")
    # API Keys
    rapid_key = get_api_key("rapidapi_key")
    news_key = get_api_key("news_api_key")
    gemini_key = get_api_key("gemini_api_key")
    cg_key = get_api_key("coingecko_key") 
    fred_key = get_api_key("fred_api_key")
    
    if st.button(">> REFRESH FEED"): 
        st.cache_data.clear()
        st.rerun()

# ==============================================================================
# 1. ASYNC DATA ENGINE (PARALLEL FETCHING)
# ==============================================================================
@st.cache_data(ttl=300) # Cache high-level data bundle for 5 mins
def fetch_dashboard_data(ticker, asset_config, _fred_key, _news_key, _rapid_key, _cg_key, _use_demo):
    """Fetches independent IO-bound data in parallel."""
    data_bundle = {}
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Define Futures
        f_daily = executor.submit(de.get_daily_data, ticker)
        f_intra = executor.submit(de.get_intraday_data, ticker)
        f_dxy = executor.submit(de.get_dxy_data, _fred_key)
        f_eco = executor.submit(de.get_economic_calendar, _rapid_key, _use_demo)
        f_news_gen = executor.submit(de.get_financial_news_general, _news_key, asset_config.get('news_query', 'Finance'))
        f_news_ff = executor.submit(de.get_forex_factory_news, _rapid_key, 'breaking')
        f_gex = executor.submit(qe.get_gex_profile, asset_config['opt_ticker'])
        f_ml = executor.submit(qe.get_ml_prediction, ticker)
        
        # Conditional Futures
        f_yield = executor.submit(de.get_fred_series, "T10Y2Y", _fred_key) if _fred_key else None
        f_ff = executor.submit(de.get_fred_series, "FEDFUNDS", _fred_key) if _fred_key else None
        f_cpi = executor.submit(de.get_fred_series, "CPIAUCSL", _fred_key) if _fred_key else None
        f_m2 = executor.submit(de.get_fred_series, "M2SL", _fred_key) if _fred_key else None
        f_cot = executor.submit(de.fetch_cot_history, selected_asset, 2024) if de.HAS_COT_LIB else None
        f_cg = executor.submit(de.get_coingecko_stats, asset_config.get('cg_id'), _cg_key) if asset_config.get('cg_id') else None

        # Resolve Futures (Blocking)
        data_bundle['daily'] = f_daily.result()
        data_bundle['intraday'] = f_intra.result()
        data_bundle['dxy'] = f_dxy.result()
        data_bundle['eco'] = f_eco.result()
        data_bundle['news'] = f_news_gen.result()[:5] + f_news_ff.result()[:5]
        data_bundle['news_all_gen'] = f_news_gen.result()
        data_bundle['news_all_ff'] = f_news_ff.result()
        data_bundle['gex'] = f_gex.result()
        data_bundle['ml_pred'] = f_ml.result()
        
        data_bundle['fred_yield'] = f_yield.result() if f_yield else pd.DataFrame()
        data_bundle['fred_ff'] = f_ff.result() if f_ff else pd.DataFrame()
        data_bundle['fred_cpi'] = f_cpi.result() if f_cpi else pd.DataFrame()
        data_bundle['fred_m2'] = f_m2.result() if f_m2 else pd.DataFrame()
        data_bundle['cot'] = f_cot.result() if f_cot else pd.DataFrame()
        data_bundle['cg'] = f_cg.result() if f_cg else None

    return data_bundle

# --- EXECUTE FETCH ---
with st.spinner(f"⚡ Establishing secure link to {selected_asset} feed..."):
    # 1. Fetch Raw IO Data
    db = fetch_dashboard_data(asset_info['ticker'], asset_info, fred_key, news_key, rapid_key, cg_key, use_demo_data)
    
    # 2. Process CPU Bound Analytics (Dependent on Raw Data)
    daily_data = db['daily']
    intraday_data = db['intraday']
    
    # Safety check for empty data
    if daily_data.empty:
        st.error("Failed to fetch market data. Please check API connections.")
        st.stop()

    # Derived Analytics
    hurst = qe.calculate_hurst(daily_data['Close'].values)
    regime_data = qe.get_market_regime(asset_info['ticker'])
    ms_df, ms_trend, ms_last_sh, ms_last_sl = qe.detect_market_structure(daily_data)
    vol_cone = qe.get_volatility_cone(daily_data)
    of_df, of_bias = qe.calculate_order_flow_proxy(daily_data)
    active_fvgs = qe.detect_fair_value_gaps(daily_data)
    rs_data = qe.get_relative_strength(asset_info['ticker'])
    key_levels = qe.get_key_levels(daily_data)
    vwap_df = qe.calculate_vwap_bands(intraday_data)
    vol_profile, poc_price = qe.calculate_volume_profile(intraday_data)
    seasonality_stats = qe.get_seasonality_stats(daily_data, asset_info['ticker'])
    
    # Macro Context Builder
    macro_context_data = {}
    if not db['fred_yield'].empty: macro_context_data['yield_curve'] = f"{db['fred_yield']['value'].iloc[-1]:.2f}"
    if not db['fred_cpi'].empty: macro_context_data['cpi'] = f"{(db['fred_cpi']['value'].pct_change(12).iloc[-1]*100):.2f}"

# ==============================================================================
# 2. HEAD-UP DISPLAY (HUD)
# ==============================================================================
st.markdown(f"<h1 style='border-bottom: 2px solid #00FFFF;'>{selected_asset} <span style='font-size:0.5em; color:#AAAAAA;'>TERMINAL PRO V6.1</span></h1>", unsafe_allow_html=True)

close, high, low = daily_data['Close'], daily_data['High'], daily_data['Low']
curr, pct = close.iloc[-1], ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("LAST PX", f"{curr:,.2f}", f"{pct:.2f}%")

# UI Component Usage
_, ml_prob = db['ml_pred']
ml_bias = "BULLISH" if ml_prob > 0.55 else "BEARISH" if ml_prob < 0.45 else "NEUTRAL"
with c2: UI.card("AI PREDICTION", ml_bias, f"CONF: {abs(ml_prob - 0.5) * 200:.0f}%", ml_bias.lower())

with c3:
    hurst_type = "TRENDING" if hurst > 0.55 else "MEAN REV"
    UI.card("QUANT REGIME", regime_data['regime'], f"Fractal: {hurst_type}", "bullish" if "BULL" in regime_data['regime'] else "neutral")

c4.metric("SESSION H/L", f"{high.max():,.2f} / {low.min():,.2f}")

# ==============================================================================
# 3. MACRO & SENTIMENT
# ==============================================================================
st.markdown("---")
UI.header("PHASE 1: MACRO & SENTIMENT CONTEXT")

macro_tab1, macro_tab2 = st.tabs(["🇺🇸 MACRO DASHBOARD", "📰 NEWS & EVENTS"])

with macro_tab1:
    if fred_key:
        mc1, mc2 = st.columns([3, 1])
        with mc1:
            t1, t2 = st.tabs(["YIELD & RATES", "LIQUIDITY"])
            with t1:
                col_y, col_ff = st.columns(2)
                # Yield Curve Chart
                if not db['fred_yield'].empty:
                    fig_yc = go.Figure(go.Scatter(x=db['fred_yield'].index, y=db['fred_yield']['value'], fill='tozeroy', line=dict(color='#00FFFF')))
                    col_y.plotly_chart(terminal_chart_layout(fig_yc, title="10Y-2Y SPREAD", height=200), use_container_width=True)
                # Fed Funds
                if not db['fred_ff'].empty:
                    fig_ff = go.Figure(go.Scatter(x=db['fred_ff'].index, y=db['fred_ff']['value'], line=dict(color='#40E0FF')))
                    col_ff.plotly_chart(terminal_chart_layout(fig_ff, title="FED FUNDS RATE", height=200), use_container_width=True)
            with t2:
                 col_m2, col_cpi = st.columns(2)
                 if not db['fred_m2'].empty:
                     fig_m2 = go.Figure(go.Scatter(x=db['fred_m2'].index, y=db['fred_m2']['value'], line=dict(color='#00FFFF')))
                     col_m2.plotly_chart(terminal_chart_layout(fig_m2, title="M2 MONEY SUPPLY", height=200), use_container_width=True)
        
        with mc2:
            st.markdown("**MACRO ML ENGINE**")
            macro_regime = qe.get_macro_ml_regime(db['fred_cpi'], db['fred_ff']) if not db['fred_cpi'].empty else None
            if macro_regime:
                UI.card("ECON CYCLE", macro_regime['regime'], f"Rate: {macro_regime['rate']:.1f}%", "neutral")
    else:
        st.info("Fred API Key missing.")

with macro_tab2:
    ec1, ec2 = st.columns([1, 1])
    with ec1:
        st.markdown("**📅 USD HIGH IMPACT**")
        if db['eco']:
            df_cal = pd.DataFrame(db['eco'])
            st.dataframe(df_cal[['time','event_name','actual','forecast']], use_container_width=True, hide_index=True)
        else: st.caption("No upcoming high impact events.")
        
    with ec2:
        st.markdown("**📰 WIRE**")
        for n in db['news'][:3]:
            st.markdown(f"<div style='border-left:2px solid #00FFFF; padding-left:5px; margin-bottom:5px;'><a href='{n['url']}' style='color:white; text-decoration:none;'>{n['title']}</a><br><span style='font-size:0.7em; color:#888'>{n['time']}</span></div>", unsafe_allow_html=True)

# ==============================================================================
# 4. STRATEGIC ANALYSIS
# ==============================================================================
st.markdown("---")
UI.header("PHASE 2: STRATEGIC ANALYSIS")

sc1, sc2 = st.columns([2, 1])
with sc1:
    # --- MASTER CHART ---
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=daily_data.index, open=daily_data['Open'], high=high, low=low, close=close, name="Price", increasing_line_color="#00FFFF", decreasing_line_color="#405060"))
    
    # FVGs
    for fvg in active_fvgs:
        fig.add_shape(type="rect", x0=fvg['date'], x1=daily_data.index[-1], y0=fvg['bottom'], y1=fvg['top'], fillcolor="rgba(0,255,255,0.1)", line_width=0)
    
    # DXY Overlay
    if not db['dxy'].empty:
         dxy_aligned = db['dxy']['Close'].reindex(daily_data.index, method='ffill')
         fig.add_trace(go.Scatter(x=dxy_aligned.index, y=dxy_aligned.values, name="DXY", line=dict(color='#8080FF', width=1), yaxis="y2", opacity=0.5))

    fig = terminal_chart_layout(fig, height=500)
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False))
    st.plotly_chart(fig, use_container_width=True)

with sc2:
    UI.card("MARKET STRUCTURE", ms_trend, f"Last Low: {ms_last_sl:,.2f}", "bullish" if "BULL" in ms_trend else "bearish")
    
    # Seasonality
    if seasonality_stats:
        st.markdown("**⏳ SEASONALITY**")
        t_s1, t_s2 = st.tabs(["DAY", "HOUR"])
        with t_s1: 
             st.bar_chart(seasonality_stats.get('day_high', pd.Series()), color="#00FFFF")
        with t_s2:
             st.bar_chart(seasonality_stats.get('hourly_perf', pd.Series()), color="#8080FF")

# COT Section (Safe)
if db['cot'] is not None and not db['cot'].empty:
    with st.expander("🏛️ INSTITUTIONAL POSITIONING (COT)", expanded=False):
        cot = db['cot']
        try:
            # Safe extraction of latest row
            latest = cot.iloc[-1]
            
            # Safe conversion to numeric, coercing bad data to NaN, then 0
            s_long = pd.to_numeric(latest.get('spec_long', 0), errors='coerce')
            s_short = pd.to_numeric(latest.get('spec_short', 0), errors='coerce')
            
            # Handle potential NaNs
            s_long = 0 if pd.isna(s_long) else s_long
            s_short = 0 if pd.isna(s_short) else s_short
            
            net_spec = s_long - s_short
            
            # Series for Z-Score
            series_diff = cot['spec_long'] - cot['spec_short']
            z_score = qe.calculate_z_score(series_diff)
            if pd.isna(z_score): z_score = 0.0
            
            c_cot1, c_cot2 = st.columns([1, 2])
            with c_cot1:
                UI.card(
                    "NET SPECULATOR", 
                    f"{int(net_spec):,}", 
                    f"Z-Score: {z_score:.2f}σ", 
                    "bullish" if net_spec > 0 else "bearish"
                )
            with c_cot2:
                fig_cot = go.Figure()
                fig_cot.add_trace(go.Scatter(x=cot['date'], y=series_diff, fill='tozeroy', line=dict(color='#00FFFF'), name="Net Spec"))
                st.plotly_chart(terminal_chart_layout(fig_cot, title="COT NET POS", height=150), use_container_width=True)

        except Exception as e:
            st.error(f"⚠️ COT Data Error: {str(e)}")


# ==============================================================================
# 5. DYNAMICS & EXECUTION
# ==============================================================================
st.markdown("---")
UI.header("PHASE 3: DYNAMICS & EXECUTION")

dc1, dc2, dc3 = st.columns(3)

# GEX
with dc1:
    gex_df, _, _, _ = db['gex']
    if gex_df is not None:
        total_gex = gex_df['gex'].sum() / 1e6
        UI.card("NET GAMMA", f"${total_gex:.1f}M", "Sticky" if total_gex > 0 else "Slippery", "bullish" if total_gex > 0 else "bearish")
    else: st.info("No Options Data")

# Volatility
with dc2:
    hv = close.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100
    UI.card("VOLATILITY", f"{hv:.1f}%", f"Regime: {vol_cone.get('regime','N/A')}", "neutral")

# Order Flow
with dc3:
    UI.card("ORDER FLOW", of_bias, "Volume Impulse", "bullish" if "Buying" in of_bias else "bearish")

# Execution Levels
st.markdown("#### 🎯 TACTICAL LEVELS")
lvl_cols = st.columns(5)
levels_map = [("R1", key_levels['R1']), ("PDH", key_levels['PDH']), ("PIVOT", key_levels['Pivot']), ("PDL", key_levels['PDL']), ("S1", key_levels['S1'])]

for i, (label, val) in enumerate(levels_map):
    with lvl_cols[i]:
        # Highlight if price is close (within 0.1%)
        is_close = abs(curr - val) / curr < 0.001
        st.metric(label, f"{val:,.2f}", delta="NEAR" if is_close else None, delta_color="inverse" if is_close else "off")

# ==============================================================================
# 6. AI SYNTHESIS & EXPORT
# ==============================================================================
st.markdown("---")
UI.header("🧠 PHASE 4: AI COMMAND CENTER")

ai_col1, ai_col2 = st.columns([2, 1])

with ai_col1:
    chat_container = st.container(height=400)
    for msg in st.session_state.chat_history:
        chat_container.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask about structure, macro, or key levels..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        chat_container.chat_message("user").write(prompt)
        
        with chat_container.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                # Assemble context for the AI
                context = {
                    "price": curr, "structure": ms_trend, "levels": key_levels,
                    "macro": macro_context_data, "news": db['news'][:3],
                    "ml_signal": ml_bias
                }
                response = ai.chat_with_data(prompt, context, gemini_key)
                st.write(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})

with ai_col2:
    st.markdown("**ACTIONS**")
    if st.button("📝 GENERATE EXECUTIVE BRIEF", use_container_width=True):
        brief = ai.get_technical_narrative(selected_asset, curr, pct, regime_data, ml_bias, db['gex'][0], None, key_levels, macro_context_data, gemini_key)
        st.session_state.chat_history.append({"role": "assistant", "content": brief})
        st.rerun()

    # MT5 Export Bridge
    st.markdown("---")
    st.markdown("**🤖 ALGO BRIDGE**")
    
    export_payload = {
        "asset": selected_asset,
        "timestamp": str(datetime.now()),
        "price": curr,
        "bias": ml_bias,
        "levels": key_levels,
        "vol_regime": vol_cone.get('regime', 'N/A')
    }
    
    st.download_button(
        label="⬇️ EXPORT SIGNAL (JSON)",
        data=json.dumps(export_payload, indent=4),
        file_name=f"{selected_asset}_signal_{datetime.now().strftime('%Y%m%d')}.json",
        mime="application/json",
        use_container_width=True
    )

if not gemini_key:
    st.warning("⚠️ Connect Gemini API Key in sidebar for AI features.")
