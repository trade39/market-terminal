import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --- MODULES ---
import config
from utils import get_api_key, terminal_chart_layout
import data_engine as de
import quant_engine as qe
import ai_engine as ai

# --- APP CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Bloomberg Terminal Pro V5.21", page_icon="⚡")
st.markdown(config.CSS_STYLE, unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'gemini_calls' not in st.session_state: st.session_state['gemini_calls'] = 0
if 'news_calls' not in st.session_state: st.session_state['news_calls'] = 0
if 'rapid_calls' not in st.session_state: st.session_state['rapid_calls'] = 0
if 'coingecko_calls' not in st.session_state: st.session_state['coingecko_calls'] = 0
if 'fred_calls' not in st.session_state: st.session_state['fred_calls'] = 0
if 'narrative_cache' not in st.session_state: st.session_state['narrative_cache'] = None
if 'thesis_cache' not in st.session_state: st.session_state['thesis_cache'] = None

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h3 style='color: #00FFFF;'>COMMAND LINE</h3>", unsafe_allow_html=True)
    selected_asset = st.selectbox("SEC / Ticker", list(config.ASSETS.keys()))
    asset_info = config.ASSETS[selected_asset]
    
    use_demo_data = st.checkbox("🛠️ USE DEMO DATA (Save Quota)", value=True, help="Use mock data for Calendar to save RapidAPI credits.")
    
    st.markdown("---")
    
    with st.expander("📡 API QUOTA MONITOR", expanded=True):
        st.markdown("<div style='font-size:0.7em; color:#AAAAAA;'>Session Usage vs Hard Limits</div>", unsafe_allow_html=True)
        st.write(f"**NewsAPI** ({st.session_state['news_calls']} / 100)")
        st.progress(min(st.session_state['news_calls'] / 100, 1.0))
        st.write(f"**Gemini AI** ({st.session_state['gemini_calls']} / 20)")
        st.progress(min(st.session_state['gemini_calls'] / 20, 1.0))
        st.write(f"**RapidAPI** ({st.session_state['rapid_calls']} / 10)")
        st.progress(min(st.session_state['rapid_calls'] / 10, 1.0))
        st.write(f"**FRED** ({st.session_state['fred_calls']} calls)")
        st.write(f"**CoinGecko** ({st.session_state['coingecko_calls']} calls)")
        
        if use_demo_data:
            st.success("🟢 DEMO MODE ACTIVE")
        else:
            st.warning("🔴 LIVE API MODE")
        
        # Dependency Status
        if not ai.HAS_NLP:
            st.warning("NLP Disabled: `textblob` missing")
        if not de.HAS_COT_LIB:
            st.info("Using Direct COT Fetch (No Lib)")
            
    st.markdown("---")
    rapid_key = get_api_key("rapidapi_key")
    news_key = get_api_key("news_api_key")
    gemini_key = get_api_key("gemini_api_key")
    cg_key = get_api_key("coingecko_key") 
    fred_key = get_api_key("fred_api_key")
    
    if st.button(">> REFRESH DATA"): 
        st.cache_data.clear()
        st.rerun()

# ==============================================================================
# 1. CENTRALIZED DATA FETCHING ENGINE
# ==============================================================================
# Fetch Market Data
daily_data = de.get_daily_data(asset_info['ticker'])
dxy_data = de.get_dxy_data(fred_key) 
intraday_data = de.get_intraday_data(asset_info['ticker'])
eco_events = de.get_economic_calendar(rapid_key, use_demo=use_demo_data)

# Fetch News
news_general = de.get_financial_news_general(news_key, query=asset_info.get('news_query', 'Finance'))
news_ff = de.get_forex_factory_news(rapid_key, 'breaking')
combined_news_for_llm = news_general[:5] + news_ff[:5]

# Fetch FRED Macro
df_yield = de.get_fred_series("T10Y2Y", fred_key)
df_ff = de.get_fred_series("FEDFUNDS", fred_key)
df_cpi = de.get_fred_series("CPIAUCSL", fred_key)
df_m2 = de.get_fred_series("M2SL", fred_key)

# Advanced Engines
_, ml_prob = qe.get_ml_prediction(asset_info['ticker'])
gex_df, gex_date, gex_spot, current_iv = qe.get_gex_profile(asset_info['opt_ticker'])
vol_profile, poc_price = qe.calculate_volume_profile(intraday_data)
hurst = qe.calculate_hurst(daily_data['Close'].values) if not daily_data.empty else 0.5
regime_data = qe.get_market_regime(asset_info['ticker'])
correlations = qe.get_correlations(asset_info['ticker'], fred_key)
news_sentiment_df = ai.calculate_news_sentiment(combined_news_for_llm)
macro_regime = qe.get_macro_ml_regime(df_cpi, df_ff) if fred_key else None

# Multilayer Technicals
ms_df, ms_trend, ms_last_sh, ms_last_sl = qe.detect_market_structure(daily_data)
vol_cone = qe.get_volatility_cone(daily_data)
of_df, of_bias = qe.calculate_order_flow_proxy(daily_data)
active_fvgs = qe.detect_fair_value_gaps(daily_data)
rs_data = qe.get_relative_strength(asset_info['ticker'])
key_levels = qe.get_key_levels(daily_data)
vwap_df = qe.calculate_vwap_bands(intraday_data)
pred_dates, pred_paths = qe.generate_monte_carlo(daily_data)
seasonality_stats = qe.get_seasonality_stats(daily_data, asset_info['ticker']) 

# Initialize COT Data for AI (Fetched later in the specific section but defined here)
cot_data = None 
macro_context_data = {}

# Populate Macro Context for AI
if not df_yield.empty: macro_context_data['yield_curve'] = f"{df_yield['value'].iloc[-1]:.2f}"
if not df_cpi.empty: macro_context_data['cpi'] = f"{(df_cpi['value'].pct_change(12).iloc[-1]*100):.2f}"
if not df_ff.empty: macro_context_data['rates'] = f"{df_ff['value'].iloc[-1]:.2f}"
if macro_regime: macro_context_data['regime'] = macro_regime['regime']

# ==============================================================================
# 2. HEAD-UP DISPLAY (HUD) - Immediate Situational Awareness
# ==============================================================================
st.markdown(f"<h1 style='border-bottom: 2px solid #00FFFF;'>{selected_asset} <span style='font-size:0.5em; color:#AAAAAA;'>TERMINAL PRO V5.21</span></h1>", unsafe_allow_html=True)

if not daily_data.empty:
    close, high, low = daily_data['Close'], daily_data['High'], daily_data['Low']
    curr = close.iloc[-1]
    pct = ((curr - close.iloc[-2]) / close.iloc[-2]) * 100
    
    # HUD Layout
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LAST PX", f"{curr:,.2f}", f"{pct:.2f}%")
    
    # ML Signal
    ml_bias = "BULLISH" if ml_prob > 0.55 else "BEARISH" if ml_prob < 0.45 else "NEUTRAL"
    ml_conf = abs(ml_prob - 0.5) * 200
    ml_color = "bullish" if ml_bias == "BULLISH" else "bearish" if ml_bias == "BEARISH" else "neutral"
    
    c2.markdown(f"""
    <div class='terminal-box' style="text-align:center; padding:5px;">
        <div style="font-size:0.8em; color:#00FFFF;">AI PREDICTION</div>
        <span class='{ml_color}'>{ml_bias}</span>
        <div style="font-size:0.8em; margin-top:5px; color:#AAAAAA;">CONF: {ml_conf:.0f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Regime
    hurst_type = "TRENDING" if hurst > 0.55 else "MEAN REVERT" if hurst < 0.45 else "RANDOM WALK"
    h_color = "#00FFFF" if hurst > 0.55 else "#8080FF" if hurst < 0.45 else "gray"
    
    if regime_data:
        c3.markdown(f"""
        <div class='terminal-box' style="padding:10px;">
            <div style="font-size:0.8em; color:#00FFFF;">QUANT REGIME</div>
            <div style="font-size:1.1em; font-weight:bold;" class='{regime_data['color']}'>{regime_data['regime']}</div>
            <div style="font-size:0.7em; display:flex; justify-content:space-between; margin-top:5px;">
                <span>FRACTAL:</span>
                <span style="color:{h_color}">{hurst_type}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        c3.info("Calculating Regime...")
    
    c4.metric("HIGH/LOW", f"{high.max():,.2f} / {low.min():,.2f}")

# ==============================================================================
# 3. MACRO & SENTIMENT CONTEXT (THE "WEATHER")
# ==============================================================================
st.markdown("---")
st.markdown("### 🌎 PHASE 1: MACRO & SENTIMENT CONTEXT")

macro_tab1, macro_tab2 = st.tabs(["🇺🇸 MACRO DASHBOARD (FRED)", "📰 NEWS & CALENDAR"])

with macro_tab1:
    if fred_key:
        macro_col_main, macro_col_ml = st.columns([3, 1])
        with macro_col_main:
            # RESTORED: Inner tabs to fit 4 charts properly
            mt_inner_1, mt_inner_2 = st.tabs(["YIELD CURVE & RATES", "INFLATION & LIQUIDITY"])
            
            with mt_inner_1:
                c_m1, c_m2 = st.columns(2)
                with c_m1:
                    # Yield Curve
                    if not df_yield.empty:
                        curr_yield = df_yield['value'].iloc[-1]
                        yield_color = "#8080FF" if curr_yield < 0 else "#00FFFF"
                        fig_yc = go.Figure()
                        fig_yc.add_trace(go.Scatter(x=df_yield.index, y=df_yield['value'], fill='tozeroy', fillcolor='rgba(102, 204, 255, 0.2)', line=dict(color=yield_color)))
                        fig_yc.add_hline(y=0, line_dash="dash", line_color="white")
                        fig_yc = terminal_chart_layout(fig_yc, title=f"10Y-2Y SPREAD: {curr_yield:.2f}%", height=200)
                        st.plotly_chart(fig_yc, use_container_width=True)
                        st.caption("CONTEXT: " + ("⚠️ RECESSION SIGNAL" if curr_yield < 0 else "NORMAL GROWTH"))
                with c_m2:
                     # RESTORED: Fed Funds Chart
                    if not df_ff.empty:
                        fig_ff = go.Figure()
                        fig_ff.add_trace(go.Scatter(x=df_ff.index, y=df_ff['value'], line=dict(color="#40E0FF")))
                        fig_ff = terminal_chart_layout(fig_ff, title=f"FED FUNDS: {df_ff['value'].iloc[-1]:.2f}%", height=200)
                        st.plotly_chart(fig_ff, use_container_width=True)
                        st.caption("CONTEXT: BASELINE RISK-FREE RATE")

            with mt_inner_2:
                c_m3, c_m4 = st.columns(2)
                with c_m3:
                    # CPI
                    if not df_cpi.empty:
                        df_cpi['YoY'] = df_cpi['value'].pct_change(12) * 100
                        fig_cpi = go.Figure()
                        fig_cpi.add_trace(go.Bar(x=df_cpi.index, y=df_cpi['YoY'], marker_color='#00FFFF'))
                        fig_cpi = terminal_chart_layout(fig_cpi, title=f"CPI (YoY): {df_cpi['YoY'].iloc[-1]:.2f}%", height=200)
                        st.plotly_chart(fig_cpi, use_container_width=True)
                        st.caption("TARGET: 2.0%")
                with c_m4:
                     # M2
                    if not df_m2.empty:
                        fig_m2 = go.Figure()
                        fig_m2.add_trace(go.Scatter(x=df_m2.index, y=df_m2['value'], line=dict(color="#00FFFF")))
                        fig_m2 = terminal_chart_layout(fig_m2, title="M2 LIQUIDITY", height=200)
                        st.plotly_chart(fig_m2, use_container_width=True)
                        st.caption("CONTEXT: MONEY SUPPLY")

        with macro_col_ml:
            st.markdown("**MACRO ML ENGINE**")
            if macro_regime:
                st.markdown(f"""
                <div class='terminal-box'>
                    <div style='color:#AAAAAA; font-size:0.8em;'>ECONOMIC REGIME</div>
                    <div style='color:#00FFFF; font-size:1.1em; font-weight:bold;'>{macro_regime['regime']}</div>
                    <hr>
                    <div style='font-size:0.8em;'>RATES: {macro_regime['rate']:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            # DXY COMPARISON
            if not correlations.empty and 'Dollar' in correlations:
                dxy_corr = correlations['Dollar']
                corr_color = "#00FFFF" if dxy_corr > 0.5 else "#8080FF" if dxy_corr < -0.5 else "white"
                st.markdown(f"""
                <div class='terminal-box' style='margin-top:10px;'>
                    <div style='color:#AAAAAA; font-size:0.8em;'>DXY CORRELATION</div>
                    <div style='color:{corr_color}; font-size:1.5em;'>{dxy_corr:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("Add `fred_api_key` to view Fed Macro Data.")

with macro_tab2:
    col_eco, col_news = st.columns([1, 1])
    
    # Economic Calendar Logic
    def parse_eco_value(val_str):
        if not isinstance(val_str, str) or val_str == '': return None
        clean = val_str.replace('%', '').replace(',', '')
        multiplier = 1.0
        if 'K' in clean.upper(): multiplier = 1000.0; clean = clean.upper().replace('K', '')
        elif 'M' in clean.upper(): multiplier = 1000000.0; clean = clean.upper().replace('M', '')
        elif 'B' in clean.upper(): multiplier = 1000000000.0; clean = clean.upper().replace('B', '')
        try: return float(clean) * multiplier
        except: return None

    def analyze_eco_context(actual_str, forecast_str, previous_str):
        is_happened = actual_str is not None and actual_str != ""
        val_actual = parse_eco_value(actual_str)
        val_forecast = parse_eco_value(forecast_str)
        val_prev = parse_eco_value(previous_str)
        context_str = ""
        bias = "Neutral"
        if is_happened:
            if val_actual is not None and val_forecast is not None:
                context_str = f"Act {actual_str} / Est {forecast_str}"
                delta = val_actual - val_forecast
                if delta > 0: bias = "Bullish"
                else: bias = "Bearish"
            else: context_str = f"Actual: {actual_str}"
        else:
             context_str = f"Est {forecast_str}" if forecast_str else "Waiting..."
        return context_str, bias

    with col_eco:
        st.markdown("**📅 HIGH IMPACT EVENTS (USD)**")
        if eco_events:
            cal_data = []
            for event in eco_events:
                context, bias = analyze_eco_context(event.get('actual', ''), event.get('forecast', ''), event.get('previous', ''))
                cal_data.append({"TIME": event.get('time', 'N/A'), "EVENT": event.get('event_name', 'Unknown'), "CONTEXT": context, "BIAS": bias})
            df_cal = pd.DataFrame(cal_data)
            def color_bias(val):
                color = '#CCCCCC'
                if 'Bullish' in val: color = '#00FFFF' 
                elif 'Bearish' in val: color = '#8080FF'
                return f'color: {color}'
            if not df_cal.empty: 
                st.dataframe(df_cal.style.applymap(color_bias, subset=['BIAS']), use_container_width=True, hide_index=True)
        else: st.info("No high impact events.")

    with col_news:
        st.markdown(f"**📰 {asset_info.get('news_query', 'LATEST')} WIRE**")
        if ai.HAS_NLP and not news_sentiment_df.empty:
             fig_sent = go.Figure()
             fig_sent.add_trace(go.Scatter(
                 x=news_sentiment_df.index, y=news_sentiment_df['cumulative'],
                 mode='lines+markers', line=dict(color='#00FFFF', width=2), name="Sentiment"))
             fig_sent = terminal_chart_layout(fig_sent, title="SENTIMENT VELOCITY", height=150)
             fig_sent.update_layout(xaxis=dict(showgrid=False, visible=False))
             st.plotly_chart(fig_sent, use_container_width=True)
        
        # RESTORED: Tabs for General vs Forex Factory
        tab_gen, tab_ff = st.tabs(["📰 GENERAL", "⚡ FOREX FACTORY"])
        
        def render_news(items):
            if items:
                for news in items:
                    st.markdown(f"""
                    <div style="border-bottom:1px solid #333; padding-bottom:5px; margin-bottom:5px;">
                        <a class='news-link' href='{news['url']}' target='_blank'>▶ {news['title']}</a><br>
                        <span style='font-size:0.7em; color:#AAAAAA;'>{news['time']} | {news['source']}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else: st.markdown("<div style='color:gray;'>No data.</div>", unsafe_allow_html=True)
        
        with tab_gen: render_news(news_general)
        with tab_ff: render_news(news_ff)

# ==============================================================================
# 4. STRATEGIC ANALYSIS (TREND & POSITIONING)
# ==============================================================================
st.markdown("---")
st.markdown("### 🔭 PHASE 2: STRATEGIC ANALYSIS (Trend & Positioning)")

strat_col1, strat_col2 = st.columns([2, 1])

with strat_col1:
    # --- CHART: MULTILAYER LIQUIDITY MAP ---
    if not daily_data.empty:
        fig = go.Figure()
        
        # Trace 1: Candles
        fig.add_trace(go.Candlestick(
            x=daily_data.index, open=daily_data['Open'], high=high, low=low, close=close, name="Price",
            increasing_line_color="#00FFFF", decreasing_line_color="#405060"
        ))
        
        # Trace 2: FVG
        for fvg in active_fvgs:
            color = "rgba(0, 255, 255, 0.15)" if "Bullish" in fvg['type'] else "rgba(128, 128, 255, 0.15)"
            fig.add_shape(type="rect", x0=fvg['date'], x1=daily_data.index[-1], y0=fvg['bottom'], y1=fvg['top'], fillcolor=color, line_width=0)

        # Trace 3: Swing Points
        sh_mask = ms_df['Structure'] == 'SH'
        sl_mask = ms_df['Structure'] == 'SL'
        fig.add_trace(go.Scatter(x=ms_df[sh_mask].index, y=ms_df[sh_mask]['High'], mode='markers', marker=dict(symbol='triangle-down', size=8, color='#8080FF'), name='Swing High'))
        fig.add_trace(go.Scatter(x=ms_df[sl_mask].index, y=ms_df[sl_mask]['Low'], mode='markers', marker=dict(symbol='triangle-up', size=8, color='#00FFFF'), name='Swing Low'))

        if poc_price: fig.add_hline(y=poc_price, line_dash="dash", line_color="#CCCCCC", annotation_text="POC")

        # Trace 4: DXY Overlay
        if not dxy_data.empty:
            dxy_aligned = dxy_data['Close'].reindex(daily_data.index, method='ffill')
            fig.add_trace(go.Scatter(x=dxy_aligned.index, y=dxy_aligned.values, name="DXY", line=dict(color='#8080FF', width=2), opacity=0.5, yaxis="y2"))

        fig = terminal_chart_layout(fig, height=500)
        fig.update_layout(
            yaxis=dict(title="Price"),
            yaxis2=dict(title="DXY", overlaying="y", side="right", showgrid=False, tickfont=dict(color="#8080FF")),
            legend=dict(orientation="h", y=1.02, x=0, bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig, use_container_width=True)

with strat_col2:
    # --- MARKET STRUCTURE DETAILS ---
    trend_color = "bullish" if "BULLISH" in ms_trend else "bearish" if "BEARISH" in ms_trend else "neutral"
    st.markdown(f"""
    <div class='terminal-box'>
        <div style='color:#AAAAAA; font-size:0.8em;'>MARKET STRUCTURE (Trend)</div>
        <div style='font-size:1.2em; font-weight:bold;'>{ms_trend}</div>
        <hr style='margin:5px 0;'>
        <div style='font-size:0.8em;'>Last High: <span style='color:#8080FF'>{ms_last_sh:,.2f}</span></div>
        <div style='font-size:0.8em;'>Last Low: <span style='color:#00FFFF'>{ms_last_sl:,.2f}</span></div>
    </div>
    """, unsafe_allow_html=True)
    
    # --- MONTE CARLO ---
    st.markdown("**🎲 PROBABILITY PATH**")
    if pred_dates is not None and pred_paths is not None:
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(x=pred_dates, y=np.mean(pred_paths, axis=1), name='Avg Path', line=dict(color='#00FFFF', dash='dash')))
        fig_pred = terminal_chart_layout(fig_pred, title="MC FORECAST (126 Days)", height=200)
        st.plotly_chart(fig_pred, use_container_width=True)
        
    # --- RESTORED: SEASONALITY TABS ---
    if seasonality_stats:
        st.markdown("**⏳ SEASONAL TENDENCIES**")
        tab_hour, tab_day, tab_week = st.tabs(["HOUR (NY)", "DAY", "WEEK"])
        
        with tab_hour:
            if 'hourly_perf' in seasonality_stats and seasonality_stats['hourly_perf'] is not None:
                hp = seasonality_stats['hourly_perf']
                fig_h = go.Figure()
                colors = ['#00FFFF' if v > 0 else '#8080FF' for v in hp.values]
                fig_h.add_trace(go.Bar(x=[f"{h:02d}:00" for h in hp.index], y=hp.values, marker_color=colors))
                fig_h = terminal_chart_layout(fig_h, title="AVG RETURN BY HOUR", height=200)
                st.plotly_chart(fig_h, use_container_width=True)
        
        with tab_day:
            if 'day_high' in seasonality_stats:
                fig_d = go.Figure()
                fig_d.add_trace(go.Bar(x=seasonality_stats['day_high'].index, y=seasonality_stats['day_high'].values, marker_color='#00FFFF'))
                fig_d = terminal_chart_layout(fig_d, title="PROB OF HIGH OF WEEK", height=200)
                st.plotly_chart(fig_d, use_container_width=True)

        with tab_week:
            if 'week_returns' in seasonality_stats:
                 wr = seasonality_stats['week_returns']
                 fig_w = go.Figure()
                 colors = ['#00FFFF' if v > 0 else '#8080FF' for v in wr.values]
                 fig_w.add_trace(go.Bar(x=["Wk 1", "Wk 2", "Wk 3", "Wk 4", "Wk 5"], y=wr.values, marker_color=colors))
                 fig_w = terminal_chart_layout(fig_w, title="MONTHLY SEASONALITY", height=200)
                 st.plotly_chart(fig_w, use_container_width=True)

# --- 4B. COT & FUNDAMENTALS (RESTORED FULL DETAIL) ---
with st.expander("🏛️ INSTITUTIONAL POSITIONING (COT) & FUNDAMENTALS", expanded=True):
    
    # HELPER FUNCTION FOR TEXT ANALYSIS
    def generate_cot_analysis(spec_net, hedge_net, spec_label, hedge_label):
        spec_sent = "🟢 BULLISH" if spec_net > 0 else "🔴 BEARISH"
        hedge_sent = "🟢 BULLISH" if hedge_net > 0 else "🔴 BEARISH"
        if (spec_net > 0 and hedge_net < 0) or (spec_net < 0 and hedge_net > 0):
            structure = "✅ **Healthy Structure:** Risk Transfer active."
        else:
            structure = "⚠️ **Anomaly:** Groups positioned same side."
        return f"* **{spec_label}:** {spec_sent} (Net: {int(spec_net):,})\n* **{hedge_label}:** {hedge_sent}\n{structure}"

    cot_col, fund_col = st.columns([2, 1])
    
    with cot_col:
        st.markdown("**COT FUTURES POSITIONING**")
        # 1. Fetch COT
        cot_history = de.fetch_cot_history(selected_asset, start_year=2024)
        if cot_history is not None and not cot_history.empty:
            cot_config = config.COT_MAPPING[selected_asset]
            spec_label, hedge_label = cot_config['labels']
            
            # Calculations
            cot_history['Net Speculator'] = cot_history['spec_long'] - cot_history['spec_short']
            cot_history['Net Hedger'] = cot_history['hedge_long'] - cot_history['hedge_short']
            cot_history['Spec Z-Score'] = qe.calculate_z_score(cot_history['Net Speculator'])
            latest_cot = cot_history.iloc[-1]
            prev_cot = cot_history.iloc[-2] if len(cot_history) > 1 else latest_cot
            
            # Capture for AI
            cot_data = {"sentiment": "BULLISH" if latest_cot['Net Speculator'] > 0 else "BEARISH", "net_spec": latest_cot['Net Speculator'], "z_score": latest_cot['Spec Z-Score']}
            
            # RESTORED: METRICS ROW
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric(f"{spec_label} (Net)", f"{int(latest_cot['Net Speculator']):,}", f"{int(latest_cot['Net Speculator'] - prev_cot['Net Speculator']):,}")
            mc2.metric(f"{hedge_label} (Net)", f"{int(latest_cot['Net Hedger']):,}", f"{int(latest_cot['Net Hedger'] - prev_cot['Net Hedger']):,}", delta_color="inverse")
            mc3.metric("Z-Score", f"{latest_cot['Spec Z-Score']:.2f}σ", "Extreme" if abs(latest_cot['Spec Z-Score']) > 2 else "Neutral", delta_color="off")
            
            # RESTORED: Interpretation Box
            cot_txt = generate_cot_analysis(latest_cot['Net Speculator'], latest_cot['Net Hedger'], spec_label, hedge_label)
            st.info(cot_txt)

            # RESTORED: TABS FOR CHARTS
            tab_trend, tab_struct, tab_osc = st.tabs(["📈 NET TREND", "🦋 STRUCTURE", "📊 Z-SCORE"])
            
            with tab_trend:
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(x=cot_history['date'], y=cot_history['Net Speculator'], name=spec_label, line=dict(color='#00FFFF', width=2)))
                fig_trend.add_trace(go.Scatter(x=cot_history['date'], y=cot_history['Net Hedger'], name=hedge_label, line=dict(color='#8080FF', width=2)))
                fig_trend.add_hline(y=0, line_dash="dash", line_color="gray")
                fig_trend = terminal_chart_layout(fig_trend, title="NET POSITIONING HISTORY", height=300)
                st.plotly_chart(fig_trend, use_container_width=True)

            with tab_struct:
                fig_struct = go.Figure()
                fig_struct.add_trace(go.Bar(x=cot_history['date'], y=cot_history['spec_long'], name=f"{spec_label} Longs", marker_color='#00FFFF'))
                fig_struct.add_trace(go.Bar(x=cot_history['date'], y=-cot_history['spec_short'], name=f"{spec_label} Shorts", marker_color='#8080FF'))
                fig_struct.update_layout(barmode='overlay')
                fig_struct = terminal_chart_layout(fig_struct, title="BUTTERFLY CHART (Long vs Short)", height=300)
                st.plotly_chart(fig_struct, use_container_width=True)

            with tab_osc:
                fig_z = go.Figure()
                colors = ['#8080FF' if val > 2 or val < -2 else '#333' for val in cot_history['Spec Z-Score']]
                fig_z.add_trace(go.Bar(x=cot_history['date'], y=cot_history['Spec Z-Score'], marker_color=colors))
                fig_z.add_hline(y=2, line_dash="dot", line_color="#8080FF")
                fig_z.add_hline(y=-2, line_dash="dot", line_color="#8080FF")
                fig_z = terminal_chart_layout(fig_z, title="OSCILLATOR (Z-Score)", height=300)
                st.plotly_chart(fig_z, use_container_width=True)

        else:
            st.info("COT Data unavailable for this asset.")

    with fund_col:
        cg_id = asset_info.get('cg_id')
        if cg_id and cg_key:
            st.markdown("**🦎 COINGECKO FUNDAMENTALS**")
            cg_data = de.get_coingecko_stats(cg_id, cg_key)
            if cg_data:
                st.metric("Market Rank", f"#{cg_data['rank']}")
                ath_color = "#8080FF" if cg_data['ath_change'] < -20 else "#00FFFF"
                st.markdown(f"**ATH Drawdown:** <span style='color:{ath_color}'>{cg_data['ath_change']:.2f}%</span>", unsafe_allow_html=True)
                st.progress(cg_data['sentiment'])
                st.caption(f"Sentiment: {cg_data['sentiment']}% Bullish")
                
                # RESTORED: Algo & Description
                st.markdown(f"**Algorithm:** `{cg_data['hashing']}`")
                with st.expander("Asset Description"):
                    st.write(cg_data['desc'])

# ==============================================================================
# 5. MARKET DYNAMICS (VOLATILITY & FLOW)
# ==============================================================================
st.markdown("---")
st.markdown("### ⚙️ PHASE 3: MARKET DYNAMICS (Vol & Flow)")

dyn_col1, dyn_col2, dyn_col3 = st.columns(3)

with dyn_col1:
    # GEX
    if gex_df is not None:
        total_gex = gex_df['gex'].sum() / 1_000_000
        sent_color = "bullish" if total_gex > 0 else "bearish"
        st.markdown(f"""
        <div class='terminal-box'>
            <div style='color:#00FFFF;'>NET GAMMA EXPOSURE</div>
            <div style='font-size:1.5em; color:white;'>${total_gex:.1f}M</div>
            <span class='{sent_color}'>{"STICKY (Low Vol)" if total_gex > 0 else "SLIPPERY (High Vol)"}</span>
        </div>
        """, unsafe_allow_html=True)
        
        center_strike = gex_spot 
        gex_zoom = gex_df[(gex_df['strike'] > center_strike * 0.9) & (gex_df['strike'] < center_strike * 1.1)]
        fig_gex = go.Figure()
        colors = ['#00FFFF' if x > 0 else '#8080FF' for x in gex_zoom['gex']]
        fig_gex.add_trace(go.Bar(x=gex_zoom['strike'], y=gex_zoom['gex'], marker_color=colors))
        fig_gex.add_vline(x=center_strike, line_dash="dot", line_color="white")
        fig_gex = terminal_chart_layout(fig_gex, title="GAMMA PROFILE", height=200)
        st.plotly_chart(fig_gex, use_container_width=True)
    else: st.info("No Options Data")

with dyn_col2:
    # VOL CONE & IV
    if not daily_data.empty:
        hv_current = daily_data['Close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100
    else: hv_current = 0
    iv_display = current_iv if current_iv else 0
    vol_premium = iv_display - hv_current
    
    vol_regime = vol_cone.get('regime', 'N/A')
    vol_rank = vol_cone.get('rank', 0.5)
    
    st.markdown(f"""
    <div class='terminal-box'>
        <div style='color:#AAAAAA; font-size:0.8em;'>VOLATILITY SURFACE</div>
        <div style='font-size:1.1em; font-weight:bold;'>{vol_regime}</div>
        <div style='font-size:0.8em;'>Rank: {vol_rank*100:.0f}%</div>
        <hr style='margin:5px 0;'>
        <div style='display:flex; justify-content:space-between;'>
            <span>IV: {iv_display:.1f}%</span>
            <span>HV: {hv_current:.1f}%</span>
        </div>
        <div style='color:{"#8080FF" if vol_premium > 0 else "#00FFFF"}; font-weight:bold;'>Prem: {vol_premium:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with dyn_col3:
    # ORDER FLOW PROXY
    of_color = "bullish" if "Buying" in of_bias else "bearish"
    st.markdown(f"""
    <div class='terminal-box'>
        <div style='color:#AAAAAA; font-size:0.8em;'>ORDER FLOW PRESSURE</div>
        <div style='font-size:1.2em; font-weight:bold;' class='{of_color}'>{of_bias}</div>
        <div style='font-size:0.8em; color:gray;'>Volume-Weighted Impulse</div>
    </div>
    """, unsafe_allow_html=True)
    if vol_profile is not None:
         fig_vp = go.Figure()
         colors = ['#00FFFF' if x == poc_price else '#333' for x in vol_profile['PriceLevel']]
         fig_vp.add_trace(go.Bar(y=vol_profile['PriceLevel'], x=vol_profile['Volume'], orientation='h', marker_color='#40E0FF', opacity=0.4))
         fig_vp.add_hline(y=poc_price, line_dash="dash", line_color="#FFFFFF", annotation_text="POC")
         fig_vp = terminal_chart_layout(fig_vp, title="INTRADAY VOLUME PROFILE", height=200)
         st.plotly_chart(fig_vp, use_container_width=True)

# ==============================================================================
# 6. TACTICAL EXECUTION (ENTRY/EXIT)
# ==============================================================================
st.markdown("---")
st.markdown("### 🎯 PHASE 4: TACTICAL EXECUTION")

exe_col1, exe_col2 = st.columns([2, 1])

with exe_col1:
    st.markdown("**SESSION VWAP + KEY LEVELS**")
    if not vwap_df.empty:
        fig_vwap = go.Figure()
        fig_vwap.add_trace(go.Candlestick(x=vwap_df.index, open=vwap_df['Open'], high=vwap_df['High'], low=vwap_df['Low'], close=vwap_df['Close'], name="Price", increasing_line_color="#00FFFF", decreasing_line_color="#405060"))
        fig_vwap.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df['VWAP'], name="Session VWAP", line=dict(color='#FFFFFF', width=2)))
        fig_vwap.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df['Upper_Band_1'], name="+1 STD", line=dict(color='gray', width=1), opacity=0.3))
        fig_vwap.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df['Lower_Band_1'], name="-1 STD", line=dict(color='gray', width=1), opacity=0.3))
        
        if key_levels:
            fig_vwap.add_hline(y=key_levels['PDH'], line_dash="dot", line_color="#8080FF", annotation_text="PDH")
            fig_vwap.add_hline(y=key_levels['PDL'], line_dash="dot", line_color="#00FFFF", annotation_text="PDL")
            fig_vwap.add_hline(y=key_levels['Pivot'], line_width=1, line_color="#40E0FF", annotation_text="DAILY PIVOT")
            
        fig_vwap = terminal_chart_layout(fig_vwap, height=450)
        st.plotly_chart(fig_vwap, use_container_width=True)

        # RESTORED: Key Levels Text List
        if key_levels:
            st.markdown("#### 🔑 ALGO LEVELS")
            cur_price = intraday_data['Close'].iloc[-1] if not intraday_data.empty else 0
            levels_list = [("R1 (Resist)", key_levels['R1']), ("PDH (High)", key_levels['PDH']), ("PIVOT", key_levels['Pivot']), ("PDL (Low)", key_levels['PDL']), ("S1 (Support)", key_levels['S1'])]
            
            c_lvl_cols = st.columns(5)
            for i, (name, price) in enumerate(levels_list):
                 dist = abs(price - cur_price) / cur_price
                 color = "#FFFF00" if dist < 0.002 else "#8080FF" if price > cur_price else "#00FFFF"
                 with c_lvl_cols[i]:
                     st.markdown(f"<div style='font-size:0.8em; color:gray;'>{name}</div><div style='color:{color}; font-family:monospace;'>{price:,.2f}</div>", unsafe_allow_html=True)

with exe_col2:
    st.markdown("**INTRADAY ALPHA (vs SPY)**")
    if not rs_data.empty:
        curr_rs = rs_data['RS_Score'].iloc[-1]
        rs_color = "#00FFFF" if curr_rs > 0 else "#8080FF"
        rs_text = "OUTPERFORMING" if curr_rs > 0 else "UNDERPERFORMING"
        st.markdown(f"<span style='color:{rs_color}; font-weight:bold;'>{rs_text}</span>", unsafe_allow_html=True)
        
        fig_rs = go.Figure()
        fig_rs.add_hline(y=0, line_color="#333", line_dash="dash")
        fig_rs.add_trace(go.Scatter(x=rs_data.index, y=rs_data['RS_Score'], mode='lines', line=dict(color=rs_color, width=2), fill='tozeroy', fillcolor='rgba(102, 204, 255, 0.2)'))
        fig_rs = terminal_chart_layout(fig_rs, height=150)
        st.plotly_chart(fig_rs, use_container_width=True)
    
    st.markdown("**RISK / BACKTEST**")
    strat_perf = qe.run_strategy_backtest(asset_info['ticker'])
    if strat_perf:
        sig_color = "#00FFFF" if "LONG" in strat_perf['signal'] else "#8080FF"
        st.markdown(f"Signal: <span style='color:{sig_color}; font-weight:bold;'>{strat_perf['signal']}</span>", unsafe_allow_html=True)
        st.metric("Sharpe", f"{strat_perf['sharpe']:.2f}")
        
        # RESTORED: Equity Curve Chart
        ec_df = pd.DataFrame({"Strategy": strat_perf['equity_curve'], "Buy & Hold": strat_perf['df']['Cum_BnH']})
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(x=ec_df.index, y=ec_df['Buy & Hold'], name="Buy & Hold", line=dict(color='#8080FF', dash='dot')))
        fig_perf.add_trace(go.Scatter(x=ec_df.index, y=ec_df['Strategy'], name="Active Strat", line=dict(color='#00FFFF', width=2), fill='tozeroy', fillcolor='rgba(102, 204, 255, 0.1)'))
        fig_perf = terminal_chart_layout(fig_perf, title="STRATEGY EDGE", height=200)
        st.plotly_chart(fig_perf, use_container_width=True)

# ==============================================================================
# 7. AI SYNTHESIS (THE CONCLUSION)
# ==============================================================================
st.markdown("---")
st.markdown("### 🧠 PHASE 5: AI SYNTHESIS")

# Prepare Data for LLM
gex_summary = gex_df if gex_df is not None else None
ml_signal_str = "BULLISH" if ml_prob > 0.55 else "BEARISH" if ml_prob < 0.45 else "NEUTRAL"
news_text_summary = "\n".join([f"- {n['title']} ({n['source']})" for n in combined_news_for_llm])

if gemini_key:
    c_ai1, c_ai2 = st.columns([1, 4])
    with c_ai1:
        if st.button("📝 GENERATE EXECUTIVE BRIEF"):
            with st.spinner("Synthesizing..."):
                narrative = ai.get_technical_narrative(
                    ticker=selected_asset, price=curr, daily_pct=pct, regime=regime_data,
                    ml_signal=ml_signal_str, gex_data=gex_summary, cot_data=cot_data,
                    levels=key_levels, macro_data=macro_context_data, api_key=gemini_key
                )
                st.session_state['narrative_cache'] = narrative
                st.rerun()
                
        if st.button("🔎 GENERATE DEEP THESIS"):
             with st.spinner("Writing Thesis..."):
                thesis_text = ai.generate_deep_dive_thesis(
                    ticker=selected_asset, price=curr, change=pct, regime=regime_data,
                    ml_signal=ml_signal_str, gex_data=gex_summary, cot_data=cot_data,
                    levels=key_levels, news_summary=news_text_summary, macro_data=macro_context_data, api_key=gemini_key
                )
                st.session_state['thesis_cache'] = thesis_text
                st.rerun()

    with c_ai2:
        if st.session_state['narrative_cache']:
             st.markdown(f"""
            <div class='terminal-box' style='border-left: 4px solid #00FFFF; margin-bottom:10px;'>
                <div style='font-family: monospace; white-space: pre-wrap;'>{st.session_state['narrative_cache']}</div>
            </div>
            """, unsafe_allow_html=True)
            
        if st.session_state['thesis_cache']:
            st.markdown(f"""
            <div class='terminal-box' style='padding: 20px;'>
                {st.session_state['thesis_cache']}
            </div>
            """, unsafe_allow_html=True)
else:
    st.warning("Add GEMINI_API_KEY to enable AI Synthesis.")
