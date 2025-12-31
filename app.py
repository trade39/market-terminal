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
st.set_page_config(layout="wide", page_title="Bloomberg Terminal Pro V5.23", page_icon="⚡")
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

# --- DATA LOADING ---
daily_data = de.get_daily_data(asset_info['ticker'])
dxy_data = de.get_dxy_data(fred_key) 
intraday_data = de.get_intraday_data(asset_info['ticker'])
eco_events = de.get_economic_calendar(rapid_key, use_demo=use_demo_data)

# Fetch News
news_general = de.get_financial_news_general(news_key, query=asset_info.get('news_query', 'Finance'))
news_ff = de.get_forex_factory_news(rapid_key, 'breaking')
combined_news_for_llm = news_general[:5] + news_ff[:5]

# Quant Engines
_, ml_prob = qe.get_ml_prediction(asset_info['ticker'])
gex_df, gex_date, gex_spot, current_iv = qe.get_gex_profile(asset_info['opt_ticker'])
vol_profile, poc_price = qe.calculate_volume_profile(intraday_data)
hurst = qe.calculate_hurst(daily_data['Close'].values) if not daily_data.empty else 0.5
regime_data = qe.get_market_regime(asset_info['ticker'])
correlations = qe.get_correlations(asset_info['ticker'], fred_key)
news_sentiment_df = ai.calculate_news_sentiment(combined_news_for_llm)
seasonality_stats = qe.get_seasonality_stats(daily_data, asset_info['ticker'])

# Multilayer Setup Engines
ms_df, ms_trend, ms_last_sh, ms_last_sl = qe.detect_market_structure(daily_data)
vol_cone = qe.get_volatility_cone(daily_data)
of_df, of_bias = qe.calculate_order_flow_proxy(daily_data)
active_fvgs = qe.detect_fair_value_gaps(daily_data)

# FRED Engines
if fred_key:
    df_yield = de.get_fred_series("T10Y2Y", fred_key)
    df_ff = de.get_fred_series("FEDFUNDS", fred_key)
    df_cpi = de.get_fred_series("CPIAUCSL", fred_key)
    df_m2 = de.get_fred_series("M2SL", fred_key)
    macro_regime = qe.get_macro_ml_regime(df_cpi, df_ff)
else:
    df_yield, df_ff, df_cpi, df_m2, macro_regime = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None

# Prepare Context Data for AI
macro_context_data = {}
if not df_yield.empty: macro_context_data['yield_curve'] = f"{df_yield['value'].iloc[-1]:.2f}"
if not df_cpi.empty: macro_context_data['cpi'] = f"{(df_cpi['value'].pct_change(12).iloc[-1]*100):.2f}"
if not df_ff.empty: macro_context_data['rates'] = f"{df_ff['value'].iloc[-1]:.2f}"
if macro_regime: macro_context_data['regime'] = macro_regime['regime']

cot_data = None # Placeholder

# --- 0. HEADS UP DISPLAY (HUD) ---
st.markdown(f"<h1 style='border-bottom: 2px solid #00FFFF;'>{selected_asset} <span style='font-size:0.5em; color:#AAAAAA;'>TERMINAL PRO V5.23</span></h1>", unsafe_allow_html=True)

if not daily_data.empty:
    close, high, low = daily_data['Close'], daily_data['High'], daily_data['Low']
    curr = close.iloc[-1]
    pct = ((curr - close.iloc[-2]) / close.iloc[-2]) * 100
    
    # HUD Layout
    hud1, hud2, hud3, hud4 = st.columns(4)
    hud1.metric("LAST PX", f"{curr:,.2f}", f"{pct:.2f}%")
    
    ml_bias = "BULLISH" if ml_prob > 0.55 else "BEARISH" if ml_prob < 0.45 else "NEUTRAL"
    ml_conf = abs(ml_prob - 0.5) * 200
    ml_color = "bullish" if ml_bias == "BULLISH" else "bearish" if ml_bias == "BEARISH" else "neutral"
    
    hud2.markdown(f"""
    <div class='terminal-box' style="text-align:center; padding:5px;">
        <div style="font-size:0.8em; color:#00FFFF;">AI FORECAST</div>
        <span class='{ml_color}'>{ml_bias}</span>
        <div style="font-size:0.8em; margin-top:5px; color:#AAAAAA;">CONF: {ml_conf:.0f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    if regime_data:
        hud3.markdown(f"""
        <div class='terminal-box' style="text-align:center; padding:5px;">
            <div style="font-size:0.8em; color:#00FFFF;">QUANT REGIME</div>
            <div style="font-size:1.0em; font-weight:bold; color:white;">{regime_data['regime']}</div>
            <div style="font-size:0.7em; color:#AAAAAA;">VOL: {regime_data['color'].upper()}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # AI BRIEF BUTTON
    if gemini_key:
        if hud4.button("📝 GENERATE BRIEF"):
            with st.spinner("Analyzing Technicals + Macro..."):
                narrative = ai.get_technical_narrative(
                    ticker=selected_asset, price=curr, daily_pct=pct, regime=regime_data,
                    ml_signal=ml_bias, gex_data=gex_df, cot_data=cot_data,
                    levels=qe.get_key_levels(daily_data), macro_data=macro_context_data, api_key=gemini_key
                )
                st.session_state['narrative_cache'] = narrative
                st.rerun()
    
    # Display AI Result immediately if exists
    if st.session_state['narrative_cache']:
        if "⚠️" in st.session_state['narrative_cache']: st.error(st.session_state['narrative_cache'])
        else:
            st.markdown(f"""
            <div class='terminal-box' style='border-left: 4px solid #00FFFF; margin-bottom: 20px;'>
                <div style='font-family: monospace; font-size: 0.9em; white-space: pre-wrap;'>{st.session_state['narrative_cache']}</div>
            </div>
            """, unsafe_allow_html=True)

# --- SYSTEMATIC DECISION FUNNEL ---
tab_macro, tab_struct, tab_liq, tab_exec = st.tabs([
    "1. MACRO & CONTEXT (WHY)", 
    "2. STRUCTURE & TREND (WHAT)", 
    "3. LIQUIDITY & LEVELS (WHERE)", 
    "4. EXECUTION & RISK (WHEN)"
])

# ==========================================
# PHASE 1: MACRO & CONTEXT (The "Why")
# ==========================================
with tab_macro:
    m_col1, m_col2 = st.columns([2, 1])
    
    with m_col1:
        st.markdown("#### 📅 ECONOMIC EVENTS & NEWS")
        
        # Economic Calendar
        if eco_events:
            cal_data = []
            for event in eco_events:
                # Basic parsing for display
                cal_data.append({
                    "TIME": event.get('time', 'N/A'), 
                    "EVENT": event.get('event_name', 'Unknown'), 
                    "ACTUAL": event.get('actual', '-'),
                    "FORECAST": event.get('forecast', '-')
                })
            df_cal = pd.DataFrame(cal_data)
            st.dataframe(df_cal, use_container_width=True, hide_index=True)
        else: st.info("No high-impact USD events scheduled.")
        
        # News Sentiment Chart
        if not news_sentiment_df.empty:
             fig_sent = go.Figure()
             fig_sent.add_trace(go.Scatter(
                 x=news_sentiment_df.index, y=news_sentiment_df['cumulative'],
                 mode='lines+markers', line=dict(color='#00FFFF', width=2), name="Sentiment"
             ))
             fig_sent = terminal_chart_layout(fig_sent, title="NEWS SENTIMENT VELOCITY", height=200)
             fig_sent.update_layout(xaxis=dict(showgrid=False, visible=False))
             st.plotly_chart(fig_sent, use_container_width=True)
        
        # News Feed Lists
        subtab_gen, subtab_ff = st.tabs(["GENERAL WIRE", "FOREX FACTORY"])
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
        
        with subtab_gen: render_news(news_general)
        with subtab_ff: render_news(news_ff)

    with m_col2:
        st.markdown("#### 🇺🇸 MACRO DASHBOARD")
        if macro_regime:
            st.markdown(f"""
            <div class='terminal-box'>
                <div style='color:#AAAAAA; font-size:0.8em;'>MACRO REGIME (ML)</div>
                <div style='color:#00FFFF; font-size:1.1em; font-weight:bold;'>{macro_regime['regime']}</div>
                <hr>
                <div style='font-size:0.8em;'>CPI YoY: {macro_regime['cpi']:.1f}%</div>
                <div style='font-size:0.8em;'>Fed Rates: {macro_regime['rate']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        # DXY Correlation
        if not correlations.empty and 'Dollar' in correlations:
            dxy_corr = correlations['Dollar']
            corr_color = "#00FFFF" if dxy_corr > 0.5 else "#8080FF" if dxy_corr < -0.5 else "white"
            st.markdown(f"""
            <div class='terminal-box'>
                <div style='color:#AAAAAA; font-size:0.8em;'>DXY CORRELATION</div>
                <div style='color:{corr_color}; font-size:1.5em;'>{dxy_corr:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        # CoinGecko Fundamentals (RESTORED)
        cg_id = asset_info.get('cg_id')
        if cg_id and cg_key:
            cg_data = de.get_coingecko_stats(cg_id, cg_key)
            if cg_data:
                st.markdown(f"""
                <div class='terminal-box'>
                    <div style='color:#AAAAAA; font-size:0.8em;'>COINGECKO RANK</div>
                    <div style='color:#FFFFFF; font-size:1.2em;'>#{cg_data['rank']}</div>
                    <hr>
                    <div style='font-size:0.8em;'>Sentiment: {cg_data['sentiment']}% Bullish</div>
                    <div style='font-size:0.8em;'>ATH Drop: {cg_data['ath_change']:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

        # COT Data
        with st.spinner("Loading COT Data..."):
            cot_history = de.fetch_cot_history(selected_asset, start_year=2024)
        if cot_history is not None and not cot_history.empty:
            cot_config = config.COT_MAPPING[selected_asset]
            cot_history['Net Speculator'] = cot_history['spec_long'] - cot_history['spec_short']
            latest_net = cot_history['Net Speculator'].iloc[-1]
            st.markdown(f"""
            <div class='terminal-box'>
                <div style='color:#AAAAAA; font-size:0.8em;'>COT NET SPECULATOR</div>
                <div style='color:#FFFFFF; font-size:1.2em;'>{int(latest_net):,}</div>
                <div style='font-size:0.7em; color:#AAAAAA;'>{cot_config['labels'][0]}</div>
            </div>
            """, unsafe_allow_html=True)
            cot_data = {"sentiment": "BULLISH" if latest_net > 0 else "BEARISH", "net_spec": latest_net}

# ==========================================
# PHASE 2: STRUCTURE & TREND (The "What")
# ==========================================
with tab_struct:
    s_col1, s_col2 = st.columns([3, 1])
    
    with s_col1:
        # --- PRIMARY CHART WITH STRUCTURE ---
        fig = go.Figure()
        
        # Candlesticks
        fig.add_trace(go.Candlestick(
            x=daily_data.index, open=daily_data['Open'], high=high, low=low, close=close, name="Price",
            increasing_line_color="#00FFFF", decreasing_line_color="#405060"
        ))
        
        # Swing Points
        sh_mask = ms_df['Structure'] == 'SH'
        sl_mask = ms_df['Structure'] == 'SL'
        fig.add_trace(go.Scatter(x=ms_df[sh_mask].index, y=ms_df[sh_mask]['High'], mode='markers', marker=dict(symbol='triangle-down', size=8, color='#8080FF'), name='Swing High'))
        fig.add_trace(go.Scatter(x=ms_df[sl_mask].index, y=ms_df[sl_mask]['Low'], mode='markers', marker=dict(symbol='triangle-up', size=8, color='#00FFFF'), name='Swing Low'))
        
        # FVG Rectangles
        for fvg in active_fvgs:
            color = "rgba(0, 255, 255, 0.15)" if "Bullish" in fvg['type'] else "rgba(128, 128, 255, 0.15)"
            fig.add_shape(type="rect", x0=fvg['date'], x1=daily_data.index[-1], y0=fvg['bottom'], y1=fvg['top'], fillcolor=color, line_width=0)

        # DXY Overlay
        if not dxy_data.empty:
            dxy_aligned = dxy_data['Close'].reindex(daily_data.index, method='ffill')
            fig.add_trace(go.Scatter(x=dxy_aligned.index, y=dxy_aligned.values, name="DXY (FRED)", line=dict(color='#8080FF', width=2), opacity=0.5, yaxis="y2"))

        fig = terminal_chart_layout(fig, height=500, title="MARKET STRUCTURE & LIQUIDITY MAP")
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, title="DXY", title_font=dict(color="#8080FF"), tickfont=dict(color="#8080FF")))
        st.plotly_chart(fig, use_container_width=True)
        
    with s_col2:
        st.markdown("#### 🧬 QUANT METRICS")
        
        # Market Structure
        ms_color = "bullish" if "BULLISH" in ms_trend else "bearish" if "BEARISH" in ms_trend else "neutral"
        st.markdown(f"""
        <div class='terminal-box'>
            <div style='color:#AAAAAA; font-size:0.8em;'>STRUCTURE (BOS/CHoCH)</div>
            <div style='font-size:1.1em; font-weight:bold;'>{ms_trend}</div>
            <span class='{ms_color}' style='font-size:0.7em;'>{ms_color.upper()}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Volatility
        vol_regime = vol_cone.get('regime', 'N/A')
        st.markdown(f"""
        <div class='terminal-box'>
            <div style='color:#AAAAAA; font-size:0.8em;'>VOLATILITY (Garman-Klass)</div>
            <div style='font-size:1.1em; font-weight:bold;'>{vol_regime}</div>
            <progress value="{int(vol_cone.get('rank', 0.5)*100)}" max="100" style="width:100%; height:5px;"></progress>
        </div>
        """, unsafe_allow_html=True)
        
        # Order Flow
        of_color = "bullish" if "Buying" in of_bias else "bearish"
        st.markdown(f"""
        <div class='terminal-box'>
            <div style='color:#AAAAAA; font-size:0.8em;'>ORDER FLOW PROXY</div>
            <div style='font-size:1.1em; font-weight:bold;' class='{of_color}'>{of_bias}</div>
        </div>
        """, unsafe_allow_html=True)

    # --- SEASONALITY CHARTS (RESTORED) ---
    st.markdown("#### ⏳ SEASONAL TENDENCIES")
    if seasonality_stats:
        t_sea1, t_sea2 = st.tabs(["TIME OF DAY (NY)", "DAY OF WEEK"])
        with t_sea1:
            if 'hourly_perf' in seasonality_stats and seasonality_stats['hourly_perf'] is not None:
                hp = seasonality_stats['hourly_perf']
                fig_h = go.Figure()
                colors = ['#00FFFF' if v > 0 else '#8080FF' for v in hp.values]
                fig_h.add_trace(go.Bar(x=[f"{h:02d}:00" for h in hp.index], y=hp.values, marker_color=colors))
                fig_h = terminal_chart_layout(fig_h, title="AVG RETURN BY HOUR", height=250)
                st.plotly_chart(fig_h, use_container_width=True)
        with t_sea2:
            fig_d = go.Figure()
            fig_d.add_trace(go.Bar(x=seasonality_stats['day_high'].index, y=seasonality_stats['day_high'].values, marker_color='#00FFFF'))
            fig_d = terminal_chart_layout(fig_d, title="PROBABILITY OF WEEKLY HIGH", height=250)
            st.plotly_chart(fig_d, use_container_width=True)

# ==========================================
# PHASE 3: LIQUIDITY & LEVELS (The "Where")
# ==========================================
with tab_liq:
    l_col1, l_col2, l_col3 = st.columns(3)
    
    with l_col1:
        st.markdown("#### 🏦 GEX PROFILE")
        if gex_df is not None and gex_spot is not None:
            center_strike = gex_spot 
            gex_zoom = gex_df[(gex_df['strike'] > center_strike * 0.9) & (gex_df['strike'] < center_strike * 1.1)]
            fig_gex = go.Figure()
            colors = ['#00FFFF' if x > 0 else '#8080FF' for x in gex_zoom['gex']]
            fig_gex.add_trace(go.Bar(x=gex_zoom['strike'], y=gex_zoom['gex'], marker_color=colors))
            fig_gex.add_vline(x=center_strike, line_dash="dot", line_color="white", annotation_text="SPOT")
            fig_gex = terminal_chart_layout(fig_gex, title=f"NET GAMMA (EXP: {gex_date})", height=300)
            st.plotly_chart(fig_gex, use_container_width=True)
            
            total_gex = gex_df['gex'].sum() / 1_000_000
            st.caption(f"NET GEX: ${total_gex:.1f}M ({'STICKY' if total_gex > 0 else 'VOLATILE'})")
        else:
            st.warning("GEX Data Unavailable")
            
    with l_col2:
        st.markdown("#### 📊 VOLUME PROFILE")
        if vol_profile is not None:
            fig_vp = go.Figure()
            colors = ['#00FFFF' if x == poc_price else '#333' for x in vol_profile['PriceLevel']]
            fig_vp.add_trace(go.Bar(y=vol_profile['PriceLevel'], x=vol_profile['Volume'], orientation='h', marker_color='#40E0FF', opacity=0.4))
            fig_vp.add_hline(y=poc_price, line_dash="dash", line_color="#FFFFFF", annotation_text="POC")
            fig_vp = terminal_chart_layout(fig_vp, title="INTRADAY DISTRIBUTION", height=300)
            st.plotly_chart(fig_vp, use_container_width=True)
            st.caption(f"Point of Control (POC): {poc_price:,.2f}")
            
    with l_col3:
        st.markdown("#### 🔑 KEY LEVELS")
        key_levels = qe.get_key_levels(daily_data)
        if key_levels:
            cur_price = intraday_data['Close'].iloc[-1] if not intraday_data.empty else 0
            levels_list = [("R1 (Resist)", key_levels['R1']), ("PDH (High)", key_levels['PDH']), ("PIVOT (Daily)", key_levels['Pivot']), ("PDL (Low)", key_levels['PDL']), ("S1 (Support)", key_levels['S1'])]
            for name, price in levels_list:
                dist = abs(price - cur_price) / cur_price
                c_code = "#00FFFF" if price < cur_price else "#8080FF"
                if dist < 0.002: c_code = "#FFFF00" # Active
                
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; border-bottom:1px solid #333; padding:5px;">
                    <span style="color:#AAAAAA;">{name}</span>
                    <span style="color:{c_code}; font-family:monospace;">{price:,.2f}</span>
                </div>
                """, unsafe_allow_html=True)

# ==========================================
# PHASE 4: EXECUTION & RISK (The "When")
# ==========================================
with tab_exec:
    e_col1, e_col2 = st.columns([2, 1])
    
    with e_col1:
        st.markdown("#### 🎯 INTRADAY TACTICAL (VWAP)")
        vwap_df = qe.calculate_vwap_bands(intraday_data)
        if not vwap_df.empty:
            fig_vwap = go.Figure()
            fig_vwap.add_trace(go.Candlestick(
                x=vwap_df.index, open=vwap_df['Open'], high=vwap_df['High'], low=vwap_df['Low'], close=vwap_df['Close'], name="Price",
                increasing_line_color="#00FFFF", decreasing_line_color="#405060"
            ))
            fig_vwap.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df['VWAP'], name="VWAP", line=dict(color='#FFFFFF', width=2)))
            fig_vwap.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df['Upper_Band_1'], name="+1 STD", line=dict(color='gray', width=1), opacity=0.3))
            fig_vwap.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df['Lower_Band_1'], name="-1 STD", line=dict(color='gray', width=1), opacity=0.3))
            fig_vwap = terminal_chart_layout(fig_vwap, height=400)
            st.plotly_chart(fig_vwap, use_container_width=True)
            
    with e_col2:
        st.markdown("#### ⚡ RISK & STRATEGY")
        
        # Relative Strength
        rs_data = qe.get_relative_strength(asset_info['ticker'])
        if not rs_data.empty:
            curr_rs = rs_data['RS_Score'].iloc[-1]
            rs_color = "#00FFFF" if curr_rs > 0 else "#8080FF"
            st.markdown(f"**RELATIVE STRENGTH (vs SPY)**: <span style='color:{rs_color}'>{curr_rs:.2%}</span>", unsafe_allow_html=True)
        
        # Backtest
        strat_perf = qe.run_strategy_backtest(asset_info['ticker'])
        if strat_perf:
            st.markdown("---")
            sig_color = "#00FFFF" if "LONG" in strat_perf['signal'] else "#8080FF"
            st.markdown(f"ALGO SIGNAL: <span style='color:{sig_color}; font-weight:bold;'>{strat_perf['signal']}</span>", unsafe_allow_html=True)
            st.metric("Sharpe Ratio", f"{strat_perf['sharpe']:.2f}")
            
        # Monte Carlo
        st.markdown("---")
        pred_dates, pred_paths = qe.generate_monte_carlo(daily_data)
        if pred_dates is not None:
            fig_mc = go.Figure()
            fig_mc.add_trace(go.Scatter(x=pred_dates, y=np.mean(pred_paths, axis=1), name='Forecast', line=dict(color='#00FFFF', dash='dash')))
            fig_mc = terminal_chart_layout(fig_mc, title="126-DAY FORECAST", height=150)
            fig_mc.update_layout(xaxis=dict(showgrid=False, visible=False))
            st.plotly_chart(fig_mc, use_container_width=True)

# --- 9. THESIS GENERATOR ---
st.markdown("---")
col_thesis_btn, col_thesis_info = st.columns([1, 4])
with col_thesis_btn:
    if gemini_key and st.button("🔎 DEEP DIVE THESIS"):
        with st.spinner("Compiling full institutional report..."):
            news_text = "\n".join([f"- {n['title']}" for n in combined_news_for_llm])
            thesis_text = ai.generate_deep_dive_thesis(
                ticker=selected_asset, price=curr, change=pct, regime=regime_data,
                ml_signal=ml_bias, gex_data=gex_df, cot_data=cot_data,
                levels=qe.get_key_levels(daily_data), news_summary=news_text, macro_data=macro_context_data, api_key=gemini_key
            )
            st.session_state['thesis_cache'] = thesis_text
            st.rerun()

if st.session_state['thesis_cache']:
    st.markdown(f"""
    <div class='terminal-box' style='border: 1px solid #444; padding: 20px;'>
        {st.session_state['thesis_cache']}
    </div>
    """, unsafe_allow_html=True)
