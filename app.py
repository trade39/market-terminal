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
st.set_page_config(layout="wide", page_title="Bloomberg Terminal Pro V5.20", page_icon="⚡")
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
    st.markdown("<h3 style='color: #ff9900;'>COMMAND LINE</h3>", unsafe_allow_html=True)
    selected_asset = st.selectbox("SEC / Ticker", list(config.ASSETS.keys()))
    asset_info = config.ASSETS[selected_asset]
    
    use_demo_data = st.checkbox("🛠️ USE DEMO DATA (Save Quota)", value=True, help="Use mock data for Calendar to save RapidAPI credits.")
    
    st.markdown("---")
    
    with st.expander("📡 API QUOTA MONITOR", expanded=True):
        st.markdown("<div style='font-size:0.7em; color:gray;'>Session Usage vs Hard Limits</div>", unsafe_allow_html=True)
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

# --- MAIN DASHBOARD ---
st.markdown(f"<h1 style='border-bottom: 2px solid #ff9900;'>{selected_asset} <span style='font-size:0.5em; color:white;'>TERMINAL PRO V5.20</span></h1>", unsafe_allow_html=True)

# Fetch Data
daily_data = de.get_daily_data(asset_info['ticker'])
dxy_data = de.get_dxy_data(fred_key) 
intraday_data = de.get_intraday_data(asset_info['ticker'])
eco_events = de.get_economic_calendar(rapid_key, use_demo=use_demo_data)

# Fetch News
news_general = de.get_financial_news_general(news_key, query=asset_info.get('news_query', 'Finance'))
news_ff = de.get_forex_factory_news(rapid_key, 'breaking')
combined_news_for_llm = news_general[:5] + news_ff[:5]

# Engines (Advanced)
_, ml_prob = qe.get_ml_prediction(asset_info['ticker'])
gex_df, gex_date, gex_spot, current_iv = qe.get_gex_profile(asset_info['opt_ticker'])
vol_profile, poc_price = qe.calculate_volume_profile(intraday_data)
hurst = qe.calculate_hurst(daily_data['Close'].values) if not daily_data.empty else 0.5
regime_data = qe.get_market_regime(asset_info['ticker'])
correlations = qe.get_correlations(asset_info['ticker'], fred_key)
news_sentiment_df = ai.calculate_news_sentiment(combined_news_for_llm)

# NEW: Multilayer Engines
ms_df, ms_trend, ms_last_sh, ms_last_sl = qe.detect_market_structure(daily_data)
vol_cone = qe.get_volatility_cone(daily_data)
of_df, of_bias = qe.calculate_order_flow_proxy(daily_data)
active_fvgs = qe.detect_fair_value_gaps(daily_data)

# Initialize COT Data for AI
cot_data = None 

# --- 1. OVERVIEW & MULTILAYER QUANT SETUP ---
if not daily_data.empty:
    close, high, low = daily_data['Close'], daily_data['High'], daily_data['Low']
    curr = close.iloc[-1]
    pct = ((curr - close.iloc[-2]) / close.iloc[-2]) * 100
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("LAST PX", f"{curr:,.2f}", f"{pct:.2f}%")
    
    ml_bias = "BULLISH" if ml_prob > 0.55 else "BEARISH" if ml_prob < 0.45 else "NEUTRAL"
    ml_conf = abs(ml_prob - 0.5) * 200
    ml_color = "bullish" if ml_bias == "BULLISH" else "bearish" if ml_bias == "BEARISH" else "neutral"
    
    c2.markdown(f"""
    <div class='terminal-box' style="text-align:center; padding:5px;">
        <div style="font-size:0.8em; color:#ff9900;">AI PREDICTION</div>
        <span class='{ml_color}'>{ml_bias}</span>
        <div style="font-size:0.8em; margin-top:5px; color:#aaa;">CONF: {ml_conf:.0f}%</div>
    </div>
    """, unsafe_allow_html=True)
    
    hurst_type = "TRENDING" if hurst > 0.55 else "MEAN REVERT" if hurst < 0.45 else "RANDOM WALK"
    h_color = "#00ff00" if hurst > 0.55 else "#ff3333" if hurst < 0.45 else "gray"
    
    if regime_data:
        c3.markdown(f"""
        <div class='terminal-box' style="padding:10px;">
            <div style="font-size:0.8em; color:#ff9900;">QUANT REGIME</div>
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
    
    # --- CHART: MULTILAYER LIQUIDITY MAP ---
    fig = go.Figure()
    
    # Trace 1: Asset Candlesticks
    fig.add_trace(go.Candlestick(
        x=daily_data.index, 
        open=daily_data['Open'], 
        high=high, 
        low=low, 
        close=close, 
        name="Price"
    ))
    
    # Trace 2: Fair Value Gaps (Rectangles)
    for fvg in active_fvgs:
        color = "rgba(0, 255, 0, 0.2)" if "Bullish" in fvg['type'] else "rgba(255, 0, 0, 0.2)"
        fig.add_shape(type="rect",
            x0=fvg['date'], x1=daily_data.index[-1],
            y0=fvg['bottom'], y1=fvg['top'],
            fillcolor=color, line_width=0,
        )

    # Trace 3: Swing Points (Structure)
    sh_mask = ms_df['Structure'] == 'SH'
    sl_mask = ms_df['Structure'] == 'SL'
    fig.add_trace(go.Scatter(
        x=ms_df[sh_mask].index, y=ms_df[sh_mask]['High'], 
        mode='markers', marker=dict(symbol='triangle-down', size=8, color='red'), name='Swing High'
    ))
    fig.add_trace(go.Scatter(
        x=ms_df[sl_mask].index, y=ms_df[sl_mask]['Low'], 
        mode='markers', marker=dict(symbol='triangle-up', size=8, color='green'), name='Swing Low'
    ))

    if poc_price:
        fig.add_hline(y=poc_price, line_dash="dash", line_color="yellow", annotation_text="POC", annotation_position="bottom right")

    # Trace 4: DXY Overlay (FRED)
    if not dxy_data.empty:
        dxy_aligned = dxy_data['Close'].reindex(daily_data.index, method='ffill')
        fig.add_trace(go.Scatter(
            x=dxy_aligned.index, 
            y=dxy_aligned.values, 
            name="DXY (FRED)", 
            line=dict(color='orange', width=2),
            opacity=0.7,
            yaxis="y2"
        ))

    # Layout Updates
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#000000",
        plot_bgcolor="#000000",
        height=500,
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(showgrid=True, gridcolor="#222", zerolinecolor="#222"),
        yaxis=dict(showgrid=True, gridcolor="#222", zerolinecolor="#222", title="Asset Price"),
        yaxis2=dict(
            title="DXY Index (FRED)",
            overlaying="y",
            side="right",
            showgrid=False,
            title_font=dict(color="orange"),
            tickfont=dict(color="orange")
        ),
        font=dict(family="Courier New", color="#e0e0e0"),
        legend=dict(orientation="h", y=1.02, x=0, bgcolor="rgba(0,0,0,0)")
    )
    
    fig.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# --- 1B. QUANT MULTILAYER SETUP (REPLACES RETAIL RADAR) ---
st.markdown("---")
st.markdown("### 🧬 QUANT MULTILAYER SETUP")

ms_col, vol_col, of_col = st.columns(3)

with ms_col:
    trend_color = "bullish" if "BULLISH" in ms_trend else "bearish" if "BEARISH" in ms_trend else "neutral"
    st.markdown(f"""
    <div class='terminal-box'>
        <div style='color:gray; font-size:0.8em;'>MARKET STRUCTURE (BOS/CHoCH)</div>
        <div style='font-size:1.2em; font-weight:bold;'>{ms_trend}</div>
        <hr style='margin:5px 0;'>
        <div style='font-size:0.8em;'>Last Swing High: <span style='color:#ff3333'>{ms_last_sh:,.2f}</span></div>
        <div style='font-size:0.8em;'>Last Swing Low: <span style='color:#00ff00'>{ms_last_sl:,.2f}</span></div>
    </div>
    """, unsafe_allow_html=True)

with vol_col:
    vol_regime = vol_cone.get('regime', 'N/A')
    vol_rank = vol_cone.get('rank', 0.5)
    v_color = "bullish" if "COMPRESSED" in vol_regime else "neutral"
    st.markdown(f"""
    <div class='terminal-box'>
        <div style='color:gray; font-size:0.8em;'>VOLATILITY SURFACE (Garman-Klass)</div>
        <div style='font-size:1.1em; font-weight:bold;'>{vol_regime}</div>
        <hr style='margin:5px 0;'>
        <div style='font-size:0.8em;'>Percentile Rank: {vol_rank*100:.0f}%</div>
        <progress value="{int(vol_rank*100)}" max="100" style="width:100%; height:5px;"></progress>
    </div>
    """, unsafe_allow_html=True)

with of_col:
    of_color = "bullish" if "Buying" in of_bias else "bearish"
    st.markdown(f"""
    <div class='terminal-box'>
        <div style='color:gray; font-size:0.8em;'>ORDER FLOW PROXY (Pressure)</div>
        <div style='font-size:1.2em; font-weight:bold;' class='{of_color}'>{of_bias}</div>
        <hr style='margin:5px 0;'>
        <div style='font-size:0.8em;'>Vol-Weighted Impulse logic</div>
        <div style='font-size:0.8em; color:gray;'>Detects Aggression vs Absorption</div>
    </div>
    """, unsafe_allow_html=True)

# --- 1C. COINGECKO INTEGRATION ---
cg_id = asset_info.get('cg_id')
if cg_id and cg_key:
    st.markdown("---")
    st.markdown("### 🦎 COINGECKO FUNDAMENTALS")
    
    with st.spinner("Fetching CoinGecko Data..."):
        cg_data = de.get_coingecko_stats(cg_id, cg_key)
    
    if cg_data:
        c_cg1, c_cg2, c_cg3, c_cg4 = st.columns(4)
        c_cg1.metric("Market Rank", f"#{cg_data['rank']}")
        
        ath_color = "red" if cg_data['ath_change'] < -20 else "orange"
        c_cg2.markdown(f"""
        <div class='terminal-box'>
            <div style='font-size:0.8em; color:gray;'>ATH DRAWDOWN</div>
            <div style='color:{ath_color}; font-size:1.2em;'>{cg_data['ath_change']:.2f}%</div>
            <div style='font-size:0.7em;'>High: ${cg_data['ath']:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
        sent_val = cg_data['sentiment']
        sent_color = "#00ff00" if sent_val > 60 else "#ff3333" if sent_val < 40 else "gray"
        c_cg3.markdown(f"""
        <div class='terminal-box'>
            <div style='font-size:0.8em; color:gray;'>COMMUNITY SENTIMENT</div>
            <div style='color:{sent_color}; font-size:1.2em;'>{sent_val}% Bullish</div>
            <progress value="{sent_val}" max="100" style="width:100%; height:5px;"></progress>
        </div>
        """, unsafe_allow_html=True)
        
        c_cg4.markdown(f"""
        <div class='terminal-box'>
            <div style='font-size:0.8em; color:gray;'>ALGORITHM</div>
            <div style='color:white; font-size:1em;'>{cg_data['hashing']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("Asset Description"):
            st.write(cg_data['desc'])
    else:
        st.warning("CoinGecko API Limit Reached (30 calls/min). Please wait.")

# --- 2. INTRADAY TACTICAL FEED ---
st.markdown("---")
st.markdown("### 🔭 INTRADAY TACTICAL FEED")
rs_data = qe.get_relative_strength(asset_info['ticker'])
key_levels = qe.get_key_levels(daily_data)
col_intra_1, col_intra_2 = st.columns([2, 1])
with col_intra_1:
    if not rs_data.empty:
        curr_rs = rs_data['RS_Score'].iloc[-1]
        rs_color = "#00ff00" if curr_rs > 0 else "#ff3333"
        rs_text = "OUTPERFORMING SPY" if curr_rs > 0 else "UNDERPERFORMING SPY"
        st.markdown(f"**RELATIVE STRENGTH (vs SPY)**: <span style='color:{rs_color}'>{rs_text}</span>", unsafe_allow_html=True)
        fig_rs = go.Figure()
        fig_rs.add_hline(y=0, line_color="#333", line_dash="dash")
        fig_rs.add_trace(go.Scatter(x=rs_data.index, y=rs_data['RS_Score'], mode='lines', name='Alpha', line=dict(color=rs_color, width=2), fill='tozeroy'))
        terminal_chart_layout(fig_rs, title="INTRADAY ALPHA (Real-Time)", height=250)
        st.plotly_chart(fig_rs, use_container_width=True)
with col_intra_2:
    st.markdown("**🔑 KEY ALGO LEVELS**")
    if key_levels:
        cur_price = intraday_data['Close'].iloc[-1] if not intraday_data.empty else 0
        def get_lvl_color(level, current):
            if current == 0: return "white"
            dist = abs(level - current) / current
            if dist < 0.002: return "#ffff00" # Near
            if level > current: return "#ff3333" # Resistance
            return "#00ff00" # Support
        levels_list = [("R1 (Resist)", key_levels['R1']), ("PDH (High)", key_levels['PDH']), ("PIVOT (Daily)", key_levels['Pivot']), ("PDL (Low)", key_levels['PDL']), ("S1 (Support)", key_levels['S1'])]
        for name, price in levels_list:
            c_code = get_lvl_color(price, cur_price)
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; border-bottom:1px solid #222; padding:5px;">
                <span style="color:#aaa;">{name}</span>
                <span style="color:{c_code}; font-family:monospace;">{price:,.2f}</span>
            </div>
            """, unsafe_allow_html=True)

# --- 3. EVENTS & NEWS & MACRO ---
st.markdown("---")
col_eco, col_news = st.columns([2, 1])

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
            context_str = f"Actual ({actual_str}) vs Forecast ({forecast_str})"
            delta = val_actual - val_forecast
            pct_dev = abs(delta / val_forecast) if val_forecast != 0 else 1.0
            if pct_dev < 0.02: bias = "Mean Reverting"
            elif delta > 0: bias = "Bullish"
            else: bias = "Bearish"
        else: context_str = f"Actual: {actual_str}"
    else:
        if val_forecast is not None and val_prev is not None:
            context_str = f"Forecast ({forecast_str}) vs Prev ({previous_str})"
            delta = val_forecast - val_prev
            pct_dev = abs(delta / val_prev) if val_prev != 0 else 1.0
            if pct_dev < 0.02: bias = "Mean Reverting"
            elif delta > 0: bias = "Bullish Exp."
            else: bias = "Bearish Exp."
        else: context_str = "Waiting for Data..."
    return context_str, bias

with col_eco:
    st.markdown("### 📅 ECONOMIC EVENTS (USD)")
    if use_demo_data:
        st.caption("🟢 USING DEMO DATA (API Quota Saver Active)")
    
    if eco_events:
        cal_data = []
        for event in eco_events:
            context, bias = analyze_eco_context(event.get('actual', ''), event.get('forecast', ''), event.get('previous', ''))
            cal_data.append({"TIME": event.get('time', 'N/A'), "EVENT": event.get('event_name', 'Unknown'), "DATA CONTEXT": context, "BIAS": bias})
        df_cal = pd.DataFrame(cal_data)
        def color_bias(val):
            color = 'white'
            if 'Bullish' in val: color = '#00ff00' 
            elif 'Bearish' in val: color = '#ff3333'
            elif 'Mean' in val: color = '#cccc00'
            return f'color: {color}'
        if not df_cal.empty: 
            st.dataframe(df_cal.style.applymap(color_bias, subset=['BIAS']), use_container_width=True, hide_index=True)
    else: st.info("NO HIGH IMPACT USD EVENTS SCHEDULED.")
with col_news:
    st.markdown(f"### 📰 {asset_info.get('news_query', 'LATEST')} WIRE & SENTIMENT")
    
    if ai.HAS_NLP and not news_sentiment_df.empty:
         fig_sent = go.Figure()
         fig_sent.add_trace(go.Scatter(
             x=news_sentiment_df.index, 
             y=news_sentiment_df['cumulative'],
             mode='lines+markers',
             line=dict(color='#00e6ff', width=2, shape='spline'),
             name="Sentiment"
         ))
         fig_sent.update_layout(
             title="NLP SENTIMENT VELOCITY (Current Batch)",
             height=150,
             margin=dict(l=10, r=10, t=30, b=10),
             paper_bgcolor="#111", plot_bgcolor="#111",
             font=dict(size=10, color="white"),
             xaxis=dict(showgrid=False, visible=False),
             yaxis=dict(showgrid=True, gridcolor="#333")
         )
         st.plotly_chart(fig_sent, use_container_width=True)
    elif not ai.HAS_NLP:
        st.warning("Install `textblob` to enable Sentiment Analysis.")
    
    tab_gen, tab_ff = st.tabs(["📰 GENERAL", "⚡ FOREX FACTORY"])
    def render_news(items):
        if items:
            for news in items:
                st.markdown(f"""
                <div style="border-bottom:1px solid #333; padding-bottom:5px; margin-bottom:5px;">
                    <a class='news-link' href='{news['url']}' target='_blank'>▶ {news['title']}</a><br>
                    <span style='font-size:0.7em; color:gray;'>{news['time']} | {news['source']}</span>
                </div>
                """, unsafe_allow_html=True)
        else: st.markdown("<div style='color:gray;'>No data.</div>", unsafe_allow_html=True)
    with tab_gen: render_news(news_general)
    with tab_ff: render_news(news_ff)

# --- FRED MACRO DASHBOARD ---
st.markdown("---")
st.markdown("### 🇺🇸 FED LIQUIDITY & MACRO (FRED)")
macro_context_data = {} 
if fred_key:
    # 1. Fetch Key Series
    df_yield = de.get_fred_series("T10Y2Y", fred_key)
    df_ff = de.get_fred_series("FEDFUNDS", fred_key)
    df_cpi = de.get_fred_series("CPIAUCSL", fred_key)
    df_m2 = de.get_fred_series("M2SL", fred_key)
    
    # 2. Macro ML Regime Engine
    macro_regime = qe.get_macro_ml_regime(df_cpi, df_ff)
    
    # 3. Store for AI
    if not df_yield.empty: macro_context_data['yield_curve'] = f"{df_yield['value'].iloc[-1]:.2f}"
    if not df_cpi.empty: macro_context_data['cpi'] = f"{(df_cpi['value'].pct_change(12).iloc[-1]*100):.2f}"
    if not df_ff.empty: macro_context_data['rates'] = f"{df_ff['value'].iloc[-1]:.2f}"
    if macro_regime: macro_context_data['regime'] = macro_regime['regime']
    
    # --- UI LAYOUT ---
    macro_col_main, macro_col_ml = st.columns([3, 1])
    
    with macro_col_main:
        macro_tab1, macro_tab2 = st.tabs(["YIELD CURVE & RATES", "INFLATION & LIQUIDITY"])
        
        with macro_tab1:
            c_m1, c_m2 = st.columns(2)
            with c_m1:
                # Yield Curve
                if not df_yield.empty:
                    curr_yield = df_yield['value'].iloc[-1]
                    yield_color = "red" if curr_yield < 0 else "green"
                    fig_yc = go.Figure()
                    fig_yc.add_trace(go.Scatter(x=df_yield.index, y=df_yield['value'], fill='tozeroy', line=dict(color=yield_color)))
                    fig_yc.add_hline(y=0, line_dash="dash", line_color="white")
                    terminal_chart_layout(fig_yc, title=f"10Y-2Y SPREAD: {curr_yield:.2f}%", height=250)
                    st.plotly_chart(fig_yc, use_container_width=True)
                    yc_msg = "⚠️ INVERTED: RECESSION SIGNAL ALERT" if curr_yield < 0 else "NORMAL: GROWTH EXPECTATIONS"
                    st.caption(f"CONTEXT: {yc_msg}")
            
            with c_m2:
                # Fed Funds
                if not df_ff.empty:
                    fig_ff = go.Figure()
                    fig_ff.add_trace(go.Scatter(x=df_ff.index, y=df_ff['value'], line=dict(color="#00e6ff")))
                    terminal_chart_layout(fig_ff, title=f"FED FUNDS RATE: {df_ff['value'].iloc[-1]:.2f}%", height=250)
                    st.plotly_chart(fig_ff, use_container_width=True)
                    st.caption("CONTEXT: BASELINE RISK-FREE RATE (Cost of Capital)")
        with macro_tab2:
            c_m3, c_m4 = st.columns(2)
            with c_m3:
                # CPI
                if not df_cpi.empty:
                    df_cpi['YoY'] = df_cpi['value'].pct_change(12) * 100
                    fig_cpi = go.Figure()
                    fig_cpi.add_trace(go.Bar(x=df_cpi.index, y=df_cpi['YoY'], marker_color='#ff9900'))
                    terminal_chart_layout(fig_cpi, title=f"CPI INFLATION (YoY): {df_cpi['YoY'].iloc[-1]:.2f}%", height=250)
                    st.plotly_chart(fig_cpi, use_container_width=True)
                    cpi_trend = "RISING" if df_cpi['YoY'].iloc[-1] > df_cpi['YoY'].iloc[-2] else "FALLING"
                    st.caption(f"CONTEXT: INFLATION IS {cpi_trend} (Target: 2.0%)")
            
            with c_m4:
                # M2
                if not df_m2.empty:
                    fig_m2 = go.Figure()
                    fig_m2.add_trace(go.Scatter(x=df_m2.index, y=df_m2['value'], line=dict(color="#00ff00")))
                    terminal_chart_layout(fig_m2, title="M2 MONEY SUPPLY (Liquidity)", height=250)
                    st.plotly_chart(fig_m2, use_container_width=True)
                    st.caption("CONTEXT: FUEL FOR ASSET PRICES")
    
    with macro_col_ml:
        st.markdown("#### 🤖 MACRO ML ENGINE")
        if macro_regime:
            st.markdown(f"""
            <div class='terminal-box'>
                <div style='color:#aaa; font-size:0.8em;'>ECONOMIC REGIME (GMM)</div>
                <div style='color:#00e6ff; font-size:1.1em; font-weight:bold;'>{macro_regime['regime']}</div>
                <hr>
                <div style='font-size:0.8em;'>CPI: {macro_regime['cpi']:.1f}%</div>
                <div style='font-size:0.8em;'>RATES: {macro_regime['rate']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Insufficient data for Macro ML.")
        # DXY COMPARISON
        st.markdown("#### 💵 DOLLAR (DXY) COMPARE")
        if not correlations.empty and 'Dollar' in correlations:
            dxy_corr = correlations['Dollar']
            corr_color = "#00ff00" if dxy_corr > 0.5 else "#ff3333" if dxy_corr < -0.5 else "white"
            st.markdown(f"""
            <div class='terminal-box'>
                <div style='color:#aaa; font-size:0.8em;'>CORRELATION TO DXY</div>
                <div style='color:{corr_color}; font-size:1.5em;'>{dxy_corr:.2f}</div>
                <div style='font-size:0.7em; color:gray;'>1.0 = Moves with DXY<br>-1.0 = Inverse to DXY</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            if not fred_key:
                st.warning("MISSING API KEY: Add 'fred_api_key' to secrets.")
            else:
                st.warning("Data Unavailable: DXY (FRED) merge failed.")
else:
    st.info("FRED API Key not found. Add `fred_api_key` to secrets to view Fed Macro Data.")

# --- 4. RISK ANALYSIS & BACKTEST ---
st.markdown("---")
st.markdown("### ⚡ QUANTITATIVE RISK & EXECUTION")
strat_perf = qe.run_strategy_backtest(asset_info['ticker'])
if not intraday_data.empty and strat_perf:
    q1, q2, q3 = st.columns([1, 2, 2])
    with q1:
        st.markdown("**STRATEGY SIGNAL**")
        sig_color = "#00ff00" if "LONG" in strat_perf['signal'] else "#ffff00"
        st.markdown(f"<span style='color:{sig_color}; font-size:1.8em; font-weight:bold;'>{strat_perf['signal']}</span>", unsafe_allow_html=True)
        st.markdown("---")
        st.metric("Total Return", f"{strat_perf['return']*100:.1f}%")
        st.metric("Sharpe Ratio", f"{strat_perf['sharpe']:.2f}")
    with q2:
        ec_df = pd.DataFrame({"Strategy": strat_perf['equity_curve'], "Buy & Hold": strat_perf['df']['Cum_BnH']})
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(x=ec_df.index, y=ec_df['Buy & Hold'], name="Buy & Hold", line=dict(color='gray', dash='dot')))
        fig_perf.add_trace(go.Scatter(x=ec_df.index, y=ec_df['Strategy'], name="Active Strat", line=dict(color='#00e6ff', width=2)))
        terminal_chart_layout(fig_perf, title="STRATEGY EDGE VALIDATION", height=300)
        st.plotly_chart(fig_perf, use_container_width=True)
    with q3:
        if vol_profile is not None:
            fig_vp = go.Figure()
            colors = ['#00e6ff' if x == poc_price else '#333' for x in vol_profile['PriceLevel']]
            fig_vp.add_trace(go.Bar(y=vol_profile['PriceLevel'], x=vol_profile['Volume'], orientation='h', marker_color='#ff9900', opacity=0.4))
            fig_vp.add_hline(y=poc_price, line_dash="dash", line_color="yellow", annotation_text="POC")
            terminal_chart_layout(fig_vp, title="INTRADAY VOLUME PROFILE", height=300)
            st.plotly_chart(fig_vp, use_container_width=True)

# --- 5. VWAP EXECUTION ---
st.markdown("#### 🎯 SESSION VWAP + KEY LEVELS")
vwap_df = qe.calculate_vwap_bands(intraday_data)
if not vwap_df.empty:
    fig_vwap = go.Figure()
    fig_vwap.add_trace(go.Candlestick(x=vwap_df.index, open=vwap_df['Open'], high=vwap_df['High'], low=vwap_df['Low'], close=vwap_df['Close'], name="Price"))
    fig_vwap.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df['VWAP'], name="Session VWAP", line=dict(color='#ff9900', width=2)))
    fig_vwap.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df['Upper_Band_1'], name="+1 STD", line=dict(color='gray', width=1), opacity=0.3))
    fig_vwap.add_trace(go.Scatter(x=vwap_df.index, y=vwap_df['Lower_Band_1'], name="-1 STD", line=dict(color='gray', width=1), opacity=0.3))
    if key_levels:
        fig_vwap.add_hline(y=key_levels['PDH'], line_dash="dot", line_color="#ff3333", annotation_text="PDH")
        fig_vwap.add_hline(y=key_levels['PDL'], line_dash="dot", line_color="#00ff00", annotation_text="PDL")
        fig_vwap.add_hline(y=key_levels['Pivot'], line_width=1, line_color="#00e6ff", annotation_text="DAILY PIVOT")
    terminal_chart_layout(fig_vwap, height=500)
    st.plotly_chart(fig_vwap, use_container_width=True)

# --- 6. GEX & VOLATILITY ---
st.markdown("---")
st.markdown("### 🏦 INSTITUTIONAL GEX & VOLATILITY")
if gex_df is not None and gex_spot is not None:
    g1, g2, g3 = st.columns([2, 1, 1])
    
    with g1:
        center_strike = gex_spot 
        gex_zoom = gex_df[(gex_df['strike'] > center_strike * 0.9) & (gex_df['strike'] < center_strike * 1.1)]
        fig_gex = go.Figure()
        colors = ['#00ff00' if x > 0 else '#ff3333' for x in gex_zoom['gex']]
        fig_gex.add_trace(go.Bar(x=gex_zoom['strike'], y=gex_zoom['gex'], marker_color=colors))
        fig_gex.add_vline(x=center_strike, line_dash="dot", line_color="white", annotation_text="ETF SPOT")
        terminal_chart_layout(fig_gex, title=f"NET GAMMA PROFILE (EXP: {gex_date})")
        st.plotly_chart(fig_gex, use_container_width=True)
        
    with g2:
        total_gex = gex_df['gex'].sum() / 1_000_000
        sentiment = "LOW VOL (Sticky)" if total_gex > 0 else "HIGH VOL (Slippery)"
        sent_color = "bullish" if total_gex > 0 else "bearish"
        st.markdown(f"""
        <div class='terminal-box'>
            <div style='color:#ff9900;'>NET GAMMA</div>
            <div style='font-size:1.5em; color:white;'>${total_gex:.1f}M</div>
            <div style='margin-top:10px;'><span class='{sent_color}'>{sentiment}</span></div>
        </div>
        """, unsafe_allow_html=True)
    
    with g3:
        # Calculate HV (20D Annualized)
        if not daily_data.empty:
            hv_current = daily_data['Close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100
        else:
            hv_current = 0
            
        iv_display = current_iv if current_iv else 0
        vol_premium = iv_display - hv_current
        prem_color = "#ff3333" if vol_premium > 5 else "#00ff00" if vol_premium < 0 else "white"
        
        st.markdown(f"""
        <div class='terminal-box'>
            <div style='color:#aaa; font-size:0.8em;'>IMPLIED VOL (IV)</div>
            <div style='font-size:1.2em; color:#00e6ff;'>{iv_display:.1f}%</div>
            <hr style='margin:5px 0; border-color:#333;'>
            <div style='color:#aaa; font-size:0.8em;'>HISTORICAL VOL (HV)</div>
            <div style='font-size:1.2em; color:#e0e0e0;'>{hv_current:.1f}%</div>
            <hr style='margin:5px 0; border-color:#333;'>
            <div style='color:#aaa; font-size:0.8em;'>VOL PREMIUM (IV-HV)</div>
            <div style='font-size:1.2em; color:{prem_color};'>{vol_premium:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

else: st.warning("GEX Data Unavailable for this asset.")

# --- 7. MONTE CARLO & ADVANCED SEASONALITY ---
st.markdown("---")
st.markdown("### 🎲 SIMULATION & TIME ANALYSIS")
pred_dates, pred_paths = qe.generate_monte_carlo(daily_data)
stats = qe.get_seasonality_stats(daily_data, asset_info['ticker']) 
if stats:
    st.markdown("#### ⏳ SEASONAL TENDENCIES")
    tab_hour, tab_day, tab_week = st.tabs(["HOUR (NY)", "DAY", "WEEK"])
    with tab_hour:
        if 'hourly_perf' in stats and stats['hourly_perf'] is not None:
            hp = stats['hourly_perf']
            fig_h = go.Figure()
            colors = ['#00ff00' if v > 0 else '#ff3333' for v in hp.values]
            fig_h.add_trace(go.Bar(x=[f"{h:02d}:00" for h in hp.index], y=hp.values, marker_color=colors))
            terminal_chart_layout(fig_h, title="AVG RETURN BY HOUR (NY TIME)", height=350)
            st.plotly_chart(fig_h, use_container_width=True)
    with tab_day:
        fig_d = go.Figure()
        fig_d.add_trace(go.Bar(x=stats['day_high'].index, y=stats['day_high'].values, marker_color='#00ff00'))
        terminal_chart_layout(fig_d, title="PROBABILITY OF WEEKLY HIGH", height=350)
        st.plotly_chart(fig_d, use_container_width=True)
    with tab_week:
        if 'week_returns' in stats:
            wr = stats['week_returns']
            fig_w = go.Figure()
            colors = ['#00ff00' if v > 0 else '#ff3333' for v in wr.values]
            fig_w.add_trace(go.Bar(x=["Wk 1", "Wk 2", "Wk 3", "Wk 4", "Wk 5"], y=wr.values, marker_color=colors))
            terminal_chart_layout(fig_w, title="AVG RETURN BY WEEK OF MONTH", height=350)
            st.plotly_chart(fig_w, use_container_width=True)
st.markdown("#### 🎲 MONTE CARLO PROJECTION")
if pred_dates is not None and pred_paths is not None:
    fig_pred = go.Figure()
    hist_slice = daily_data['Close'].tail(90)
    fig_pred.add_trace(go.Scatter(x=hist_slice.index, y=hist_slice.values, name='History', line=dict(color='white')))
    fig_pred.add_trace(go.Scatter(x=pred_dates, y=np.mean(pred_paths, axis=1), name='Avg Path', line=dict(color='#ff9900', dash='dash')))
    terminal_chart_layout(fig_pred, title="MONTE CARLO PROJECTION (126 Days)", height=400)
    st.plotly_chart(fig_pred, use_container_width=True)

# --- 8. COT QUANT TERMINAL ---
st.markdown("---")
st.markdown("### 🏛️ COT QUANT TERMINAL")

def generate_cot_analysis(spec_net, hedge_net, spec_label, hedge_label):
    spec_sent = "🟢 BULLISH" if spec_net > 0 else "🔴 BEARISH"
    hedge_sent = "🟢 BULLISH" if hedge_net > 0 else "🔴 BEARISH"
    if (spec_net > 0 and hedge_net < 0) or (spec_net < 0 and hedge_net > 0):
        structure = "✅ **Healthy Structure:** Speculators and Hedgers are on opposite sides (Risk Transfer active)."
    else:
        structure = "⚠️ **Anomaly:** Both groups are positioned in the same direction. Watch for liquidity gaps."
    return f"""
    * **{spec_label}:** {spec_sent} (Net: {int(spec_net):,})
    * **{hedge_label}:** {hedge_sent} (Net: {int(hedge_net):,})
    {structure}
    """

# 1. Fetch Historical Data
with st.spinner("Analyzing CFTC Data..."):
    cot_history = de.fetch_cot_history(selected_asset, start_year=2024)

if cot_history is not None and not cot_history.empty:
    
    # 2. Process Data for Metrics
    cot_config = config.COT_MAPPING[selected_asset]
    spec_label, hedge_label = cot_config['labels']
    
    if all(c in cot_history.columns for c in ['spec_long', 'spec_short', 'hedge_long', 'hedge_short']):
        
        # Calculations
        cot_history['Net Speculator'] = cot_history['spec_long'] - cot_history['spec_short']
        cot_history['Net Hedger'] = cot_history['hedge_long'] - cot_history['hedge_short']
        cot_history['Spec Z-Score'] = qe.calculate_z_score(cot_history['Net Speculator'])
        
        latest_cot = cot_history.iloc[-1]
        prev_cot = cot_history.iloc[-2] if len(cot_history) > 1 else latest_cot
        
        # 3. METRICS ROW
        c_cot1, c_cot2, c_cot3, c_cot4 = st.columns(4)
        
        # Speculator Net
        spec_delta = latest_cot['Net Speculator'] - prev_cot['Net Speculator']
        c_cot1.metric(f"{spec_label} (Net)", f"{int(latest_cot['Net Speculator']):,}", f"{int(spec_delta):,}")
        
        # Hedger Net
        hedge_delta = latest_cot['Net Hedger'] - prev_cot['Net Hedger']
        c_cot2.metric(f"{hedge_label} (Net)", f"{int(latest_cot['Net Hedger']):,}", f"{int(hedge_delta):,}", delta_color="inverse")
        
        # Z-Score
        z_val = latest_cot['Spec Z-Score']
        c_cot3.metric("Z-Score (52wk)", f"{z_val:.2f}σ", "Extreme" if abs(z_val) > 2 else "Neutral", delta_color="off")
        
        # Date
        c_cot4.metric("Report Date", latest_cot['date'].strftime('%Y-%m-%d'))
        
        # 4. AI Interpretation Box
        cot_analysis_txt = generate_cot_analysis(latest_cot['Net Speculator'], latest_cot['Net Hedger'], spec_label, hedge_label)
        st.info(cot_analysis_txt)
        
        # FIX 4: DEFINE COT DATA FOR AI ENGINE
        cot_data = {
            "sentiment": "BULLISH" if latest_cot['Net Speculator'] > 0 else "BEARISH",
            "net_spec": latest_cot['Net Speculator'],
            "z_score": z_val
        }
        
        # 5. VISUALIZATION TABS
        tab_trend, tab_struct, tab_osc = st.tabs(["📈 NET TREND", "🦋 LONG/SHORT STRUCTURE", "📊 Z-SCORE OSCILLATOR"])
        
        with tab_trend:
            fig_trend = go.Figure()
            fig_trend.add_trace(go.Scatter(x=cot_history['date'], y=cot_history['Net Speculator'], name=spec_label, line=dict(color='#00FF00', width=2)))
            fig_trend.add_trace(go.Scatter(x=cot_history['date'], y=cot_history['Net Hedger'], name=hedge_label, line=dict(color='#FF0000', width=2)))
            fig_trend.add_hline(y=0, line_dash="dash", line_color="gray")
            terminal_chart_layout(fig_trend, title="NET POSITIONING HISTORY (Smart Money vs Specs)", height=400)
            st.plotly_chart(fig_trend, use_container_width=True)
            
        with tab_struct:
            fig_struct = go.Figure()
            # Speculator Longs (Green)
            fig_struct.add_trace(go.Bar(
                x=cot_history['date'], 
                y=cot_history['spec_long'], 
                name=f"{spec_label} Longs", 
                marker_color='#00C805'
            ))
            # Speculator Shorts (Red) - Negative for Butterfly
            fig_struct.add_trace(go.Bar(
                x=cot_history['date'], 
                y=-cot_history['spec_short'], 
                name=f"{spec_label} Shorts", 
                marker_color='#FF4B4B'
            ))
            fig_struct.update_layout(barmode='overlay')
            terminal_chart_layout(fig_struct, title=f"{spec_label.upper()} STRUCTURE (Butterfly Chart)", height=400)
            st.caption("Green Bars = Long Contracts. Red Bars = Short Contracts (Plotted Inversely).")
            st.plotly_chart(fig_struct, use_container_width=True)
            
        with tab_osc:
            fig_z = go.Figure()
            colors = ['red' if val > 2 or val < -2 else 'gray' for val in cot_history['Spec Z-Score']]
            fig_z.add_trace(go.Bar(x=cot_history['date'], y=cot_history['Spec Z-Score'], marker_color=colors, name="Z-Score"))
            fig_z.add_hline(y=2, line_dash="dot", line_color="red", annotation_text="Overbought (+2σ)")
            fig_z.add_hline(y=-2, line_dash="dot", line_color="red", annotation_text="Oversold (-2σ)")
            terminal_chart_layout(fig_z, title="POSITIONING EXTREMES (Z-Score)", height=400)
            st.plotly_chart(fig_z, use_container_width=True)
            
    else:
        st.warning(f"COT Data retrieved but missing required columns for {selected_asset}. Try another asset.")
else:
    st.info(f"No COT Data available for {selected_asset}. This asset might not be a futures contract or mapping is missing.")

# --- 9. INTELLIGENT EXECUTIVE SUMMARY & THESIS ---
st.markdown("---")
st.markdown("### 🧠 AI QUANT ANALYST")
# Prepare Data for LLM
gex_summary = gex_df if gex_df is not None else None
ml_signal_str = "BULLISH" if ml_prob > 0.55 else "BEARISH" if ml_prob < 0.45 else "NEUTRAL"
news_text_summary = "\n".join([f"- {n['title']} ({n['source']})" for n in combined_news_for_llm])
if gemini_key:
    col_exec_btn, col_exec_info = st.columns([1, 4])
    with col_exec_btn:
        if st.button("📝 GENERATE BRIEF"):
            with st.spinner("Analyzing Technicals + Macro..."):
                narrative = ai.get_technical_narrative(
                    ticker=selected_asset, price=curr, daily_pct=pct, regime=regime_data,
                    ml_signal=ml_signal_str, gex_data=gex_summary, cot_data=cot_data,
                    levels=key_levels, macro_data=macro_context_data, api_key=gemini_key
                )
                st.session_state['narrative_cache'] = narrative
                st.rerun()
    with col_exec_info:
        st.markdown("<span style='color:gray; vertical-align:middle;'>3-bullet Executive Summary (Costs 1 Gemini Call)</span>", unsafe_allow_html=True)
    if st.session_state['narrative_cache']:
        if "⚠️" in st.session_state['narrative_cache']: st.error(st.session_state['narrative_cache'])
        else:
            st.markdown(f"""
            <div class='terminal-box' style='border-left: 4px solid #00e6ff;'>
                <div style='font-family: monospace; font-size: 0.95em; white-space: pre-wrap;'>{st.session_state['narrative_cache']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    col_thesis_btn, col_thesis_info = st.columns([1, 4])
    with col_thesis_btn:
        if st.button("🔎 DEEP DIVE THESIS"):
            with st.spinner("Analyzing Macro, Gamma, and Order Flow..."):
                thesis_text = ai.generate_deep_dive_thesis(
                    ticker=selected_asset, price=curr, change=pct, regime=regime_data,
                    ml_signal=ml_signal_str, gex_data=gex_summary, cot_data=cot_data,
                    levels=key_levels, news_summary=news_text_summary, macro_data=macro_context_data, api_key=gemini_key
                )
                st.session_state['thesis_cache'] = thesis_text
                st.rerun()
    with col_thesis_info:
        st.markdown("<span style='color:gray;'>Full 4-paragraph Hedge Fund style report. (Costs 1 Gemini Call)</span>", unsafe_allow_html=True)
    if st.session_state['thesis_cache']:
        if "⚠️" in st.session_state['thesis_cache']: st.error(st.session_state['thesis_cache'])
        else:
            st.markdown(f"""
            <div class='terminal-box' style='border: 1px solid #444; padding: 20px;'>
                {st.session_state['thesis_cache']}
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("Add GEMINI_API_KEY to see the AI Analyst Report.")
