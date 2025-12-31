import streamlit as st
import pandas as pd
import google.generativeai as genai

# --- SAFE IMPORT SYSTEM ---
try:
    from textblob import TextBlob
    HAS_NLP = True
except ImportError:
    HAS_NLP = False

def get_technical_narrative(ticker, price, daily_pct, regime, ml_signal, gex_data, cot_data, levels, macro_data, api_key):
    if not api_key: return "AI Analyst unavailable (No Key)."
    if 'gemini_calls' in st.session_state: st.session_state['gemini_calls'] += 1
    
    gex_text = "N/A"
    if gex_data is not None:
        total_gex = gex_data['gex'].sum()
        gex_text = f"Net Gamma: ${total_gex/1_000_000:.1f}M ({'Long/Sticky' if total_gex>0 else 'Short/Volatile'})"
    lvl_text = "N/A"
    if levels:
        lvl_text = f"Pivot: {levels['Pivot']:.2f}, R1: {levels['R1']:.2f}, S1: {levels['S1']:.2f}"
    
    macro_str = "N/A"
    if macro_data:
        macro_str = f"YieldCurve: {macro_data.get('yield_curve', 'N/A')}, Inflation(CPI): {macro_data.get('cpi', 'N/A')}%, Rates: {macro_data.get('rates', 'N/A')}%, MacroRegime: {macro_data.get('regime', 'N/A')}"
    
    cot_str = "N/A"
    if cot_data and 'sentiment' in cot_data:
        cot_str = cot_data['sentiment']

    prompt = f"""
    You are a Senior Portfolio Manager. Analyze data for {ticker} and write a 3-bullet executive summary.
    DATA: Price: {price:,.2f} ({daily_pct:.2f}%), Regime: {regime['regime'] if regime else 'Unknown'}, 
    ML: {ml_signal}, GEX: {gex_text}, COT: {cot_str}, Levels: {lvl_text}
    MACRO CONTEXT: {macro_str}
    TASK:
    1. Synthesize Technicals + Macro.
    2. Identify key trigger level.
    3. Final Execution bias ("Buy Dips", "Fade", etc).
    Keep it concise. Bloomberg Terminal style.
    """
    try:
        genai.configure(api_key=api_key)
        
        # Auto-discover models
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m)
        
        if not available_models: return "Error: No valid models found. Check API Key permissions."
            
        available_models.sort(key=lambda x: 'flash' not in x.name)
        chosen_model_name = available_models[0].name
        
        model = genai.GenerativeModel(chosen_model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        if "429" in str(e): return "⚠️ API LIMIT REACHED."
        return f"AI Analyst unavailable: {str(e)}"

def generate_deep_dive_thesis(ticker, price, change, regime, ml_signal, gex_data, cot_data, levels, news_summary, macro_data, api_key):
    if not api_key: return "API Key Missing."
    if 'gemini_calls' in st.session_state: st.session_state['gemini_calls'] += 1
    
    gex_text = "N/A"
    if gex_data is not None:
        total_gex = gex_data['gex'].sum()
        gex_text = f"Net Gamma: ${total_gex/1_000_000:.1f}M"
    macro_str = "N/A"
    if macro_data:
        macro_str = f"YieldCurve: {macro_data.get('yield_curve', 'N/A')}, CPI: {macro_data.get('cpi', 'N/A')}%, Rates: {macro_data.get('rates', 'N/A')}%, Regime: {macro_data.get('regime', 'N/A')}"
    
    cot_str = "N/A"
    if cot_data and 'sentiment' in cot_data:
        cot_str = cot_data['sentiment']

    prompt = f"""
    Write a detailed Investment Thesis for {ticker}.
    DATA: Price: {price:,.2f} ({change:.2f}%), Regime: {regime['regime'] if regime else 'Unknown'}, 
    ML: {ml_signal}, GEX: {gex_text}, COT: {cot_str}
    MACRO: {macro_str}
    NEWS: {news_summary}
    OUTPUT FORMAT (Markdown):
    ### 1. THE MACRO & TECHNICAL CROSSROADS
    ### 2. CORE ARGUMENT (Long/Short/Neutral)
    ### 3. THE BEAR/BULL CASE (Risks)
    ### 4. KEY LEVELS (Invalidation)
    """
    try:
        genai.configure(api_key=api_key)
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m)
        if not available_models: return "Error: No valid models found."
        available_models.sort(key=lambda x: 'flash' not in x.name)
        chosen_model_name = available_models[0].name
        
        model = genai.GenerativeModel(chosen_model_name)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Thesis Generation Failed: {str(e)}"

def calculate_news_sentiment(news_items):
    if not HAS_NLP or not news_items: return pd.DataFrame()
    
    scores = []
    for news in news_items:
        try:
            blob = TextBlob(f"{news['title']} {news['title']}") 
            score = blob.sentiment.polarity
            scores.append({
                "title": news['title'],
                "score": score,
                "time": news['time']
            })
        except: continue
        
    df = pd.DataFrame(scores)
    if df.empty: return pd.DataFrame()
    
    df = df.iloc[::-1].reset_index(drop=True) 
    df['cumulative'] = df['score'].cumsum()
    return df
