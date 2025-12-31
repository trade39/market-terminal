import streamlit as st
import pandas as pd
import requests
import io
import zipfile
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from utils import safe_yf_download, flatten_dataframe
from config import COT_MAPPING

# --- COT IMPORT SAFETY ---
try:
    import cot_reports as cot
    HAS_COT_LIB = True
except ImportError:
    HAS_COT_LIB = False

# --- FETCHERS ---
@st.cache_data(ttl=300)
def get_daily_data(ticker):
    try: return safe_yf_download(ticker, period="10y", interval="1d")
    except: return pd.DataFrame()

@st.cache_data(ttl=300)
def get_dxy_data():
    try: return safe_yf_download("DX-Y.NYB", period="10y", interval="1d")
    except: return pd.DataFrame()

@st.cache_data(ttl=300)
def get_intraday_data(ticker):
    try: return safe_yf_download(ticker, period="5d", interval="15m")
    except: return pd.DataFrame()

@st.cache_data(ttl=86400)
def get_fred_series(series_id, api_key, observation_start=None):
    if not api_key: return pd.DataFrame()
    if 'fred_calls' in st.session_state: st.session_state['fred_calls'] += 1
    
    base_url = "https://api.stlouisfed.org/fred/series/observations"
    if not observation_start:
        observation_start = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json", "observation_start": observation_start}
    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        if "observations" in data:
            df = pd.DataFrame(data["observations"])
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            return df[['date', 'value']].set_index('date').dropna()
    except: return pd.DataFrame()
    return pd.DataFrame()

@st.cache_data(ttl=300) 
def get_coingecko_stats(cg_id, api_key):
    if not cg_id or not api_key: return None
    if 'coingecko_calls' in st.session_state: st.session_state['coingecko_calls'] += 1
    
    url = f"https://api.coingecko.com/api/v3/coins/{cg_id}"
    params = {
        "localization": "false", "tickers": "false", "market_data": "true",
        "community_data": "true", "developer_data": "true", "sparkline": "false"
    }
    headers = {"x-cg-demo-api-key": api_key}
    try:
        response = requests.get(url, params=params, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return {
                "rank": data.get('market_cap_rank', 'N/A'),
                "sentiment": data.get('sentiment_votes_up_percentage', 50),
                "hashing": data.get('hashing_algorithm', 'N/A'),
                "ath": data['market_data']['ath']['usd'],
                "ath_change": data['market_data']['ath_change_percentage']['usd'],
                "desc": data.get('description', {}).get('en', '').split('.')[0] + "." 
            }
        return None
    except Exception: return None

# --- NEWS ENGINES ---
@st.cache_data(ttl=14400) 
def get_financial_news_general(api_key, query="Finance"):
    if not api_key: return []
    if 'news_calls' in st.session_state: st.session_state['news_calls'] += 1
    try:
        newsapi = NewsApiClient(api_key=api_key)
        all_articles = newsapi.get_everything(q=query, language='en', sort_by='publishedAt')
        articles = []
        if all_articles['status'] == 'ok':
            for art in all_articles['articles'][:6]:
                articles.append({"title": art['title'], "source": art['source']['name'], "url": art['url'], "time": art['publishedAt']})
        return articles
    except: return []

@st.cache_data(ttl=14400)
def get_forex_factory_news(api_key, news_type='breaking'):
    if not api_key: return []
    if 'rapid_calls' in st.session_state: st.session_state['rapid_calls'] += 1
    base_url = "https://forex-factory-scraper1.p.rapidapi.com/"
    endpoints = {'breaking': "latest_breaking_news", 'fundamental': "latest_fundamental_analysis_news"}
    url = base_url + endpoints.get(news_type, "latest_breaking_news")
    headers = {"x-rapidapi-key": api_key, "x-rapidapi-host": "forex-factory-scraper1.p.rapidapi.com"}
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        normalized_news = []
        if isinstance(data, list):
            for item in data[:6]:
                normalized_news.append({
                    "title": item.get('title', 'No Title'),
                    "url": item.get('link', item.get('url', '#')),
                    "source": "ForexFactory",
                    "time": item.get('date', item.get('time', 'Recent'))
                })
        return normalized_news
    except: return []

# --- CALENDAR MOCK & FETCH ---
def get_mock_calendar_data():
    return [
        {"currency": "USD", "impact": "High", "event_name": "[DEMO] Unemployment Claims", "actual": "210K", "forecast": "215K", "previous": "208K", "time": "8:30am"},
        {"currency": "USD", "impact": "High", "event_name": "[DEMO] Philly Fed Mfg Index", "actual": "15.5", "forecast": "8.0", "previous": "12.2", "time": "8:30am"},
        {"currency": "USD", "impact": "Medium", "event_name": "[DEMO] Natural Gas Storage", "actual": "85B", "forecast": "82B", "previous": "79B", "time": "10:30am"},
        {"currency": "USD", "impact": "High", "event_name": "[DEMO] Powell Speaks", "actual": "", "forecast": "", "previous": "", "time": "2:00pm"},
    ]

@st.cache_data(ttl=21600)
def get_economic_calendar(api_key, use_demo=False):
    if use_demo: return get_mock_calendar_data()
    if not api_key: return get_mock_calendar_data() 

    if 'rapid_calls' in st.session_state: st.session_state['rapid_calls'] += 1
    try:
        url = "https://forex-factory-scraper1.p.rapidapi.com/get_real_time_calendar_details"
        now = datetime.now()
        querystring = {"calendar": "Forex", "year": str(now.year), "month": str(now.month), "day": str(now.day), "currency": "ALL", "event_name": "ALL", "timezone": "GMT-04:00 Eastern Time (US & Canada)", "time_format": "12h"}
        headers = {"x-rapidapi-host": "forex-factory-scraper1.p.rapidapi.com", "x-rapidapi-key": api_key}
        
        response = requests.get(url, headers=headers, params=querystring)
        if response.status_code == 429: return get_mock_calendar_data() 
            
        data = response.json()
        raw_events = data if isinstance(data, list) else data.get('data', [])
        
        filtered_events = []
        for e in raw_events:
            if e.get('currency') == 'USD' and (e.get('impact') == 'High' or e.get('impact') == 'Medium'):
                filtered_events.append(e)
                
        if filtered_events: return filtered_events
    except: pass 
    return []

# --- COT HELPERS ---
def clean_headers(df):
    if isinstance(df.columns[0], int):
        for i in range(20):
            row_str = " ".join(df.iloc[i].astype(str).tolist()).lower()
            if "market" in row_str and ("long" in row_str or "positions" in row_str):
                df.columns = df.iloc[i]
                return df.iloc[i+1:].reset_index(drop=True)
    return df

def map_columns(df, report_type):
    col_map = {}
    def get_col(keywords, exclude=None):
        for col in df.columns:
            c_str = str(col).lower()
            if all(k in c_str for k in keywords):
                if exclude and any(x in c_str for x in exclude): continue
                return col
        return None

    col_map['date'] = get_col(['report', 'date']) or get_col(['as', 'of', 'date']) or get_col(['date'])
    col_map['market'] = get_col(['market'])

    if "disaggregated" in report_type:
        col_map['spec_long'] = get_col(['money', 'long'], exclude=['lev'])
        col_map['spec_short'] = get_col(['money', 'short'], exclude=['lev'])
        col_map['hedge_long'] = get_col(['prod', 'merc', 'long'])
        col_map['hedge_short'] = get_col(['prod', 'merc', 'short'])
        
    elif "financial" in report_type:
        col_map['spec_long'] = get_col(['lev', 'money', 'long'])
        col_map['spec_short'] = get_col(['lev', 'money', 'short'])
        col_map['hedge_long'] = get_col(['asset', 'mgr', 'long'])
        col_map['hedge_short'] = get_col(['asset', 'mgr', 'short'])

    final_map = {v: k for k, v in col_map.items() if v}
    df = df.rename(columns=final_map)
    
    for c in ['spec_long', 'spec_short', 'hedge_long', 'hedge_short']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            
    return df

@st.cache_data(ttl=86400)
def fetch_cot_history(asset_name, start_year=2024):
    if asset_name not in COT_MAPPING: return None
    config = COT_MAPPING[asset_name]
    keywords = config['keywords']
    report_type = config['report_type']
    
    master_df = pd.DataFrame()
    current_year = datetime.now().year
    
    for y in range(start_year, current_year + 1):
        df_year = None
        if HAS_COT_LIB:
            try: df_year = cot.cot_year(year=y, cot_report_type=report_type)
            except: pass
        
        # Fallback Direct
        if df_year is None or df_year.empty:
            try:
                url = f"https://www.cftc.gov/files/dea/history/deahistfo{y}.zip" 
                r = requests.get(url)
                if r.status_code == 200:
                    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                        filename = z.namelist()[0]
                        with z.open(filename) as f:
                            df_year = pd.read_csv(f, low_memory=False)
            except: pass
            
        if df_year is not None and not df_year.empty:
            df_year = clean_headers(df_year)
            df_year = map_columns(df_year, report_type)
            master_df = pd.concat([master_df, df_year])
            
    if master_df.empty: return None
    
    # Filter
    if 'market' not in master_df.columns: return None
    mask = master_df['market'].astype(str).apply(lambda x: all(k.lower() in x.lower() for k in keywords))
    df_asset = master_df[mask].copy()
    
    if 'date' in df_asset.columns:
        df_asset['date'] = pd.to_datetime(df_asset['date'], errors='coerce')
        df_asset = df_asset.sort_values('date')
        
    return df_asset
