import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from datetime import datetime
from utils import safe_yf_download
from data_engine import get_fred_series

# --- MATH ---
def calculate_hurst(series, lags=range(2, 20)):
    try:
        tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    except: return 0.5

def calculate_z_score(series, window=52):
    roll_mean = series.rolling(window=window).mean()
    roll_std = series.rolling(window=window).std()
    return (series - roll_mean) / roll_std

# --- ML REGIMES ---
@st.cache_data(ttl=3600)
def get_market_regime(ticker):
    try:
        df = safe_yf_download(ticker, period="5y", interval="1d")
        if df.empty: return None
        data = df.copy()
        data['Returns'] = data['Close'].pct_change()
        data['Volatility'] = data['Returns'].rolling(20).std()
        data = data.dropna()
        X = data[['Returns', 'Volatility']].values
        gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42)
        gmm.fit(X)
        current_state = gmm.predict(X[[-1]])[0]
        probs = gmm.predict_proba(X[[-1]])[0]
        means = gmm.means_
        state_order = np.argsort(means[:, 1]) 
        regime_map = {state_order[0]: "LOW VOL (Trend)", state_order[1]: "NEUTRAL (Chop)", state_order[2]: "HIGH VOL (Crisis)"}
        regime_desc = regime_map.get(current_state, "Unknown")
        color = "bullish" if "LOW VOL" in regime_desc else "bearish" if "HIGH VOL" in regime_desc else "neutral"
        return {"regime": regime_desc, "color": color, "confidence": max(probs)}
    except: return None

@st.cache_data(ttl=86400)
def get_macro_ml_regime(cpi_df, rate_df):
    if cpi_df.empty or rate_df.empty: return None
    try:
        df = pd.merge(cpi_df, rate_df, left_index=True, right_index=True, how='inner')
        df.columns = ['CPI', 'Rates']
        df['CPI_YoY'] = df['CPI'].pct_change(12) * 100
        df = df.dropna()
        X = df[['CPI_YoY', 'Rates']].values
        if len(X) < 12: return None
        gmm = GaussianMixture(n_components=4, random_state=42)
        gmm.fit(X)
        curr_cpi = df['CPI_YoY'].iloc[-1]
        curr_rate = df['Rates'].iloc[-1]
        regime_name = "Neutral"
        if curr_cpi > 4 and curr_rate < curr_cpi: regime_name = "INFLATIONARY (Neg Real Rates)"
        elif curr_cpi > 4 and curr_rate > curr_cpi: regime_name = "TIGHTENING (Pos Real Rates)"
        elif curr_cpi < 2: regime_name = "DEFLATIONARY / RISK OFF"
        else: regime_name = "GOLDILOCKS / STABLE"
        return {"regime": regime_name, "cpi": curr_cpi, "rate": curr_rate}
    except: return None

@st.cache_data(ttl=3600)
def get_ml_prediction(ticker):
    try:
        df = safe_yf_download(ticker, period="2y", interval="1d") 
        if df.empty: return None, 0.5
        data = df.copy()
        data['Returns'] = data['Close'].pct_change()
        data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
        data['Vol_5d'] = data['Returns'].rolling(5).std()
        data['Mom_5d'] = data['Close'].pct_change(5)
        data = data.dropna()
        if len(data) < 50: return None, 0.5
        X = data[['Vol_5d', 'Mom_5d']]
        y = data['Target']
        model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
        model.fit(X, y)
        prob_up = model.predict_proba(X.iloc[[-1]])[0][1]
        return model, prob_up
    except: return None, 0.5

# --- GAMMA EXPOSURE ---
def calculate_black_scholes_gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0: return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

@st.cache_data(ttl=3600)
def get_gex_profile(opt_ticker):
    if opt_ticker is None: return None, None, None, None
    try:
        tk = yf.Ticker(opt_ticker)
        try:
            hist = tk.history(period="1d")
            spot_price = hist['Close'].iloc[-1] if not hist.empty else tk.fast_info.last_price
        except: return None, None, None, None
        
        if spot_price is None: return None, None, None, None
        exps = tk.options
        if not exps: return None, None, None, None
        
        target_exp = exps[1] if len(exps) > 1 else exps[0]
        chain = tk.option_chain(target_exp)
        calls, puts = chain.calls, chain.puts
        
        if calls.empty or puts.empty: return None, None, None, None
        
        # Calculate ATM IV
        atm_mask = (calls['strike'] > spot_price * 0.95) & (calls['strike'] < spot_price * 1.05)
        atm_calls = calls[atm_mask]
        avg_iv = atm_calls['impliedVolatility'].mean() * 100 if not atm_calls.empty else 0
        
        exp_date = datetime.strptime(target_exp, "%Y-%m-%d")
        days_to_exp = (exp_date - datetime.now()).days
        T = 0.001 if days_to_exp <= 0 else days_to_exp / 365.0
        gex_data = []
        strikes = sorted(list(set(calls['strike'].tolist() + puts['strike'].tolist())))
        for K in strikes:
            if K < spot_price * 0.75 or K > spot_price * 1.25: continue
            c_row = calls[calls['strike'] == K]
            p_row = puts[puts['strike'] == K]
            c_oi = c_row['openInterest'].iloc[0] if not c_row.empty else 0
            p_oi = p_row['openInterest'].iloc[0] if not p_row.empty else 0
            c_iv = c_row['impliedVolatility'].iloc[0] if not c_row.empty and 'impliedVolatility' in c_row.columns else 0.2
            p_iv = p_row['impliedVolatility'].iloc[0] if not p_row.empty and 'impliedVolatility' in p_row.columns else 0.2
            c_gamma = calculate_black_scholes_gamma(spot_price, K, T, 0.045, c_iv)
            p_gamma = calculate_black_scholes_gamma(spot_price, K, T, 0.045, p_iv)
            net_gex = (c_gamma * c_oi - p_gamma * p_oi) * spot_price * 100
            gex_data.append({"strike": K, "gex": net_gex})
            
        df = pd.DataFrame(gex_data, columns=['strike', 'gex'])
        return df, target_exp, spot_price, avg_iv
    except: return None, None, None, None

# --- TECHNICALS & SIMULATION ---
def calculate_volume_profile(df, bins=50):
    if df.empty: return None, None
    price_range = df['High'].max() - df['Low'].min()
    if price_range == 0: return None, None
    bin_size = price_range / bins
    df['PriceBin'] = ((df['Close'] - df['Low'].min()) // bin_size).astype(int)
    vol_profile = df.groupby('PriceBin')['Volume'].sum().reset_index()
    vol_profile['PriceLevel'] = df['Low'].min() + (vol_profile['PriceBin'] * bin_size)
    poc_idx = vol_profile['Volume'].idxmax()
    return vol_profile, vol_profile.loc[poc_idx, 'PriceLevel']

@st.cache_data(ttl=3600)
def get_seasonality_stats(daily_data, ticker_name):
    stats = {}
    try:
        df = daily_data.copy()
        df['Week_Num'] = df.index.to_period('W')
        high_days = df.groupby('Week_Num')['High'].idxmax().apply(lambda x: df.loc[x].name.day_name())
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        stats['day_high'] = high_days.value_counts().reindex(days_order, fill_value=0) / len(high_days) * 100
        df['Day'] = df.index.day
        df['Month_Week'] = np.ceil(df['Day'] / 7).astype(int)
        df['Returns'] = df['Close'].pct_change()
        stats['week_returns'] = df.groupby('Month_Week')['Returns'].mean() * 100
        try:
            intra = safe_yf_download(ticker_name, period="60d", interval="1h")
            if not intra.empty:
                if intra.index.tz is None: intra.index = intra.index.tz_localize('UTC')
                intra.index = intra.index.tz_convert('America/New_York')
                intra['Hour'] = intra.index.hour
                intra['Return'] = intra['Close'].pct_change()
                target_hours = [2,3,4,5,6, 8,9,10,11, 14,15,16,17,18, 20,21,22,23]
                stats['hourly_perf'] = intra[intra['Hour'].isin(target_hours)].groupby('Hour')['Return'].mean() * 100
        except: stats['hourly_perf'] = None
        return stats
    except: return None, None

@st.cache_data(ttl=3600)
def generate_monte_carlo(stock_data, days=126, simulations=1000):
    if stock_data is None or stock_data.empty or len(stock_data) < 2: return None, None
    try:
        close = stock_data['Close']
        log_returns = np.log(1 + close.pct_change())
        u, var = log_returns.mean(), log_returns.var()
        drift = u - (0.5 * var)
        stdev = log_returns.std()
        price_paths = np.zeros((days + 1, simulations))
        price_paths[0] = close.iloc[-1]
        daily_returns = np.exp(drift + stdev * np.random.normal(0, 1, (days, simulations)))
        for t in range(1, days + 1): price_paths[t] = price_paths[t - 1] * daily_returns[t - 1]
        return pd.date_range(start=close.index[-1], periods=days + 1, freq='B'), price_paths
    except: return None, None

def calculate_technical_radar(df):
    if df.empty or len(df) < 30: return None
    data = df.copy()
    close = data['Close']
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    k = close.ewm(span=12, adjust=False, min_periods=12).mean()
    d = close.ewm(span=26, adjust=False, min_periods=26).mean()
    data['MACD'] = k - d
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False, min_periods=9).mean()
    data['EMA_20'] = close.ewm(span=20, adjust=False).mean()
    data['EMA_50'] = close.ewm(span=50, adjust=False).mean()
    last = data.iloc[-1]
    signals = {}
    
    if last['RSI'] < 30: signals['RSI'] = {"val": f"{last['RSI']:.0f}", "bias": "OVERSOLD (Bull)", "col": "bullish"}
    elif last['RSI'] > 70: signals['RSI'] = {"val": f"{last['RSI']:.0f}", "bias": "OVERBOUGHT (Bear)", "col": "bearish"}
    else: signals['RSI'] = {"val": f"{last['RSI']:.0f}", "bias": "NEUTRAL", "col": "neutral"}
    
    macd_hist = last['MACD'] - last['MACD_Signal']
    if macd_hist > 0 and last['MACD'] > 0: signals['MACD'] = {"val": f"{macd_hist:.2f}", "bias": "BULLISH", "col": "bullish"}
    elif macd_hist < 0 and last['MACD'] < 0: signals['MACD'] = {"val": f"{macd_hist:.2f}", "bias": "BEARISH", "col": "bearish"}
    else: signals['MACD'] = {"val": f"{macd_hist:.2f}", "bias": "NEUTRAL", "col": "neutral"}
    
    if last['Close'] > last['EMA_20'] and last['EMA_20'] > last['EMA_50']:
        signals['Trend'] = {"val": "Uptrend", "bias": "STRONG BULL", "col": "bullish"}
    elif last['Close'] < last['EMA_20'] and last['EMA_20'] < last['EMA_50']:
        signals['Trend'] = {"val": "Downtrend", "bias": "STRONG BEAR", "col": "bearish"}
    else: signals['Trend'] = {"val": "Chop", "bias": "WEAK/MIXED", "col": "neutral"}
    return signals

@st.cache_data(ttl=3600)
def get_correlations(base_ticker, api_key):
    try:
        tickers = {"Base": base_ticker, "VIX": "^VIX", "10Y Yield": "^TNX", "Gold": "GC=F"}
        unique_tickers = list(set(tickers.values()))
        yf_data = safe_yf_download(unique_tickers, period="6mo", interval="1d")
        
        # FIX: Fetch DXY from FRED using DTWEXAFEGS
        fred_data = get_fred_series("DTWEXAFEGS", api_key) 
        
        if yf_data.empty: return pd.Series()
        
        # Handle Close column extraction safely
        if isinstance(yf_data.columns, pd.MultiIndex): 
            if 'Close' in yf_data.columns.get_level_values(0):
                 yf_df = yf_data.xs('Close', axis=1, level=0)
            else:
                 yf_df = yf_data['Close'].copy()
        elif 'Close' in yf_data.columns: 
            yf_df = yf_data['Close'].copy()
        else: 
            yf_df = yf_data.copy()

        if isinstance(yf_df, pd.Series): yf_df = yf_df.to_frame(name=unique_tickers[0])

        for label, ticker in tickers.items():
            if ticker in yf_df.columns:
                yf_df.rename(columns={ticker: label}, inplace=True)
        
        combined = yf_df
        if not fred_data.empty:
            fred_data = fred_data.rename(columns={'value': 'Dollar'})
            if yf_df.index.tz is not None: yf_df.index = yf_df.index.tz_localize(None)
            combined = pd.concat([yf_df, fred_data], axis=1).dropna()
            
        if combined.empty or 'Base' not in combined.columns: return pd.Series()
        corrs = combined.pct_change().rolling(20).corr(combined['Base'].pct_change()).iloc[-1]
        return corrs.drop('Base', errors='ignore') 
    except Exception as e: 
        return pd.Series()

@st.cache_data(ttl=300)
def run_strategy_backtest(ticker):
    try:
        df = safe_yf_download(ticker, period="2y", interval="1d")
        if df.empty: return None
        df['Returns'] = df['Close'].pct_change()
        df['Range'] = df['High'] - df['Low']
        df['TR'] = pd.concat([df['Range'], (df['High'] - df['Close'].shift(1)).abs(), (df['Low'] - df['Close'].shift(1)).abs()], axis=1).max(axis=1)
        df['Log_TR'] = np.log(df['TR'] / df['Close'])
        df['Vol_Forecast'] = df['Log_TR'].ewm(span=10).mean()
        df['Vol_Baseline'] = df['Log_TR'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['Signal'] = np.where((df['Vol_Forecast'] > df['Vol_Baseline']) & (df['Close'] > df['SMA_50']), 1, 0)
        df['Strategy_Returns'] = df['Signal'].shift(1) * df['Returns']
        df['Cum_BnH'] = (1 + df['Returns']).cumprod()
        df['Cum_Strat'] = (1 + df['Strategy_Returns']).cumprod()
        total_return = df['Cum_Strat'].iloc[-1] - 1
        sharpe = (df['Strategy_Returns'].mean() / df['Strategy_Returns'].std()) * np.sqrt(252) if df['Strategy_Returns'].std() != 0 else 0
        current_signal = "LONG" if df['Signal'].iloc[-1] == 1 else "CASH/NEUTRAL"
        return {"df": df, "signal": current_signal, "return": total_return, "sharpe": sharpe, "equity_curve": df['Cum_Strat']}
    except: return None

def calculate_vwap_bands(df):
    if df.empty: return df
    df = df.copy()
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VP'] = df['TP'] * df['Volume']
    df['Date'] = df.index.date
    df['Cum_VP'] = df.groupby('Date')['VP'].cumsum()
    df['Cum_Vol'] = df.groupby('Date')['Volume'].cumsum()
    df['VWAP'] = df['Cum_VP'] / df['Cum_Vol']
    df['Sq_Dist'] = df['Volume'] * (df['TP'] - df['VWAP'])**2
    df['Cum_Sq_Dist'] = df.groupby('Date')['Sq_Dist'].cumsum()
    df['Std_Dev'] = np.sqrt(df['Cum_Sq_Dist'] / df['Cum_Vol'])
    df['Upper_Band_1'] = df['VWAP'] + df['Std_Dev']
    df['Lower_Band_1'] = df['VWAP'] - df['Std_Dev']
    return df

@st.cache_data(ttl=300) 
def get_relative_strength(asset_ticker, benchmark_ticker="SPY"):
    try:
        asset = safe_yf_download(asset_ticker, period="5d", interval="15m")
        bench = safe_yf_download(benchmark_ticker, period="5d", interval="15m")
        if asset.empty or bench.empty: return pd.DataFrame()
        df = pd.DataFrame(index=asset.index)
        df['Asset_Close'] = asset['Close']
        df['Bench_Close'] = bench['Close']
        df = df.dropna()
        current_date = df.index[-1].date()
        session_data = df[df.index.date == current_date].copy()
        if session_data.empty: return pd.DataFrame()
        session_data['Asset_Pct'] = (session_data['Asset_Close'] / session_data['Asset_Close'].iloc[0]) - 1
        session_data['Bench_Pct'] = (session_data['Bench_Close'] / session_data['Bench_Close'].iloc[0]) - 1
        session_data['RS_Score'] = session_data['Asset_Pct'] - session_data['Bench_Pct']
        return session_data
    except: return pd.DataFrame()

def get_key_levels(daily_df):
    if daily_df.empty: return {}
    try: last_complete_day = daily_df.iloc[-2]
    except: return {}
    high, low, close = last_complete_day['High'], last_complete_day['Low'], last_complete_day['Close']
    pivot = (high + low + close) / 3
    return {
        "PDH": high, "PDL": low, "PDC": close, "Pivot": pivot, 
        "R1": (2 * pivot) - low, "S1": (2 * pivot) - high
    }
