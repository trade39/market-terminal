import streamlit as st
import yfinance as yf
import pandas as pd
import time
import os
import plotly.graph_objects as go

def get_api_key(key_name):
    """Securely retrieve API keys from secrets or environment variables."""
    if "api_keys" in st.secrets and key_name in st.secrets["api_keys"]:
        return st.secrets["api_keys"][key_name]
    if key_name in st.secrets:
        return st.secrets[key_name]
    if key_name == "gemini_api_key":
        if "GOOGLE_API_KEY" in st.secrets: return st.secrets["GOOGLE_API_KEY"]
    if key_name in os.environ:
        return os.environ[key_name]
    return None

def flatten_dataframe(df):
    """Prevents MultiIndex crashes in yfinance."""
    if df.empty: return df
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        if 'Close' in df.columns.get_level_values(0):
            df.columns = df.columns.get_level_values(0)
        elif df.columns.nlevels > 1 and 'Close' in df.columns.get_level_values(1):
            df.columns = df.columns.get_level_values(1)
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def safe_yf_download(tickers, period, interval, retries=3):
    """Robust yfinance download with retries and flattening."""
    for i in range(retries):
        try:
            time.sleep(0.1) 
            df = yf.download(tickers, period=period, interval=interval, progress=False)
            if not df.empty:
                return flatten_dataframe(df)
        except Exception as e:
            if i == retries - 1: return pd.DataFrame()
            time.sleep(2 ** i)
    return pd.DataFrame()

def terminal_chart_layout(fig, title="", height=350):
    """
    Standardized Chart Layout - Monochromatic Navy/Cyan Theme
    Background: #12161F (Matches Cards)
    Grid: #333333
    Text: #CCCCCC
    """
    fig.update_layout(
        title=dict(text=title, font=dict(color="#FFFFFF", family="Arial")),
        template="plotly_dark",
        paper_bgcolor="#12161F", # Card background
        plot_bgcolor="#12161F",  # Chart area background
        height=height,
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(
            showgrid=True, 
            gridcolor="#333333", 
            zerolinecolor="#444444",
            tickfont=dict(color="#AAAAAA")
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor="#333333", 
            zerolinecolor="#444444",
            tickfont=dict(color="#AAAAAA")
        ),
        font=dict(family="Courier New", color="#CCCCCC"),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CCCCCC")
        )
    )
    return fig
