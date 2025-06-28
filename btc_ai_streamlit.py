import streamlit as st
import pandas as pd
import numpy as np
import joblib
import ccxt
from keras.models import load_model
from ta import add_all_ta_features
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator, ROCIndicator
from ta.volatility import AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from datetime import datetime, timedelta
import pytz

# Constants
COIN = "BTC/USDT"
INTERVAL = '5m'
TIMEZONE = 'America/New_York'
LOOKBACK = 150

# Load model and scaler
model = load_model("btc_lstm_model.keras")
scaler = joblib.load("scaler.joblib")

# Fetch historical OHLCV data
def fetch_ohlcv(symbol, timeframe='5m', limit=150):
    exchange = ccxt.coinbase()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert(TIMEZONE)
    return df

# Add all required indicators
def add_indicators(df):
    df['EMA9'] = EMAIndicator(close=df['close'], window=9).ema_indicator()
    df['EMA21'] = EMAIndicator(close=df['close'], window=21).ema_indicator()
    df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    macd = MACD(close=df['close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['RSI'] = RSIIndicator(close=df['close']).rsi()
    df['ATR'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
    df['ROC'] = ROCIndicator(close=df['close']).roc()
    df['OBV'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
    df = df.dropna()
    return df

# Prepare features
def prepare_input(df):
    features = ['EMA9', 'EMA21', 'VWAP', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'ROC', 'OBV']
    X = scaler.transform(df[features])
    return np.expand_dims(X, axis=0)

# Predict class
def predict_class(df):
    x_input = prepare_input(df.tail(LOOKBACK))
    pred = model.predict(x_input)
    pred_class = np.argmax(pred, axis=1)[0]
    return pred_class

# Map prediction class to text
def class_to_label(pred_class):
    return {0: "Short", 1: "Neutral", 2: "Long"}.get(pred_class, "Unknown")

# Streamlit dashboard
st.set_page_config(page_title="BTC AI", layout="wide", page_icon="🧠")
st.title("🧠 BTC AI Trading Dashboard")

tab1, tab2, tab3 = st.tabs(["BTC", "ETH", "SOL"])

with tab1:
    st.subheader("🔍 BTC/USDT 5-Minute Prediction")
    df = fetch_ohlcv("BTC/USDT", INTERVAL, LOOKBACK + 50)
    df = add_indicators(df)
    pred_class = predict_class(df)
    label = class_to_label(pred_class)
    st.metric("📈 AI Prediction", label)
    st.dataframe(df.tail(5).set_index("timestamp"), use_container_width=True)

# You can duplicate this logic in tab2 and tab3 for ETH and SOL with minor edits
