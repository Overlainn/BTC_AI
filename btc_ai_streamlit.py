# Re-deploying with Python 3.10

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from keras.models import load_model
from datetime import datetime, timedelta
import ccxt
import ta

# Load scaler and model
scaler = joblib.load("scaler.npy")
model = load_model("btc_lstm_model.keras")

# Fetch latest data
def fetch_data():
    exchange = ccxt.coinbasepro()
    now = exchange.milliseconds()
    since = now - 300 * 5 * 60 * 1000  # 5min * 300 = ~25hrs
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '5m', since=since)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Add indicators
def add_indicators(df):
    df['EMA9'] = ta.trend.ema_indicator(df['close'], window=9)
    df['EMA21'] = ta.trend.ema_indicator(df['close'], window=21)
    df['VWAP'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])
    df['RSI'] = ta.momentum.rsi(df['close'])
    macd = ta.trend.macd(df['close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
    df['ROC'] = ta.momentum.roc(df['close'])
    df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    return df.dropna()

# Predict latest point
def predict(df):
    features = ['EMA9', 'EMA21', 'VWAP', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'ROC', 'OBV']
    last_sequence = df[features].values[-60:]
    scaled_sequence = scaler.transform(last_sequence)
    input_data = np.expand_dims(scaled_sequence, axis=0)
    prediction = model.predict(input_data)
    return np.argmax(prediction), prediction

# Streamlit layout
st.title("BTC AI Price Movement Predictor")
df = fetch_data()
df = add_indicators(df)

pred_class, prob = predict(df)

classes = {0: "Short", 1: "Neutral", 2: "Long"}
colors = {"Short": "🔴", "Neutral": "🟡", "Long": "🟢"}

st.subheader("Prediction (next ~15min):")
st.markdown(f"### {colors[classes[pred_class]]} {classes[pred_class]}")

st.subheader("Confidence")
st.write({classes[i]: f"{100 * p:.2f}%" for i, p in enumerate(prob[0])})
