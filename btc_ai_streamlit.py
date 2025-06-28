# Force rebuild with Python 3.10

import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import plotly.graph_objs as go
import ccxt
import ta
from sklearn.preprocessing import StandardScaler

# ✅ Load scaler manually from .npy files (no joblib needed)
scaler = StandardScaler()
scaler.mean_ = np.load("scaler_mean.npy", allow_pickle=True)
scaler.scale_ = np.load("scaler_scale.npy", allow_pickle=True)

# ✅ Load the trained model
model = load_model("btc_lstm_model.keras")

# Fetch latest BTC/USDT 15‑minute data
@st.cache_data(ttl=300)
def fetch_data():
    exchange = ccxt.coinbasepro()
    now = exchange.milliseconds()
    since = now - 15 * 60 * 1000 * 500  # ~500 intervals (~125 hours)
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '15m', since=since)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Add technical indicators
def add_indicators(df):
    df = df.copy()
    df['EMA9'] = ta.trend.ema_indicator(df['close'], window=9)
    df['EMA21'] = ta.trend.ema_indicator(df['close'], window=21)
    df['VWAP'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])
    df['RSI'] = ta.momentum.rsi(df['close'])
    macd = ta.trend.MACD(df['close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
    df['ROC'] = ta.momentum.roc(df['close'])
    df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    return df.dropna().reset_index(drop=True)

# Prepare input and predict
def predict(df):
    features = ['EMA9', 'EMA21', 'VWAP', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'ROC', 'OBV']
    seq = df[features].values[-60:]
    seq_scaled = scaler.transform(seq)
    X = np.expand_dims(seq_scaled, axis=0)
    probs = model.predict(X)[0]
    pred = np.argmax(probs)
    return pred, probs

# Streamlit UI
st.title("BTC 15‑Min AI Predictor")
df = fetch_data()
df = add_indicators(df)

pred, probs = predict(df)
cls = {0: "Short (🔴)", 1: "Neutral (⚪)", 2: "Long (🟢)"}

st.subheader("Latest Prediction")
st.markdown(f"### {cls[pred]}")

st.subheader("Confidence Stats")
conf = {cls[i]: f"{probs[i]*100:.1f}%" for i in range(len(probs))}
st.write(conf)

# Price chart with signal markers
df_plot = df.iloc[-1:]
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['close'], mode='lines', name='Close'))
fig.add_trace(go.Scatter(x=df_plot['timestamp'], y=df_plot['close'],
                         mode='markers', name='Signal',
                         marker=dict(color=['red','white','green'][pred], size=12)))
st.plotly_chart(fig, use_container_width=True)
