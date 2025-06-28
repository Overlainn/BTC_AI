import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objs as go
import ta

# ✅ Load scaler using joblib
scaler = joblib.load("scaler.joblib")

# ✅ Load trained LSTM model
model = load_model("btc_lstm_model.keras")

# ✅ Load BTC/USDT 15-min historical data from CSV
@st.cache_data(ttl=300)
def load_data():
    df = pd.read_csv("btc_15min_data.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# ✅ Add technical indicators using `ta` library
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

# ✅ Prepare sequence, scale, and predict class
def predict(df):
    features = ['EMA9', 'EMA21', 'VWAP', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'ROC', 'OBV']
    seq = df[features].values[-60:]  # Last 60 intervals (~15 hours)
    seq_scaled = scaler.transform(seq)
    X = np.expand_dims(seq_scaled, axis=0)  # shape: (1, 60, num_features)
    probs = model.predict(X)[0]
    pred = np.argmax(probs)
    return pred, probs

# ✅ Streamlit UI setup
st.set_page_config(page_title="BTC 15-min AI Predictor", layout="wide")
st.title("📈 BTC/USDT 15-Minute AI Prediction")

# ✅ Load and prepare data
df = load_data()
df = add_indicators(df)

# ✅ Run prediction
pred, probs = predict(df)
class_map = {0: "Short (🔴)", 1: "Neutral (⚪)", 2: "Long (🟢)"}

# ✅ Show prediction
st.subheader("Latest Prediction")
st.markdown(f"## {class_map[pred]}")

# ✅ Show prediction probabilities
st.subheader("Confidence Levels")
conf = {class_map[i]: f"{probs[i]*100:.2f}%" for i in range(len(probs))}
st.write(conf)

# ✅ Plot recent price with signal marker
df_plot = df.iloc[-1:]
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df['timestamp'], y=df['close'],
    mode='lines', name='Close'
))
fig.add_trace(go.Scatter(
    x=df_plot['timestamp'], y=df_plot['close'],
    mode='markers', name='AI Signal',
    marker=dict(color=['red', 'white', 'green'][pred], size=12)
))
st.plotly_chart(fig, use_container_width=True)
