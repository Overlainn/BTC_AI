import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import ta

st.set_page_config(page_title="BTC AI Dashboard", layout="wide")
st.title("📈 BTC AI Trading Dashboard (30-min Intervals)")

# ✅ Load data
data = pd.read_csv("btc_15min_data.csv")
data.rename(columns={"timestamp": "datetime"}, inplace=True)  # Ensure correct datetime column
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)
data = data.sort_index()

# ✅ Resample to 30-minute intervals
data_30min = data.resample('30min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

# ✅ Compute technical indicators
data_30min['EMA9'] = ta.trend.ema_indicator(data_30min['close'], window=9).fillna(0)
data_30min['EMA21'] = ta.trend.ema_indicator(data_30min['close'], window=21).fillna(0)
data_30min['VWAP'] = ta.volume.volume_weighted_average_price(
    data_30min['high'], data_30min['low'], data_30min['close'], data_30min['volume'], window=14
).fillna(0)
data_30min['RSI'] = ta.momentum.RSIIndicator(data_30min['close']).rsi().fillna(0)
macd_ind = ta.trend.MACD(data_30min['close'])
data_30min['MACD'] = macd_ind.macd_diff().fillna(0)
data_30min['MACD_Signal'] = macd_ind.macd_signal().fillna(0)
data_30min['ATR'] = ta.volatility.average_true_range(
    data_30min['high'], data_30min['low'], data_30min['close']
).fillna(0)
data_30min['ROC'] = ta.momentum.ROCIndicator(data_30min['close']).roc().fillna(0)
data_30min['OBV'] = ta.volume.on_balance_volume(data_30min['close'], data_30min['volume']).fillna(0)

# ✅ Select features for model input
feature_cols = ['EMA9', 'EMA21', 'VWAP', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'ROC', 'OBV']
X = data_30min[feature_cols].copy().dropna()

# ✅ Load scaler from .npy files
scaler = StandardScaler()
scaler.mean_ = np.load("scaler_mean.npy")
scaler.scale_ = np.load("scaler_scale.npy")
X_scaled = scaler.transform(X)

# ✅ Load trained LSTM model (use compile=False to avoid errors)
model = load_model("btc_lstm_model.keras", compile=False)

# ✅ Create sequences (last 20 steps for LSTM input)
SEQUENCE_LENGTH = 20
sequences = []
timestamps = []

for i in range(SEQUENCE_LENGTH, len(X_scaled)):
    sequences.append(X_scaled[i-SEQUENCE_LENGTH:i])
    timestamps.append(X.index[i])

X_seq = np.array(sequences)

# ✅ Predict
preds = model.predict(X_seq, verbose=0)
classes = np.argmax(preds, axis=1)  # 0=Short, 1=Neutral, 2=Long

# ✅ Build final table
results = pd.DataFrame({
    'Timestamp': timestamps,
    'Prediction': classes
})
results['Prediction Label'] = results['Prediction'].map({0: 'Short 📉', 1: 'Neutral ⚖️', 2: 'Long 📈'})

# ✅ Display table
st.subheader("🔮 BTC Class Predictions (Next 3 Candles / 90 Min Horizon)")
st.dataframe(results.set_index('Timestamp').tail(20), use_container_width=True)
