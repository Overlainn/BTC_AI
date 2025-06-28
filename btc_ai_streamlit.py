# btc_ai_streamlit.py

import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# Constants
MODEL_PATH = 'btc_lstm_model.keras'
SCALER_MEAN_PATH = 'scaler_mean.npy'
SCALER_SCALE_PATH = 'scaler_scale.npy'
WINDOW_SIZE = 10

# Load model and scaler
@st.cache_resource
def load_lstm_model():
    return load_model(MODEL_PATH)

@st.cache_resource
def load_scaler():
    scaler = StandardScaler()
    scaler.mean_ = joblib.load(SCALER_MEAN_PATH)
    scaler.scale_ = joblib.load(SCALER_SCALE_PATH)
    return scaler

# Add indicators (assuming they’re already computed in CSV)
def fetch_data():
    df = pd.read_csv("btc_15min_data.csv")
    return df.dropna()

def create_sequences(data, window_size=10):
    sequences = []
    for i in range(len(data) - window_size):
        seq = data[i:i+window_size]
        sequences.append(seq)
    return np.array(sequences)

def make_predictions(df, model, scaler, features):
    df_scaled = df.copy()
    df_scaled[features] = scaler.transform(df_scaled[features])
    sequences = create_sequences(df_scaled[features], window_size=WINDOW_SIZE)

    if len(sequences) == 0:
        return pd.DataFrame()

    preds = model.predict(sequences)
    preds_classes = np.argmax(preds, axis=1)

    pred_df = df.iloc[WINDOW_SIZE:].copy()
    pred_df['Prediction'] = preds_classes
    pred_df['Signal'] = pred_df['Prediction'].map({0: 'Short', 1: 'Neutral', 2: 'Long'})
    return pred_df[['timestamp', 'close', 'Prediction', 'Signal']]

# Streamlit UI
st.set_page_config(layout="wide", page_title="BTC AI Predictions")

st.title("🔮 BTC AI Live Prediction (15-min intervals)")

df = fetch_data()
features = ['EMA9', 'EMA21', 'VWAP', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'ROC', 'OBV']

model = load_lstm_model()
scaler = load_scaler()

pred_df = make_predictions(df, model, scaler, features)

if pred_df.empty:
    st.warning("Not enough data to make predictions.")
else:
    st.dataframe(pred_df.tail(50), use_container_width=True)
