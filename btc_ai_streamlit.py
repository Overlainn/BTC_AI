# Deployed with Python3.10 runtime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import datetime
import os

# Load the scaler
scaler = StandardScaler()
scaler.mean_ = np.load('scaler_mean.npy')
scaler.scale_ = np.load('scaler_scale.npy')

# Load model
model = load_model('btc_lstm_model.keras')

# Define features
FEATURES = ['EMA9', 'EMA21', 'VWAP', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'ROC', 'OBV']

def preprocess_input(df):
    df = df.copy()
    df = df[FEATURES]
    data_scaled = scaler.transform(df)
    sequences = []
    for i in range(len(data_scaled) - 10):
        sequences.append(data_scaled[i:i + 10])
    return np.array(sequences)

# Load your crypto data
@st.cache_data
def load_data():
    return pd.read_csv('btc_15min_data.csv')  # Ensure this file exists

# Streamlit UI
st.title("BTC AI 15-Min Prediction")
df = load_data()

# Display last few rows
st.subheader("Recent Data")
st.write(df.tail())

# Prediction
if st.button("Predict"):
    input_data = preprocess_input(df)
    preds = model.predict(input_data)
    pred_labels = np.argmax(preds, axis=1)

    latest_preds = pred_labels[-5:]
    st.subheader("Last 5 Predictions")
    st.write(latest_preds)

    class_map = {0: 'Short (↓)', 1: 'Neutral (→)', 2: 'Long (↑)'}
    mapped_preds = [class_map[i] for i in latest_preds]
    st.write("Signals:", mapped_preds)

    # Plot signal zones
    df_plot = df.iloc[-len(latest_preds):].copy()
    df_plot['Prediction'] = mapped_preds

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_plot['timestamp'], y=df_plot['close'], mode='lines+markers', name='Close Price'))
    fig.add_trace(go.Scatter(x=df_plot['timestamp'], y=df_plot['close'],
                             mode='markers', name='Signal',
                             marker=dict(size=10,
                                         color=[{'Short (↓)': 'red', 'Neutral (→)': 'gray', 'Long (↑)': 'green'}[p] for p in mapped_preds])))
    st.plotly_chart(fig, use_container_width=True)
