# Predictor-
import streamlit as st
import yfinance as yf
import pandas as pd
import ta
from sklearn.ensemble import RandomForestClassifier

# App title and setup
st.set_page_config(page_title="Nifty Next Candle Predictor", layout="centered")
st.title("📈 Nifty Next 5-Min Candle Predictor")

# When button is clicked
if st.button("🔮 Predict Next Candle"):

    with st.spinner("Fetching data and predicting..."):

        # Step 1: Download last 5 days of 5-minute Nifty data
        data = yf.download("^NSEI", period="5d", interval="5m")

        # Step 2: Feature Engineering
        data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()
        data['EMA_10'] = ta.trend.EMAIndicator(data['Close'], window=10).ema_indicator()
        data['EMA_20'] = ta.trend.EMAIndicator(data['Close'], window=20).ema_indicator()
        data['MACD'] = ta.trend.MACD(data['Close']).macd_diff()
        data['Returns'] = data['Close'].pct_change()

        # Step 3: Drop missing values
        data.dropna(inplace=True)

        # Step 4: Create features and labels
        features = data[['RSI', 'EMA_10', 'EMA_20', 'MACD', 'Returns']]
        targets = (data['Close'].shift(-1) > data['Close']).astype(int)
        features = features.iloc[:-1]
        targets = targets.iloc[:-1]

        # Step 5: Train the model
        X_train = features[:-50]
        y_train = targets[:-50]
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Step 6: Make prediction on latest candle
        latest = features.tail(1)
        prediction = model.predict(latest)[0]

        # Step 7: Display result
        if prediction == 1:
            st.success("📈 Prediction: Next 5‑minute Nifty candle is likely to go **UP**.")
        else:
            st.error("📉 Prediction: Next 5‑minute Nifty candle is likely to go **DOWN**.")
            
