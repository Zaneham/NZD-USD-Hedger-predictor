# -*- coding: utf-8 -*-
"""
Created on Thu Sep 11 20:17:48 2025

@author: GGPC
"""

import pandas as pd
import numpy as np
import requests
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from dotenv import load_dotenv
import os
from jb_news.news import CJBNews
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# --- Load API keys ---
load_dotenv()
JB_API_KEY = os.getenv("JB_API_KEY")      # jblanked.com key
NEWS_API_KEY = os.getenv("NEWS_API_KEY")  # newsapi.org key
FX_API_KEY = os.getenv("FX_API_KEY")      # exchangerate-api key

# --- Helper functions ---
def fetch_jb_calendar(api_key, offset=15):
    jb = CJBNews()
    jb.offset = offset
    if jb.calendar(api_key, today=False):
        return pd.DataFrame([
            {
                "Date": pd.to_datetime(event.date).date(),
                "Event": event.name,
                "Currency": event.currency,
                "Forecast": event.forecast,
                "Actual": event.actual,
                "Previous": event.previous,
                "Outcome": event.outcome,
                "Strength": event.strength,
                "Quality": event.quality,
                "Projection": event.projection
            }
            for event in jb.calendar_info
        ])
    return pd.DataFrame()

def engineer_jb_features(jb_df):
    if jb_df.empty:
        return jb_df
    jb_df["Forecast"] = pd.to_numeric(jb_df["Forecast"], errors="coerce")
    jb_df["Actual"] = pd.to_numeric(jb_df["Actual"], errors="coerce")
    jb_df["Surprise"] = jb_df["Actual"] - jb_df["Forecast"]
    outcome_map = {"Better": 1, "Worse": -1, "In-Line": 0}
    jb_df["Outcome_num"] = jb_df["Outcome"].map(outcome_map).fillna(0)
    strength_map = {"Low": 1, "Medium": 2, "High": 3}
    jb_df["Strength_num"] = jb_df["Strength"].map(strength_map).fillna(0)
    quality_map = {"Low": 1, "Medium": 2, "High": 3}
    jb_df["Quality_num"] = jb_df["Quality"].map(quality_map).fillna(0)
    return jb_df[["Date", "Surprise", "Outcome_num", "Strength_num", "Quality_num"]]

def fetch_news_sentiment(api_key):
    url = "https://newsapi.org/v2/everything?q=forex&language=en"
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        news_data = response.json()
        sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
        return pd.DataFrame([
            {
                "Date": pd.to_datetime(article["publishedAt"]).date(),
                "Sentiment": sentiment_map.get(article.get("sentiment", "neutral"), 0)
            }
            for article in news_data.get("articles", [])
        ])
    else:
        print(f"NewsAPI error: {response.status_code}")
        return pd.DataFrame(columns=["Date", "Sentiment"])

# --- Load FX price data ---
df = pd.read_csv(r"C:\Users\GGPC\OneDrive\Documents\New folder\NZD_USD Historical Data.csv")
df.columns = df.columns.str.strip()
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df['Date_only'] = df['Date'].dt.date

# --- Fetch jb-news features ---
try:
    jb_calendar_df = fetch_jb_calendar(JB_API_KEY, offset=15)
    jb_features = engineer_jb_features(jb_calendar_df)

    if not jb_features.empty:
        df = pd.merge(df, jb_features, left_on="Date_only", right_on="Date", how="left")
        df.drop(columns=["Date_y"], inplace=True)
        df.rename(columns={"Date_x": "Date"}, inplace=True)
        print("jb-news features merged successfully.")
    else:
        print("jb-news returned no data, skipping.")
except Exception as e:
    print(f"Skipping jb-news features due to error: {e}")
# --- Fetch NewsAPI sentiment ---
sentiment_df = fetch_news_sentiment(NEWS_API_KEY)

# --- Merge sentiment ---
df = pd.merge(df, sentiment_df, left_on="Date_only", right_on="Date", how="left")
df['Sentiment'] = df['Sentiment'].fillna(0)
df.drop(columns=["Date_only", "Date_y"], inplace=True, errors="ignore")
df.rename(columns={"Date_x": "Date"}, inplace=True)

# --- Diagnostics ---
plt.figure(figsize=(8, 4))
plt.hist(df['Price'], bins=50, color='skyblue', edgecolor='black')
plt.title("Distribution of NZD/USD Prices in Training Data")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.show()

# --- Prepare data for LSTM ---
window_size = 90
recent_df = df[df['Date'] >= df['Date'].max() - pd.Timedelta(days=window_size)]

features = ['Price', 'Sentiment']
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(recent_df[features])

look_back = 60
X, y = [], []
for i in range(look_back, len(scaled_data)):
    X.append(scaled_data[i - look_back:i])
    y.append(scaled_data[i, 0])
X, y = np.array(X), np.array(y)

# --- Build and train LSTM ---
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

early_stop = EarlyStopping(monitor='loss', patience=5)
model.fit(X, y, epochs=100, batch_size=32, callbacks=[early_stop])

# --- Live FX rate ---
url = f"https://v6.exchangerate-api.com/v6/{FX_API_KEY}/latest/NZD"
response = requests.get(url)
data_api = response.json()
live_rate = data_api['conversion_rates']['USD']
print(f"Live NZD/USD rate: {live_rate:.5f}")

# --- Predict next rate ---
recent_scaled = scaled_data[-look_back + 1:]
new_point = scaler.transform([[live_rate, 0]])  # neutral sentiment
updated_sequence = np.append(recent_scaled, new_point, axis=0)
X_live = np.reshape(updated_sequence, (1, look_back, 2))

predicted_scaled = model.predict(X_live)
predicted_scaled_padded = np.array([[predicted_scaled[0][0], 0]])
predicted_rate = scaler.inverse_transform(predicted_scaled_padded)[0][0]
print(f"Predicted next rate: {predicted_rate:.5f}")

# --- Sensitivity analysis ---
print("\nSentiment sensitivity:")
for s in [-1, 0, 1]:
    test_point = scaler.transform([[live_rate, s]])
    updated_seq = np.append(recent_scaled, test_point, axis=0)
    X_test = np.reshape(updated_seq, (1, look_back, 2))
    pred_scaled = model.predict(X_test)
    pred_padded = np.array([[pred_scaled[0][0], 0]])
    pred_unscaled = scaler.inverse_transform(pred_padded)[0][0]
    print(f"Sentiment {s}: Predicted Rate = {pred_unscaled:.5f}")

decision = "Hedge now" if predicted_rate < live_rate else "Wait"
print(f"Recommendation: {decision}")

with open('hedge_log.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), live_rate, predicted_rate, decision])


# --- Plot predictions vs actual ---
predicted_scaled = model.predict(X)
predicted_padded = np.hstack([predicted_scaled, np.zeros_like(predicted_scaled)])
predicted_unscaled = scaler.inverse_transform(predicted_padded)[:, 0]

plt.figure(figsize=(10, 6))
recent_dates = recent_df['Date'].values[look_back:]
recent_prices = recent_df['Price'].values[look_back:]
plt.plot(recent_dates, predicted_unscaled, label='Predicted')
plt.plot(recent_dates, recent_prices, label='Actual')
plt.legend()
plt.title('NZD/USD Prediction vs Actual')
plt.xlabel('Date')
plt.ylabel('Rate')
plt.show()




