# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 19:47:25 2025

@author: GGPC
"""
#All packages are imported here:
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests


def get_live_rate():
    url = "https://api.exchangerate-api.com/v4/latest/NZD"
    try:
        response = requests.get(url)
        data = response.json()
        return data["rates"]["USD"]
    except Exception as e:
        return None

# Load hedge log 
log = pd.read_csv(r"C:\Users\GGPC\OneDrive\Documents\New folder\hedge_log.csv")

log.columns = log.columns.str.strip().str.replace(" ", "_")

st.set_page_config(page_title="FX Hedging Dashboard", layout="wide")
st.title("📊 FX Hedging Dashboard 📊 ")

#  Dropdown Filter 
st.subheader("🔍 Filter Hedge Log by Decision Type🪄")

# Get unique decision types
decision_types = log.Decision.dropna().unique().tolist()
selected_decision = st.selectbox("Select decision type:", ["All"] + decision_types)

# Filter the log
if selected_decision != "All":
    filtered_log = log[log.Decision == selected_decision]
else:
    filtered_log = log

# Display filtered table
st.dataframe(filtered_log.tail(20), use_container_width=True)

#  Latest Prediction Panel 
st.subheader("🔮 Latest Hedge Decision")
latest = log.iloc[-1]
col1, col2, col3 = st.columns(3)
live_rate = get_live_rate()
if live_rate:
    col1.metric("Live NZD/USD Rate", f"{live_rate:.5f}")
else:
    col1.warning("Live rate unavailable")
    
    st.download_button(
        label="📥 Download filtered hedge log",
        data=filtered_log.to_csv(index=False),
        file_name="filtered_hedge_log.csv",
        mime="text/csv"
    ) ## This is a download button
##This uses the csv file fyi
col2.metric("Predicted Rate", f"{latest.Predicted_Rate:.5f}")
col3.metric("Decision", latest.Decision)

#  Hedge Log Table 
st.subheader("📋 Recent Hedge Log Entries 📊")
st.dataframe(log.tail(20), use_container_width=True)




#  Performance Metrics 
st.subheader("📉 Performance Summary 📈")
rmse = log.Error.dropna().pow(2).mean()**0.5
mae = log.Error.dropna().abs().mean()
dir_acc = log.CorrectDirection.dropna().mean() * 100

col1, col2, col3 = st.columns(3)
col1.metric("RMSE", f"{rmse:.5f}")
col2.metric("MAE", f"{mae:.5f}")
col3.metric("Directional Accuracy", f"{dir_acc:.2f}%")

st.subheader("📊 Hedge Outcome Over Time")
outcome_by_decision = filtered_log.groupby(["Decision", "HedgeOutcome"]).size().unstack(fill_value=0)
st.bar_chart(outcome_by_decision)


#  Outcome Breakdown 
st.subheader("📊 Hedge Outcome Breakdown 📊")
outcome_counts = log.HedgeOutcome.value_counts()
st.bar_chart(outcome_counts)

#  Prediction vs Actual Chart 
st.subheader("📉 Predicted vs Actual Rates 📈")
fig, ax = plt.subplots()
log.plot(x="Timestamp", y=["Live_Rate", "Predicted_Rate", "Actual"], ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# Simulator Panel 
st.subheader("🧪 Hedge Decision Simulator 🧪")
hypothetical_rate = st.number_input("Enter hypothetical live NZD/USD rate:", value=latest.Live_Rate)
predicted_rate = latest.Predicted_Rate

if st.button("Simulate Decision"):
    if predicted_rate > hypothetical_rate:
        st.success("Model would recommend: Hedge now")
    elif predicted_rate < hypothetical_rate:
        st.info("Model would recommend: Wait")
    else:
        st.warning("Model is neutral — no clear signal")
