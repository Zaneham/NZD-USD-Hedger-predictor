# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 11:11:06 2025

@author: GGPC
"""
import pandas as pd
import requests
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load hedge log
log_path = (r"C:\Users\GGPC\OneDrive\Documents\New folder\hedge_log.csv")
log = pd.read_csv(log_path)

# Normalize column names (underscores, no spaces)
log.columns = log.columns.str.strip().str.replace(" ", "_")

# Ensure required columns exist
for col in ["Actual", "Error", "CorrectDirection", "HedgeOutcome"]:
    if col not in log.columns:
        log[col] = None

# API key 
load_dotenv()
API_KEY = os.getenv("FX_API_KEY")
url = f"https://v6.exchangerate-api.com/v6/{API_KEY}/latest/NZD"

#  Fetch current actual rate 
try:
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    actual_rate_now = data['conversion_rates']['USD']
    print(f"Latest NZD/USD rate: {actual_rate_now:.5f}")
except Exception as e:
    print(f"Error fetching actual rate: {e}")
    actual_rate_now = None

def evaluate_entry(row, actual_rate):
    if pd.isna(row.Predicted_Rate) or pd.isna(row.Live_Rate) or actual_rate is None:
        return row  # skip if it is missing data

    error = row.Predicted_Rate - actual_rate
    correct_direction = (row.Predicted_Rate > row.Live_Rate) == (actual_rate > row.Live_Rate)

    if row.Decision == "Hedge now":
        hedge_outcome = "Profitable" if actual_rate < row.Live_Rate else "Missed"
    elif row.Decision == "Wait":
        hedge_outcome = "Good Wait" if actual_rate > row.Live_Rate else "Should've Hedged"
    else:
        hedge_outcome = "Unknown"

    row.Actual = actual_rate
    row.Error = error
    row.CorrectDirection = correct_direction
    row.HedgeOutcome = hedge_outcome
    return row

# Updating yesterday’s entry 
yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
mask = log['Timestamp'].astype(str).str.startswith(yesterday)

if mask.any() and actual_rate_now is not None:
    idx = log[mask].index[-1]  # last entry from yesterday
    log.loc[idx] = evaluate_entry(log.loc[idx], actual_rate_now)
    print(f"Updated yesterday's entry ({yesterday}).")
else:
    print(f"No prediction found for {yesterday} or missing actual rate.")

#  Backfill all past entries (if Actual is missing)
if actual_rate_now is not None:
    for idx, row in log.iterrows():
        if pd.isna(row.Actual):
            log.loc[idx] = evaluate_entry(row, actual_rate_now)

# Save updated log 
log.to_csv(log_path, index=False)
print("Log updated with actuals, errors, directions, and hedge outcomes.")
