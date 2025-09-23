# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 11:11:06 2025

@author: GGPC
"""

import pandas as pd
import requests
from datetime import datetime, timedelta

# Load and clean the log
log = pd.read_csv(r"C:\Users\GGPC\OneDrive\Documents\New folder\hedge_log.csv")
log = log.dropna(subset=['Timestamp'])  # Fix for NaN issue

# Filter for yesterday's prediction
yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
filtered_log = log[log['Timestamp'].str.startswith(yesterday)]

if filtered_log.empty:
    print(f"No prediction found for {yesterday}. Skipping evaluation.")
else:
    entry = filtered_log.iloc[-1]

    # Fetch today's actual rate
    from dotenv import load_dotenv
    import os

    load_dotenv()
    API_KEY = os.getenv("FX_API_KEY")
    url = f"https://v6.exchangerate-api.com/v6/{API_KEY}/latest/NZD"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        actual_rate = data['conversion_rates']['USD']
        print(f"Actual NZD/USD rate (T+1): {actual_rate:.5f}")
    except Exception as e:
        print(f"Error fetching actual rate: {e}")
        actual_rate = None

    # Evaluate prediction
    if 'Predicted Rate' in entry and pd.notna(entry['Predicted Rate']) and actual_rate is not None:
        error = entry['Predicted Rate'] - actual_rate
        correct_direction = (entry['Predicted Rate'] > entry['Live Rate']) == (actual_rate > entry['Live Rate'])

        # Determine hedge outcome
        if entry['Decision'] == "Hedge now":
            hedge_outcome = "Profitable" if actual_rate < entry['Live Rate'] else "Missed"
        elif entry['Decision'] == "Wait":
            hedge_outcome = "Good Wait" if actual_rate > entry['Live Rate'] else "Should've Hedged"
        else:
            hedge_outcome = "Unknown"

        # Update log
        log.loc[entry.name, 'Actual'] = actual_rate
        log.loc[entry.name, 'Error'] = error
        log.loc[entry.name, 'CorrectDirection'] = correct_direction
        log.loc[entry.name, 'HedgeOutcome'] = hedge_outcome

        log.to_csv(r"C:\Users\GGPC\OneDrive\Documents\New folder\hedge_log.csv", index=False)
        print("Log updated with actual rate, error, direction, and hedge outcome.")
    else:
        print("Missing 'Predicted Rate' or actual rate. Skipping evaluation.")

