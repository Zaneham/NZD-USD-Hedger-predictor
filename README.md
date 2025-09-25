# NZD-USD-Hedger-predictor


This repository contains my experiments with **Long Short-Term Memory (LSTM) networks** for foreign exchange (FX) prediction. The goal is to explore how deep learning can be applied to time series forecasting in financial markets, and to build reproducible results that can be extended to live trading or hedging strategies. I'm currently using it for a side case study analysis I'll be publishing. I'm constantly learning so any feedback is appreciated

---

##  Overview
This project applies LSTM models to FX data with the aim of predicting exchange rate movements.  
It focuses on reproducibility, automation, and integration of external features such as sentiment and macroeconomic indicators.

---

##  Features
- Multivariate LSTM model for FX rate prediction  
- Integration of price data with external features (e.g., sentiment, macro indicators)  
- Automated logging and evaluation of predictions vs. actual outcomes  
- Configurable workflow using `.env` and YAML for reproducibility  
- Backtesting framework to benchmark model performance  

---

##  Tech Stack
- **Python** (TensorFlow / Keras, NumPy, Pandas, Scikit-learn)  
- **YAML & .env** for configuration management  
- **Matplotlib** for plotting results
- ***Streamlit*** for a user friendly app

---

## Getting Started with the LSTM

### 1. Clone the repository
git clone https://github.com/Zaneham/NZD-USD-Hedger-predictor.git
cd NZD-USD-Hedger-predictor

2. Set up environment
bash
Make sure you've got all the packaged used


3. Configure
Add your API keys and environment variables to a .env file

Adjust model and data settings in config.yaml


4. Run training
bash
python train.py


5. Evaluate
bash
python evaluate.py


##Results##

Model performance is logged automatically



 Contributing
Feedback and collaboration are welcome! If you have ideas for improving the model, feel free to open an issue or submit a pull request.

📬 Contact
GitHub: Zaneham

LinkedIn:(https://www.linkedin.com/in/zane-hambly-11a69b166/)
