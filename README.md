# NZD-USD-Hedger-predictor

# LSTM FX Prediction Model 

This repository contains my experiments with **Long Short-Term Memory (LSTM) networks** for foreign exchange (FX) prediction. The goal is to explore how deep learning can be applied to time series forecasting in financial markets, and to build reproducible workflows that can be extended to live trading or hedging strategies.

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
- **R** (for visualization and statistical analysis)  
- **YAML & .env** for configuration management  
- **Matplotlib / Seaborn** for plotting results  

---

##  Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Zaneham/NZD-USD-Hedger-predictor.git
cd NZD-USD-Hedger-predictor

###2. Set up the environment
Install all packages used like Pandas, matplotlib etc

###3 Configure
You'll have to use your own API keys and put it in a .env
I've used a news GPT to help give sentiment to my predictions

###4Run training script
type: python train.py
in console/cmd

###Run evaluation script

python evaluate.py

---

###Results

Model performance is logged automatically
Evaluation includes error metrics (MAE, RMSE) and visualizations of predicted vs. actual FX rates
Backtesting framework allows comparison with baseline strategies

Feedback is welcome and appreciated!
