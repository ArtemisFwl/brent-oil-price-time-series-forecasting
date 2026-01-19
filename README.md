# Brent Oil Price Time Series Forecasting

## Overview
This project implements an end-to-end time series forecasting pipeline for **Brent crude oil prices**, covering classical statistical models and deep learning approaches. The goal is to evaluate and compare different forecasting techniques on a highly volatile, real-world energy dataset.

The project follows **industry best practices** including baseline modeling, proper time-aware validation, and honest performance comparison.

---

## Dataset
- **Source:** Investing.com (Brent Oil Historical Prices)
- **Frequency:** Daily
- **Target Variable:** Brent crude oil price
- **Time Span:** Multiple decades (long-term historical data)

---

## Project Structure

brent-oil-price-time-series-forecasting/
│
├── data/
│ ├── raw/ # Original downloaded data
│ └── processed/ # Cleaned & preprocessed data
│
├── notebooks/
│ ├── 01_eda.ipynb # Data cleaning, EDA, baseline model
│ ├── 02_arima_model.ipynb # ARIMA modeling & evaluation
│ └── 03_lstm_model.ipynb # LSTM deep learning model
│
├── .gitignore
├── requirements.txt
└── README.md



---

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Datetime parsing and standardization
- Enforcing daily frequency and handling missing dates
- Time-aware train–validation split
- **Naive baseline model** for benchmarking

### 2. ARIMA Modeling
- Stationarity testing using Augmented Dickey-Fuller (ADF)
- First-order differencing to achieve stationarity
- ACF and PACF analysis for parameter selection
- ARIMA(1,1,1) model fitting and diagnostics
- Comparison against naive baseline

### 3. LSTM Deep Learning Model
- Data scaling using MinMaxScaler
- Sliding window sequence generation
- Multi-layer LSTM architecture with dropout
- Model training and validation
- Inverse scaling and final evaluation

---

## Results Summary

| Model | MAE | RMSE |
|------|-----|------|
| Naive Baseline | ~13.19 | ~18.25 |
| ARIMA(1,1,1) | ~13.19 | ~18.26 |
| **LSTM** | **~2.14** | **~2.79** |

### Key Insight
Classical ARIMA models struggle with the high volatility and non-linearity of crude oil prices.  
The LSTM model significantly outperforms both ARIMA and the naive baseline, effectively capturing complex temporal patterns and price shocks.

---

## Key Learnings
- Importance of **baseline models** for honest evaluation
- Limitations of classical time series models on volatile financial data
- Strength of deep learning models in capturing non-linear dependencies
- Proper handling of time series data prevents leakage and misleading results

---

## Tech Stack
- **Python**
- **Pandas, NumPy**
- **Matplotlib**
- **scikit-learn**
- **statsmodels**
- **TensorFlow / Keras**

---

## Conclusion
This project demonstrates a complete, industry-aligned workflow for time series forecasting in the energy domain. By combining statistical and deep learning approaches, it highlights why modern deep learning models such as LSTM are better suited for complex, real-world forecasting problems like crude oil prices.

---

## Next Steps
- Experiment with SARIMA for seasonal effects
- Incorporate exogenous variables (macroeconomic indicators)
- Deploy the LSTM model using Streamlit or FastAPI
