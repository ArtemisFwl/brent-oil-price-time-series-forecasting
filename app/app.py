import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import streamlit as st
import pandas as pd

from src.inference import BrentPriceForecaster 

st.set_page_config(page_title="Brent Oil Price Forecast", layout="centered")

st.title("Brent Oil Price Forecasting")
st.write("LSTM based next day price prediction")


st.write("Click the button to predict the next day Brent Oil Price")

if st.button("Predict Next Day Price"):
    with st.spinner("Loading model and generating prediction..."):
        df = pd.read_csv("data/raw/brent_daily_price.csv")

        forecaster = BrentPriceForecaster()
        prediction = forecaster.predict_next(df)

    st.success(f"📌 Predicted Next-Day Price: **${prediction:.2f}**")