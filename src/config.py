import os

#Project Root

os.path.abspath(os.join(os.path.dirname(__file__), ".."))

#Data Paths

RAW_DATA_PATH =os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_PATH=os.path.join(PROJECT_ROOT, "data", "processed")


#Model Paths 

MODEL_DIR= os.path.join(PROJECT_ROOT, "models")
LSTM_MODEL_PATH= os.path.join(MODEL_DIR, "lstm_brent_model")
SCALER_PATH= os.path.join(MODEL_DIR, "price_scaler.pkl")

# Time series parameters
WINDOW_SIZE = 60

# Forecast settings
FORECAST_HORIZON = 1  # next-day forecast