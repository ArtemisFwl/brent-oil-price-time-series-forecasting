from tensorflow.keras.models import load_model
import joblib

from src.config import LSTM_MODEL_PATH, SCALER_PATH

def load_lstm_model():
    """
    Load the trained LSTM model from the disk
    """
    model=load_model(LSTM_MODEL_PATH, compile=False)
    return model

def load_scaler():
    """
    Load the fitted MinMaxScaler used during training 
    """
    scaler=joblib.load(SCALER_PATH)
    return scaler

