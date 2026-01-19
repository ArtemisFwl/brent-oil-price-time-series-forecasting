import numpy as np
import pandas as pd

from src.model_loader import load_lstm_model, load_scaler 
from src.data_preprocessing import preprocess_price_series
from src.feature_engineering import create_lstm_sequences 
from src.config import WINDOW_SIZE

class BrentPriceForecaster:
    """
    End to end inference pipeline for Brent oil Price forecasting using LSTM
    """

    def __init__(self):
        self.model=load_lstm_model()
        self.scaler=load_scaler()

    def predict_next(self, df):
        """
        Predict next-day Brent oil price.

        Parameters
        ----------
        df : pd.DataFrame
            Raw DataFrame with columns ['date', 'price']

        Returns
        -------
        float
            Predicted next-day price
        """

        # Preprocess Raw Data 
        df_clean=preprocess_price_series(df)

        # Use last WINDOW_SIZE Prices 
        recent_price= df_clean[["price"]].values[-WINDOW_SIZE:]

        #Scale 
        recent_scaled=self.scaler.transform(recent_price)

        #Create LSTM input 
        X=create_lstm_sequences(recent_scaled, WINDOW_SIZE)

        #Predict 
        pred_scaled=self.model.predict(X[-1].reshape(1, WINDOW_SIZE, 1))

        #Inverse Scale 
        pred=self.scaler.inverse_transform(pred_scaled)

        return float(pred[0][0])