import numpy as np

from src.config import WINDOW_SIZE

def create_lstm_sequences(data, window_size=WINDOW_SIZE):
    """
    Create LSTM input sequences from scaled time series data.

    Parameters
    ----------
    data : np.ndarray
        Scaled price data of shape (n_samples, 1)
    window_size : int
        Number of past timesteps used for prediction

    Returns
    -------
    X : np.ndarray
        LSTM input of shape (samples, window_size, 1)n
    """

    X=[]

    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])

    X=np.array(X)
    X=X.reshape((X.reshape[0],X.shape[1],1))

    return X