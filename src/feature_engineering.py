import numpy as np

def create_lstm_input(data, window_size):
    """
    Create a single LSTM input window for inference.
    """
    if len(data) < window_size:
        raise ValueError("Not enough data to create LSTM input window")

    X = data[-window_size:]          # shape (window_size, 1)
    X = X.reshape((1, window_size, 1))

    return X
