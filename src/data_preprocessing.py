import pandas as pd


def preprocess_price_series(df):
    """
    Clean and prepare raw price time series for inference.

    Steps:
    - Ensure datetime index
    - Sort by time
    - Enforce daily frequency
    - Handle missing values

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ['date', 'price']

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame indexed by datetime
    """
    df = df.copy()

    # Standardize columns
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # Parse date
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    df = df.sort_values('date')
    df = df.set_index('date')

    # Enforce daily frequency
    df = df.asfreq("D")

    # Fill missing prices
    df['price'] = df['price'].interpolate(method='time')

    return df
