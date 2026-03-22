import numpy as np

def forecast_naive_mean(series, steps_ahead):
    """
    Naive Mean forecast: predicts the average of the last 24 hours of data.
    """
    lookback = min(24, len(series))
    daily_mean = series.iloc[-lookback:].mean()
    return np.full(steps_ahead, daily_mean)
