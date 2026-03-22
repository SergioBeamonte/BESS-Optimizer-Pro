import numpy as np

def forecast_seasonal_naive(series, steps_ahead):
    """
    Seasonal Naive forecast (24h periodicity).
    """
    seasonal_period = 24
    if len(series) < seasonal_period:
        return np.full(steps_ahead, series.iloc[-1])
        
    preds = []
    for i in range(steps_ahead):
        past_idx = -(seasonal_period - (i % seasonal_period))
        preds.append(series.iloc[past_idx])
    return np.array(preds)
