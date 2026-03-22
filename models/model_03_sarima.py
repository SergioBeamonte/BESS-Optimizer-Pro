import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
import pmdarima as pm

def check_stationarity(series):
    """
    Performs ADF and KPSS tests to determine stationarity.
    Returns a dictionary with results.
    """
    res = {}
    # ADF test (Null: Unit root / Non-stationary)
    adf_res = adfuller(series.dropna())
    res['adf_p'] = adf_res[1]
    res['adf_stationary'] = adf_res[1] < 0.05
    
    # KPSS test (Null: Trend stationary / Stationary)
    kpss_res = kpss(series.dropna(), regression='c')
    res['kpss_p'] = kpss_res[1]
    res['kpss_stationary'] = kpss_res[1] > 0.05
    
    return res

def forecast_sarima(series, steps_ahead):
    """
    Fits an Auto-SARIMA model and returns the forecast and diagnostics.
    """
    # 1. Variance and Mean Tests (Diagnostics)
    diag = check_stationarity(series)
    
    try:
        # 2. Auto-Fit
        # m=24 for hourly seasonality. Quality-First search.
        model = pm.auto_arima(series, 
                              seasonal=True, m=24,
                              start_p=1, start_q=1,
                              max_p=3, max_q=3,
                              max_P=1, max_Q=1,
                              information_criterion='aic',
                              stepwise=True, 
                              suppress_warnings=True, 
                              error_action='ignore')
        
        forecast = model.predict(n_periods=steps_ahead)
        
        # 3. Term justification
        diag['order'] = model.order
        diag['seasonal_order'] = model.seasonal_order
        diag['aic'] = model.aic()
        
        return forecast.values, diag
    except Exception as e:
        # Fallback to simple SARIMA if auto fails
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24))
        fit_model = model.fit(disp=False)
        diag['fallback'] = str(e)
        return fit_model.forecast(steps_ahead).values, diag
