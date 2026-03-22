from statsmodels.tsa.holtwinters import ExponentialSmoothing

def forecast_holt_winters(series, steps_ahead):
    try:
        model = ExponentialSmoothing(series, seasonal_periods=24, trend='add', seasonal='add').fit()
        return model.forecast(steps_ahead).values
    except:
        model = ExponentialSmoothing(series).fit()
        return model.forecast(steps_ahead).values
