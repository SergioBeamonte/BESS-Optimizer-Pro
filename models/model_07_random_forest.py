import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def create_features(df, target_var):
    df_feat = df.copy()
    
    # Features de tiempo
    df_feat['hour'] = df_feat.index.hour
    df_feat['dayofweek'] = df_feat.index.dayofweek
    
    n = len(df)
    
    # 🔹 Lags cortos 
    short_lags = [1, 2, 3, 6, 12]
    for lag in short_lags:
        if n > lag:
            df_feat[f'lag_{lag}'] = df_feat[target_var].shift(lag)
    
    # 🔹 Lags diarios 
    daily_lags = [24, 48, 72]
    for lag in daily_lags:
        if n > lag:
            df_feat[f'lag_{lag}h'] = df_feat[target_var].shift(lag)
    
    # 🔹 Lags semanales
    weekly_lags = [24*7, 24*14]
    for lag in weekly_lags:
        if n > lag:
            df_feat[f'lag_{lag//24}d'] = df_feat[target_var].shift(lag)
            
    # 🔹 Lag anual 
    annual_lag = 24 * 365
    if n > annual_lag:
        df_feat['lag_annual'] = df_feat[target_var].shift(annual_lag)
    
    return df_feat


def forecast_random_forest(df_full, target_var, steps_ahead):
    df_feat = create_features(df_full, target_var)
    train = df_feat.iloc[:-steps_ahead].dropna()
    test = df_feat.iloc[-steps_ahead:]
    
    if train.empty:
        # Fallback if dropna cleared everything (too short dataset)
        return np.full(steps_ahead, df_full[target_var].iloc[-steps_ahead-1])
        
    features = [c for c in train.columns if c != target_var]
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(train[features], train[target_var])
    return model.predict(test[features])
