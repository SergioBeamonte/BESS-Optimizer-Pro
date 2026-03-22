import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from sklearn.preprocessing import StandardScaler

def forecast_sarimax(df_full, target_var, steps_ahead):
    """
    Fits an Auto-SARIMAX model with feature selection and scaling.
    """
    # 1. Feature Selection (K=3 to keep it robust for small datasets)
    all_num_cols = df_full.select_dtypes(include=[np.number]).columns.tolist()
    redundant = ['generacion_total', 'generación_total']
    candidate_cols = [c for c in all_num_cols if c not in redundant and c != target_var]
    
    # Select top 3 correlated features
    corrs = df_full[candidate_cols].corrwith(df_full[target_var]).abs().sort_values(ascending=False)
    exog_cols = corrs.head(3).index.tolist()
    
    # 2. Split train/validation
    df_train = df_full.iloc[:-steps_ahead]
    df_val = df_full.iloc[-steps_ahead:]
    
    y_train = df_train[target_var]
    x_train_raw = df_train[exog_cols]
    x_val_raw = df_val[exog_cols]
    
    # 3. Scaling
    scaler = StandardScaler()
    try:
        x_train = scaler.fit_transform(x_train_raw)
        x_val = scaler.transform(x_val_raw)
        x_train_df = pd.DataFrame(x_train, index=y_train.index, columns=exog_cols)
        x_val_df = pd.DataFrame(x_val, index=df_val.index, columns=exog_cols)
    except:
        x_train_df, x_val_df = x_train_raw, x_val_raw

    try:
        # 4. Auto-SARIMAX (Using pmdarima with exog)
        # Optimized for speed.
        model = pm.auto_arima(y_train, 
                              exogenous=x_train_df,
                              seasonal=True, m=24,
                              start_p=0, start_q=0, max_p=2, max_q=2,
                              max_P=1, max_Q=1,
                              suppress_warnings=True, 
                              error_action='ignore', 
                              stepwise=True)
        
        # 5. Forecast
        forecast = model.predict(n_periods=steps_ahead, exogenous=x_val_df)
        
        return forecast.values, {
            "exog_used": exog_cols, 
            "order": model.order,
            "seasonal_order": model.seasonal_order,
            "aic": model.aic()
        }
        
    except Exception as e:
        # Fallback to simple SARIMA
        try:
            model = pm.auto_arima(y_train, seasonal=True, m=24, error_action='ignore')
            return model.predict(n_periods=steps_ahead).values, {"fallback": f"Auto-SARIMAX failed: {str(e)}"}
        except:
            return np.full(steps_ahead, y_train.iloc[-1]), {"fallback": "Critical Failure"}
