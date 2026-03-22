import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Ignorar las advertencias de convergencia
warnings.simplefilter('ignore', ConvergenceWarning)

def forecast_varima(df_full, target_var, steps_ahead, p=1, q=1):
    """
    Modelo VARIMA (Vector Autoregresivo Integrado de Media Móvil).
    Puramente endógeno: sin variables exógenas ni estacionalidad determinista.
    """
    if len(df_full) < steps_ahead + 2:
         return np.full(steps_ahead, df_full[target_var].iloc[-1]), {"fallback": "Data insuficiente"}

    try:
        # 1. Selección de Variables (Correlación)
        exclude = [x.lower() for x in ['datetime', 'index', 'precio_mwh', 'demanda', 'generacion_total', 'generación_total']]
        num_candidates = df_full.select_dtypes(include=[np.number]).columns.tolist()
        valid_candidates = [c for c in num_candidates if c.lower() not in exclude and c != target_var]
        
        df_train_raw_full = df_full.iloc[:-steps_ahead]
        
        # Mantenemos un máximo de 2 variables correlacionadas para estabilidad de la matriz
        if valid_candidates:
            corrs = df_train_raw_full[valid_candidates].corrwith(df_train_raw_full[target_var]).abs().sort_values(ascending=False)
            top_vars = corrs.head(2).index.tolist()
            var_cols = [target_var] + top_vars
        else:
            var_cols = [target_var]

        df_train_raw = df_train_raw_full[var_cols].copy()
        
        # 2. Estacionariedad (Diferenciación manual - La 'I' de VARIMA)
        best_d = 0
        work_data = df_train_raw.copy()
        
        for d in range(2): 
            if work_data[target_var].std() > 0:
                p_val = adfuller(work_data[target_var].dropna())[1]
            else:
                p_val = 0
                
            if p_val < 0.05:
                best_d = d
                break
            
            if d == 0: 
                work_data = work_data.diff().dropna()
                best_d = 1 
        
        # 3. Escalado (Crítico para que el optimizador del MA no colapse)
        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(work_data)
        scaled_df = pd.DataFrame(scaled_train, index=work_data.index, columns=var_cols)

        # 4. Ajuste del modelo VARMA
        # Al no pasar 'exog', esto actúa como un VARMA estándar sobre datos diferenciados (VARIMA)
        varma_model = VARMAX(scaled_df, order=(p, q), trend='c')
        varma_fit = varma_model.fit(maxiter=50, disp=False)

        # 5. Predicción pura
        forecast_scaled_diff = varma_fit.forecast(steps=steps_ahead)
        
        # 6. Inversión del Escalado y Diferenciación
        forecast_diff = scaler.inverse_transform(forecast_scaled_diff)
        
        if best_d == 0:
            forecast_levels = forecast_diff
        else:
            forecast_levels = np.cumsum(forecast_diff, axis=0) + df_train_raw.iloc[-1].values

        return forecast_levels[:, 0], {
            "best_p": p,
            "best_q": q,
            "best_d": best_d,
            "vars_used": var_cols,
            "aic": varma_fit.aic,
            "seasonal": "Ninguna (Puro Endógeno)",
            "summary_snippet": str(varma_fit.summary().tables[1])[:500] + "..." 
        }
        
    except Exception as e:
        last_val = df_full[target_var].iloc[-steps_ahead-1] if len(df_full) > steps_ahead else df_full[target_var].iloc[-1]
        return np.full(steps_ahead, last_val), {"fallback": f"VARIMA Error: {str(e)}", "best_p": 0, "best_q": 0, "best_d": 0}