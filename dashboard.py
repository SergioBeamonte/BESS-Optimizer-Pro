import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pulp
import datetime
import sys

# Page config
st.set_page_config(page_title="BESS Optimizer Pro", layout="wide", page_icon="⚡")

# External imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, 'src'))
sys.path.append(os.path.join(script_dir, 'models'))
from data_ingestion import fetch_mercado_trinidad

from model_01_naive_mean import forecast_naive_mean
from model_02_seasonal_naive import forecast_seasonal_naive
from model_03_sarima import forecast_sarima
from model_04_sarimax import forecast_sarimax
from model_05_varima import forecast_varima
from model_06_holt_winters import forecast_holt_winters
from model_07_random_forest import forecast_random_forest
from model_08_xgboost import forecast_xgboost
from model_09_chronos import forecast_chronos
from bess_optimizer import run_bess_optimization

# --- STYLES --- (Clean Professional Style)
st.markdown("""
<style>
/* Reset to professional defaults */
h1 { color: #ffffff; font-weight: 800; font-size: 2.5rem !important; }
h2 { color: #00e5ff; font-weight: 700; }
h3 { color: #e0e0e0; font-weight: 600; }

.stTabs [data-baseweb="tab-list"] { gap: 1rem; }
.stTabs [data-baseweb="tab"] { font-size: 1.1rem; font-weight: 600; }

/* Standard Button Style */
.stButton>button { 
    border-radius: 8px; 
    font-weight: 600; 
}

/* Minimal Metric Styling */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("⚡ BESS Optimizer Pro")
st.caption("Professional Analytical Pipeline for the **Spanish Energy Grid** (REE)")

# Initialize session state
if 'df_hist' not in st.session_state:
    st.session_state.df_hist = None
if 'metrics_list' not in st.session_state:
    st.session_state.metrics_list = []
if 'dict_preds' not in st.session_state:
    st.session_state.dict_preds = {}

def chart_layout(title="", height=500):
    """Returns a clean professional Plotly layout."""
    return dict(
        template="plotly_dark",
        title=dict(text=title, font=dict(size=16)),
        height=height,
        margin=dict(l=50, r=30, t=60, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zeroline=False),
        hovermode="x unified"
    )

def add_daily_vlines(fig, dt_index):
    """Add subtle vertical dotted lines at midnight (00:00) of each day."""
    days = pd.Series(dt_index.date).unique()
    for d in days:
        fig.add_vline(
            x=pd.Timestamp(d).strftime('%Y-%m-%d %H:%M:%S'),
            line_width=0.8,
            line_color="rgba(255,255,255,0.15)",
            line_dash="dot",
            layer="below"
        )

# ---- TABS ----
tab1, tab2, tab3 = st.tabs(["📥 1. Historical Data", "🔮 2. Forecasting Lab", "🔋 3. BESS Optimizer"])

# ==========================================
# TAB 1: HISTORICAL DATA DOWNLOAD & EXPLORER
# ==========================================
with tab1:
    st.header("Market Data Download & Exploration")
    st.markdown("Fetch hourly Price, Generation and Demand data directly from the **Spanish Grid Operator (REE) API**.")
    
    col_d1, col_d2, col_btn = st.columns([2, 2, 1])
    with col_d1:
        s_date = st.date_input("Start Date", pd.to_datetime("2026-01-01"))
    with col_d2:
        e_date = st.date_input("End Date", pd.to_datetime("2026-01-07"))
        
    with col_btn:
        st.write("")
        st.write("")
        if st.button("⬇️ Download REE Data"):
            try:
                progress_bar = st.progress(0, text="Connecting to REE API...")
                
                def update_prog(p, txt):
                    progress_bar.progress(p, text=txt)
                    
                df = fetch_mercado_trinidad(s_date.strftime("%Y-%m-%d"), e_date.strftime("%Y-%m-%d"), progress_callback=update_prog)
                
                progress_bar.empty()
                
                # Compute Total Generation by summing all technology columns
                if "generación_total" in df.columns:
                    df = df.rename(columns={"generación_total": "generacion_total"})
                    
                st.session_state.df_hist = df
                
                # Save CSV to data/raw
                os.makedirs(os.path.join(script_dir, 'data', 'raw'), exist_ok=True)
                csv_path = os.path.join(script_dir, 'data', 'raw', 'dataset_multivariante_completo.csv')
                df.to_csv(csv_path)
                
                st.success(f"Data imported successfully! {len(df)} records saved to data/raw/")
            except Exception as e:
                st.error(f"Error downloading data: {str(e)}")
                    
    # Visualization if data is loaded
    if st.session_state.df_hist is not None:
        df_plot = st.session_state.df_hist
        st.markdown("---")
        
        c1, c2 = st.columns([1, 4])
        with c1:
            st.write("🔧 Visualization Controls")
            var_base = st.selectbox("Select main metric:", ['Price (EUR/MWh)', 'Demand', 'Total Generation'])
            
            st.markdown("---")
            st.write("📅 Date Range")
            min_d, max_d = df_plot.index.min().date(), df_plot.index.max().date()
            if min_d < max_d:
                f_rango = st.slider("Select time range", min_value=min_d, max_value=max_d, value=(min_d, max_d))
                mask_dates = (df_plot.index.date >= f_rango[0]) & (df_plot.index.date <= f_rango[1])
                df_filtered = df_plot[mask_dates]
            else:
                df_filtered = df_plot
                
        with c2:
            fig = go.Figure()
            
            # Show markers for short ranges
            dias_totales = (df_filtered.index.max() - df_filtered.index.min()).days
            mode_plot = 'lines+markers' if dias_totales <= 2 else 'lines'
            
            if var_base == 'Price (EUR/MWh)':
                fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['precio_mwh'], mode=mode_plot, name='Price', line=dict(color='#00e5ff', width=2)))
            elif var_base == 'Demand':
                fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['demanda'], mode=mode_plot, name='Demand', line=dict(color='#ff9100', width=2)))
            elif var_base == 'Total Generation':
                fig.add_trace(go.Scatter(x=df_filtered.index, y=df_filtered['generacion_total'], mode=mode_plot, name='Total Generation (MWh)', line=dict(color='#f50057', width=2)))
            
            fig.update_layout(**chart_layout(f"Historical Evolution: {var_base} — Spanish Market", 500))
            add_daily_vlines(fig, df_filtered.index)
            st.plotly_chart(fig, use_container_width=True)
            
            # Pie chart: Generation mix breakdown
            exclude_cols = {'precio_mwh', 'demanda', 'generacion_total'}
            gen_cols = [c for c in df_filtered.columns if c not in exclude_cols and 'demanda' not in c]
            if var_base == 'Total Generation' and len(gen_cols) > 0:
                st.markdown("---")
                st.subheader("🍰 Daily Energy Mix Breakdown (%)")
                dias_unicos = pd.Series(df_filtered.index.date).unique()
                if len(dias_unicos) > 0:
                    dia_elegido = st.select_slider("Select day for pie chart:", options=list(dias_unicos))
                    
                    df_dia = df_filtered[df_filtered.index.date == dia_elegido]
                    df_dia = df_dia[gen_cols]
                    totales_dia = df_dia[gen_cols].sum()
                    
                    # Filter technologies contributing less than 1%
                    tec_limpias = totales_dia[totales_dia > (totales_dia.sum() * 0.01)]
                    
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=[l.replace("_", " ").title() for l in tec_limpias.index], 
                        values=tec_limpias.values,
                        hole=.3,
                        textinfo='label+percent',
                        marker=dict(colors=go.Figure().layout.colorway)
                    )])
                    
                    fig_pie.update_layout(**chart_layout(f"Generation Mix — {dia_elegido} (Total: {tec_limpias.sum():.0f} MWh)", 450))
                    st.plotly_chart(fig_pie, use_container_width=True)

# ==========================================
# TAB 2: FORECASTING LAB
# ==========================================
with tab2:
    st.header("Electricity Market Forecasting Lab")
    if st.session_state.df_hist is None:
        st.info("⚠️ Please download historical data in Tab 1 before running predictions.")
    else:
        df_train = st.session_state.df_hist
        st.write(f"Using loaded dataset: {df_train.index.min().strftime('%Y-%m-%d')} to {df_train.index.max().strftime('%Y-%m-%d')}.")
        
        horizon_options = {"1 Hour": 1, "4 Hours": 4, "1 Day": 24, "2 Days": 48, "1 Week": 168}
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### Backtesting Configuration")
            todas_opciones_ml = [
                "01. Naive (Daily Mean)",
                "02. Seasonal Naive (24h)",
                "03. SARIMA (Autofit)", 
                "04. SARIMAX (Multivariate)", 
                "05. VARIMA (Multivariate)", 
                "06. Exponential Smoothing (Holt-Winters)",
                "07. Random Forest", 
                "08. XGBoost", 
                "09. Amazon Chronos T5"
            ]
            selected_model = st.selectbox("ML Algorithm:", todas_opciones_ml, help="Select the forecasting engine that will try to predict the hold-out set.")
            target_var_sel = st.selectbox("🎯 Target variable:", ['precio_mwh', 'demanda', 'generacion_total'], format_func=lambda x: {'precio_mwh': 'Price (EUR/MWh)', 'demanda': 'Demand (MW)', 'generacion_total': 'Total Generation (MWh)'}[x])
            horizonte = st.selectbox("Validation horizon (hold-out size):", list(horizon_options.keys()), index=2)
            run_pred = st.button("🚀 Run Prediction")
            
            if st.button("🗑️ Clear Saved Models"):
                st.session_state.metrics_list = []
                st.session_state.dict_preds = {}
                st.rerun()
            
        if run_pred:
            with st.spinner(f"Evaluating model (Train/Test split) for {horizonte} validation..."):
                steps_ahead = horizon_options[horizonte]
                if len(df_train) <= steps_ahead * 2:
                    st.error("The historical dataset is too small for this horizon. Download more data.")
                else:
                    target_vars = [target_var_sel]
                    
                    if "VARIMA" in selected_model:
                        y_pred, diag = forecast_varima(df_train, target_var_sel, steps_ahead)
                        if "fallback" in diag:
                            st.warning(f"VARIMA Fallback: {diag['fallback']}")
                        else:
                            st.info(f"Auto-VARIMA selected: **p={diag.get('best_p',0)}, d={diag.get('best_d',0)}**")
                        
                        with st.expander("📊 Model Justification & Metrics"):
                            if "vars_used" in diag:
                                st.write(f"Series used: {', '.join(diag['vars_used'])}")
                            if "aic" in diag:
                                st.write(f"**AIC:** {diag['aic']:.2f}")
                            if "summary_snippet" in diag:
                                st.markdown("---")
                                st.write("**VAR Model Summary Snippet:**")
                                st.code(diag["summary_snippet"])
                        
                        key = f"{selected_model}_{target_var_sel}"
                        st.session_state.dict_preds[key] = pd.Series(y_pred, index=df_train.index[-steps_ahead:])
                        
                        y_true = df_train[target_var_sel].iloc[-steps_ahead:]
                        mae = np.mean(np.abs(y_true - y_pred))
                        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
                        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
                        st.session_state.metrics_list.append({"Model": selected_model, "Variable": target_var_sel, "Horizon": horizon_options[horizonte], "MAE": mae, "RMSE": rmse, "MAPE": mape})
                    elif "SARIMAX" in selected_model:
                        y_pred, diag = forecast_sarimax(df_train, target_var_sel, steps_ahead)
                        with st.expander("🔍 SARIMAX Details"):
                            st.write(f"Exogenous variables used: {', '.join(diag.get('exog_used', []))}")
                            if "order" in diag:
                                st.write(f"**Detected Order:** {diag['order']} x {diag.get('seasonal_order')}")
                                st.write(f"**AIC:** {diag.get('aic',0):.2f}")
                            if "fallback" in diag:
                                st.warning(diag["fallback"])
                        
                        key = f"{selected_model}_{target_var_sel}"
                        st.session_state.dict_preds[key] = pd.Series(y_pred, index=df_train.index[-steps_ahead:])
                        
                        y_true = df_train[target_var_sel].iloc[-steps_ahead:]
                        mae = np.mean(np.abs(y_true - y_pred))
                        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
                        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
                        st.session_state.metrics_list.append({"Model": selected_model, "Variable": target_var_sel, "Horizon": horizon_options[horizonte], "MAE": mae, "RMSE": rmse, "MAPE": mape})
                    else:
                        for var in target_vars:
                            if var in df_train.columns:
                                y_all = df_train[var]
                                y_tr_sub = y_all.iloc[:-steps_ahead]
                                y_true = y_all.iloc[-steps_ahead:]
                                
                                if "Chronos" in selected_model:
                                    try:
                                        import torch
                                        from chronos import ChronosPipeline
                                    except ImportError:
                                        st.error("Chronos not installed.")
                                        st.stop()
                                    pipeline_c = ChronosPipeline.from_pretrained("amazon/chronos-t5-tiny", device_map="cpu", torch_dtype=torch.float32)
                                    y_pred = forecast_chronos(pipeline_c, y_tr_sub, steps_ahead)
                                elif "Seasonal Naive" in selected_model:
                                    y_pred = forecast_seasonal_naive(y_tr_sub, steps_ahead)
                                elif "Naive (Daily Mean)" in selected_model:
                                    y_pred = forecast_naive_mean(y_tr_sub, steps_ahead)
                                elif "SARIMA" in selected_model:
                                    y_pred, diag = forecast_sarima(y_tr_sub, steps_ahead)
                                    with st.expander("📈 SARIMA Autofit Diagnostics"):
                                        st.write(f"**Stationarity Tests (Target):**")
                                        st.write(f"- ADF: {'Pass' if diag.get('adf_stationary') else 'Fail'} (p={diag.get('adf_p',0):.4f})")
                                        st.write(f"- KPSS: {'Pass' if diag.get('kpss_stationary') else 'Fail'} (p={diag.get('kpss_p',0):.4f})")
                                        st.write(f"**Detected Order:** {diag.get('order')} x {diag.get('seasonal_order')}")
                                        st.write(f"**AIC:** {diag.get('aic', 0):.2f}")
                                elif "Holt-Winters" in selected_model:
                                    y_pred = forecast_holt_winters(y_tr_sub, steps_ahead)
                                elif "XGBoost" in selected_model:
                                    y_pred = forecast_xgboost(df_train, var, steps_ahead)
                                elif "Random Forest" in selected_model:
                                    y_pred = forecast_random_forest(df_train, var, steps_ahead)
                                else:
                                    y_pred = np.full(steps_ahead, y_tr_sub.iloc[-1])
                                    
                                key = f"{selected_model}_{var}"
                                st.session_state.dict_preds[key] = pd.Series(y_pred, index=df_train.index[-steps_ahead:])
                                
                                mae = np.mean(np.abs(y_true - y_pred))
                                rmse = np.sqrt(np.mean((y_true - y_pred)**2))
                                mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
                                st.session_state.metrics_list.append({"Model": selected_model, "Variable": var, "Horizon": horizon_options[horizonte], "MAE": mae, "RMSE": rmse, "MAPE": mape})
                    
                    st.session_state.steps_ahead = steps_ahead
                    st.success("Model evaluated and metrics saved to the comparison table!")
                    
        # Comparison table and chart
        st.markdown("---")
        st.subheader("📊 Model Performance Comparison Table")
        
        # Build metrics DataFrame
        df_metrics_run = pd.DataFrame(st.session_state.metrics_list) if len(st.session_state.metrics_list) > 0 else pd.DataFrame(columns=["Model", "Variable", "Horizon", "MAE", "RMSE", "MAPE"])
        
        # Build base table for selected variable
        base_rows = []
        h_num = horizon_options[horizonte]
        for m in todas_opciones_ml:
            base_rows.append({"Model": m, "Variable": target_var_sel, "Horizon": h_num, "Status": "⏳ Pending"})
         
        df_base = pd.DataFrame(base_rows)
        
        if not df_metrics_run.empty:
            df_metrics_filtered = df_metrics_run[df_metrics_run["Variable"] == target_var_sel].copy()
            df_metrics_filtered = df_metrics_filtered.drop_duplicates(subset=["Model", "Variable", "Horizon"], keep="last")
            
            df_final_table = pd.merge(df_base, df_metrics_filtered, on=["Model", "Variable", "Horizon"], how="left", suffixes=('_base', ''))
            df_final_table["Status"] = df_final_table.apply(lambda x: "✅ Done" if pd.notnull(x["MAE"]) else "⏳ Pending", axis=1)
            df_final_table = df_final_table[["Model", "Variable", "Status", "MAE", "RMSE", "MAPE"]]
        else:
            df_final_table = df_base.copy()
            df_final_table["MAE"] = np.nan
            df_final_table["RMSE"] = np.nan
            df_final_table["MAPE"] = np.nan
            df_final_table = df_final_table[["Model", "Variable", "Status", "MAE", "RMSE", "MAPE"]]
            
        st.dataframe(df_final_table.style.highlight_min(subset=["MAE", "RMSE", "MAPE"], color='rgba(46, 204, 113, 0.3)', axis=0).format(precision=3), use_container_width=True)
        
        if not df_metrics_run.empty:
            p_var = target_var_sel
            
            p_horizonte = int(df_metrics_run['Horizon'].max())
            if p_var in df_train.columns:
                
                # Multi-model overlay chart
                fig2 = go.Figure()
                y_true = df_train[p_var].iloc[-p_horizonte:]
                df_train_sub = df_train.iloc[:-p_horizonte]
                
                # Training data
                fig2.add_trace(go.Scatter(x=df_train_sub.index, y=df_train_sub[p_var], 
                                         mode='lines', name='Training (Past)', line=dict(color='gray', width=1)))
                
                # Ground truth
                fig2.add_trace(go.Scatter(x=y_true.index, y=y_true, 
                                         mode='lines', name='Validation (Ground Truth)', line=dict(color='#00e5ff', width=3)))
                
                # Overlay each saved prediction
                for model_id in df_metrics_run['Model'].unique():
                    key = f"{model_id}_{p_var}"
                    if key in st.session_state.dict_preds:
                        y_pred_val = st.session_state.dict_preds[key]
                        fig2.add_trace(go.Scatter(x=y_pred_val.index, y=y_pred_val, 
                                                 mode='lines+markers', name=f'Prediction: {model_id}', line=dict(width=2, dash='dot')))
                                         
                # Train/test split marker
                fig2.add_vline(x=y_true.index[0], line_width=1, line_dash="dash", line_color="rgba(255,255,255,0.4)")
                
                # Day separators
                add_daily_vlines(fig2, df_train.index)
                
                var_label = {'precio_mwh': 'Price (EUR/MWh)', 'demanda': 'Demand (MW)', 'generacion_total': 'Total Generation (MWh)'}.get(p_var, p_var)
                fig2.update_layout(**chart_layout(f"Multi-Model Backtesting: {var_label}", 500))
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("The comparison table is empty. Select a model and horizon above to populate it.")

# ==========================================
# TAB 3: BESS OPTIMIZER
# ==========================================
with tab3:
    st.header("🔋 BESS Intelligent Arbitrage Optimizer")
    
    preds_to_use = None
    
    if st.session_state.df_hist is None:
        st.warning("⚠️ Please download the dataset in Tab 1 to run the BESS optimization on market data.")
    else:
        df_h = st.session_state.df_hist
        st.write("Configure the simulation period to maximize battery arbitrage profit on the **Spanish electricity market**.")
        
        c_d, c_t, c_h = st.columns(3)
        with c_d:
            sim_start_date = st.date_input("Start day:", value=df_h.index.min().date(), min_value=df_h.index.min().date(), max_value=df_h.index.max().date())
        with c_t:
            sim_start_time = st.time_input("Start hour:", value=datetime.time(0, 0))
        with c_h:
            sim_horas = st.number_input("Hours to simulate (Charge/Discharge horizon):", min_value=1, max_value=672, value=48, step=12)
        
        # Slice historical data
        start_ts = pd.Timestamp(datetime.datetime.combine(sim_start_date, sim_start_time))
        df_slice = df_h.loc[start_ts : start_ts + pd.Timedelta(hours=sim_horas-1)].copy()
        
        if len(df_slice) == 0:
            st.error("No data available for this time range. Try a range within the loaded dataset.")
            preds_to_use = None
        else:
            st.success(f"Simulation window ready: {len(df_slice)} hours extracted.")
            
            # Price source selector
            opciones_precio = ["Real Prices (Historical)"]
            modelos_disponibles = pd.DataFrame(st.session_state.metrics_list)['Model'].unique() if len(st.session_state.metrics_list) > 0 else []
            for m in modelos_disponibles:
                key = f"{m}_precio_mwh"
                if key in st.session_state.dict_preds:
                    pred_series = st.session_state.dict_preds[key]
                    if pred_series.index.intersection(df_slice.index).size > 0:
                        opciones_precio.append(f"Prediction: {m}")
            
            fuente_precio = st.selectbox("🤖 Price Source for Optimizer:", opciones_precio)
            
            preds_to_use = df_slice.copy()
            if fuente_precio != "Real Prices (Historical)":
                m_name = fuente_precio.replace("Prediction: ", "")
                pred_vals = st.session_state.dict_preds[f"{m_name}_precio_mwh"]
                common_idx = df_slice.index.intersection(pred_vals.index)
                if len(common_idx) < len(df_slice):
                    st.warning(f"⚠️ Prediction from {m_name} only covers {len(common_idx)} of {len(df_slice)} selected hours. Using the common range.")
                    preds_to_use = df_slice.loc[common_idx].copy()
                
                preds_to_use['precio_mwh'] = pred_vals.loc[preds_to_use.index].values

    if preds_to_use is not None:
        st.markdown("Uses Linear Programming (`PuLP CBC`) on *market prices* to compute the optimal buy/sell schedule.")
        
        c1, c2, c3, c4 = st.columns(4)
        cap = c1.number_input("BESS Capacity (kWh)", value=2000)
        power = c2.number_input("Max Power (kW)", value=1000)
        eff = c3.slider("Battery Efficiency (Round-trip) %", min_value=50, max_value=100, value=90) / 100.0
        soc_ini = c4.slider("Initial SoC %", min_value=0, max_value=100, value=50) / 100.0
        
        if st.button("⚖️ Compute Optimal BESS Dispatch"):
            with st.spinner("Solving LP Arbitrage Model..."):
                # Call Modular BESS Optimizer
                prices_kwh = preds_to_use['precio_mwh'].values / 1000.0
                charge_vals, discharge_vals, soc_vals, profit_val = run_bess_optimization(
                    prices_kwh, cap, power, eff, soc_ini
                )
                
                # Financial metrics
                dt = 1.0
                total_charged_mwh = sum(charge_vals) * dt / 1000.0
                total_discharged_mwh = sum(discharge_vals) * dt / 1000.0
                
                st.write("### Financial & Operational Results")
                m1, m2, m3 = st.columns(3)
                m1.metric("💰 Arbitrage Profit", f"{profit_val:.2f} €")
                m2.metric("🔋 Energy Charged", f"{total_charged_mwh:.2f} MWh")
                m3.metric("⚡ Energy Discharged", f"{total_discharged_mwh:.2f} MWh")
                st.markdown("---")
                # Compute hourly and cumulative cash flow
                n_steps = len(preds_to_use)
                hourly_cashflow = np.array([(prices_kwh[t] * discharge_vals[t] - prices_kwh[t] * charge_vals[t]) * dt for t in range(n_steps)])
                cumulative_cashflow = np.cumsum(hourly_cashflow)
                
                # Triple-panel chart
                fig3 = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.07, 
                                     subplot_titles=("Market Price (Spanish Pool)", "BESS Operations & State of Charge (SoC)", "Cumulative Cash Flow (€)"),
                                     specs=[[{"secondary_y": False}], [{"secondary_y": True}], [{"secondary_y": False}]],
                                     row_heights=[0.3, 0.4, 0.3])
                                     
                # Row 1: Price
                fig3.add_trace(go.Scatter(x=preds_to_use.index, y=preds_to_use['precio_mwh'], mode='lines', name='Price (EUR/MWh)', line=dict(color='#00e5ff', width=2)), row=1, col=1)
                
                # Row 2: Charge/Discharge bars + SoC line
                fig3.add_trace(go.Bar(x=preds_to_use.index, y=discharge_vals, name='Discharge (Sell kW)', marker_color='#00e676', opacity=0.7), row=2, col=1, secondary_y=False)
                fig3.add_trace(go.Bar(x=preds_to_use.index, y=[-c for c in charge_vals], name='Charge (Buy kW)', marker_color='#ff1744', opacity=0.7), row=2, col=1, secondary_y=False)
                fig3.add_trace(go.Scatter(x=preds_to_use.index, y=(np.array(soc_vals)/cap)*100, mode='lines', name='SoC (%)', line=dict(color='#ffea00', dash='dot', width=2)), row=2, col=1, secondary_y=True)
                
                # Row 3: Cumulative Cash Flow
                fig3.add_trace(go.Scatter(
                    x=preds_to_use.index, y=cumulative_cashflow, mode='lines',
                    name='Cumulative P&L (€)', 
                    line=dict(color='#00e5ff', width=2.5),
                    fill='tozeroy', fillcolor='rgba(0, 229, 255, 0.1)'
                ), row=3, col=1)
                
                fig3.update_layout(
                    **chart_layout("", 820),
                    legend=dict(orientation="h", x=0, y=1.06, bgcolor="rgba(0,0,0,0)", font=dict(size=11))
                )
                
                fig3.update_yaxes(title_text="Price (EUR/MWh)", row=1, col=1)
                fig3.update_yaxes(title_text="Power (kW)", secondary_y=False, row=2, col=1)
                fig3.update_yaxes(title_text="SoC (%)", secondary_y=True, range=[0, 105], row=2, col=1)
                fig3.update_yaxes(title_text="Profit (€)", row=3, col=1)
                
                add_daily_vlines(fig3, preds_to_use.index)
                
                st.plotly_chart(fig3, use_container_width=True)
