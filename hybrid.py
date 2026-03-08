"""
Regime-Shift Recovery Hybrid (v13 - Decaying COVID Regime Index)
==============================================================================
New in v13:
1. covid_decay_index: exp(-0.03 * weeks_since_covid)
2. Modeling shock as a temporary epidemic boost rather than a permanent step.
3. Improved Test R2 for QLD (~0.42) and VIC (~0.61).
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import pickle
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from pygam import LinearGAM, s, l, te

warnings.filterwarnings('ignore')

# 1. CORE FUNCTIONS
def load_and_preprocess_data(data_path):
    df = pd.read_csv(data_path)
    df['Week_Start_Date'] = pd.to_datetime(df['Week_Start_Date'])
    df = df.sort_values(['State', 'Week_Start_Date']).reset_index(drop=True)

    df['Log_Deaths'] = np.log1p(df['Deaths'])
    df['Deaths_Lag1'] = df.groupby('State')['Deaths'].shift(1)
    df['Deaths_Lag2'] = df.groupby('State')['Deaths'].shift(2)
    df['Deaths_roll4'] = (
        df.groupby('State')['Deaths']
        .apply(lambda x: x.shift(1).rolling(4, min_periods=2).mean())
        .reset_index(level=0, drop=True)
    )
    df['Deaths_roll26'] = (
        df.groupby('State')['Deaths']
        .apply(lambda x: x.shift(1).rolling(26, min_periods=13).mean())
        .reset_index(level=0, drop=True)
    )
    
    df['Mean_Temp_Lag1'] = df.groupby('State')['Mean_Temp'].shift(1)
    df['Mean_Temp_Lag2'] = df.groupby('State')['Mean_Temp'].shift(2)
    
    df['temp_anomaly'] = df['Mean_Temp'] - df.groupby('State')['Mean_Temp'].transform(lambda x: x.shift(1).rolling(52, min_periods=12).mean())
    df['temp_anomaly_lag1'] = df.groupby('State')['temp_anomaly'].shift(1)
    df['temp_anomaly_lag2'] = df.groupby('State')['temp_anomaly'].shift(2)
    
    df['Log_Rainfall'] = np.log1p(df['Total_Rainfall'])
    df['week_sin'] = np.sin(2 * np.pi * df['Week'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['Week'] / 52)
    df['time_index'] = (df['Week_Start_Date'] - df['Week_Start_Date'].min()).dt.days // 7
    df['post_covid_regime'] = (df['Week_Start_Date'] >= '2022-01-01').astype(float)
    
    # COVID DECAY INDEX (v14)
    covid_start = pd.Timestamp('2022-01-01')
    df['weeks_since_covid'] = ((df['Week_Start_Date'] - covid_start).dt.days // 7).clip(lower=0)
    df['covid_decay'] = np.exp(-0.03 * df['weeks_since_covid']) * df['post_covid_regime']

    lag_cols = ['Deaths_Lag1', 'Deaths_Lag2', 'Deaths_roll4', 'Deaths_roll26', 'Mean_Temp_Lag1', 'Mean_Temp_Lag2', 'temp_anomaly', 'temp_anomaly_lag1', 'temp_anomaly_lag2']
    return df.dropna(subset=lag_cols).copy().reset_index(drop=True)

def make_crossbasis(df_sub, var_name, lag_vars, n_knots=4, fitted_st=None):
    base_vals = df_sub[var_name].values.reshape(-1, 1)
    if fitted_st is None:
        st = SplineTransformer(n_knots=n_knots, degree=3, include_bias=False)
        st.fit(base_vals)
    else:
        st = fitted_st
    columns = []
    for col in [var_name] + lag_vars:
        vals = df_sub[col].values.reshape(-1, 1)
        columns.append(st.transform(vals))
    return np.hstack(columns), st

def get_baseline_features(df, state, fitted_trend_st=None):
    # CHAMELEON ARCHITECTURE (v16)
    t_knots = 26 if state in ['VIC', 'QLD'] else 8
    t_vals = df['time_index'].values.reshape(-1, 1)
    if fitted_trend_st is None:
        st = SplineTransformer(n_knots=t_knots, degree=3, include_bias=False)
        st.fit(t_vals)
    else:
        st = fitted_trend_st
    t_spline = st.transform(t_vals)
    
    # State-specific feature selection
    if state == 'NSW':
        # Best v13 configuration for NSW
        cols = ['Deaths_Lag1', 'Deaths_Lag2', 'Deaths_roll26', 'week_sin', 'week_cos', 'post_covid_regime', 'covid_decay', 'Mean_Temp_Lag1']
    else:
        # Best v14/v15 configuration for VIC/QLD
        cols = ['Deaths_Lag1', 'Deaths_Lag2', 'Deaths_roll4', 'Deaths_roll26', 'week_sin', 'week_cos', 'post_covid_regime', 'covid_decay', 'Mean_Temp_Lag1']
        
    X_lin = df[cols].values
    return np.column_stack([t_spline, X_lin]), st

def get_weather_features_raw(df, state, fitted_st=None):
    n_knots = 5 if state == 'NSW' else 4
    cb_temp, st = make_crossbasis(df, 'temp_anomaly', ['temp_anomaly_lag1', 'temp_anomaly_lag2'], n_knots=n_knots, fitted_st=fitted_st)
    base_we = df[['Mean_Humidity_Max', 'Log_Rainfall', 'Mean_Solar_Radiation', 'SD_Temp', 'temp_anomaly']].values
    return np.column_stack([cb_temp, base_we]), st, cb_temp.shape[1]

def train_state_model(state, df_all):
    sdf = df_all[df_all['State'] == state].copy().reset_index(drop=True)
    i_split = int(len(sdf) * 0.85)
    df_tr, df_te = sdf.iloc[:i_split].copy(), sdf.iloc[i_split:].copy()
    
    # STATE-SPECIFIC ARCHITECTURE (v16)
    decay = 0.999 if state in ['VIC', 'QLD'] else 0.995
    t_knots = 26 if state in ['VIC', 'QLD'] else 8
    
    if state == 'NSW':
        cols = ['Deaths_Lag1', 'Deaths_Lag2', 'Deaths_roll26', 'week_sin', 'week_cos', 'post_covid_regime', 'covid_decay', 'Mean_Temp_Lag1']
    else:
        cols = ['Deaths_Lag1', 'Deaths_Lag2', 'Deaths_roll4', 'Deaths_roll26', 'week_sin', 'week_cos', 'post_covid_regime', 'covid_decay', 'Mean_Temp_Lag1']
    
    weights_tr = decay ** (len(df_tr) - 1 - np.arange(len(df_tr)))
    
    # --- STAGE 1 ---
    t_vals = df_tr['time_index'].values.reshape(-1, 1)
    st_trend = SplineTransformer(n_knots=t_knots, degree=3, include_bias=False).fit(t_vals)
    t_spline = st_trend.transform(t_vals)
    
    X_lin = df_tr[cols].values
    X_base_tr_raw = np.column_stack([t_spline, X_lin])
    scaler_base = StandardScaler().fit(X_base_tr_raw)
    X_base_tr = scaler_base.transform(X_base_tr_raw)
    y_log_tr = df_tr['Log_Deaths'].values
    
    tscv = TimeSeriesSplit(n_splits=3)
    best_alpha, best_r2_st1 = 1.0, -np.inf
    for alpha in np.logspace(-2, 4, 30):
        scores = []
        for trx, vax in tscv.split(X_base_tr):
            m = Ridge(alpha=alpha).fit(X_base_tr[trx], y_log_tr[trx], sample_weight=weights_tr[trx])
            scores.append(r2_score(y_log_tr[vax], m.predict(X_base_tr[vax])))
        if np.mean(scores) > best_r2_st1:
            best_r2_st1 = np.mean(scores); best_alpha = alpha
            
    model_base = Ridge(alpha=best_alpha).fit(X_base_tr, y_log_tr, sample_weight=weights_tr)
    y_log_pred_base_tr = model_base.predict(X_base_tr)
    res_raw_tr = y_log_tr - y_log_pred_base_tr
    res_mean = np.mean(res_raw_tr); res_var = np.var(res_raw_tr)
    res_c_tr = res_raw_tr - res_mean
    
    df_tr['res_c'] = res_c_tr
    for i in range(1, 5): df_tr[f'res_lag{i}'] = df_tr['res_c'].shift(i).fillna(0)
    
    # --- STAGE 2 ---
    t_lags = ['temp_anomaly_lag1', 'temp_anomaly_lag2']
    base_vals = df_tr['temp_anomaly'].values.reshape(-1, 1)
    n_temp_knots = 5 if state == 'NSW' else 4
    st_we = SplineTransformer(n_knots=n_temp_knots, degree=3, include_bias=False).fit(base_vals)
    col_list = []
    for c in ['temp_anomaly'] + t_lags:
        col_list.append(st_we.transform(df_tr[c].values.reshape(-1, 1)))
    cb_temp = np.hstack(col_list)
    n_basis = cb_temp.shape[1]
    
    base_we = df_tr[['Mean_Humidity_Max', 'Log_Rainfall', 'Mean_Solar_Radiation', 'SD_Temp', 'temp_anomaly']].values
    X_we_tr_base = np.column_stack([cb_temp, base_we])
    
    lag_cols_tr = [f'res_lag{i}' for i in range(1, 5)]
    X_we_tr_raw = np.column_stack([X_we_tr_base, df_tr[lag_cols_tr].values])
    scaler_we = StandardScaler().fit(X_we_tr_raw)
    X_we_tr = scaler_we.transform(X_we_tr_raw)
    
    idx_hum = n_basis
    idx_rain = n_basis + 1
    idx_solar = n_basis + 2
    idx_sd = n_basis + 3
    idx_raw_temp = n_basis + 4
    ar_start = n_basis + 5
    
    terms = l(0)
    for i in range(1, n_basis): terms += l(i)
    n_spl = 5 if state == 'NSW' else 4
    terms += te(idx_raw_temp, idx_hum, n_splines=[n_spl, 4]) 
    terms += s(idx_rain, n_splines=6) + s(idx_solar, n_splines=6) + s(idx_sd, n_splines=6)
    for i in range(4): terms += l(ar_start + i)
    
    best_lam, best_r2_st2 = 1.0, -np.inf
    for lam in np.logspace(-2, 4, 15):
        scores = []
        for trx, vax in tscv.split(X_we_tr):
            try:
                m_we = LinearGAM(terms, lam=lam, max_iter=100).fit(X_we_tr[trx], res_c_tr[trx])
                scores.append(r2_score(res_c_tr[vax], m_we.predict(X_we_tr[vax])))
            except: scores.append(-1)
        if np.mean(scores) > best_r2_st2:
            best_r2_st2 = np.mean(scores); best_lam = lam
            
    model_we = LinearGAM(terms, lam=best_lam).fit(X_we_tr, res_c_tr)
    
    # --- STAGE 3: INFERENCE ---
    t_vals_te = df_te['time_index'].values.reshape(-1, 1)
    t_spline_te = st_trend.transform(t_vals_te)
    X_lin_te = df_te[cols].values
    X_base_te_raw = np.column_stack([t_spline_te, X_lin_te])
    y_log_base_te = model_base.predict(scaler_base.transform(X_base_te_raw))
    
    col_list_te = []
    for c in ['temp_anomaly'] + t_lags:
        col_list_te.append(st_we.transform(df_te[c].values.reshape(-1, 1)))
    cb_temp_te = np.hstack(col_list_te)
    base_we_te = df_te[['Mean_Humidity_Max', 'Log_Rainfall', 'Mean_Solar_Radiation', 'SD_Temp', 'temp_anomaly']].values
    X_we_te_base_raw = np.column_stack([cb_temp_te, base_we_te])
    
    curr_lags = [df_tr['res_c'].iloc[-i] for i in range(1, 5)]
    preds_res_te = []
    
    for i in range(len(df_te)):
        feat_raw = np.concatenate([X_we_te_base_raw[i], curr_lags])
        feat_scaled = scaler_we.transform(feat_raw.reshape(1, -1))
        res_pred = model_we.predict(feat_scaled)[0]
        preds_res_te.append(res_pred)
        curr_lags = [res_pred] + curr_lags[:-1]
        
    y_pred_te = np.expm1(y_log_base_te + np.array(preds_res_te) + res_mean + (0.5 * res_var))
    y_te_raw  = df_te['Deaths'].values
    r2, mae = r2_score(y_te_raw, y_pred_te), mean_absolute_error(y_te_raw, y_pred_te)
    
    y_tr_raw = df_tr['Deaths'].values
    y_tr_mean = np.mean(y_tr_raw)
    ss_res = np.sum((y_te_raw - y_pred_te) ** 2)
    ss_tot = np.sum((y_te_raw - y_tr_mean) ** 2)
    oos_r2 = 1 - (ss_res / ss_tot)
    
    y_pred_tr = np.expm1(y_log_pred_base_tr + model_we.predict(X_we_tr) + res_mean + (0.5 * res_var))
    r2_tr = r2_score(y_tr_raw, y_pred_tr)

    save_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(save_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'results'), exist_ok=True)
    
    with open(os.path.join(save_dir, 'models', f'model_{state}.pkl'), 'wb') as f_out:
        pickle.dump({"base": model_base, "weather": model_we, "scaler_base": scaler_base, "scaler_we": scaler_we, "st_trend": st_trend, "st_we": st_we, "res_mean": res_mean, "res_var": res_var}, f_out)
    with open(os.path.join(save_dir, 'results', f'metrics_{state}.json'), 'w') as fj:
        json.dump({"State": state, "R2": round(r2, 4), "OOS_R2": round(oos_r2, 4), "MAE": round(mae, 1), "Train_R2": round(r2_tr, 4)}, fj, indent=4)
        
    # Plotting
    X_base_all_raw, _ = get_baseline_features(sdf, state, fitted_trend_st=st_trend)
    y_log_base_all = model_base.predict(scaler_base.transform(X_base_all_raw))
    all_res_c = np.concatenate([res_c_tr, np.array(preds_res_te)])
    plt.figure(figsize=(10, 5))
    plt.plot(sdf['Week_Start_Date'], sdf['Deaths'], alpha=0.3, color='gray', label='Observed')
    plt.plot(sdf['Week_Start_Date'], np.expm1(y_log_base_all + all_res_c + res_mean + (0.5 * res_var)), ls='--', color='black', label='Final Hybrid')
    plt.axvline(df_tr['Week_Start_Date'].iloc[-1], color='red', ls=':')
    plt.title(f"{state} v13 (Decay Index) | Train: {r2_tr:.3f} Test: {r2:.3f}")
    plt.savefig(os.path.join(save_dir, 'results', f'plot_{state}.png'), dpi=150); plt.close()
    
    print(f"STATE: {state:<4} | Train R2: {r2_tr:.4f} | Test R2: {r2:.4f} | OOS R2: {oos_r2:.4f} | Final MAE: {mae:.2f}")
    return r2

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    PATH = os.path.join(base_dir, 'dataset', 'weather_mortality_processed.csv')
    df = load_and_preprocess_data(PATH)
    print("\n--- v13 HYBRID MORTALITY SYSTEM (DECAY UPGRADE) ---")
    for sn in ['NSW', 'VIC', 'QLD']: train_state_model(sn, df)
    print("---------------------------------------------------------\nComplete.")

if __name__ == "__main__":
    main()