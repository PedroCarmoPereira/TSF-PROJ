import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from statsmodels.stats.diagnostic import acorr_ljungbox

from statsmodels.graphics.tsaplots import plot_acf
from plots import * 
from pmdarima import auto_arima
import pandas as pd

def sarimax_experiment(
    df,
    target,
    p, d, q,
    P, D, Q, s,
    forecast_window=12,
    no_windows=10,
    exog=None,
):
    """
    Fit one SARIMAX model, evaluate on last forecast_window points,
    and RETURN metrics & fitted model (no plotting).
    """

    # Train/test split
    train = df[target][no_windows * -forecast_window : -forecast_window]
    test = df[target][-forecast_window:]

    exog_train = exog_test = None
    if exog is not None:
        exog_train = df[exog][no_windows * -forecast_window : -forecast_window]
        exog_test = df[exog][-forecast_window:]

    # Fit model
    model_train = SARIMAX(
        train,
        exog=exog_train,
        order=(p, d, q),
        seasonal_order=(P, D, Q, s),
    )
    results_train = model_train.fit(disp=False)

    # Forecast
    forecast = results_train.forecast(
        steps=forecast_window,
        exog=exog_test
    )

    # Metrics
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)

    return rmse, mae, results_train, forecast, test, train

def sarimax_grid_search(
    df,
    target,
    candidates,
    forecast_window=12,
    no_windows=10,
    exog=None,
    metric="rmse",   # "rmse" or "mae"
    verbose=True,
):
    """
    Grid search over SARIMAX orders and keep the best performing model.
    """
    assert metric in {"rmse", "mae"}

    best_score = np.inf
    best_info = None
    records = []

    for (p, d, q, P, D, Q, s) in candidates:
        try:
            rmse, mae, results, forecast, test, train = sarimax_experiment(
                df=df,
                target=target,
                p=p, d=d, q=q,
                P=P, D=D, Q=Q, s=s,
                forecast_window=forecast_window,
                no_windows=no_windows,
                exog=exog,   
            )

            record = {
                "p": p, "d": d, "q": q,
                "P": P, "D": D, "Q": Q, "s": s,
                "rmse": rmse,
                "mae": mae,
            }
            records.append(record)

            score = rmse if metric == "rmse" else mae

            if verbose:
                print(
                    f"Tried SARIMA({p},{d},{q})({P},{D},{Q},{s}) "
                    f"-> RMSE={rmse:.4f}, MAE={mae:.4f}"
                )

            if score < best_score:
                best_score = score
                best_info = {
                    "order": (p, d, q),
                    "seasonal_order": (P, D, Q, s),
                    "rmse": rmse,
                    "mae": mae,
                    "results": results,
                    "forecast": forecast,
                    "test": test,
                    "train": train,
                }

        except Exception as e:
            if verbose:
                print(
                    f"FAILED SARIMA({p},{d},{q})({P},{D},{Q},{s}) → {e}"
                )
            continue

    scores_df = pd.DataFrame(records).sort_values(by=metric, ascending=True)

    if verbose and best_info is not None:
        print("\nBest model found:")
        print(
            f"(p,d,q) = {best_info['order']}, "
            f"(P,D,Q,s) = {best_info['seasonal_order']}, "
            f"RMSE = {best_info['rmse']:.4f}, MAE = {best_info['mae']:.4f}"
        )

    return best_info, scores_df

def auto_sarima_experiment(
    df,
    target,
    m,
    forecast_window=12,
    no_windows=10,
    exog=None,
    max_p=5, max_q=5,
    max_P=5, max_Q=5,
    metric="rmse",
    trace=True,

):
    """
    Run AutoSARIMA and evaluate it on a train/test split exactly like sarimax_experiment.

    Returns
    -------
    result : dict with keys
        {
          "order": (p,d,q),
          "seasonal_order": (P,D,Q,m),
          "rmse": float,
          "mae": float,
          "model": fitted auto_arima model,
          "forecast": forecast series,
          "test": test series
        }
    """

    # --- Train / Test split ---
    train = df[target][no_windows * -forecast_window : -forecast_window]
    test = df[target][-forecast_window:]

    exog_train = exog_test = None
    if exog is not None:
        exog_train = df[exog][no_windows * -forecast_window : -forecast_window]
        exog_test  = df[exog][-forecast_window:]

    # --- Fit AutoSARIMA ---
    model = auto_arima(
        train,
        exogenous=exog_train,
        seasonal=True,
        m=m,
        d=None,
        D=None,
        start_p=0, max_p=max_p,
        start_q=0, max_q=max_q,
        start_P=0, max_P=max_P,
        start_Q=0, max_Q=max_Q,
        stepwise=False,
        trace=trace,
        
    )

    # Extract orders
    order = model.order
    seasonal_order = model.seasonal_order

    # --- Forecast ---
    forecast = model.predict(n_periods=forecast_window, exogenous=exog_test)
    forecast = pd.Series(forecast, index=test.index)

    # --- Metrics ---
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)

    # --- Print results ---
    print("\n✅ AutoSARIMA result")
    print(f"Order:        {order}")
    print(f"Seasonal:     {seasonal_order}")
    print(f"RMSE:         {rmse:.4f}")
    print(f"MAE:          {mae:.4f}")

    return rmse, mae, model, forecast, test, train, order , seasonal_order



def sarimax_time_series_cv(
    df, target, p, d, q, P, D, Q, s,
    train_years=9, forecast_months=12,
    test_start_year=2010, exog=None,
    *, verbose=True, alpha=0.05,
):
    """
    Perform time series cross-validation for SARIMAX models.
    Returns a dict with fold metrics + concatenated predictions
    + per-fold artifacts (y_true/y_pred/resid/conf_int) for plotting.
    """

    df = df.copy()
    df = df.sort_index()

    # --- Build exog matrix aligned to df.index (supports multiple input types) ---
    exog_mat = None
    if exog is not None:
        if isinstance(exog, (pd.Series, pd.DataFrame)):
            exog_mat = exog.copy().sort_index().reindex(df.index)
        else:
            exog_mat = df[exog].copy()

    train_months = train_years * 12

    test_start_pos = df.index.get_indexer_for(df.index[df.index.year >= test_start_year])
    if len(test_start_pos) == 0:
        if verbose:
            print("No test data at/after test_start_year.")
        return {
            "split_metrics": [],
            "overall_rmse": np.inf, "overall_mae": np.inf,
            "avg_rmse": np.inf, "avg_mae": np.inf,
            "forecasts": [], "actuals": [],
            "train_months": train_months, "forecast_months": forecast_months,
            "residuals": [], "lb_results": [],
            "folds": [],  # NEW
        }

    test_start_idx = test_start_pos[0]

    # --- Determine splits safely: only create folds where full horizon fits ---
    n = len(df)
    n_splits = 0
    while True:
        train_start_idx = test_start_idx + n_splits * forecast_months
        train_end_idx = train_start_idx + train_months
        test_end_idx = train_end_idx + forecast_months
        if test_end_idx <= n:
            n_splits += 1
        else:
            break

    if verbose:
        print(f"Data range: {df.index.min()} to {df.index.max()}")
        print(f"Training window: {train_years} years ({train_months} months)")
        print(f"Forecast horizon: {forecast_months} months")
        print(f"Testing period: {test_start_year} onwards")
        print(f"Number of CV splits: {n_splits}\n")

    all_forecasts = []
    all_actuals = []
    all_residuals = []
    lb_results = []
    split_metrics = []
    folds = []  # NEW

    for split_idx in range(n_splits):
        train_start_idx = test_start_idx + split_idx * forecast_months
        train_end_idx   = train_start_idx + train_months
        test_end_idx    = train_end_idx + forecast_months

        y_train = df[target].iloc[train_start_idx:train_end_idx]
        y_test  = df[target].iloc[train_end_idx:test_end_idx]

        X_train = X_test = None
        if exog_mat is not None:
            X_train = exog_mat.iloc[train_start_idx:train_end_idx]
            X_test  = exog_mat.iloc[train_end_idx:test_end_idx]

            if len(X_train) != len(y_train) or len(X_test) != len(y_test):
                if verbose:
                    print(
                        f"Split {split_idx+1} failed: exog length mismatch "
                        f"(train {len(X_train)} vs {len(y_train)}, test {len(X_test)} vs {len(y_test)})"
                    )
                continue

            if np.any(pd.isna(X_train).to_numpy()) or np.any(pd.isna(X_test).to_numpy()):
                if verbose:
                    print(f"Split {split_idx+1} failed: NaNs in exog within fold window.")
                continue

        try:
            model = SARIMAX(
                y_train,
                exog=X_train,
                order=(p, d, q),
                seasonal_order=(P, D, Q, s),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            results = model.fit(disp=False)

            fcst = results.get_forecast(steps=len(y_test), exog=X_test)
            y_hat = fcst.predicted_mean
            y_hat.index = y_test.index  # ensure alignment

            # optional confidence intervals for plotting
            conf_int = fcst.conf_int(alpha=alpha)
            conf_int.index = y_test.index  # align to test index

            rmse = float(np.sqrt(mean_squared_error(y_test, y_hat)))
            mae  = float(mean_absolute_error(y_test, y_hat))

            # residuals as Series (keeps dates!)
            resid_s = (y_test - y_hat)

            # Ljung-Box on residual values
            residuals_np = resid_s.to_numpy()
            max_lag = len(residuals_np) - 1
            if max_lag >= 1:
                lag = min(12, max_lag)
                lb = acorr_ljungbox(residuals_np, lags=[lag], return_df=True)
                lb_results.append({
                    "ds": y_test.index[-1],
                    "lb_stat": float(lb["lb_stat"].iloc[0]),
                    "p_value": float(lb["lb_pvalue"].iloc[0]),
                    "lag": int(lag),
                })
            else:
                lb_results.append({
                    "ds": y_test.index[-1],
                    "lb_stat": np.nan,
                    "p_value": np.nan,
                    "lag": 0,
                })

            # concatenated storage
            all_forecasts.extend(y_hat.to_numpy())
            all_actuals.extend(y_test.to_numpy())
            all_residuals.extend(resid_s.to_numpy().tolist())

            split_metrics.append({
                "split": split_idx + 1,
                "train_end": y_train.index[-1],
                "test_start": y_test.index[0],
                "test_end": y_test.index[-1],
                "rmse": rmse,
                "mae": mae,
                "n_test": len(y_test),
            })

            # NEW: per-fold artifacts for plotting later
            folds.append({
                "split": split_idx + 1,
                "train_start": y_train.index[0],
                "train_end": y_train.index[-1],
                "test_start": y_test.index[0],
                "test_end": y_test.index[-1],
                "y_true": y_test.copy(),
                "y_pred": y_hat.copy(),
                "resid": resid_s.copy(),
                "conf_int": conf_int.copy(),
                # optionally:
                # "model_results": results,
            })

            if verbose:
                print(
                    f"Split {split_idx + 1}/{n_splits} - "
                    f"Train: {y_train.index[0].strftime('%Y-%m')} to {y_train.index[-1].strftime('%Y-%m')}, "
                    f"Test: {y_test.index[0].strftime('%Y-%m')} to {y_test.index[-1].strftime('%Y-%m')} - "
                    f"RMSE: {rmse:.4f}, MAE: {mae:.4f}"
                )

        except Exception as e:
            if verbose:
                print(f"Split {split_idx + 1} failed: {e}")
            continue

    # Metrics
    if len(all_actuals) == 0:
        overall_rmse = overall_mae = avg_rmse = avg_mae = np.inf
        std_rmse = std_mae = np.inf
    else:
        overall_rmse = float(np.sqrt(mean_squared_error(all_actuals, all_forecasts)))
        overall_mae  = float(mean_absolute_error(all_actuals, all_forecasts))
        rmse_vals = np.array([m["rmse"] for m in split_metrics])
        mae_vals  = np.array([m["mae"]  for m in split_metrics])
        avg_rmse = float(rmse_vals.mean())
        avg_mae  = float(mae_vals.mean())
        std_rmse = float(rmse_vals.std(ddof=0))
        std_mae  = float(mae_vals.std(ddof=0))

    if verbose:
        print(f"\n{'='*70}")
        print("OVERALL METRICS (concatenated predictions):")
        print(f"RMSE: {overall_rmse:.4f}")
        print(f"MAE: {overall_mae:.4f}")
        print("\nAVERAGE METRICS (across splits):")
        print(f"Average RMSE: {avg_rmse:.4f}")
        print(f"Average MAE: {avg_mae:.4f}")
        print(f"Standard Deviation RMSE: {std_rmse:.4f}")
        print(f"Standard Deviation MAE: {std_mae:.4f}")
        print(f"{'='*70}\n")

    return {
        "split_metrics": split_metrics,
        "overall_rmse": overall_rmse,
        "overall_mae": overall_mae,
        "avg_rmse": avg_rmse,
        "avg_mae": avg_mae,
        "std_rmse": std_rmse,
        "std_mae": std_mae,
        "forecasts": all_forecasts,
        "actuals": all_actuals,
        "train_months": train_months,
        "forecast_months": forecast_months,
        "residuals": all_residuals,
        "lb_results": lb_results,
        "folds": folds,  # NEW
    }


import numpy as np
import pandas as pd

def select_best_sarima_cv(
    df,
    target,
    candidates,
    train_years=8,
    forecast_months=24,
    test_start_year=1995,
    exog=None,
    metric="rmse",
    use="avg",
    verbose=True,
    alpha=0.05,
    folds_key_candidates=("folds", "fold_results", "cv_folds"),  # supports different names
):
    """
    Select best SARIMA/SARIMAX by time-series CV, and return:
      - best_info: includes best params + metrics + last_fold artifacts (y_true/y_pred/resid/ci/...)
      - scores_df: table of candidate scores

    REQUIREMENT (to get last_fold):
      sarimax_time_series_cv must return per-fold artifacts under one of folds_key_candidates,
      e.g. cv["folds"] = [ { "y_true":..., "y_pred":..., "resid":..., ... }, ... ]
    """
    assert metric in {"rmse", "mae"}
    assert use in {"avg", "overall"}
    metric_key = f"{metric}_mean"

    records = []
    best_params = None
    best_score = np.inf

    # PASS 1: score all candidates
    for i, (p, d, q, P, D, Q, s) in enumerate(candidates, 1):
        try:
            cv = sarimax_time_series_cv(
                df=df, target=target,
                p=p, d=d, q=q, P=P, D=D, Q=Q, s=s,
                train_years=train_years,
                forecast_months=forecast_months,
                test_start_year=test_start_year,
                exog=exog,
                alpha=alpha,
                verbose=True,
            )
        except Exception as e:
            if verbose:
                print(f"{i:02d}. SARIMA({p},{d},{q})({P},{D},{Q},{s}) failed: {e}")
            continue

        avg_rmse     = cv.get("avg_rmse", np.inf)
        avg_mae      = cv.get("avg_mae", np.inf)
        std_rmse     = cv.get("std_rmse", np.inf)
        std_mae      = cv.get("std_mae", np.inf)
        overall_rmse = cv.get("overall_rmse", np.inf)
        overall_mae  = cv.get("overall_mae", np.inf)

        rmse_mean, mae_mean = (avg_rmse, avg_mae) if use == "avg" else (overall_rmse, overall_mae)
        score = rmse_mean if metric == "rmse" else mae_mean

        records.append({
            "p": p, "d": d, "q": q,
            "P": P, "D": D, "Q": Q, "s": s,
            "rmse_mean": rmse_mean,
            "mae_mean": mae_mean,
            "avg_rmse": avg_rmse,
            "avg_mae": avg_mae,
            "std_rmse": std_rmse,
            "std_mae": std_mae,
            "overall_rmse": overall_rmse,
            "overall_mae": overall_mae,
        })

        if np.isfinite(score) and score < best_score:
            best_score = score
            best_params = (p, d, q, P, D, Q, s)

    if not records or best_params is None:
        return None, pd.DataFrame()

    scores_df = pd.DataFrame(records).sort_values(by=metric_key, ascending=True)

    # PASS 2: rerun best candidate (so we can keep folds for plotting)
    p, d, q, P, D, Q, s = best_params
    best_cv = sarimax_time_series_cv(
        df=df, target=target,
        p=p, d=d, q=q, P=P, D=D, Q=Q, s=s,
        train_years=train_years,
        forecast_months=forecast_months,
        test_start_year=test_start_year,
        exog=exog,
        alpha=alpha,
        verbose=False,
    )

    avg_rmse     = best_cv.get("avg_rmse", np.inf)
    avg_mae      = best_cv.get("avg_mae", np.inf)
    std_rmse     = best_cv.get("std_rmse", np.inf)
    std_mae      = best_cv.get("std_mae", np.inf)
    overall_rmse = best_cv.get("overall_rmse", np.inf)
    overall_mae  = best_cv.get("overall_mae", np.inf)

    rmse_mean, mae_mean = (avg_rmse, avg_mae) if use == "avg" else (overall_rmse, overall_mae)
    score = rmse_mean if metric == "rmse" else mae_mean

    # --- grab folds + last fold (if available)
    folds = None
    for k in folds_key_candidates:
        if k in best_cv and isinstance(best_cv[k], (list, tuple)) and len(best_cv[k]) > 0:
            folds = best_cv[k]
            break

    last_fold = folds[-1] if folds else None
    overall_std = np.nan
    if "residuals" in best_cv and len(best_cv["residuals"]) > 0:
        overall_std = float(np.std(best_cv["residuals"], ddof=0))

    # --- last fold RMSE
    last_fold_rmse = np.nan
    if last_fold is not None:
        last_fold_rmse = float(
        np.sqrt(mean_squared_error(last_fold["y_true"], last_fold["y_pred"]))
        )
    best_info = {
        "order": (p, d, q),
        "seasonal_order": (P, D, Q, s),
        "metric": metric,
        "use": use,
        "score": score,

        # selection view
        "rmse_mean": rmse_mean,
        "mae_mean": mae_mean,

        # keep these
        "avg_rmse": avg_rmse,
        "avg_mae": avg_mae,
        "std_rmse": std_rmse,
        "std_mae": std_mae,
        "overall_rmse": overall_rmse,
        "overall_mae": overall_mae,

        "overall_std": overall_std,
        "last_fold_rmse": last_fold_rmse,

        # raw CV result dict
        "cv_results": best_cv,

        # if your CV returns this already
        "split_metrics": best_cv.get("split_metrics", []),

        # NEW: fold artifacts for plotting
        "folds": folds,               # may be None if CV doesn't provide them
        "last_fold": last_fold,       # dict: y_true/y_pred/resid/conf_int/...
    }

    if verbose:
        print("\n=== BEST BY TIME-SERIES CV ===")
        print("Selection metric:", metric, f"({use})")
        print("Best order:       ", best_info["order"])
        print("Best seasonal:    ", best_info["seasonal_order"])
        print("Best score:       ", best_info["score"])

        print("\n--- Average across folds ---")
        print("Avg RMSE:         ", best_info["avg_rmse"])
        print("Std RMSE:         ", best_info["std_rmse"])
        print("Avg MAE:          ", best_info["avg_mae"])
        print("Std MAE:          ", best_info["std_mae"])

        print("\n--- Overall (concatenated) ---")
        print("Overall RMSE:     ", best_info["overall_rmse"])
        print("Overall MAE:      ", best_info["overall_mae"])
        print("Overall STD:      ", best_info["overall_std"])
        print("Last fold RMSE:   ", best_info["last_fold_rmse"])
    if best_info["last_fold"] is None:
        print("NOTE: No per-fold artifacts found (add cv['folds'] in sarimax_time_series_cv).")
    else:
        lf = best_info["last_fold"]
        if "test_start" in lf and "test_end" in lf:
            print("Last fold test:   ", lf["test_start"], "→", lf["test_end"])

    return best_info, scores_df






class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                           num_layers=num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Take the last time step
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

class TimeSeriesDataset(Dataset):
    def __init__(self, target_data, exog_data, seq_length):
        """
        Dataset for time series with optional exogenous variables.
        
        Args:
            target_data: Scaled target variable data
            exog_data: Scaled exogenous variables data (can be None)
            seq_length: Length of input sequences
        """
        self.target_data = target_data
        self.exog_data = exog_data
        self.seq_length = seq_length
        self.has_exog = exog_data is not None
    
    def __len__(self):
        return len(self.target_data) - self.seq_length
    
    def __getitem__(self, idx):
        # Target sequence and label
        target_seq = self.target_data[idx:idx+self.seq_length]
        y = self.target_data[idx+self.seq_length]
        
        if self.has_exog:
            # Combine target with exogenous variables
            exog_seq = self.exog_data[idx:idx+self.seq_length]
            # Stack target and exog features: shape (seq_length, 1 + n_exog)
            x = np.column_stack([target_seq.reshape(-1, 1), exog_seq])
        else:
            x = target_seq.reshape(-1, 1)
        
        return torch.FloatTensor(x), torch.FloatTensor([y])

def lstm_experiment(df, target, forecast_window=12, no_windows=10, 
                   seq_length=12, epochs=100, batch_size=32, lr=0.001, dropout=0.2, hidden_size=100, num_layers=1,  exog=None, plot=False):
    """
    LSTM time series forecasting experiment
    
    Args:
        df: DataFrame with time series data
        target: Column name of target variable
        forecast_window: Number of steps to forecast
        no_windows: Number of forecast windows for train/test split
        seq_length: Length of input sequences (lookback window)
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        exog: List of exogenous variable column names
    """
    # Split data
    train = df[target][no_windows*-forecast_window:-forecast_window].values
    test = df[target][-forecast_window:].values
    
    # Normalize data
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.reshape(-1, 1)).flatten()

    exog_scaler = None
    exog_train_scaled = None
    exog_test_scaled = None
    
    if exog is not None:
        exog_train = df[exog][no_windows*-forecast_window:-forecast_window].values
        exog_test = df[exog][-forecast_window:].values
        exog_scaler = MinMaxScaler()
        exog_train_scaled = exog_scaler.fit_transform(exog_train)
        exog_test_scaled = exog_scaler.transform(exog_test)

    input_size = 1 + (len(exog) if exog else 0)
    
    # Create datasets - FIXED: Pass scaled exog data
    train_dataset = TimeSeriesDataset(train_scaled, exog_train_scaled, seq_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        if plot and (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}')
    
    # Forecasting
    model.eval()
    forecast = []
    
    # Use the last seq_length points from training data as initial input
    current_seq = train_scaled[-seq_length:].tolist()
    if exog is not None:
        # FIXED: Convert to list properly
        current_exog_seq = exog_train_scaled[-seq_length:].tolist()
    
    with torch.no_grad():
        for step in range(forecast_window):
            # Prepare input
            if exog is not None:
                # Use known future exog values
                exog_future = exog_test_scaled[step]
                # Combine target sequence with exog
                current_seq_arr = np.array(current_seq).reshape(-1, 1)
                current_exog_arr = np.array(current_exog_seq)
                x = np.column_stack([current_seq_arr, current_exog_arr])
            else:
                x = np.array(current_seq).reshape(-1, 1)
            
            x = torch.FloatTensor(x).unsqueeze(0).to(device)
            
            # Predict next value
            pred = model(x).cpu().numpy()[0, 0]
            forecast.append(pred)
            
            # Update sequence (sliding window)
            current_seq = current_seq[1:] + [pred]
            if exog is not None:
                # FIXED: Properly update exog sequence
                current_exog_seq = current_exog_seq[1:] + [exog_future.tolist()]
    
    # Inverse transform predictions
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
    
    # Evaluate
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)
    
    if plot:
        # Visualize results
        _, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot forecast vs actual
        train_index = df[target][no_windows*-forecast_window:-forecast_window].index
        test_index = df[target][-forecast_window:].index
        
        ax1.plot(train_index, train, label='Train', alpha=0.7)
        ax1.plot(test_index, test, label='Actual Test', linewidth=2)
        ax1.plot(test_index, forecast, label='Forecast', linestyle='--', linewidth=2)
        ax1.legend()
        ax1.set_title('LSTM Forecast vs Actual')
        ax1.set_xlabel('Time')
        ax1.set_ylabel(target)
        ax1.grid(True, alpha=0.3)
        
        # Plot training loss
        ax2.plot(train_losses)
        ax2.set_title('Training Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MSE Loss')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return model, scaler, exog_scaler, forecast, rmse, mae, test

def lstm_time_series_cv(df, target, train_years=9, forecast_months=12, 
                        test_start_year=2010, exog=None, seq_length=12, 
                        epochs=100, batch_size=32, lr=0.001, 
                        hidden_size=50, num_layers=1, dropout=0.2, verbose=True, plot=False):
    """
    Perform time series cross-validation for LSTM models with exogenous variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with datetime index
    target : str
        Name of target column
    train_years : int
        Number of years to use for training (default: 9)
    forecast_months : int
        Number of months to forecast ahead (default: 12)
    test_start_year : int
        Year to start testing from (default: 2010)
    exog : list of str, optional
        Names of exogenous variables
    seq_length : int
        Length of input sequences (lookback window)
    epochs : int
        Number of training epochs per split
    batch_size : int
        Batch size for training
    lr : float
        Learning rate
    hidden_size : int
        LSTM hidden size
    num_layers : int
        Number of LSTM layers
    verbose : bool
        Print progress
        
    Returns:
    --------
    dict : Dictionary containing results and metrics
    """
    
    # Calculate number of splits
    test_data = df[df.index.year >= test_start_year]
    total_test_months = len(test_data)
    train_months = train_years * 12
    n_splits = (total_test_months - train_months) // forecast_months
    
    if verbose:
        print(f"Data range: {df.index.min().year} to {df.index.max().year}")
        print(f"Training window: {train_years} years ({train_months} months)")
        print(f"Forecast horizon: {forecast_months} months")
        print(f"Sequence length: {seq_length} months")
        print(f"Testing period: {test_start_year} onwards")
        print(f"Number of CV splits: {n_splits}")
        if exog:
            print(f"Exogenous variables: {', '.join(exog)}")
        print()
    
    # Determine input size
    input_size = 1 + (len(exog) if exog else 0)
    
    # Initialize storage
    all_forecasts = []
    all_actuals = []
    all_residuals = []
    split_metrics = []
    lb_results = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Perform cross-validation
    for split_idx in range(n_splits):
        if verbose:
            print(f"{'='*70}")
            print(f"Split {split_idx + 1}/{n_splits}")
            print(f"{'='*70}")
        
        # Calculate indices
        train_start_idx = (test_start_year - df.index.min().year) * 12 + split_idx * forecast_months
        train_end_idx = train_start_idx + train_months
        test_end_idx = train_end_idx + forecast_months
        
        
        df_split = df.iloc[train_start_idx: test_end_idx]
        
        model, scaler, escaler, forecast, rmse, mae, test = lstm_experiment(df_split, target, hidden_size=hidden_size, dropout=dropout, num_layers=num_layers, forecast_window=forecast_months, no_windows=train_years+1, seq_length=seq_length, epochs=epochs, batch_size=batch_size, lr=lr, exog=exog, plot=False)
        # Store results
        all_forecasts.extend(forecast)
        all_actuals.extend(test)
        
        residuals = [a - f for a, f in zip(test, forecast)]
        all_residuals.extend(residuals)
        
        lb = acorr_ljungbox(
            residuals,
            lags=[12],
            return_df=True
        )  
        lb_results.append({
            'ds': df.index[test_end_idx - 1],
            'lb_stat': lb['lb_stat'].iloc[0],
            'p_value': lb['lb_pvalue'].iloc[0]
        })


        train_start_date = df.index[train_start_idx]
        train_end_date = df.index[train_end_idx - 1]
        test_start_date = df.index[train_end_idx]
        test_end_date = df.index[test_end_idx - 1]
        
        split_metrics.append({
            'split': split_idx + 1,
            'train_start': train_start_date,
            'train_end': train_end_date,
            'test_start': test_start_date,
            'test_end': test_end_date,
            'rmse': rmse,
            'mae': mae
        })
        
        del model, scaler, escaler
        if verbose:
            print(f"\n  Train: {train_start_date.strftime('%Y-%m')} to {train_end_date.strftime('%Y-%m')}")
            print(f"  Test:  {test_start_date.strftime('%Y-%m')} to {test_end_date.strftime('%Y-%m')}")
            print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}\n")
    
    # Calculate overall metrics
    overall_rmse = np.sqrt(mean_squared_error(all_actuals, all_forecasts))
    overall_mae = mean_absolute_error(all_actuals, all_forecasts)
    avg_rmse = np.mean([m['rmse'] for m in split_metrics])
    avg_rmse_std = np.std([m['rmse'] for m in split_metrics])
    avg_mae = np.mean([m['mae'] for m in split_metrics])
    avg_mae_std = np.std([m['mae'] for m in split_metrics])
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"OVERALL METRICS (concatenated predictions):")
        print(f"RMSE: {overall_rmse:.4f}")
        print(f"MAE: {overall_mae:.4f}")
        print(f"\nAVERAGE METRICS (across splits):")
        print(f"Average RMSE: {avg_rmse:.4f}, STD:  {avg_rmse_std:.4f}")
        print(f"Average MAE: {avg_mae:.4f}, STD:  {avg_mae_std:.4f}")
        print(f"{'='*70}\n")
    
    # Visualize
    if plot:
        _plot_lstm_cv_results(df, target, split_metrics, all_forecasts, 
                            all_actuals, forecast_months, residuals, pd.DataFrame(lb_results))
        
    return {
        'split_metrics': pd.DataFrame(split_metrics),
        'overall_rmse': overall_rmse,
        'overall_mae': overall_mae,
        'avg_rmse': avg_rmse,
        'avg_mae': avg_mae,
        'forecasts': all_forecasts,
        'actuals': all_actuals,
        'residuals': all_residuals,
        'lb_results': lb_results
    }


def _plot_lstm_cv_results(df, target, split_metrics, forecasts, actuals, forecast_months, residuals, lb):
    """Helper function to visualize LSTM cross-validation results."""
    
    forecast_indices = []
    for split_info in split_metrics:
        test_start = split_info['test_start']
        test_end = split_info['test_end']
        indices = df.loc[test_start:test_end].index
        forecast_indices.extend(indices[:forecast_months])
    
    plt.plot(forecast_indices, actuals, label='Actual', marker='o', 
            markersize=3, linewidth=1.5)
    plt.plot(forecast_indices, forecasts, label='Forecast', marker='x', 
            markersize=3, linewidth=1.5, alpha=0.7)
    plt.title('LSTM Forecast vs Actual (All Test Periods)')
    plt.xlabel('Date')
    plt.ylabel(target)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Metrics across splits
    residuals = [a - f for a, f in zip(actuals, forecasts)]
    plot_acf(pd.Series(residuals))

    plot_lb_test(lb)
    plt.tight_layout()
    plt.show()

import itertools
import time

def print_grid_search_results(results_df, best_overall_score, best_overall_results, best_overall_params, best_avg_score, best_avg_results, best_avg_params, best_last_fold_score, best_last_fold_results, best_last_fold_params):
    print(f"\n{'='*80}")
    print(f"GRID SEARCH COMPLETE")
    print(f"{'='*80}")
    
    print(f"\n{'='*80}")
    print(f"BEST MODEL BY OVERALL RMSE (concatenated predictions)")
    print(f"{'='*80}")
    print(f"Overall RMSE: {best_overall_score:.4f}")
    print(f"Overall MAE: {best_overall_results['overall_mae']:.4f}")
    print(f"Average RMSE: {best_overall_results['avg_rmse']:.4f}")
    print(f"Average MAE: {best_overall_results['avg_mae']:.4f}")
    print(f"\nParameters:")
    for key, val in best_overall_params.items():
        print(f"  {key}: {val}")
    
    print(f"\n{'='*80}")
    print(f"BEST MODEL BY AVERAGE RMSE (across folds)")
    print(f"{'='*80}")
    print(f"Average RMSE: {best_avg_score:.4f}")
    print(f"Average MAE: {best_avg_results['avg_mae']:.4f}")
    print(f"Overall RMSE: {best_avg_results['overall_rmse']:.4f}")
    print(f"Overall MAE: {best_avg_results['overall_mae']:.4f}")
    print(f"\nParameters:")
    for key, val in best_avg_params.items():
        print(f"  {key}: {val}")
    
    print(f"\n{'='*80}")
    print(f"BEST MODEL BY LAST FOLD RMSE (most recent period)")
    print(f"{'='*80}")
    last_fold_info = best_last_fold_results['split_metrics'].iloc[-1]
    print(f"Last Fold RMSE: {best_last_fold_score:.4f}")
    print(f"Last Fold MAE: {last_fold_info['mae']:.4f}")
    print(f"Test Period: {last_fold_info['test_start'].strftime('%Y-%m')} to {last_fold_info['test_end'].strftime('%Y-%m')}")
    print(f"Overall RMSE: {best_last_fold_results['overall_rmse']:.4f}")
    print(f"Average RMSE: {best_last_fold_results['avg_rmse']:.4f}")
    print(f"\nParameters:")
    for key, val in best_last_fold_params.items():
        print(f"  {key}: {val}")
    
    print(f"\n{'='*80}")
    print(f"TOP 10 MODELS BY AVG RMSE")
    print(f"{'='*80}")
    top_10_display = results_df[['combination', 'exog', 'lr', 'hidden_size', 
                                    'num_layers', 'dropout', 'overall_rmse', 
                                    'avg_rmse', 'last_fold_rmse']].head(10)
    print(top_10_display.to_string(index=False))
    print(f"{'='*80}\n")

def lstm_grid_search_cv(
    df,
    target,
    train_years=9,
    forecast_months=12,
    test_start_year=2010,
    seq_length=12,
    epochs=100,
    batch_size=32,
    param_grid= None,
    verbose= 1
):
    """
    Perform grid search with time series cross-validation for LSTM models.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with datetime index
    target : str
        Name of target column
    train_years : int
        Number of years to use for training
    forecast_months : int
        Number of months to forecast ahead
    test_start_year : int
        Year to start testing from
    seq_length : int
        Length of input sequences (lookback window)
    epochs : int
        Number of training epochs per split
    batch_size : int
        Batch size for training
    param_grid : dict, optional
        Dictionary with parameters names (str) as keys and lists of 
        parameter settings to try as values. If None, uses default grid.
        Possible keys:
        - 'exog': list of lists of exogenous variable names
        - 'lr': list of learning rates
        - 'hidden_size': list of hidden sizes
        - 'num_layers': list of number of layers
        - 'dropout': list of dropout rates
    verbose : int
        Verbosity level (0=silent, 1=progress, 2=detailed)
        
    Returns:
    --------
    dict : Dictionary containing:
        - 'results': DataFrame with all combinations and their metrics
        - 'best_overall_params': Parameters for best model by overall RMSE
        - 'best_overall_score': Best overall RMSE achieved
        - 'best_overall_results': Full CV results for best overall model
        - 'best_avg_params': Parameters for best model by average RMSE
        - 'best_avg_score': Best average RMSE achieved
        - 'best_avg_results': Full CV results for best average model
        - 'best_last_fold_params': Parameters for best model on last fold
        - 'best_last_fold_score': Best last fold RMSE achieved
        - 'best_last_fold_results': Full CV results for best last fold model
        
    Example:
    --------
    param_grid = {
        'exog': [
            None,
            ['lagged_tmed', 'lagged_tmin'],
            ['lagged_tmed', 'lagged_tmin', 'lagged_prec', 'lagged_tmax']
        ],
        'lr': [0.001, 0.01],
        'hidden_size': [50, 100],
        'num_layers': [1, 2],
        'dropout': [0.1, 0.2]
    }
    
    results = lstm_grid_search_cv(
        df=data,
        target='tdiff',
        train_years=8,
        forecast_months=24,
        test_start_year=1996,
        param_grid=param_grid,
        verbose=1
    )
    """
    
    # Default parameter grid if none provided
    if param_grid is None:
        param_grid = {
            'exog': [None],
            'lr': [0.001],
            'hidden_size': [50],
            'num_layers': [1],
            'dropout': [0.2]
        }
    
    # Ensure all required keys exist
    default_params = {
        'exog': [None],
        'lr': [0.001],
        'hidden_size': [50],
        'num_layers': [1],
        'dropout': [0.2]
    }
    
    for key, default_val in default_params.items():
        if key not in param_grid:
            param_grid[key] = default_val
    
    # Generate all parameter combinations
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    total_combinations = len(param_combinations)
    
    if verbose >= 1:
        print(f"{'='*80}")
        print(f"LSTM GRID SEARCH CROSS-VALIDATION")
        print(f"{'='*80}")
        print(f"Target: {target}")
        print(f"Training window: {train_years} years")
        print(f"Forecast horizon: {forecast_months} months")
        print(f"Sequence length: {seq_length} months")
        print(f"Total parameter combinations: {total_combinations}")
        print(f"\nParameter grid:")
        for key, vals in param_grid.items():
            print(f"  {key}: {vals}")
        print(f"{'='*80}\n")
    
    # Store results for all combinations
    results_list = []
    best_overall_score = float('inf')
    best_overall_params = None
    best_overall_results = None
    
    best_avg_score = float('inf')
    best_avg_params = None
    best_avg_results = None
    
    best_last_fold_score = float('inf')
    best_last_fold_params = None
    best_last_fold_results = None
    
    # Try each parameter combination
    for idx, params in enumerate(param_combinations, 1):
        if verbose >= 1:
            print(f"\n{'='*80}")
            print(f"Combination {idx}/{total_combinations}")
            print(f"{'='*80}")
            print(f"Parameters:")
            for key, val in params.items():
                if key == 'exog' and val is not None:
                    print(f"  {key}: {val}")
                elif key != 'exog':
                    print(f"  {key}: {val}")
            print(f"{'='*80}\n")
        
        try:
            # Run cross-validation with current parameters
            start_time = time.time()
            
            cv_results = lstm_time_series_cv(
                df=df,
                target=target,
                train_years=train_years,
                forecast_months=forecast_months,
                test_start_year=test_start_year,
                exog=params['exog'],
                seq_length=seq_length,
                epochs=epochs,
                batch_size=batch_size,
                lr=params['lr'],
                hidden_size=params['hidden_size'],
                num_layers=params['num_layers'],
                dropout=params['dropout'],
                verbose=False,
                plot=False
            )
            
            elapsed_time = time.time() - start_time
            
            # Extract metrics
            overall_rmse = cv_results['overall_rmse']
            overall_mae = cv_results['overall_mae']
            avg_rmse = cv_results['avg_rmse']
            avg_mae = cv_results['avg_mae']
            
            # Get last fold metrics
            split_metrics_df = cv_results['split_metrics']
            last_fold_rmse = split_metrics_df.iloc[-1]['rmse']
            last_fold_mae = split_metrics_df.iloc[-1]['mae']
            
            # Store results
            result_dict = {
                'combination': idx,
                'exog': str(params['exog']) if params['exog'] else 'None',
                'n_exog': len(params['exog']) if params['exog'] else 0,
                'lr': params['lr'],
                'hidden_size': params['hidden_size'],
                'num_layers': params['num_layers'],
                'dropout': params['dropout'],
                'overall_rmse': overall_rmse,
                'overall_mae': overall_mae,
                'avg_rmse': avg_rmse,
                'avg_mae': avg_mae,
                'last_fold_rmse': last_fold_rmse,
                'last_fold_mae': last_fold_mae,
                'time_seconds': elapsed_time
            }
            
            results_list.append(result_dict)
            
            # Check if this is the best model by overall RMSE
            if overall_rmse < best_overall_score:
                best_overall_score = overall_rmse
                best_overall_params = params.copy()
                best_overall_results = cv_results
            
            # Check if this is the best model by average RMSE
            if avg_rmse < best_avg_score:
                best_avg_score = avg_rmse
                best_avg_params = params.copy()
                best_avg_results = cv_results
            
            # Check if this is the best model by last fold RMSE
            if last_fold_rmse < best_last_fold_score:
                best_last_fold_score = last_fold_rmse
                best_last_fold_params = params.copy()
                best_last_fold_results = cv_results
            
            if verbose >= 1:
                print(f"\nResults for combination {idx}:")
                print(f"  Overall RMSE: {overall_rmse:.4f}")
                print(f"  Overall MAE: {overall_mae:.4f}")
                print(f"  Average RMSE: {avg_rmse:.4f}")
                print(f"  Average MAE: {avg_mae:.4f}")
                print(f"  Last Fold RMSE: {last_fold_rmse:.4f}")
                print(f"  Last Fold MAE: {last_fold_mae:.4f}")
                print(f"  Time elapsed: {elapsed_time:.2f}s")
                
                # Mark which best models this is
                best_markers = []
                if overall_rmse == best_overall_score:
                    best_markers.append("BEST OVERALL")
                if avg_rmse == best_avg_score:
                    best_markers.append("BEST AVG")
                if last_fold_rmse == best_last_fold_score:
                    best_markers.append("BEST LAST FOLD")
                if best_markers:
                    print(f"  *** {' | '.join(best_markers)} ***")
        
        except Exception as e:
            if verbose >= 1:
                print(f"\nERROR in combination {idx}: {str(e)}")
            
            result_dict = {
                'combination': idx,
                'exog': str(params['exog']) if params['exog'] else 'None',
                'n_exog': len(params['exog']) if params['exog'] else 0,
                'lr': params['lr'],
                'hidden_size': params['hidden_size'],
                'num_layers': params['num_layers'],
                'dropout': params['dropout'],
                'overall_rmse': np.nan,
                'overall_mae': np.nan,
                'avg_rmse': np.nan,
                'avg_mae': np.nan,
                'last_fold_rmse': np.nan,
                'last_fold_mae': np.nan,
                'time_seconds': np.nan,
                'error': str(e)
            }
            results_list.append(result_dict)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values('avg_rmse', ascending=True).reset_index(drop=True)
    
    if verbose >= 1:
        print_grid_search_results(results_df, best_overall_score, best_overall_results, best_overall_params, best_avg_score, best_avg_results, best_avg_params, best_last_fold_score, best_last_fold_results, best_last_fold_params)

    return {
        'results': results_df,
        'best_overall_params': best_overall_params,
        'best_overall_score': best_overall_score,
        'best_overall_results': best_overall_results,
        'best_avg_params': best_avg_params,
        'best_avg_score': best_avg_score,
        'best_avg_results': best_avg_results,
        'best_last_fold_params': best_last_fold_params,
        'best_last_fold_score': best_last_fold_score,
        'best_last_fold_results': best_last_fold_results,
        'param_grid': param_grid,
        'n_combinations': total_combinations
    }
