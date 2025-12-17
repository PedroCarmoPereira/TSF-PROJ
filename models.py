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
    max_p=3, max_q=3,
    max_P=2, max_Q=2,
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
        stepwise=True,
        trace=trace,
        error_action="ignore",
        suppress_warnings=True,
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


def sarimax_rolling_cv(
    df,
    target,
    order,
    seasonal_order,
    forecast_window=12,
    train_size=None,
    step=None,
    exog=None,
    verbose=True,
):
    """
    Rolling-window cross-validation for SARIMAX.

    - Uses a fixed-size rolling train window (train_size).
    - Forecasts 'forecast_window' steps ahead.
    - Moves the window by 'step' time steps each fold.
    - Returns only error metrics across folds.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataframe containing target (and exogenous vars if any).
    target : str
        Name of the target column in df.
    order : tuple
        (p, d, q) for SARIMAX.
    seasonal_order : tuple
        (P, D, Q, s) for SARIMAX.
    forecast_window : int
        Number of steps to forecast in each fold (h).
    train_size : int
        Number of time steps to use for training in each fold (must be > 0).
    step : int or None
        Jump between fold starts (in time steps). If None, defaults to forecast_window.
    exog : list[str] or None
        List of exogenous column names in df, or None.
    verbose : bool
        Print fold-by-fold metrics.

    Returns
    -------
    dict
        {
          "rmse_mean": float,
          "mse_mean": float,
          "mae_mean": float,
          "rmse_folds": list[float],
          "mse_folds": list[float],
          "mae_folds": list[float],
          "n_folds": int,
        }
    """
    y = df[target]
    n = len(y)
    h = forecast_window

    if train_size is None or train_size <= 0:
        raise ValueError("You must provide a positive train_size (number of time steps).")

    if step is None:
        step = h  # default: jump one forecast window each time

    if step <= 0:
        raise ValueError("step must be > 0")

    max_train_start = n - train_size - h
    if max_train_start < 0:
        raise ValueError(
            f"Not enough data: need at least train_size + forecast_window = "
            f"{train_size + h} points, but only have n = {n}."
        )

    n_folds = max_train_start // step + 1
    print(f"Running rolling CV with {n_folds} folds...")
    if verbose:
        print(f"Total length: {n}")
        print(f"Train window size: {train_size}")
        print(f"Forecast window: {h}")
        print(f"Step between folds: {step}")
        print(f"Number of folds (computed): {n_folds}")

    rmse_folds, mse_folds, mae_folds = [], [], []

    for fold in range(n_folds):
        train_start = fold * step
        train_end = train_start + train_size
        test_start = train_end
        test_end = test_start + h

        train = y.iloc[train_start:train_end]
        test = y.iloc[test_start:test_end]

        if exog is not None:
            exog_train = df[exog].iloc[train_start:train_end]
            exog_test = df[exog].iloc[test_start:test_end]
        else:
            exog_train = exog_test = None

        model = SARIMAX(
            train,
            exog=exog_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        results = model.fit(disp=False)

        fc = results.forecast(steps=h, exog=exog_test)
        fc = pd.Series(fc, index=test.index)

        mse = mean_squared_error(test, fc)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test, fc)

        rmse_folds.append(rmse)
        mse_folds.append(mse)
        mae_folds.append(mae)

        if verbose:
            print(
                f"Fold {fold+1}/{n_folds}: "
                f"RMSE={rmse:.4f}, MSE={mse:.4f}, MAE={mae:.4f}"
            )

    rmse_mean = float(np.mean(rmse_folds))
    mse_mean = float(np.mean(mse_folds))
    mae_mean = float(np.mean(mae_folds))

    if verbose:
        print("\nRolling CV summary:")
        print(f"Mean RMSE: {rmse_mean:.4f}")
        print(f"Mean MSE:  {mse_mean:.4f}")
        print(f"Mean MAE:  {mae_mean:.4f}")

    return {
        "rmse_mean": rmse_mean,
        "mse_mean": mse_mean,
        "mae_mean": mae_mean,
        "rmse_folds": rmse_folds,
        "mse_folds": mse_folds,
        "mae_folds": mae_folds,
        "n_folds": n_folds,
    }

def select_best_sarima_cv(
    df,
    target,
    candidates,
    forecast_window,
    train_size,
    step,
    exog=None,
    metric="rmse",   # "rmse" or "mae"
    verbose=True,
):
    """
    Evaluate a list of SARIMA candidates via rolling CV and pick the best.

    Parameters
    ----------
    df : pd.DataFrame
    target : str
    candidates : list[tuple]
        Each tuple is (p, d, q, P, D, Q, s).
    forecast_window : int
    train_size : int
    step : int
    exog : list[str] or None
    metric : str
        "rmse" or "mae" – which mean metric to minimize.
    verbose : bool

    Returns
    -------
    best_info : dict
        {
          "order": (p,d,q),
          "seasonal_order": (P,D,Q,s),
          "metric": metric,
          "score": best_score,
          "rmse_mean": float,
          "mae_mean": float,
          "cv_results": dict,  # full output of sarimax_rolling_cv for best model
        }
    scores_df : pd.DataFrame
        One row per candidate with mean RMSE/MAE etc.
    """
    assert metric in {"rmse", "mae"}
    metric_key = f"{metric}_mean"

    records = []
    best_info = None
    best_score = np.inf

    for i, (p, d, q, P, D, Q, s) in enumerate(candidates, 1):
        cv_results = sarimax_rolling_cv(
            df=df,
            target=target,
            order=(p, d, q),
            seasonal_order=(P, D, Q, s),
            forecast_window=forecast_window,
            train_size=train_size,
            step=step,
            exog=exog,
            verbose=False,
        )

        rmse_mean = cv_results["rmse_mean"]
        mae_mean = cv_results["mae_mean"]
        score = cv_results[metric_key]

        records.append({
            "p": p, "d": d, "q": q,
            "P": P, "D": D, "Q": Q, "s": s,
            "rmse_mean": rmse_mean,
            "mae_mean": mae_mean,
        })

        if verbose:
            print(
                f"{i:02d}. SARIMA({p},{d},{q})({P},{D},{Q},{s}) "
                f"- mean RMSE={rmse_mean:.4f}, mean MAE={mae_mean:.4f}"
            )

        if score < best_score:
            best_score = score
            best_info = {
                "order": (p, d, q),
                "seasonal_order": (P, D, Q, s),
                "metric": metric,
                "score": score,
                "rmse_mean": rmse_mean,
                "mae_mean": mae_mean,
                "cv_results": cv_results,
            }

    scores_df = pd.DataFrame(records).sort_values(by=metric_key, ascending=True)

    if verbose and best_info is not None:
        print("\n=== BEST BY ROLLING CV ===")
        print("Metric:", metric)
        print("Best order:       ", best_info["order"])
        print("Best seasonal:    ", best_info["seasonal_order"])
        print("Best mean RMSE:   ", best_info["rmse_mean"])
        print("Best mean MAE:    ", best_info["mae_mean"])

    return best_info, scores_df

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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
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
        print_grid_search_results(results_df, best_overall_score, best_overall_results, best_avg_params, best_avg_score, best_avg_results, best_avg_params, best_last_fold_score, best_last_fold_results, best_avg_score)
    
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
