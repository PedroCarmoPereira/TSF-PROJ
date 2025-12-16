import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
import plots # ensure plots.py is imported for plotting functions

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
