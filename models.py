import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
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
    plot=True,
):
    """
    Fit one SARIMAX model, evaluate on last forecast_window points,
    optionally plot, and RETURN metrics & fitted model.
    """
    # Train/test split
    train = df[target][no_windows * -forecast_window : -forecast_window]
    test = df[target][-forecast_window:]

    exog_train = exog_test = None
    if exog is not None:
        exog_train = df[exog][no_windows * -forecast_window : -forecast_window]
        exog_test = df[exog][-forecast_window:]

    # Fit on training data
    model_train = SARIMAX(train, exog_train, order=(p, d, q), seasonal_order=(P, D, Q, s))
    results_train = model_train.fit(disp=False)

    # Forecast on test period
    forecast = results_train.forecast(exog=exog_test, steps=forecast_window)

    # Evaluate forecast accuracy
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)

    if plot:
        print(f"SARIMA({p},{d},{q})({P},{D},{Q},{s})")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")

        plt.figure(figsize=(12, 6))
        plt.plot(train.index, train, label="Train")
        plt.plot(test.index, test, label="Actual Test")
        plt.plot(test.index, forecast, label="Forecast", linestyle="--")
        plt.legend()
        plt.title("SARIMA Forecast vs Actual")
        plt.show()

        # Optional diagnostics
        results_train.plot_diagnostics(figsize=(10, 8))
        plt.show()

    # IMPORTANT: return values so grid search can use them
    return rmse, mae, results_train, forecast, test


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
            rmse, mae, results, forecast, test = sarimax_experiment(
                df=df,
                target=target,
                p=p, d=d, q=q,
                P=P, D=D, Q=Q, s=s,
                forecast_window=forecast_window,
                no_windows=no_windows,
                exog=exog,
                plot=False,   # avoid spamming plots during grid search
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

import matplotlib.pyplot as plt

def plot_best_sarimax(df, target, best_info, forecast_window=12, no_windows=10):
    # reconstruct the train slice
    train = df[target][no_windows * -forecast_window : -forecast_window]
    test = best_info["test"]
    forecast = best_info["forecast"]

    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train, label="Train")
    plt.plot(test.index, test, label="Actual Test")
    plt.plot(test.index, forecast, label="Forecast (best model)", linestyle="--")
    plt.legend()
    plt.title(
        f"Best SARIMA{best_info['order']}{best_info['seasonal_order']} "
        f"RMSE={best_info['rmse']:.4f}, MAE={best_info['mae']:.4f}"
    )
    plt.show()

    # diagnostics from fitted model
    best_info["results"].plot_diagnostics(figsize=(10, 8))
    plt.show()

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
    plot=True,
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

    # --- Plot ---
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(train.index, train, label="Train")
        plt.plot(test.index, test, label="Actual Test")
        plt.plot(test.index, forecast, label="AutoSARIMA Forecast", linestyle="--")
        plt.legend()
        plt.title(f"AutoSARIMA{order}{seasonal_order}")
        plt.show()

    return {
        "order": order,
        "seasonal_order": seasonal_order,
        "rmse": rmse,
        "mae": mae,
        "model": model,
        "forecast": forecast,
        "test": test,
    }