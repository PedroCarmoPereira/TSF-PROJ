import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, month_plot
from statsmodels.graphics.gofplots import qqplot
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.diagnostic import acorr_ljungbox
from utils import get_residuals_any

def plot_time_series(df, date_col, var_col, ma_window=None):
    # 1. Set up the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # 2. Plot using the new 'date' column for the x-axis
    ax.plot(df[date_col], df[var_col], marker='o', linestyle='-')
    # 3. Optionally plot moving average
    if ma_window is not None and ma_window > 1:
        ma_series = df[var_col].rolling(window=ma_window).mean()
        ax.plot(df[date_col], ma_series, color='red', linewidth=2,
                label=f'{ma_window}-period MA')
        
    # 3. Format the date axis for clarity ✨
    # Set the major locator to find the start of each year
    ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
    # Set the format of the major labels to show just the year (e.g., "2023")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # To add ticks for every 3 months, you can use a minor locator
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))

    # 4. Add labels and a grid
    ax.set_title(var_col)
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.grid(True, which='major', alpha=0.6)
    ax.grid(True, which='minor', alpha=0.2)

    plt.tight_layout()
    plt.show()

def seasonal_plot(df, col, title):
    _, ax = plt.subplots(figsize=(16, 8))
    month_plot(df[col], ylabel='col', ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Month")
    plt.show()

def lag_plot_grid(ts, ys=None, lags=12, title="Lag Plots"):
    """Create grid of lag plots"""
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    for i in range(lags):
        row = i // 4
        col = i % 4
        
        # Create lagged series
        if ys is not None:
            lagged = ys.shift(i+1)
        else:
            lagged = ts.shift(i+1)
        
        # Remove NaN values
        mask = ~(np.isnan(ts) | np.isnan(lagged))
        x = ts[mask]
        y = lagged[mask]
        
        # Scatter plot
        axes[row, col].scatter(y, x, alpha=0.9, s=10, color="steelblue", edgecolor="black")
        axes[row, col].set_title(f'Lag {i+1}')
        axes[row, col].set_ylabel('X(t)')
        if ys is not None:
            axes[row, col].set_xlabel(f'Y(t-{i+1})')
        else:
            axes[row, col].set_xlabel(f'X(t-{i+1})')
        axes[row, col].grid(True, alpha=0.3)

        # Compute correlation
        corr = np.corrcoef(y, x)[0, 1]
        axes[row, col].text(
            0.05, 0.95,
            f"r = {corr:.3f}",
            transform=axes[row, col].transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5)
        )

        # Fit LOWESS
        smoothed = lowess(x, y, frac=0.3)  # frac controls smoothing
        axes[row, col].plot(smoothed[:,0], smoothed[:,1], color="red", linewidth=1.5)

    plt.tight_layout()
    plt.show()
    
def plot_acfs(df, col, lags=36):
    plot_acf(df[col], lags=lags, bartlett_confint=False)
    plot_pacf(df[col], lags=lags)
    plt.show()


def plot_lb_test(lb_results, lag=12, rolling_window=False):
    plt.figure(figsize=(12, 5))
    plt.plot(lb_results['ds'], lb_results['p_value'], marker='o')
    plt.axhline(0.05, linestyle='--', label='Significance level (0.05)')
    plt.xlabel('Forecast window end date')
    plt.ylabel(f'Ljung–Box p-value (lag {lag})')
    plt.title('Rolling Ljung–Box Test on Forecast Residuals')
    plt.legend()
    plt.grid(True)
    plt.show()
def plot_lb_test_not_rolling(lb_results, lag=12, rolling_window=False):
    plt.figure(figsize=(12, 5))
    plt.plot(lb_results['lb_stat'], lb_results['lb_pvalue'], marker='o')
    plt.axhline(0.05, linestyle='--', label='Significance level (0.05)')
    plt.xlabel('Forecast window end date')
    plt.ylabel(f'Ljung–Box p-value (lag {lag})')
    plt.title('Rolling Ljung–Box Test on Forecast Residuals')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_sarimax_results(
    train,
    test,
    forecast,
    results,
    order,
    seasonal_order,
    rmse,
    mae,
    rolling_window=False
):
    """
    Plot forecast vs actuals and individual SARIMAX diagnostics.
    """

    # === Forecast vs Actual ===
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train, label="Train")
    plt.plot(test.index, test, label="Actual Test")
    plt.plot(test.index, forecast, label="Forecast", linestyle="--")
    plt.legend()
    plt.title(
        f"SARIMA{order}{seasonal_order}\n"
        f"RMSE={rmse:.4f}, MAE={mae:.4f}"
    )
    plt.grid(alpha=0.3)
    plt.show()

    # === Individual diagnostics ===
    resid = get_residuals_any(results, train=train)

    # 1. Standardized residuals
    plt.figure(figsize=(10, 3))
    plt.plot(resid)
    plt.title("Standardized Residuals")
    plt.grid(alpha=0.3)
    plt.show()

    # 2. Histogram + KDE
    plt.figure(figsize=(6, 4))
    plt.hist(resid, bins=30, density=True)
    plt.title("Residual Distribution")
    plt.grid(alpha=0.3)
    plt.show()

    # 3. QQ plot
    
    qqplot(resid, line="s")
    plt.title("QQ Plot of Residuals")
    plt.show()

    # 4. ACF of residuals
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(resid, lags=24)
    plt.title("ACF of Residuals")
    plt.show()

    # 5. Ljung–Box test
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb = acorr_ljungbox(resid, lags=12, return_df=True)

    plot_lb_test_not_rolling(lb, rolling_window)




def plot_sarima_cv_results(df, target, split_metrics, forecasts, actuals, forecast_months, residuals, lb):
    """Helper function to visualize SARIMA cross-validation results."""

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
    plt.title('SARIMA Forecast vs Actual (All Test Periods)')
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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

def plot_last_fold(last_fold, *, alpha=0.05, acf_lags=12, lb_lags=12):
    """
    Plots for a single CV fold (typically the last fold):
      1) Actual vs Predicted (+ optional confidence interval band)
      2) Rolling Ljung-Box p-values across lags 1..lb_lags
      3) ACF of residuals

    Expects last_fold dict to contain:
      - "y_true": pd.Series
      - "y_pred": pd.Series
      - "resid":  pd.Series
      - optional "conf_int": pd.DataFrame (2 cols: lower/upper) aligned to index
      - optional "test_start", "test_end" for titles
    """
    y_true = last_fold["y_true"].copy()
    y_pred = last_fold["y_pred"].copy()
    resid  = last_fold["resid"].copy()
    ci     = last_fold.get("conf_int", None)

    # --- Align everything on the same index
    idx = y_true.index.intersection(y_pred.index)
    y_true = y_true.reindex(idx)
    y_pred = y_pred.reindex(idx)
    resid  = (y_true - y_pred) if resid is None else resid.reindex(idx)

    if ci is not None:
        ci = ci.reindex(idx)

    # Title helpers
    test_start = last_fold.get("test_start", idx.min())
    test_end   = last_fold.get("test_end", idx.max())
    title_suffix = f"({pd.to_datetime(test_start).date()} → {pd.to_datetime(test_end).date()})"

    # =========================
    # 1) Actual vs Predicted
    # =========================
    plt.figure(figsize=(10, 4))
    plt.plot(idx, y_true.values, label="Actual")
    plt.plot(idx, y_pred.values, label="Predicted")

    if ci is not None and isinstance(ci, pd.DataFrame) and ci.shape[1] >= 2:
        lower = ci.iloc[:, 0].to_numpy(dtype=float)
        upper = ci.iloc[:, 1].to_numpy(dtype=float)
        plt.fill_between(idx, lower, upper, alpha=0.2, label=f"{int((1-alpha)*100)}% CI")

    plt.title(f"Actual vs Predicted (Last CV Fold) {title_suffix}")
    plt.xlabel("Date")
    plt.ylabel(target := "Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # =========================
    # 2) Rolling Ljung-Box p-values
    #    (across lags 1..lb_lags)
    # =========================
    r = resid.dropna().astype(float).to_numpy()
    if len(r) < 3:
        print("Not enough residual points to compute Ljung-Box / ACF.")
        return

    max_lag_possible = max(1, len(r) - 1)
    lb_lags_eff = int(min(lb_lags, max_lag_possible))

    lags = list(range(1, lb_lags_eff + 1))
    lb_df = acorr_ljungbox(r, lags=lags, return_df=True)
    print(lb_df)

    plt.figure(figsize=(10, 3.5))
    plt.plot(lb_df.index, lb_df["lb_pvalue"].values, marker="o", label="p-value")
    plt.axhline(alpha, linestyle="--", label=f"Significance level ({alpha})")
    plt.title(f"Rolling Ljung-Box Test on Residuals {title_suffix}")
    plt.xlabel("Lag")
    plt.ylabel("p-value")
    plt.ylim(0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # =========================
    # 3) ACF of residuals
    # =========================
    acf_lags_eff = int(min(acf_lags, len(r) - 1))
    plt.figure(figsize=(10, 3.5))
    plot_acf(r, lags=acf_lags_eff)
    plt.title(f"ACF of Residuals (Last CV Fold) {title_suffix}")
    plt.tight_layout()
    plt.show()

def plots_from_ml_results(results, lb_results):
    # Preiction vs Actual plot
    plt.figure(figsize=(12, 6))
    plt.plot(results["ds"], results["actual"], label="Actual", linewidth=2)
    plt.plot(results["ds"], results["predicted"], label="Predicted", linewidth=2)
    plt.title("Actual vs Predicted values")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()

    residuals = np.array(results.actual) - np.array(results.predicted)

    plt.figure(figsize=(12, 4))
    plt.scatter(results["ds"], residuals)
    plt.axhline(0)
    plt.title("Residuals over time")
    plt.xlabel("Date")
    plt.ylabel("Residual")
    plt.show()

    lb_df = pd.DataFrame(lb_results)
    plot_lb_test(lb_df)


    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    plot_acf(residuals, lags=12, ax=axes[0])
    axes[0].set_title("ACF of Residuals")
    
    plot_pacf(residuals, lags=12, ax=axes[1], method="ywm")
    axes[1].set_title("PACF of Residuals")
    
    plt.tight_layout()
    plt.show()
