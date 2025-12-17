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

def plot_cv_results(df, target, split_metrics, forecasts, actuals, 
                     train_months, forecast_months):
    """Helper function to visualize cross-validation results."""
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: All data with CV splits highlighted
    ax1 = axes[0]
    ax1.plot(df.index, df[target], label='Full Data', alpha=0.7, color='blue')
    
    for i, split_info in enumerate(split_metrics):
        test_start = split_info['test_start']
        test_end = split_info['test_end']
        ax1.axvspan(test_start, test_end, alpha=0.2, color='red', 
                   label='Test Period' if i == 0 else '')
        ax1.axvline(split_info['train_end'], color='green', linestyle='--', 
                   alpha=0.5, label='Train End' if i == 0 else '')
    
    ax1.set_title(f'Time Series with CV Splits ({train_months//12}yr train, {forecast_months}mo forecast)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel(target)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Forecast vs Actual for test periods
    ax2 = axes[1]
    
    # Reconstruct index for forecasts
    forecast_indices = []
    for split_info in split_metrics:
        test_start = split_info['test_start']
        test_end = split_info['test_end']
        indices = df.loc[test_start:test_end].index
        forecast_indices.extend(indices[:forecast_months])
    
    ax2.plot(forecast_indices, actuals, label='Actual', marker='o', 
            markersize=3, linewidth=1.5)
    ax2.plot(forecast_indices, forecasts, label='Forecast', marker='x', 
            markersize=3, linewidth=1.5, alpha=0.7)
    
    ax2.set_title('Forecast vs Actual (SARIMAX)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel(target)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Plot metrics over splits
    metrics_df = pd.DataFrame(split_metrics)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(metrics_df['split'], metrics_df['rmse'], marker='o', label='RMSE')
    ax.plot(metrics_df['split'], metrics_df['mae'], marker='s', label='MAE')
    ax.set_xlabel('Split Number')
    ax.set_ylabel('Error')
    ax.set_title('Forecast Errors Across CV Splits')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



def plot_sarimax_cv_results(
    df,
    target,
    split_metrics,
    forecasts,
    actuals,
    forecast_months,
    residuals=None,
    lb=None,
    lags_acf=24,
    lags_lb=12,
):
    """Visualize SARIMAX cross-validation results + diagnostics (CV residuals)."""

    # --- Reconstruct index for CV test predictions (legacy approach) ---
    forecast_indices = []
    for split_info in split_metrics:
        test_start = split_info["test_start"]
        test_end = split_info["test_end"]
        indices = df.loc[test_start:test_end].index
        forecast_indices.extend(indices[:forecast_months])

    # Ensure same length
    n = min(len(forecast_indices), len(actuals), len(forecasts))
    forecast_indices = forecast_indices[:n]
    actuals = list(actuals)[:n]
    forecasts = list(forecasts)[:n]

    # --- 1) Forecast vs Actual (all CV test periods) ---
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[target], label="Full Series", alpha=0.35)
    plt.plot(forecast_indices, actuals, label="Actual (CV tests)", marker="o", markersize=3, linewidth=1.5)
    plt.plot(forecast_indices, forecasts, label="Forecast (CV)", marker="x", markersize=3, linewidth=1.5, alpha=0.7)
    plt.title("SARIMAX Forecast vs Actual (All CV Test Periods)")
    plt.xlabel("Date")
    plt.ylabel(target)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- Residuals series (test residuals) ---
    if residuals is None:
        residuals = [a - f for a, f in zip(actuals, forecasts)]

    resid = pd.Series(residuals, index=pd.Index(forecast_indices, name="date")).dropna()

    # Standardize (like typical diagnostics)
    std = resid.std(ddof=1)
    if std == 0 or np.isnan(std):
        std = 1.0
    resid_std = resid / std

    # --- 2) Standardized residuals ---
    plt.figure(figsize=(10, 3))
    plt.plot(resid_std.index, resid_std.values)
    plt.title("Standardized Residuals (CV test periods)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- 3) Histogram of residuals (standardized) ---
    plt.figure(figsize=(6, 4))
    plt.hist(resid_std.values, bins=30, density=True)
    plt.title("Residual Distribution (standardized, CV)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- 4) QQ Plot ---
    qqplot(resid_std.values, line="s")
    plt.title("QQ Plot of Residuals (CV)")
    plt.tight_layout()
    plt.show()

    # --- 5) ACF of residuals ---
    plot_acf(resid_std.values, lags=lags_acf)
    plt.title("ACF of Residuals (CV)")
    plt.tight_layout()
    plt.show()

    # --- 6) Ljung–Box test ---
    if lb is None:
        lb = acorr_ljungbox(resid_std.values, lags=lags_lb, return_df=True)

    # Use your existing helper
    plot_lb_test(lb)
    plt.tight_layout()
    plt.show()
