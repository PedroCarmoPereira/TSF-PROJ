import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, month_plot
from statsmodels.nonparametric.smoothers_lowess import lowess


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