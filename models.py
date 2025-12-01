
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

def sarimax_experiment(df, target, p, d, q, P, D, Q, s, forecast_window=12, no_windows=10, exog=None):
    train, test = df[target][no_windows*-forecast_window:-forecast_window], df[target][-forecast_window:]

    exog_train = exog_test = None
    if exog is not None:
        exog_train, exog_test = df[exog][no_windows*-forecast_window:-forecast_window], df[exog][-forecast_window:]

    # Fit on training data
    model_train = SARIMAX(train, exog_train, order=(p, d, q), seasonal_order=(P, D, Q, s))
    results_train = model_train.fit()

    # Forecast on test period
    forecast = results_train.forecast(exog=exog_test, steps=forecast_window)

    # Evaluate forecast accuracy
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    # Visualize forecast
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train, label='Train')
    plt.plot(test.index, test, label='Actual Test')
    plt.plot(test.index, forecast, label='Forecast', linestyle='--')
    plt.legend()
    plt.title('SARIMA Forecast vs Actual')
    plt.show()
    results_train.summary(), results_train.plot_diagnostics()
