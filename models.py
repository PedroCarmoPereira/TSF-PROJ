
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

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
                   seq_length=12, epochs=100, batch_size=32, lr=0.001, exog=None, plot=False):
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
    print(target)
    print(no_windows)
    print(forecast_window)
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
    model = LSTM(input_size=input_size, hidden_size=100, num_layers=1).to(device)
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
        
        if (epoch + 1) % 10 == 0:
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
    
    print(f"\nRMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
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
                        hidden_size=50, num_layers=1, verbose=True):
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
    split_metrics = []
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
        
        model, scaler, escaler, forecast, rmse, mae, test = lstm_experiment(df_split, target, forecast_window=forecast_months, no_windows=train_years+1, seq_length=seq_length, epochs=epochs, batch_size=batch_size, lr=lr, exog=exog, plot=False)
        # Store results
        all_forecasts.extend(forecast)
        all_actuals.extend(test)
        
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
    avg_mae = np.mean([m['mae'] for m in split_metrics])
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"OVERALL METRICS (concatenated predictions):")
        print(f"RMSE: {overall_rmse:.4f}")
        print(f"MAE: {overall_mae:.4f}")
        print(f"\nAVERAGE METRICS (across splits):")
        print(f"Average RMSE: {avg_rmse:.4f}")
        print(f"Average MAE: {avg_mae:.4f}")
        print(f"{'='*70}\n")
    
    # Visualize
    _plot_lstm_cv_results(df, target, split_metrics, all_forecasts, 
                          all_actuals, train_months, forecast_months)
    
    return {
        'split_metrics': pd.DataFrame(split_metrics),
        'overall_rmse': overall_rmse,
        'overall_mae': overall_mae,
        'avg_rmse': avg_rmse,
        'avg_mae': avg_mae,
        'forecasts': all_forecasts,
        'actuals': all_actuals
    }


def _plot_lstm_cv_results(df, target, split_metrics, forecasts, actuals, 
                          train_months, forecast_months):
    """Helper function to visualize LSTM cross-validation results."""
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot 1: Full timeline with CV splits
    ax1 = axes[0]
    ax1.plot(df.index, df[target], label='Full Data', alpha=0.7, color='blue')
    
    for i, split_info in enumerate(split_metrics):
        test_start = split_info['test_start']
        test_end = split_info['test_end']
        ax1.axvspan(test_start, test_end, alpha=0.2, color='red', 
                   label='Test Period' if i == 0 else '')
        ax1.axvline(split_info['train_end'], color='green', linestyle='--', 
                   alpha=0.5, label='Train End' if i == 0 else '')
    
    ax1.set_title(f'LSTM Time Series CV ({train_months//12}yr train, {forecast_months}mo forecast)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel(target)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Forecast vs Actual
    ax2 = axes[1]
    forecast_indices = []
    for split_info in split_metrics:
        test_start = split_info['test_start']
        test_end = split_info['test_end']
        indices = df.loc[test_start:test_end].index
        forecast_indices.extend(indices[:forecast_months])
    
    ax2.plot(forecast_indices, actuals, label='Actual', marker='o', 
            markersize=3, linewidth=1.5, color='blue')
    ax2.plot(forecast_indices, forecasts, label='Forecast', marker='x', 
            markersize=3, linewidth=1.5, color='red', alpha=0.7)
    ax2.set_title('LSTM Forecast vs Actual (All Test Periods)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel(target)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Metrics across splits
    ax3 = axes[2]
    metrics_df = pd.DataFrame(split_metrics)
    ax3.plot(metrics_df['split'], metrics_df['rmse'], marker='o', 
            label='RMSE', linewidth=2)
    ax3.plot(metrics_df['split'], metrics_df['mae'], marker='s', 
            label='MAE', linewidth=2)
    ax3.set_xlabel('Split Number')
    ax3.set_ylabel('Error')
    ax3.set_title('LSTM Forecast Errors Across CV Splits')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()