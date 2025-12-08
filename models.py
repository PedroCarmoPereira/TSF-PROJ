
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

from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                           num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Take the last time step
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_length]
        y = self.data[idx+self.seq_length]
        return torch.FloatTensor(x).unsqueeze(-1), torch.FloatTensor([y])

def lstm_experiment(df, target, forecast_window=12, no_windows=10, 
                   seq_length=12, epochs=100, batch_size=32, lr=0.001):
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
    """
    # Split data
    train = df[target][no_windows*-forecast_window:-forecast_window].values
    test = df[target][-forecast_window:].values
    
    # Normalize data
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.reshape(-1, 1)).flatten()
    test_scaled = scaler.transform(test.reshape(-1, 1)).flatten()
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_scaled, seq_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTM(input_size=1, hidden_size=50, num_layers=1).to(device)
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
    
    with torch.no_grad():
        for _ in range(forecast_window):
            # Prepare input
            x = torch.FloatTensor(current_seq).unsqueeze(0).unsqueeze(-1).to(device)
            
            # Predict next value
            pred = model(x).cpu().numpy()[0, 0]
            forecast.append(pred)
            
            # Update sequence (sliding window)
            current_seq = current_seq[1:] + [pred]
    
    # Inverse transform predictions
    forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
    
    # Evaluate
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mae = mean_absolute_error(test, forecast)
    
    print(f"\nRMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # Visualize results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
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
    
    return model, scaler, forecast, rmse, mae