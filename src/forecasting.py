import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential  # type: ignore
from keras.layers import LSTM, Dense  # type: ignore
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

def forecast_load():
    # Load data
    df = pd.read_csv('data/smart_meter_data.csv', parse_dates=['Datetime'])
    
    # Aggregate to total grid load per hour
    df['Hour'] = df['Datetime'].dt.floor('h')
    total_load = df.groupby('Hour')['Consumption_kWh'].sum().reset_index()
    total_load.rename(columns={'Consumption_kWh': 'Total_Grid_Load'}, inplace=True)
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_load = scaler.fit_transform(total_load[['Total_Grid_Load']])
    
    # Prepare sequences: past 24 hours to predict next 1 hour
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)
    
    seq_length = 24
    X, y = create_sequences(scaled_load, seq_length)
    
    # Split train/test (80/20)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Build LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    
    # Save model
    model.save('models/load_forecaster.h5')
    
    # Predict
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test)
    
    # Calculate MAPE
    mape = mean_absolute_percentage_error(y_test_actual, predictions)
    print(f"MAPE: {mape:.2%}")
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_actual, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title('LSTM Load Forecasting')
    plt.xlabel('Time Steps')
    plt.ylabel('Total Grid Load (kWh)')
    plt.legend()
    plt.savefig('models/forecast_plot.png')
    plt.close()
    
    return predictions, mape

if __name__ == "__main__":
    forecast_load()