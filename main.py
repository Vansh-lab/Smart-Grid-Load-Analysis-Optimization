# Main entry point for the Smart Grid Load Analysis & Optimization System

import os
import sys

# Add src to path
sys.path.append('src')

from data_gen import generate_smart_meter_data
from clustering import cluster_consumers
from forecasting import forecast_load
from optimization import optimize_load

def main():
    print("Starting Smart Grid Load Analysis & Optimization System...")

    # Step 1: Generate data
    print("Generating synthetic smart meter data...")
    generate_smart_meter_data()

    # Step 2: Cluster consumers
    print("Clustering consumers based on load profiles...")
    cluster_consumers()

    # Step 3: Forecast load
    print("Training LSTM model for load forecasting...")
    predictions, mape = forecast_load()
    print(f"Forecasting completed with MAPE: {mape:.2%}")

    # Step 4: Optimize load
    print("Running peak shaving optimization...")
    # Load sample data for optimization
    import pandas as pd
    df = pd.read_csv('data/smart_meter_data.csv', parse_dates=['Datetime'])
    df['Hour'] = df['Datetime'].dt.floor('h')
    total_load = df.groupby('Hour')['Consumption_kWh'].sum().reset_index()
    load_curve = total_load['Consumption_kWh'].values[:24]

    optimized, battery, pct = optimize_load(load_curve, battery_capacity=100, max_discharge_rate=10)
    print(f"Optimization completed. Peak load reduced by {pct:.2f}%")

    print("All processes completed successfully!")
    print("Run 'streamlit run app/main.py' to launch the dashboard.")

if __name__ == "__main__":
    main()