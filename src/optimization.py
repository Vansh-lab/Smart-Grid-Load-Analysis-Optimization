import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def optimize_load(original_load_curve, battery_capacity, max_discharge_rate):
    """
    Peak Shaving Algorithm with Battery Storage.
    
    Parameters:
    - original_load_curve: np.array of hourly load values
    - battery_capacity: float, max kWh battery can hold
    - max_discharge_rate: float, max kWh per hour discharge
    
    Returns:
    - optimized_load_curve: np.array
    - battery_status: np.array (charge level over time)
    """
    n_hours = len(original_load_curve)
    optimized_load = np.copy(original_load_curve)
    soc = np.zeros(n_hours)
    battery_level = 0
    threshold = np.mean(original_load_curve) + np.std(original_load_curve)
    for i in range(n_hours):
        load = original_load_curve[i]
        if load > threshold and battery_level > 0:
            discharge = min(max_discharge_rate, battery_level, load - threshold)
            optimized_load[i] -= discharge
            battery_level -= discharge
        elif load < threshold and battery_level < battery_capacity:
            charge = min(max_discharge_rate, battery_capacity - battery_level, threshold - load)
            optimized_load[i] += charge
            battery_level += charge
        soc[i] = battery_level
    original_peak = np.max(original_load_curve)
    optimized_peak = np.max(optimized_load)
    reduction_pct = ((original_peak - optimized_peak) / original_peak) * 100
    total_shifted = np.sum(np.abs(original_load_curve - optimized_load))
    return optimized_load, soc, reduction_pct, total_shifted

# Example usage
if __name__ == "__main__":
    # Load sample data
    df = pd.read_csv('data/smart_meter_data.csv', parse_dates=['Datetime'])
    df['Hour'] = df['Datetime'].dt.floor('h')
    total_load = df.groupby('Hour')['Consumption_kWh'].sum().reset_index()
    load_curve = total_load['Consumption_kWh'].values[:24]  # First 24 hours
    
    optimized, soc, pct, shifted = optimize_load(load_curve, battery_capacity=100, max_discharge_rate=10)
    print(f"Peak Load Reduced by {pct:.2f}% | Total Energy Shifted: {shifted:.2f} kWh")
    plt.figure(figsize=(12,6))
    plt.plot(load_curve, label='Original Load', color='red')
    plt.plot(optimized, label='Optimized Load', color='green')
    plt.title('Peak Shaving Optimization')
    plt.xlabel('Hour')
    plt.ylabel('Load (kWh)')
    plt.legend()
    plt.savefig('models/optimization_plot.png')
    plt.close()