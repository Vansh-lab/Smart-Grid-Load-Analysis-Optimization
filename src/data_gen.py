import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_smart_meter_data():
    # Define consumer types and their profiles
    consumer_types = ['Residential', 'Commercial', 'Industrial']
    num_consumers = 50
    consumers = []
    for i in range(num_consumers):
        if i < 17:
            consumers.append('Residential')
        elif i < 34:
            consumers.append('Commercial')
        else:
            consumers.append('Industrial')

    # Start date: 6 months ago
    start_date = datetime.now() - timedelta(days=180)
    end_date = datetime.now()
    # Hourly timestamps
    timestamps = pd.date_range(start=start_date, end=end_date, freq='h')
    data = []
    for consumer_id in range(1, num_consumers+1):
        ctype = consumers[consumer_id-1]
        for ts in timestamps:
            hour = ts.hour
            # Base profile
            if ctype == 'Residential':
                if 18 <= hour <= 22:
                    base = random.uniform(2, 4)
                else:
                    base = random.uniform(0.3, 1.2)
            elif ctype == 'Commercial':
                if 9 <= hour <= 18:
                    base = random.uniform(3, 7)
                else:
                    base = random.uniform(0.2, 1.0)
            else:  # Industrial
                base = random.uniform(5, 8)
            # Gaussian seasonality: weekends lower
            if ts.weekday() >= 5:
                base *= 0.7
            # Random noise
            noise = np.random.normal(0, 0.08 * base)
            consumption = max(0, base + noise)
            data.append({
                'Datetime': ts,
                'Consumer_ID': consumer_id,
                'Type': ctype,
                'Consumption_kWh': consumption
            })
    df = pd.DataFrame(data)
    df.to_csv('data/smart_meter_data.csv', index=False)
    print("Data generated and saved to data/smart_meter_data.csv")

if __name__ == "__main__":
    generate_smart_meter_data()