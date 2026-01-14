import numpy as np
import pandas as pd

def calculate_bill(load_profile, pricing_scheme):
    total_bill = 0.0
    for hour, consumption in enumerate(load_profile):
        price = pricing_scheme['offpeak']
        if 18 <= hour <= 21:
            price = pricing_scheme['peak']
        total_bill += consumption * price
    return total_bill

def main():
    # Load a sample month (e.g., last 30 days) for a random consumer
    df = pd.read_csv('data/smart_meter_data.csv', parse_dates=['Datetime'])
    consumer_id = np.random.choice(df['Consumer_ID'].unique())
    month_df = df[df['Consumer_ID'] == consumer_id].copy()
    month_df = month_df[month_df['Datetime'] >= (month_df['Datetime'].max() - pd.Timedelta(days=30))]
    month_df['Hour'] = month_df['Datetime'].dt.hour
    # Aggregate to hourly profile (average per hour)
    hourly_profile = month_df.groupby('Hour')['Consumption_kWh'].mean().reindex(range(24), fill_value=0).values
    # Scenario A: Unoptimized
    scenario_a_profile = hourly_profile.copy()
    # Scenario B: Shift 20% of peak (6-10PM) to off-peak (add to 0-6AM)
    scenario_b_profile = hourly_profile.copy()
    peak_hours = list(range(18, 22))
    offpeak_hours = list(range(0, 6))
    peak_load = scenario_b_profile[peak_hours]
    shift_amount = 0.2 * peak_load
    scenario_b_profile[peak_hours] -= shift_amount
    scenario_b_profile[offpeak_hours] += shift_amount.sum() / len(offpeak_hours)
    # Pricing
    pricing_scheme = {'peak': 0.20, 'offpeak': 0.08}
    bill_a = calculate_bill(scenario_a_profile, pricing_scheme)
    bill_b = calculate_bill(scenario_b_profile, pricing_scheme)
    savings = bill_a - bill_b
    print(f"Consumer ID: {consumer_id}")
    print(f"Unoptimized Bill: ${bill_a:.2f}")
    print(f"Optimized Bill: ${bill_b:.2f}")
    print(f"Total Money Saved ($): {savings:.2f}")
    return savings

if __name__ == "__main__":
    main()
