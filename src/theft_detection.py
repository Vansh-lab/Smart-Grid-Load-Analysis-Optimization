import pandas as pd
import numpy as np
import random
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import Input, Dense  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv('data/smart_meter_data.csv', parse_dates=['Datetime'])

# 2. Simulate Theft
np.random.seed(42)
random.seed(42)
consumer_ids = df['Consumer_ID'].unique()
theft_ids = random.sample(list(consumer_ids), 5)
df['Is_Theft'] = 0
for cid in theft_ids:
    idx = (df['Consumer_ID'] == cid) & (df['Datetime'].dt.hour.between(18, 22))
    drop_pct = np.random.uniform(0.5, 0.8)
    df.loc[idx, 'Consumption_kWh'] *= (1 - drop_pct)
    df.loc[idx, 'Is_Theft'] = 1

df.to_csv('data/smart_meter_data_with_theft.csv', index=False)

# 3. Prepare data for Autoencoder
# Pivot: each row is a consumer-hour, columns are hours 0-23
pivot = df.copy()
pivot['Hour'] = pivot['Datetime'].dt.hour
pivot = pivot.groupby(['Consumer_ID', 'Hour'])[['Consumption_kWh', 'Is_Theft']].mean().reset_index()

all_hours = list(range(24))
pivot_normal = pivot[pivot['Is_Theft'] == 0].pivot(index='Consumer_ID', columns='Hour', values='Consumption_kWh').reindex(columns=all_hours, fill_value=0)
pivot_normal = pivot_normal.loc[~((pivot_normal == 0).all(axis=1) | pivot_normal.isnull().any(axis=1))]
pivot_theft = pivot[pivot['Is_Theft'] == 1].pivot(index='Consumer_ID', columns='Hour', values='Consumption_kWh').reindex(columns=all_hours, fill_value=0)
pivot_theft = pivot_theft.loc[~((pivot_theft == 0).all(axis=1) | pivot_theft.isnull().any(axis=1))]

scaler = MinMaxScaler()
if len(pivot_normal) == 0 or len(pivot_theft) == 0:
    print("Not enough data for training or theft detection. Exiting.")
    exit(1)
X_normal = scaler.fit_transform(pivot_normal)
X_theft = scaler.transform(pivot_theft)

# 4. Build Autoencoder
input_dim = X_normal.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(64, activation='relu')(input_layer)
encoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer=Adam(0.001), loss='mse')
autoencoder.fit(X_normal, X_normal, epochs=50, batch_size=8, verbose=1)

# 5. Detect Anomalies
reconstructions = autoencoder.predict(X_theft)
mse = np.mean(np.square(X_theft - reconstructions), axis=1)
threshold = np.percentile(np.mean(np.square(X_normal - autoencoder.predict(X_normal)), axis=1), 99)
suspicious_ids = list(pivot_theft.index[mse > threshold])

print("Suspicious Customer IDs:", suspicious_ids)

# 6. Plot Normal vs Anomalous Load Pattern
plt.figure(figsize=(12,6))
for cid in suspicious_ids:
    plt.plot(pivot_theft.columns, pivot_theft.loc[cid], label=f'Suspicious ID {cid}', linestyle='--')
for cid in random.sample(list(pivot_normal.index), min(3, len(pivot_normal.index))):
    plt.plot(pivot_normal.columns, pivot_normal.loc[cid], label=f'Normal ID {cid}')
plt.xlabel('Hour of Day')
plt.ylabel('Avg Consumption (kWh)')
plt.title('Normal vs Anomalous (Theft) Load Patterns')
plt.legend()
plt.savefig('data/theft_detection_plot.png')
plt.show()
