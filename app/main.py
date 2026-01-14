import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_gen import generate_smart_meter_data
from src.clustering import cluster_consumers
from src.forecasting import forecast_load
from src.optimization import optimize_load

st.set_page_config(page_title=" Smart Grid AI: Load Analysis & Peak Shaving System", layout="wide")
st.title(" Smart Grid AI: Load Analysis & Peak Shaving System")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Grid Analysis & Clustering", "AI Forecasting", "Peak Shaving Optimization", "Dynamic Pricing Savings", "🛡️ Grid Security & Economics", "🧠 Grid Resilience & Blackout Prediction"])
import random
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import Input, Dense  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
# GNN grid resilience imports
import networkx as nx
import torch  # type: ignore
from src.grid_resilience import generate_grid_graph, predict_grid_risk, get_grid_alerts
with tab6:
    st.header("🧠 Grid Resilience & Blackout Prediction")
    st.write("This feature uses a Graph Neural Network (GNN) to predict weak points and blackout risks in the grid based on real-time load, weather, and outage data.")
    if st.button("Run Grid Resilience Scan"):
        G = generate_grid_graph(num_nodes=50, edge_prob=0.08)
        risk_scores = predict_grid_risk(G)
        alerts = get_grid_alerts(G, risk_scores, threshold=0.7)
        # Visualize grid risk
        pos = nx.spring_layout(G)
        node_colors = ["red" if score > 0.7 else "green" for score in risk_scores]
        fig, ax = plt.subplots(figsize=(8,6))
        nx.draw(G, pos, node_color=node_colors, with_labels=True, ax=ax)
        for i, (x, y) in pos.items():
            ax.text(x, y+0.03, f"{risk_scores[i]:.2f}", fontsize=8, ha='center', color='black')
        ax.set_title("Grid Node Risk Scores (Red = High Risk)")
        st.pyplot(fig)
        if alerts:
            st.warning("\n".join(alerts))
        else:
            st.success("No high-risk nodes detected. Grid is resilient!")
def calculate_bill(load_profile, pricing_scheme):
    total_bill = 0.0
    for hour, consumption in enumerate(load_profile):
        price = pricing_scheme['offpeak']
        if 18 <= hour <= 21:
            price = pricing_scheme['peak']
        total_bill += consumption * price
    return total_bill

with tab5:
    st.header("🛡️ Grid Security & Economics")
    # --- Energy Theft Detection Section ---
    st.subheader("Energy Theft Detection")
    theft_alert = ""
    theft_plot = None
    if st.button("Scan for Anomalies"):
        # Load data and simulate theft
        df = pd.read_csv('data/smart_meter_data.csv', parse_dates=['Datetime'])
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
        # Prepare data for autoencoder
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
            theft_alert = "Not enough data for training or theft detection."
        else:
            X_normal = scaler.fit_transform(pivot_normal)
            X_theft = scaler.transform(pivot_theft)
            input_dim = X_normal.shape[1]
            input_layer = Input(shape=(input_dim,))
            encoded = Dense(64, activation='relu')(input_layer)
            encoded = Dense(32, activation='relu')(encoded)
            decoded = Dense(64, activation='relu')(encoded)
            decoded = Dense(input_dim, activation='sigmoid')(decoded)
            autoencoder = Model(input_layer, decoded)
            autoencoder.compile(optimizer=Adam(0.001), loss='mse')
            autoencoder.fit(X_normal, X_normal, epochs=30, batch_size=8, verbose=0)
            reconstructions = autoencoder.predict(X_theft)
            mse = np.mean(np.square(X_theft - reconstructions), axis=1)
            threshold = np.percentile(np.mean(np.square(X_normal - autoencoder.predict(X_normal)), axis=1), 99)
            suspicious_ids = list(pivot_theft.index[mse > threshold])
            if suspicious_ids:
                theft_alert = f"⚠️ Alert: Suspicious Activity detected for Customer IDs: {suspicious_ids}"
            else:
                theft_alert = "No suspicious activity detected."
            # Plot Normal vs Theft
            fig, ax = plt.subplots(figsize=(10,5))
            for cid in suspicious_ids:
                ax.plot(pivot_theft.columns, pivot_theft.loc[cid], label=f'Theft Pattern (ID {cid})', color='red', linestyle='--')
            for cid in random.sample(list(pivot_normal.index), min(2, len(pivot_normal.index))):
                ax.plot(pivot_normal.columns, pivot_normal.loc[cid], label=f'Normal Pattern (ID {cid})', color='blue')
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel('Avg Consumption (kWh)')
            ax.set_title('Normal vs Theft Load Patterns')
            ax.legend()
            plt.tight_layout()
            theft_plot = fig
    if theft_alert:
        st.info(theft_alert)
    if theft_plot:
        st.pyplot(theft_plot)
    # --- Smart Cost Savings Section ---
    st.subheader("Smart Cost Savings (Dynamic Pricing)")
    st.write("**Dynamic Pricing Rates:**  ")
    st.write("- Peak Hours (6 PM - 10 PM): $0.20 per kWh  ")
    st.write("- Off-Peak Hours: $0.08 per kWh")
    # Calculate for a random consumer
    if os.path.exists('data/smart_meter_data.csv'):
        df = pd.read_csv('data/smart_meter_data.csv', parse_dates=['Datetime'])
        consumer_id = random.choice(df['Consumer_ID'].unique())
        month_df = df[df['Consumer_ID'] == consumer_id].copy()
        month_df = month_df[month_df['Datetime'] >= (month_df['Datetime'].max() - pd.Timedelta(days=30))]
        month_df['Hour'] = month_df['Datetime'].dt.hour
        hourly_profile = month_df.groupby('Hour')['Consumption_kWh'].mean().reindex(range(24), fill_value=0).values
        scenario_a_profile = hourly_profile.copy()
        scenario_b_profile = hourly_profile.copy()
        peak_hours = list(range(18, 22))
        offpeak_hours = list(range(0, 6))
        peak_load = scenario_b_profile[peak_hours]
        shift_amount = 0.2 * peak_load
        scenario_b_profile[peak_hours] -= shift_amount
        scenario_b_profile[offpeak_hours] += shift_amount.sum() / len(offpeak_hours)
        pricing_scheme = {'peak': 0.20, 'offpeak': 0.08}
        bill_a = calculate_bill(scenario_a_profile, pricing_scheme)
        bill_b = calculate_bill(scenario_b_profile, pricing_scheme)
        savings = bill_a - bill_b
        st.metric("💰 Potential Monthly Savings", f"${savings:.2f}")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=["Bill without AI", "Bill with AI"], y=[bill_a, bill_b], marker_color=['red', 'green']))
        fig.update_layout(title="Monthly Bill Comparison", yaxis_title="Total Bill ($)")
        st.plotly_chart(fig)

import importlib.util
with tab4:
    st.header("Dynamic Pricing Bill Savings")
    if os.path.exists('data/smart_meter_data.csv'):
        df = pd.read_csv('data/smart_meter_data.csv', parse_dates=['Datetime'])
        consumer_ids = df['Consumer_ID'].unique()
        consumer_id = st.selectbox("Select Consumer ID", consumer_ids)
        month_df = df[df['Consumer_ID'] == consumer_id].copy()
        month_df = month_df[month_df['Datetime'] >= (month_df['Datetime'].max() - pd.Timedelta(days=30))]
        month_df['Hour'] = month_df['Datetime'].dt.hour
        hourly_profile = month_df.groupby('Hour')['Consumption_kWh'].mean().reindex(range(24), fill_value=0).values
        scenario_a_profile = hourly_profile.copy()
        scenario_b_profile = hourly_profile.copy()
        peak_hours = list(range(18, 22))
        offpeak_hours = list(range(0, 6))
        peak_load = scenario_b_profile[peak_hours]
        shift_amount = 0.2 * peak_load
        scenario_b_profile[peak_hours] -= shift_amount
        scenario_b_profile[offpeak_hours] += shift_amount.sum() / len(offpeak_hours)
        pricing_scheme = {'peak': 0.20, 'offpeak': 0.08}
        bill_a = calculate_bill(scenario_a_profile, pricing_scheme)
        bill_b = calculate_bill(scenario_b_profile, pricing_scheme)
        savings = bill_a - bill_b
        st.metric("Unoptimized Bill", f"${bill_a:.2f}")
        st.metric("Optimized Bill", f"${bill_b:.2f}")
        st.metric("Total Money Saved ($)", f"{savings:.2f}")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(range(24)), y=scenario_a_profile, name='Unoptimized'))
        fig.add_trace(go.Bar(x=list(range(24)), y=scenario_b_profile, name='Optimized'))
        fig.update_layout(barmode='group', title='Hourly Load Profile: Unoptimized vs Optimized', xaxis_title='Hour', yaxis_title='Avg Consumption (kWh)')
        st.plotly_chart(fig)

with tab1:
    st.header("Grid Analysis & Clustering")
    if st.button("Generate Data"):
        generate_smart_meter_data()
        st.success("Data generated!")
    if st.button("Run Clustering"):
        cluster_consumers()
        st.success("Clustering done!")
    if os.path.exists('data/smart_meter_data.csv'):
        df = pd.read_csv('data/smart_meter_data.csv', parse_dates=['Datetime'])
        st.dataframe(df.head(100))
        if os.path.exists('data/cluster_pca.png'):
            st.image('data/cluster_pca.png', caption='PCA Scatter Plot of Clusters')
        if os.path.exists('data/avg_load_profiles.png'):
            st.image('data/avg_load_profiles.png', caption='Average Load Profiles by Cluster')
        # Average load curves for Residential vs Commercial
        df['Hour'] = df['Datetime'].dt.hour
        avg_load = df[df['Type'].isin(['Residential','Commercial'])].groupby(['Type','Hour'])['Consumption_kWh'].mean().reset_index()
        fig = px.line(avg_load, x='Hour', y='Consumption_kWh', color='Type', title='Avg Load Curves: Residential vs Commercial')
        st.plotly_chart(fig)

with tab2:
    st.header("AI Forecasting")
    if st.button("Train/Load Model"):
        preds, mape = forecast_load()
        st.success(f"Model ready! MAPE: {mape:.2%}")
    if os.path.exists('models/forecast_plot.png'):
        st.image('models/forecast_plot.png', caption='LSTM Actual vs Predicted')

with tab3:
    st.header("Peak Shaving Optimization")
    battery_cap = st.slider("Battery Capacity (kWh)", 20, 200, 100)
    discharge_rate = st.slider("Max Discharge Rate (kWh)", 2, 20, 10)
    if os.path.exists('data/smart_meter_data.csv'):
        df = pd.read_csv('data/smart_meter_data.csv', parse_dates=['Datetime'])
        df['Hour'] = df['Datetime'].dt.floor('h')
        total_load = df.groupby('Hour')['Consumption_kWh'].sum().reset_index()
        load_curve = total_load['Consumption_kWh'].values[-24:]
        optimized, soc, pct, shifted = optimize_load(load_curve, battery_capacity=battery_cap, max_discharge_rate=discharge_rate)
        st.metric("Peak Load Reduced by", f"{pct:.2f}%")
        st.metric("Total Energy Shifted", f"{shifted:.2f} kWh")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=load_curve, mode='lines', name='Original Load', line=dict(color='red')))
        fig.add_trace(go.Scatter(y=optimized, mode='lines', name='Optimized Load', line=dict(color='green')))
        fig.update_layout(title='Peak Shaving Optimization', xaxis_title='Hour', yaxis_title='Load (kWh)')
        st.plotly_chart(fig)