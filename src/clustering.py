import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def cluster_consumers():
    # Load data
    df = pd.read_csv('data/smart_meter_data.csv', parse_dates=['Datetime'])
    df['Hour'] = df['Datetime'].dt.hour
    hourly_avg = df.groupby(['Consumer_ID', 'Hour'])['Consumption_kWh'].mean().reset_index()
    pivot_df = hourly_avg.pivot(index='Consumer_ID', columns='Hour', values='Consumption_kWh').fillna(0)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pivot_df)
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    pivot_df['Cluster'] = clusters
    clustered_df = df[['Consumer_ID', 'Type']].drop_duplicates()
    clustered_df['Cluster'] = clustered_df['Consumer_ID'].map(dict(zip(pivot_df.index, clusters)))
    clustered_df.to_csv('data/clustered_users.csv', index=False)
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=pca_data[:,0], y=pca_data[:,1], hue=clusters, palette='viridis')
    plt.title('Consumer Clusters (PCA)')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend(title='Cluster')
    plt.savefig('data/cluster_pca.png')
    plt.close()
    avg_profiles = pivot_df.groupby('Cluster').mean()
    plt.figure(figsize=(12,6))
    for cluster in avg_profiles.index:
        plt.plot(avg_profiles.columns, avg_profiles.loc[cluster], label=f'Cluster {cluster}')
    plt.title('Average Load Profiles by Cluster')
    plt.xlabel('Hour of Day')
    plt.ylabel('Avg Consumption (kWh)')
    plt.legend()
    plt.savefig('data/avg_load_profiles.png')
    plt.close()

if __name__ == "__main__":
    cluster_consumers()