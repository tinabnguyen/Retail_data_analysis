import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

rfm = pd.read_csv('data/rfm_retail_data.csv')

# Features to cluster
X = rfm[["Recency", "Frequency", "Monetary"]]

# Scale (important for k-means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit k-means with, say, 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42)
rfm["Cluster"] = kmeans.fit_predict(X_scaled)

print(rfm.head())

cluster_profile = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
print(cluster_profile)

cluster_labels = {
    0: "Inactive",
    1: "At Risk",
    2: "Loyal Customer",
    3: "Potential Loyalist"
}

rfm["Persona"] = rfm["Cluster"].map(cluster_labels)

print(rfm.head())

rfm.to_csv('data/clustered_retail_data.csv', index=False)