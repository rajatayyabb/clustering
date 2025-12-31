# create_models.py - Run this locally to recreate models
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import pickle

# Create sample data
np.random.seed(42)
data = np.random.randn(200, 2) * 15 + [70, 50]
data[:, 0] = np.clip(data[:, 0], 10, 150)
data[:, 1] = np.clip(data[:, 1], 1, 100)

# Scale data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Create models
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(data)

dbscan = DBSCAN(eps=0.3, min_samples=5)
dbscan.fit(scaled_data)

# Save using Python's built-in pickle
with open('kmeans_model.pkl', 'wb') as f:
    pickle.dump(kmeans, f)

with open('dbscan_model.pkl', 'wb') as f:
    pickle.dump(dbscan, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Models created successfully!")
