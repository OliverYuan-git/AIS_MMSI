import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score
import functools

# Function to convert hh:mm:ss to seconds
def hh_mm_ss2seconds(hh_mm_ss):
    return functools.reduce(lambda acc, x: acc*60 + x, map(int, hh_mm_ss.split(':')))

# Load and prepare data
def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path, converters={'SEQUENCE_DTTM': hh_mm_ss2seconds})
    selected_features = ['SEQUENCE_DTTM', 'LAT', 'LON', 'SPEED_OVER_GROUND', 'COURSE_OVER_GROUND']
    X = df[selected_features].to_numpy()
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)
    return df, X, selected_features

# Visualize clusters in 3D for a specific dataset
def visualize_clusters(X, labels, title):
    plt.figure(figsize=(10, 7))
    ax = plt.axes(projection='3d')
    ax.scatter(X[:, 1], X[:, 2], X[:, 3], c=labels, cmap='tab10')
    ax.set_xlabel('Latitude')
    ax.set_ylabel('Longitude')
    ax.set_zlabel('Speed Over Ground')
    ax.set_title(title)
    plt.show()

# Train the DBSCAN model with appropriate parameters
def train_dbscan(X, eps=0.1, min_samples=5):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    return db

# Load data for set1 and set2
df1, X1, _ = load_and_prepare_data('./set1.csv')
df2, X2, _ = load_and_prepare_data('./set2.csv')

# Concatenate datasets for combined training
X_combined = np.vstack([X1, X2])

# Train DBSCAN model with suitable parameters
dbscan = train_dbscan(X_combined, eps=0.1, min_samples=5)

# Predict clusters for set1 and set2 separately
labels_pred1 = dbscan.labels_[:len(X1)]
labels_pred2 = dbscan.labels_[len(X1):]

# Calculate adjusted Rand index scores
labels_true1 = df1['VID'].to_numpy()
labels_true2 = df2['VID'].to_numpy()
ari1 = adjusted_rand_score(labels_true1, labels_pred1)
ari2 = adjusted_rand_score(labels_true2, labels_pred2)
print(f'Adjusted Rand Index for set1.csv (DBSCAN): {ari1:.4f}')
print(f'Adjusted Rand Index for set2.csv (DBSCAN): {ari2:.4f}')

# Visualize clusters for each dataset (use the same `visualize_clusters` function)
visualize_clusters(X1, labels_pred1, 'Clustering Visualization (DBSCAN) for set1')
visualize_clusters(X2, labels_pred2, 'Clustering Visualization (DBSCAN) for set2')

# Predict clusters for set3noVID.csv
df3, X3, _ = load_and_prepare_data('./set3noVID.csv')
labels_pred3 = dbscan.fit_predict(X3)
visualize_clusters(X3, labels_pred3, 'Clustering Visualization (DBSCAN) for set3noVID')
