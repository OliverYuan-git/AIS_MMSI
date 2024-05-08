import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN, OPTICS, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
import hdbscan
import functools

def hh_mm_ss2seconds(hh_mm_ss):
    if isinstance(hh_mm_ss, str):
        return functools.reduce(lambda acc, x: acc * 60 + x, map(int, hh_mm_ss.split(':')))
    else:
        return hh_mm_ss

def preprocess_features(df):
    df['SPEED_OVER_GROUND'] = df['SPEED_OVER_GROUND'] / 10.0
    df['COURSE_OVER_GROUND'] = np.radians(df['COURSE_OVER_GROUND'] / 10.0)
    df['VEL_X'] = df['SPEED_OVER_GROUND'] * np.cos(df['COURSE_OVER_GROUND'])
    df['VEL_Y'] = df['SPEED_OVER_GROUND'] * np.sin(df['COURSE_OVER_GROUND'])
    df['TIME_SEC'] = df['SEQUENCE_DTTM'].apply(hh_mm_ss2seconds)
    df.drop(['SPEED_OVER_GROUND', 'COURSE_OVER_GROUND', 'SEQUENCE_DTTM'], axis=1, inplace=True)
    return df

def apply_clustering_algorithms(X):
    X_scaled = preprocessing.StandardScaler().fit_transform(X)
    clustering_algorithms = {
        'K-means': KMeans(n_clusters=20, random_state=80),
        'Mean Shift': MeanShift(),
        'Spectral Clustering': SpectralClustering(n_clusters=20, random_state=80, assign_labels='discretize'),
        'Hierarchical Clustering': AgglomerativeClustering(n_clusters=20),
        'DBSCAN': DBSCAN(eps=13689, min_samples=5),
        'HDBSCAN': hdbscan.HDBSCAN(min_cluster_size=5),
        'OPTICS': OPTICS(min_samples=5),
        'BIRCH': Birch(n_clusters=20),
        'GMM': GaussianMixture(n_components=20, random_state=80)
    }
    results = {}
    for name, algorithm in clustering_algorithms.items():
        labels = algorithm.fit_predict(X_scaled)
        results[name] = labels
    return results

def evaluate_clustering(labels_true, results):
    scores = {}
    for name, labels_pred in results.items():
        score = adjusted_rand_score(labels_true, labels_pred)
        scores[name] = score
    return scores

def process_dataset(file_name):
    df = pd.read_csv(file_name)
    df = preprocess_features(df)
    selected_features = ['TIME_SEC', 'LAT', 'LON', 'VEL_X', 'VEL_Y']
    X = df[selected_features].to_numpy()
    results = apply_clustering_algorithms(X)
    labels_true = pd.read_csv(file_name)['VID'].to_numpy()
    scores = evaluate_clustering(labels_true, results)
    print(f'Results for {file_name}:')
    for name, score in scores.items():
        print(f'{name}: {score:.4f}')
    print()

if __name__ == "__main__":
    file_names = ['./set1.csv', './set2.csv']
    for file_name in file_names:
        process_dataset(file_name)
