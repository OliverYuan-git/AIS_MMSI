import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering, DBSCAN, OPTICS, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from scipy.stats import mode
import itertools
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

def apply_clustering_algorithms(X_scaled, algorithm_name, params):
    if algorithm_name == 'K-means':
        algorithm = KMeans(n_clusters=params['n_clusters'], random_state=params['random_state'])
    elif algorithm_name == 'Spectral Clustering':
        algorithm = SpectralClustering(n_clusters=params['n_clusters'], random_state=params['random_state'], assign_labels='discretize')
    elif algorithm_name == 'Hierarchical Clustering':
        algorithm = AgglomerativeClustering(n_clusters=params['n_clusters'])
    elif algorithm_name == 'DBSCAN':
        algorithm = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
    elif algorithm_name == 'OPTICS':
        algorithm = OPTICS(min_samples=params['min_samples'])
    elif algorithm_name == 'BIRCH':
        algorithm = Birch(n_clusters=params['n_clusters'])
    elif algorithm_name == 'GMM':
        algorithm = GaussianMixture(n_components=params['n_components'], random_state=params['random_state'])
    else:
        raise ValueError("Unknown algorithm name")
        
    labels = algorithm.fit_predict(X_scaled)
    return labels

def evaluate_clustering(labels_true, labels_pred):
    return adjusted_rand_score(labels_true, labels_pred)

def process_dataset(file_name):
    df = pd.read_csv(file_name)
    df = preprocess_features(df)
    selected_features = ['TIME_SEC', 'LAT', 'LON', 'VEL_X', 'VEL_Y']
    X = df[selected_features]
    labels_true = pd.read_csv(file_name)['VID']
    
    X_scaled = preprocessing.StandardScaler().fit_transform(X)
    
    best_scores = {}
    best_params = {}
    ensemble_predictions = []
    
    parameter_grid = {
        'K-means': {'n_clusters': [10, 15, 20], 'random_state': [80]},
        'Spectral Clustering': {'n_clusters': [10, 15, 20], 'random_state': [80]},
        'Hierarchical Clustering': {'n_clusters': [10, 15, 20]},
        'DBSCAN': {'eps': [0.1, 0.5, 1.0], 'min_samples': [3, 5, 7]},
        'OPTICS': {'min_samples': [3, 5, 7]},
        'BIRCH': {'n_clusters': [10, 15, 20]},
        'GMM': {'n_components': [10, 15, 20], 'random_state': [80]}
    }
    
    for algorithm_name, params in parameter_grid.items():
        best_score = float('-inf')
        for param_combination in itertools.product(*params.values()):
            param_dict = dict(zip(params.keys(), param_combination))
            labels_pred = apply_clustering_algorithms(X_scaled, algorithm_name, param_dict)
            score = evaluate_clustering(labels_true, labels_pred)
            if score > best_score:
                best_score = score
                best_params[algorithm_name] = param_dict
                if score > 0:  # Only consider adding if the algorithm performed meaningfully
                    ensemble_predictions.append(labels_pred)
        best_scores[algorithm_name] = best_score
    
    # Combine the predictions using majority vote
    if ensemble_predictions:
        ensemble_labels = mode(np.column_stack(ensemble_predictions), axis=1)[0].flatten()
        ensemble_score = evaluate_clustering(labels_true, ensemble_labels)
        print(f'Ensemble Score: {ensemble_score:.4f}')
    else:
        print("No valid ensemble could be formed.")
    
    print(f'Best parameters and scores for {file_name}:')
    for algorithm_name, params in best_params.items():
        print(f'{algorithm_name}: {params} - Score: {best_scores[algorithm_name]:.4f}')
    print()

if __name__ == "__main__":
    file_names = ['./set1.csv', './set2.csv', './set4.csv']
    for file_name in file_names:
        process_dataset(file_name)
