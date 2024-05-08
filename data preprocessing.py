import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.cluster import KMeans
import functools
from sklearn.metrics.cluster import adjusted_rand_score

def hh_mm_ss2seconds(hh_mm_ss):
    if isinstance(hh_mm_ss, str):
        return functools.reduce(lambda acc, x: acc * 60 + x, map(int, hh_mm_ss.split(':')))
    else:
        return hh_mm_ss  # Assume it is already in seconds if not a string

def preprocess_features(df):
    df['SPEED_OVER_GROUND'] = df['SPEED_OVER_GROUND'] / 10.0  # Convert tenths of knots to knots
    df['COURSE_OVER_GROUND'] = np.radians(df['COURSE_OVER_GROUND'] / 10.0)  # Convert tenths of degrees to radians

    df['VEL_X'] = df['SPEED_OVER_GROUND'] * np.cos(df['COURSE_OVER_GROUND'])  # Velocity X component
    df['VEL_Y'] = df['SPEED_OVER_GROUND'] * np.sin(df['COURSE_OVER_GROUND'])  # Velocity Y component

    df['TIME_SEC'] = df['SEQUENCE_DTTM'].apply(hh_mm_ss2seconds)  # Convert time to seconds

    df.drop(['SPEED_OVER_GROUND', 'COURSE_OVER_GROUND', 'SEQUENCE_DTTM'], axis=1, inplace=True)
    return df

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)
    df = preprocess_features(df)
    return df

def plot_feature_distributions(df):
    plt.figure(figsize=(12, 10))
    
    plt.subplot(221)
    sns.histplot(df['VEL_X'], kde=True, color='blue', bins=30)
    plt.title('Distribution of Velocity X-component')
    
    plt.subplot(222)
    sns.histplot(df['VEL_Y'], kde=True, color='green', bins=30)
    plt.title('Distribution of Velocity Y-component')

    plt.subplot(223)
    sns.histplot(df['TIME_SEC'], kde=True, color='red', bins=30)
    plt.title('Distribution of Time in Seconds')

    plt.subplot(224)
    sns.scatterplot(x='VEL_X', y='VEL_Y', data=df, alpha=0.6)
    plt.title('Velocity Vector Plot')
    
    plt.tight_layout()
    plt.show()

def plot_geographical_distribution_simple(df):
    plt.figure(figsize=(10, 6))
    plt.scatter(df['LON'], df['LAT'], alpha=0.5, c='blue', label='Vessel Positions')
    plt.title('Geographical Distribution of Vessels')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.legend()
    plt.show()

def predictor_baseline(csv_path):
    df = load_and_preprocess(csv_path)
    selected_features = ['TIME_SEC', 'LAT', 'LON', 'VEL_X', 'VEL_Y']
    X = df[selected_features].to_numpy()
    X = preprocessing.StandardScaler().fit_transform(X)
    model = KMeans(n_clusters=20, random_state=123).fit(X)
    return model.predict(X)

def get_baseline_score():
    file_names = ['set1.csv', 'set2.csv']
    for file_name in file_names:
        csv_path = file_name
        labels_true = pd.read_csv(csv_path)['VID'].to_numpy()
        labels_pred = predictor_baseline(csv_path)
        rand_index_score = adjusted_rand_score(labels_true, labels_pred)
        print(f'Adjusted Rand Index Baseline Score of {file_name}: {rand_index_score:.4f}')

if __name__ == "__main__":
    df = load_and_preprocess('./set2.csv')
    plot_feature_distributions(df)
    plot_geographical_distribution_simple(df)
    get_baseline_score()
