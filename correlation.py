import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.cluster import KMeans
import functools

def hh_mm_ss2seconds(hh_mm_ss):
    return functools.reduce(lambda acc, x: acc*60 + x, map(int, hh_mm_ss.split(':')))

def load_and_prepare_data(csv_path):
    # Load data and convert hh:mm:ss to seconds
    df = pd.read_csv(csv_path, converters={'SEQUENCE_DTTM': hh_mm_ss2seconds})
    # Select features for clustering
    selected_features = ['SEQUENCE_DTTM', 'LAT', 'LON', 'SPEED_OVER_GROUND', 'COURSE_OVER_GROUND']
    X = df[selected_features].to_numpy()
    # Standardize features
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)
    return df, X, selected_features

def perform_eda(df, selected_features):
    # Plot correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[selected_features].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.show()

    # Pair plot to visualize relationships
    sns.pairplot(df[selected_features])
    plt.show()

# Example usage:
csv_path = './set1.csv'  # Adjust as needed
df, X, selected_features = load_and_prepare_data(csv_path)
perform_eda(df, selected_features)
