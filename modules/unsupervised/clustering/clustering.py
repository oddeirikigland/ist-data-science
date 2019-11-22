import time, warnings
import numpy as np
import matplotlib.pyplot as plt, math
from itertools import cycle, islice
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cluster import DBSCAN
from constants import ROOT_DIR
import pandas as pd
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs
from sklearn import datasets, metrics, cluster, mixture
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA



def normalize_df(df_nr, columns_not_to_normalize):
    df_nr = df_nr.copy()
    df_not_normalize = df_nr[columns_not_to_normalize]

    df_nr = df_nr.drop(columns_not_to_normalize, axis=1)
    transf = Normalizer().fit(df_nr)
    df_nr = pd.DataFrame(transf.transform(df_nr, copy=True), columns=df_nr.columns)
    df_nr = pd.concat([df_not_normalize, df_nr], axis=1)
    return df_nr

def compute_Importance(X, k):
    X = normalize_df(X, [])

    pca = PCA(svd_solver="auto")
    pca.fit(X)

    T = pca.transform(X)

    # 1 scale principal components
    xvector = pca.components_[0] * max(T[:, 0])
    yvector = pca.components_[1] * max(T[:, 1])

    # 2 compute column importance and sort
    columns = X.columns.values
    impt_features = {
        columns[i]: math.sqrt(xvector[i] ** 2 + yvector[i] ** 2)
        for i in range(len(columns))
    }

    # print("Features by importance:", sorted(zip(impt_features.values(), impt_features.keys()), reverse=True))
    sortedList = sorted(zip(impt_features.values(), impt_features.keys()), reverse=True)
    doneList = sortedList[:k]
    # print("")
    features = []
    for m in range(0, len(doneList)):
        (value, feature) = doneList[m]
        features.append(feature)
    # print(features)
    X_selected_collums = X.loc[:, features]
    # print(X_selected_collums)

    return X_selected_collums


def plot_number_of_features(X):
    rand_values = []
    x_values = []
    X = df.groupby(by="id").median().reset_index()
    y = X["class"]
    X = X.drop(["class", "id"], axis=1)


    for i in range(1, 15):
        data = compute_Importance(X, i)
        kmeans_model = cluster.KMeans(n_clusters=4, random_state=1).fit(data)
        y_pred = kmeans_model.labels_
        randscore = metrics.adjusted_rand_score(y, y_pred)

        rand_values.append(randscore)
        x_values.append(i)
    plt.plot(x_values, rand_values)
    plt.ylabel("Rand")
    plt.xlabel("Number of features")
    plt.show()

def plot_number_of_features_cov(X):
    rand_values = []
    x_values = []
    y = X["Cover_Type"]
    X = X.drop(["Cover_Type"], axis=1)


    for i in range(1, 15):
        data = compute_Importance(X, i)
        kmeans_model = cluster.KMeans(n_clusters=7, random_state=1).fit(data)
        y_pred = kmeans_model.labels_
        randscore = metrics.adjusted_rand_score(y, y_pred)

        rand_values.append(randscore)
        x_values.append(i)
    plt.plot(x_values, rand_values)
    plt.ylabel("Rand")
    plt.xlabel("Number of features")
    plt.show()


def kmeans_Cluster_pd(bool, data):
    X = data.copy()
    y = X["class"]
    X = X.drop(["class"], axis=1)


    # feature selection
    # X_Best = SelectKBest(f_classif, k=10).fit_transform(X, y)
    # selector = SelectKBest(f_classif, k=10).fit(X, y)
    # features = selector.get_support()
    X_selected_collums = compute_Importance(X, 8)

    kmeans_model = cluster.KMeans(n_clusters=4, random_state=1).fit(X_selected_collums)
    y_pred = kmeans_model.labels_
    # print( y_pred)legge til connfution matrix

    matrix = confusion_matrix(y_pred, y)

    print("Kmeans")
    print("rand score: ", metrics.adjusted_rand_score(y, y_pred))
    print("adjusted mutual info score: ", metrics.adjusted_mutual_info_score(y, y_pred))
    print("mutual info score: ", metrics.mutual_info_score(y, y_pred))
    print(
        "normalized mutual info score: ",
        metrics.normalized_mutual_info_score(y, y_pred),
    )

    print("Sum of squared distances km:", kmeans_model.inertia_)
    print("Silhouette km:", metrics.silhouette_score(X_selected_collums, y_pred))
    print(
        "Calinski Harabaz km:",
        metrics.calinski_harabasz_score(X_selected_collums, y_pred),
    )
    print(
        "Davies Bouldin km:", metrics.davies_bouldin_score(X_selected_collums, y_pred)
    )
    # print("Silhouette per instance km:", metrics.silhouette_samples(X_selected_collums, y_pred))

    print("matrix:")
    print(matrix)
    print("")

    inerta_values = []
    x_values = []
    for i in range(2, 10, 1):
        kmeans_model = cluster.KMeans(n_clusters=i, random_state=1).fit(
            X_selected_collums
        )
        inerta_values.append(kmeans_model.inertia_)
        x_values.append(i)
    plt.plot(x_values, inerta_values)
    plt.ylabel("Inertia")
    plt.xlabel("Clusters")
    if bool:
        plt.show()


def DBscan_Cluster_pd(data):
    X = data.copy()
    y = X["class"]
    X = X.drop(["class"], axis=1)

    # feature selection
    X_Best = SelectKBest(f_classif, k=10).fit_transform(X, y)
    selector = SelectKBest(f_classif, k=10).fit(X, y)
    features = selector.get_support()
    X_selected_collums = X.loc[:, features]

    dbscan_model = cluster.DBSCAN(eps=0.1, min_samples=5).fit(X_selected_collums)
    y_pred = dbscan_model.fit_predict(X_selected_collums)
    print(y_pred)

    print("rand score: ", metrics.adjusted_rand_score(y, y_pred))
    print("adjusted mutual info score: ", metrics.adjusted_mutual_info_score(y, y_pred))
    print("mutual info score: ", metrics.mutual_info_score(y, y_pred))
    print(
        "normalized mutual info score: ",
        metrics.normalized_mutual_info_score(y, y_pred),
    )


def kmeans_Cluster_covtype(bool, data):
    dfCov = data.copy()
    X = dfCov
    y = X["Cover_Type"]
    X = X.drop(["Cover_Type"], axis=1)


    # feature selection
    # X_Best = SelectKBest(f_classif, k=10).fit_transform(X, y)
    # selector = SelectKBest(f_classif, k=10).fit(X, y)
    # features = selector.get_support()
    # X_selected_collums = X.loc[:, features]

    X_selected_collums = compute_Importance(X, 10)

    kmeans_model = cluster.KMeans(n_clusters=7, random_state=1).fit(X_selected_collums)
    y_pred = kmeans_model.labels_

    matrix = confusion_matrix(y_pred, y.values)

    # print( y_pred)legge til connfution matrix
    print("Kmeans:")
    print("Sum of squared distances:", kmeans_model.inertia_)
    print("Silhouette:", metrics.silhouette_score(X_selected_collums, y_pred))
    print(
        "Calinski Harabaz:", metrics.calinski_harabasz_score(X_selected_collums, y_pred)
    )
    print("Davies Bouldin:", metrics.davies_bouldin_score(X_selected_collums, y_pred))
    # print("Silhouette per instance km:", metrics.silhouette_samples(X_selected_collums, y_pred))

    print("rand score: ", metrics.adjusted_rand_score(y, y_pred))
    print("adjusted mutual info score: ", metrics.adjusted_mutual_info_score(y, y_pred))
    print("mutual info score: ", metrics.mutual_info_score(y, y_pred))
    print(
        "normalized mutual info score: ",
        metrics.normalized_mutual_info_score(y, y_pred),
    )

    print("matrix:")
    print(matrix)
    print("")

    inerta_values = []
    x_values = []
    for i in range(2, 20, 1):
        kmeans_model = cluster.KMeans(n_clusters=i, random_state=1).fit(
            X_selected_collums
        )
        inerta_values.append(kmeans_model.inertia_)
        x_values.append(i)
    plt.plot(x_values, inerta_values)
    plt.ylabel("Inertia")
    plt.xlabel("Clusters")

    if bool:
        plt.show()


def DBscan_Cluster_Covtype(data):
    dfCov = data.copy()
    X = dfCov
    y = X["Cover_Type"]
    X = X.drop(["Cover_Type"], axis=1)

    # feature selection
    # X_Best = SelectKBest(f_classif, k=10).fit_transform(X, y)
    # selector = SelectKBest(f_classif, k=10).fit(X, y)
    # features = selector.get_support()
    # X_selected_collums = X.loc[:, features]

    X_selected_collums = compute_Importance(X, 10)

    dbscan_model = cluster.DBSCAN(eps=1, min_samples=10).fit(X_selected_collums)
    y_pred = dbscan_model.labels_
    # print( y_pred)

    print("rand score: ", metrics.adjusted_rand_score(y, y_pred))
    print("adjusted mutual info score: ", metrics.adjusted_mutual_info_score(y, y_pred))
    print("mutual info score: ", metrics.mutual_info_score(y, y_pred))
    print(
        "normalized mutual info score: ",
        metrics.normalized_mutual_info_score(y, y_pred),
    )


def lek():
    noe = 0
    print(noe)
