import time, warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import datasets, metrics, cluster, mixture
from sklearn.metrics import confusion_matrix

# datasetR: pd.DataFrame = pd.read_csv("{}/data/df_without_corr.csv".format(ROOT_DIR))
# dataset_CoverType: pd.DataFrame = pd.read_csv("{}/modules/test_cases/unsupervised_covtype_sample1/input".format(ROOT_DIR))

# dfCov= dataset_CoverType.copy()
# df = datasetR.copy()


def kmeans_Cluster_pd(bool, data):
    df = data.copy()
    X = df.groupby(by="id").median().reset_index()
    y = X["class"]
    X = X.drop(["id", "class"], axis=1)

    # feature selection
    X_Best = SelectKBest(f_classif, k=10).fit_transform(X, y)
    selector = SelectKBest(f_classif, k=10).fit(X, y)
    features = selector.get_support()
    X_selected_collums = X.loc[:, features]

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
    df = data.copy()
    X = df.groupby(by="id").median().reset_index()
    y = X["class"]
    X = X.drop(["id", "class"], axis=1)

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
    X_Best = SelectKBest(f_classif, k=10).fit_transform(X, y)
    selector = SelectKBest(f_classif, k=10).fit(X, y)
    features = selector.get_support()
    X_selected_collums = X.loc[:, features]

    kmeans_model = cluster.KMeans(n_clusters=7, random_state=1).fit(X_selected_collums)
    y_pred = kmeans_model.labels_

    matrix = confusion_matrix(y_pred, y)

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
    selector = SelectKBest(f_classif, k=10).fit(X, y)
    features = selector.get_support()
    X_selected_collums = X.loc[:, features]

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


def main():
    lek()
    # kmeans_Cluster(False, )
    # DBscan_Cluster()
    # kmeans_Cluster_covtype(False, 3)
    # DBscan_Cluster_Covtype()


if __name__ == "__main__":
    main()
