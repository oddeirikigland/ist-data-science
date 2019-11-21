from modules.unsupervised.clustering.clustering import kmeans_Cluster_pd
from modules.unsupervised.clustering.clustering import kmeans_Cluster_covtype

PLOT = False


def unsupervised_report(data, source):
    print("")
    print("Clustering 4.2")
    if source == "PD":
        kmeans_Cluster_pd(PLOT, data)
    if source == "CT":
        kmeans_Cluster_covtype(PLOT,data)


