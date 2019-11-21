from modules.unsupervised.patternmining.apriori import patternmining_ct, patternmining_pd

from modules.unsupervised.clustering.clustering import kmeans_Cluster_pd
from modules.unsupervised.clustering.clustering import kmeans_Cluster_covtype

PLOT = False


def unsupervised_report(data, source):
    df=data.copy()

    print("1. Applies Pattern Mining & Rules Assosiation:")
    if source == "PD":
        print("K=35, Min_patterns=300, Discretization with qcut with bins=6")
        patternmining_pd(df)

    elif source == "CT":
        print("K=25, Min_patterns=300, Discretization with cut with bins=4")
        patternmining_ct(df)



    print("")
    print("Clustering 4.2")
    if source == "PD":
        kmeans_Cluster_pd(PLOT, data)
    if source == "CT":
        kmeans_Cluster_covtype(PLOT,data)


