import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.neighbors import KNeighborsClassifier

from modules.functions import multiple_line_chart, save_model


def knn_model(trnX, trnY, n=19, d="manhattan"):
    knn = KNeighborsClassifier(n_neighbors=n, metric=d)
    knn.fit(trnX, trnY)
    save_model(knn, "knn")
    return knn


def test_several_params(trnX, tstX, trnY, tstY):
    nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    dist = ["manhattan", "euclidean", "chebyshev"]
    values = {}
    for d in dist:
        yvalues = []
        for n in nvalues:
            knn = knn_model(trnX, trnY, n, d)
            prdY = knn.predict(tstX)
            yvalues.append(metrics.accuracy_score(tstY, prdY))
        values[d] = yvalues
    plt.figure()
    multiple_line_chart(
        plt.gca(), nvalues, values, "KNN variants", "n", "accuracy", percentage=True
    )
    plt.show()
