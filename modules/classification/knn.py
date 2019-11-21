import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.neighbors import KNeighborsClassifier

from modules.functions import multiple_line_chart, save_model, calculte_models_auc_score


def knn_model(trnX, trnY, n=19, d="manhattan", save_file=False):
    knn = KNeighborsClassifier(n_neighbors=n, metric=d)
    knn.fit(trnX, trnY)
    if save_file:
        save_model(knn, "knn")
    return knn


def knn_test_several_params(trnX, tstX, trnY, tstY, multi_class, plot=False):
    nvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    dist = ["manhattan", "euclidean", "chebyshev"]
    values = {}
    best_n_value = 0
    best_dist = 0
    best_score = 0
    best_model = None
    for d in dist:
        yvalues = []
        for n in nvalues:
            knn = knn_model(trnX, trnY, n, d)
            score = calculte_models_auc_score(knn, tstX, tstY, multi_class)
            if score > best_score:
                best_score = score
                best_n_value = n
                best_dist = d
                best_model = knn
            yvalues.append(score)
        values[d] = yvalues
    if plot:
        plt.figure()
        multiple_line_chart(
            plt.gca(), nvalues, values, "KNN variants", "n", "Sensitivty", percentage=True
        )
        plt.show()
    return best_model, best_score, best_dist, best_n_value
