import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

from constants import ROOT_DIR
from modules.functions import (
    load_model_sav,
    plot_confusion_matrix,
    calculte_models_auc_score,
)
from modules.classification.decision_tree import decision_tree
from modules.classification.knn import knn_model
from modules.classification.random_forest import random_forest
from modules.classification.naive_bayes import naive


def split_dataset(data, y_column_name="class"):
    data = data.copy()
    y: np.ndarray = data.pop(y_column_name).values
    X: np.ndarray = data.values
    labels = pd.unique(y)
    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)
    return trnX, tstX, trnY, tstY, labels


def create_classifier_models(trnX, trnY):
    dt = decision_tree(trnX, trnY)
    nb = naive(trnX, trnY)
    knn = knn_model(trnX, trnY)
    rf = random_forest(trnX, trnY)
    return nb, knn, dt, rf


def confusion_matrix_model(trnX, tstX, trnY, tstY, labels, prdY):
    cnf_mtx: np.ndarray = metrics.confusion_matrix(tstY, prdY, labels)
    plt.figure()
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), squeeze=False)
    plot_confusion_matrix(axs[0, 0], cnf_mtx, labels)
    plot_confusion_matrix(
        axs[0, 1], metrics.confusion_matrix(tstY, prdY, labels), labels, normalize=True
    )
    plt.tight_layout()


def get_accuracy_models(tstX, tstY, nb, knn, dt, rf, xg):
    accuracies = {
        "nb": nb.score(tstX, tstY),
        "knn": knn.score(tstX, tstY),
        "dt": dt.score(tstX, tstY),
        "rf": rf.score(tstX, tstY),
        "xg": xg.score(tstX, tstY),
    }
    return accuracies


def get_confusion_matrix_models(trnX, tstX, trnY, tstY, labels):
    confusion_matrix_model(
        trnX, tstX, trnY, tstY, labels, load_model_sav("naive_bayes").predict(tstX)
    )
    confusion_matrix_model(
        trnX, tstX, trnY, tstY, labels, load_model_sav("knn").predict(tstX)
    )
    confusion_matrix_model(
        trnX, tstX, trnY, tstY, labels, load_model_sav("decision_tree").predict(tstX)
    )
    confusion_matrix_model(
        trnX, tstX, trnY, tstY, labels, load_model_sav("random_forest").predict(tstX)
    )
    plt.show()


def get_sensitivity_models(tstX, tstY, nb, knn, dt, rf, xg, multi_class):
    sensitivity = {
        "nb": calculte_models_auc_score(nb, tstX, tstY, multi_class),
        "knn": calculte_models_auc_score(knn, tstX, tstY, multi_class),
        "dt": calculte_models_auc_score(dt, tstX, tstY, multi_class),
        "rf": calculte_models_auc_score(rf, tstX, tstY, multi_class),
        "xg": calculte_models_auc_score(xg, tstX, tstY, multi_class),
    }
    return sensitivity


def main():
    data: pd.DataFrame = pd.read_csv("{}/data/df_without_corr.csv".format(ROOT_DIR))
    trnX, tstX, trnY, tstY, labels = split_dataset(data)
    # create_models(trnX, tstX, trnY, tstY, labels)
    # get_accuracy_models(tstX, tstY)
    # get_confusion_matrix_models(trnX, tstX, trnY, tstY, labels)


if __name__ == "__main__":
    main()
