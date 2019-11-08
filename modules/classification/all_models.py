import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

from constants import ROOT_DIR
from modules.functions import load_model_sav, plot_confusion_matrix
from modules.classification.decision_tree import decision_tree
from modules.classification.knn import knn_model
from modules.classification.random_forest import random_forest
from modules.classification.naive_bayes import naive
from modules.preprocessing.balancing import (
    balancing_training_dataset,
    compare_balanced_scores,
)


def split_dataset(data):
    y: np.ndarray = data.pop("class").values
    X: np.ndarray = data.values
    labels = pd.unique(y)
    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)
    return trnX, tstX, trnY, tstY, labels


def create_classifier_models(trnX, trnY):
    decision_tree(trnX, trnY)
    naive(trnX, trnY)
    knn_model(trnX, trnY)
    random_forest(trnX, trnY)


def get_accuracy_models(tstX, tstY):
    accuracies = {
        "nb": load_model_sav("naive_bayes").score(tstX, tstY),
        "knn": load_model_sav("knn").score(tstX, tstY),
        "dt": load_model_sav("decision_tree").score(tstX, tstY),
        "rf": load_model_sav("random_forest").score(tstX, tstY),
    }
    return accuracies


def confusion_matrix_model(trnX, tstX, trnY, tstY, labels, prdY):
    cnf_mtx: np.ndarray = metrics.confusion_matrix(tstY, prdY, labels)
    plt.figure()
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), squeeze=False)
    plot_confusion_matrix(axs[0, 0], cnf_mtx, labels)
    plot_confusion_matrix(
        axs[0, 1], metrics.confusion_matrix(tstY, prdY, labels), labels, normalize=True
    )
    plt.tight_layout()


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


def calculte_models_auc_score(classifier, trnX, trnY, tstX, tstY):
    model = classifier(trnX, trnY)
    pred_y = model.predict(tstX)
    return roc_auc_score(tstY, pred_y)


def finds_best_data_set_balance(trnX, tstX, trnY, tstY):
    scores = {}
    under_sample_y, under_sample_x, over_sample_y, over_sample_x, smote_x, smote_y = balancing_training_dataset(
        trnX, trnY
    )

    unbalanced_naive_bayes = calculte_models_auc_score(naive, trnX, trnY, tstX, tstY)
    unbalanced_knn = calculte_models_auc_score(knn_model, trnX, trnY, tstX, tstY)
    scores["unbalanced"] = [unbalanced_naive_bayes, unbalanced_knn]

    under_sample_naive_bayes = calculte_models_auc_score(
        naive, under_sample_x, under_sample_y, tstX, tstY
    )
    under_sample_knn = calculte_models_auc_score(
        knn_model, under_sample_x, under_sample_y, tstX, tstY
    )
    scores["under_sample"] = [under_sample_naive_bayes, under_sample_knn]

    over_sample_naive_bayes = calculte_models_auc_score(
        naive, over_sample_x, over_sample_y, tstX, tstY
    )
    over_sample_knn = calculte_models_auc_score(
        knn_model, over_sample_x, over_sample_y, tstX, tstY
    )
    scores["over_sample"] = [over_sample_naive_bayes, over_sample_knn]

    smote_naive_bayes = calculte_models_auc_score(naive, smote_x, smote_y, tstX, tstY)
    smote_knn = calculte_models_auc_score(knn_model, smote_x, smote_y, tstX, tstY)
    scores["smote"] = [smote_naive_bayes, smote_knn]
    return scores


def main():
    data: pd.DataFrame = pd.read_csv("{}/data/df_without_corr.csv".format(ROOT_DIR))
    trnX, tstX, trnY, tstY, labels = split_dataset(data)
    # create_models(trnX, tstX, trnY, tstY, labels)
    # get_accuracy_models(tstX, tstY)
    # get_confusion_matrix_models(trnX, tstX, trnY, tstY, labels)

    # scores = finds_best_data_set_balance(trnX, tstX, trnY, tstY)
    # compare_balanced_scores(scores)


if __name__ == "__main__":
    main()
