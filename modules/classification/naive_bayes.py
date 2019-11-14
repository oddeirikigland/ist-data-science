import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import pylab as pl

from modules.functions import bar_chart, save_model, calculte_models_auc_score


def naive(trnX, trnY, estimator=BernoulliNB, save_file=False):
    clf = estimator()
    clf.fit(trnX, trnY)
    if save_file:
        save_model(clf, "naive_bayes")
    return clf


def naive_test_different_params(trnX, tstX, trnY, tstY, multi_class=False, plot=False):
    best_estimator = ""
    best_score = 0
    best_model = None
    estimators = {
        "GaussianNB": GaussianNB,
        #  'MultinomialNB': MultinomialNB(),
        "BernoulyNB": BernoulliNB,
    }
    xvalues = []
    yvalues = []
    for key, value in estimators.items():
        xvalues.append(key)
        model = naive(trnX, trnY, estimator=value)
        score = calculte_models_auc_score(model, tstX, tstY, multi_class)
        yvalues.append(score)
        if score > best_score:
            best_score = score
            best_estimator = key
            best_model = model

    if plot:
        plt.figure()
        bar_chart(
            plt.gca(),
            xvalues,
            yvalues,
            "Comparison of Naive Bayes Models",
            "",
            "accuracy",
            percentage=True,
        )
        plt.show()
    return best_model, best_score, best_estimator
