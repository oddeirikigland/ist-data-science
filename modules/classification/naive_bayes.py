import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import pylab as pl

from modules.functions import bar_chart, save_model


def naive(trnX, trnY, save_file=False):
    clf = BernoulliNB()
    clf.fit(trnX, trnY)
    if save_file:
        save_model(clf, "naive_bayes")
    return clf


def test_different_params(trnX, tstX, trnY, tstY, labels):
    clf = naive(trnX, trnY)
    prdY = clf.predict(tstX)
    cnf_mtx = metrics.confusion_matrix(tstY, prdY, labels)
    pl.matshow(cnf_mtx)
    pl.title("Confusion matrix")
    pl.xlabel("True label")
    pl.ylabel("predicted label ")
    pl.colorbar()
    estimators = {
        "GaussianNB": GaussianNB(),
        #  'MultinomialNB': MultinomialNB(),
        "BernoulyNB": BernoulliNB(),
    }
    xvalues = []
    yvalues = []
    for clf in estimators:
        xvalues.append(clf)
        estimators[clf].fit(trnX, trnY)
        prdY = estimators[clf].predict(tstX)
        yvalues.append(metrics.accuracy_score(tstY, prdY))

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
