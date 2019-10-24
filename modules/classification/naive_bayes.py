import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import pylab as pl
from constants import ROOT_DIR
from modules.functions import bar_chart


def naive():
    data = pd.read_csv("{}/data/df_without_corr.csv".format(ROOT_DIR))
    df = data.copy()

    y: np.ndarray = df.pop('class').values
    X: np.ndarray = df.values
    labels = pd.unique(y)

    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

    clf = BernoulliNB()
    clf.fit(trnX, trnY)
    prdY = clf.predict(tstX)
    cnf_mtx = metrics.confusion_matrix(tstY, prdY, labels)
    pl.matshow(cnf_mtx)
    pl.title('Confusion matrix')
    pl.xlabel("True label")
    pl.ylabel("predicted label ")
    pl.colorbar()


    estimators = {'GaussianNB': GaussianNB(),
                  #  'MultinomialNB': MultinomialNB(),
                  'BernoulyNB': BernoulliNB()}

    xvalues = []
    yvalues = []
    for clf in estimators:
        xvalues.append(clf)
        estimators[clf].fit(trnX, trnY)
        prdY = estimators[clf].predict(tstX)
        yvalues.append(metrics.accuracy_score(tstY, prdY))

    plt.figure()
    bar_chart(plt.gca(), xvalues, yvalues, 'Comparison of Naive Bayes Models', '', 'accuracy', percentage=True)
    plt.show()




naive()

