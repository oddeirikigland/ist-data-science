import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from subprocess import call

from constants import ROOT_DIR
from modules.functions import multiple_line_chart


def decision_tree():
    data: pd.DataFrame = pd.read_csv("{}/data/df_without_corr.csv".format(ROOT_DIR))
    y: np.ndarray = data.pop("class").values
    X: np.ndarray = data.values
    labels = pd.unique(y)

    trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)

    min_samples_leaf = [0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001]
    max_depths = [5, 10, 25, 50]
    criteria = ["entropy", "gini"]

    plt.figure()
    fig, axs = plt.subplots(1, 2, figsize=(16, 4), squeeze=False)
    for k in range(len(criteria)):
        f = criteria[k]
        values = {}
        for d in max_depths:
            yvalues = []
            for n in min_samples_leaf:
                tree = DecisionTreeClassifier(
                    min_samples_leaf=n, max_depth=d, criterion=f
                )
                tree.fit(trnX, trnY)
                prdY = tree.predict(tstX)
                yvalues.append(metrics.accuracy_score(tstY, prdY))
            values[d] = yvalues
            multiple_line_chart(
                axs[0, k],
                min_samples_leaf,
                values,
                "Decision Trees with %s criteria" % f,
                "nr estimators",
                "accuracy",
                percentage=True,
            )

    tree = DecisionTreeClassifier(max_depth=15)
    tree.fit(trnX, trnY)

    dot_data = export_graphviz(
        tree,
        out_file="{}/data/dtree.dot".format(ROOT_DIR),
        filled=True,
        rounded=True,
        special_characters=True,
    )
    # Convert to png

    call(
        [
            "dot",
            "-Tpng",
            "{}/data/dtree.dot".format(ROOT_DIR),
            "-o",
            "{}/data/dtree.png".format(ROOT_DIR),
            "-Gdpi=600",
        ]
    )

    plt.figure(figsize=(14, 18))
    plt.imshow(plt.imread("dtree.png"))
    plt.axis("off")
    plt.show()


def main():
    decision_tree()


if __name__ == "__main__":
    main()
