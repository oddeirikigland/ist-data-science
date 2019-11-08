import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from subprocess import call

from constants import ROOT_DIR
from modules.functions import multiple_line_chart, save_model


def decision_tree(trnX, trnY):
    tree = DecisionTreeClassifier(max_depth=5)
    tree.fit(trnX, trnY)
    save_model(tree, "decision_tree")
    return tree


def plot_accuracy(trnX, tstX, trnY, tstY):
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
    plt.show()


def plot_tree_structure(tree):
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
    plt.imshow(plt.imread("{}/data/dtree.png".format(ROOT_DIR)))
    plt.axis("off")
    plt.show()
