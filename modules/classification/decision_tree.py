import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from subprocess import call

from constants import ROOT_DIR
from modules.functions import multiple_line_chart, save_model, calculte_models_auc_score


def decision_tree(
    trnX, trnY, samples_leaf=0.05, depth=5, criterion="entropy", save_file=False
):
    tree = DecisionTreeClassifier(
        min_samples_leaf=samples_leaf, max_depth=depth, criterion=criterion
    )
    tree.fit(trnX, trnY)
    if save_file:
        save_model(tree, "decision_tree")
    return tree


def dt_plot_accuracy(trnX, tstX, trnY, tstY, multi_class, plot=False):
    best_samples_leaf = 0
    best_depth = 0
    best_criteria = ""
    best_score = 0
    best_model = None
    min_samples_leaf = [0.05, 0.025, 0.01, 0.0075, 0.005, 0.0025, 0.001]
    max_depths = [5, 10, 25, 50]
    criteria = ["entropy", "gini"]
    if plot:
        plt.figure()
        fig, axs = plt.subplots(1, 2, figsize=(16, 4), squeeze=False)
    for k in range(len(criteria)):
        f = criteria[k]
        values = {}
        for d in max_depths:
            yvalues = []
            for n in min_samples_leaf:
                tree = decision_tree(trnX, trnY, samples_leaf=n, depth=d, criterion=f)
                score = calculte_models_auc_score(tree, tstX, tstY, multi_class)
                if score > best_score:
                    best_score = score
                    best_depth = d
                    best_criteria = f
                    best_samples_leaf = n
                    best_model = tree
                yvalues.append(score)
            values[d] = yvalues
            if plot:
                multiple_line_chart(
                    axs[0, k],
                    min_samples_leaf,
                    values,
                    "Decision Trees with %s criteria" % f,
                    "min sample leaf",
                    "Sensitivity",
                    percentage=True,
                )
    if plot:
        plt.show()
    return best_model, best_score, best_samples_leaf, best_depth, best_criteria


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
