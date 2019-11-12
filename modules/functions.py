import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import itertools
import json
import sys
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer

from constants import ROOT_DIR

CMAP = plt.cm.Blues


def choose_grid(nr):
    return nr // 4 + 1, 4


def line_chart(
    ax: plt.Axes,
    series: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    percentage=False,
):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if percentage:
        ax.set_ylim(0.0, 1.0)
    ax.plot(series)


def multiple_line_chart(
    ax: plt.Axes,
    xvalues: list,
    yvalues: dict,
    title: str,
    xlabel: str,
    ylabel: str,
    percentage=False,
):
    legend: list = []
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if percentage:
        ax.set_ylim(0.0, 1.0)

    for name, y in yvalues.items():
        ax.plot(xvalues, y)
        legend.append(name)
    ax.legend(legend, loc="best", fancybox=True, shadow=True)


def bar_chart(
    ax: plt.Axes,
    xvalues: list,
    yvalues: list,
    title: str,
    xlabel: str,
    ylabel: str,
    percentage=False,
):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(xvalues, rotation=90, fontsize="small")
    if percentage:
        ax.set_ylim(0.0, 1.0)
    ax.bar(xvalues, yvalues, edgecolor="grey")


def multiple_bar_chart(
    ax: plt.Axes,
    xvalues: list,
    yvalues: dict,
    title: str,
    xlabel: str,
    ylabel: str,
    percentage=False,
):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    x = np.arange(len(xvalues))  # the label locations
    ax.set_xticks(x)
    ax.set_xticklabels(xvalues, fontsize="small")
    if percentage:
        ax.set_ylim(0.0, 1.0)
    width = 0.8  # the width of the bars
    step = width / len(yvalues)
    k = 0
    for name, y in yvalues.items():
        ax.bar(x + k * step, y, step, label=name)
        k += 1
    ax.legend(
        loc="lower center",
        ncol=len(yvalues),
        bbox_to_anchor=(0.5, -0.2),
        fancybox=True,
        shadow=True,
    )


def save_model(model, filename):
    filename = "{}/modules/saved_models/{}.sav".format(ROOT_DIR, filename)
    pickle.dump(model, open(filename, "wb"))


def load_model_sav(filename):
    filename = "{}/modules/saved_models/{}.sav".format(ROOT_DIR, filename)
    loaded_model = pickle.load(open(filename, "rb"))
    return loaded_model


def plot_confusion_matrix(
    ax: plt.Axes, cnf_matrix: np.ndarray, classes_names: list, normalize: bool = False
):
    if normalize:
        total = cnf_matrix.sum(axis=1)[:, np.newaxis]
        cm = cnf_matrix.astype("float") / total
        title = "Normalized confusion matrix"
    else:
        cm = cnf_matrix
        title = "Confusion matrix"
    np.set_printoptions(precision=2)
    tick_marks = np.arange(0, len(classes_names), 1)
    ax.set_title(title)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes_names)
    ax.set_yticklabels(classes_names)
    ax.imshow(cm, interpolation="nearest", cmap=CMAP)

    fmt = ".2f" if normalize else "d"
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt), horizontalalignment="center")


def write_to_json(data, filename):
    with open("{}/data/{}".format(ROOT_DIR, filename), "w") as outfile:
        json.dump(data, outfile)


def read_from_json(filename):
    filename = "{}/data/{}".format(ROOT_DIR, filename)
    try:
        with open(filename) as json_file:
            data = json.load(json_file)
        return data
    except FileNotFoundError:
        print("{} does not exist".format(filename))
        sys.exit(1)
    except json.decoder.JSONDecodeError:
        print("{} is not a json file".format(filename))
        sys.exit(1)


def calculte_models_auc_score(classifier, trnX, trnY, tstX, tstY, multi_class=False):
    model = classifier(trnX, trnY)
    pred_y = model.predict(tstX)
    return (
        multi_class_roc_auc_score(tstY, pred_y)
        if multi_class
        else roc_auc_score(tstY, pred_y)
    )


def multi_class_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)


def print_table(table):
    longest_cols = [
        (max([len(str(row[i])) for row in table]) + 3) for i in range(len(table[0]))
    ]
    row_format = "".join(
        ["{:>" + str(longest_col) + "}" for longest_col in longest_cols]
    )
    for row in table:
        print(row_format.format(*row))
