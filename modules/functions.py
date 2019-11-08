import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import itertools

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
