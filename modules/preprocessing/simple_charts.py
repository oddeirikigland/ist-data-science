import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
import seaborn as sns
import scipy.stats as _stats

from constants import ROOT_DIR
from modules.functions import bar_chart, choose_grid, multiple_line_chart

register_matplotlib_converters()


def compute_known_distributions(x_values, n_bins) -> dict:
    distributions = dict()
    # Gaussian
    mean, sigma = _stats.norm.fit(x_values)
    distributions["Normal(%.1f,%.2f)" % (mean, sigma)] = _stats.norm.pdf(
        x_values, mean, sigma
    )
    # LogNorm
    #  sigma, loc, scale = _stats.lognorm.fit(x_values)
    #  distributions['LogNor(%.1f,%.2f)'%(np.log(scale),sigma)] = _stats.lognorm.pdf(x_values, sigma, loc, scale)
    # Exponential
    loc, scale = _stats.expon.fit(x_values)
    distributions["Exp(%.2f)" % (1 / scale)] = _stats.expon.pdf(x_values, loc, scale)
    # SkewNorm
    # a, loc, scale = _stats.skewnorm.fit(x_values)
    # distributions['SkewNorm(%.2f)'%a] = _stats.skewnorm.pdf(x_values, a, loc, scale)
    return distributions


def histogram_with_distributions(ax: plt.Axes, series: pd.Series, var: str):
    values = series.sort_values().values
    n, bins, patches = ax.hist(values, 20, density=True, edgecolor="grey")
    distributions = compute_known_distributions(values, bins)
    multiple_line_chart(
        ax, values, distributions, "Best fit for %s" % var, var, "probability"
    )


def plots(data):
    fig = plt.figure(figsize=(10, 7))
    mv = {}
    for var in data:
        mv[var] = data[var].isna().sum()
        bar_chart(
            plt.gca(),
            mv.keys(),
            mv.values(),
            "Number of missing values per variable",
            var,
            "nr. missing values",
        )
    fig.tight_layout()

    columns = data.select_dtypes(include="number").columns
    rows, cols = choose_grid(len(columns))
    plt.figure()
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False)
    i, j = 0, 0
    for n in range(len(columns)):
        axs[i, j].set_title("Boxplot for %s" % columns[n])
        axs[i, j].boxplot(data[columns[n]].dropna().values)
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    fig.tight_layout()

    columns = data.select_dtypes(include="number").columns
    rows, cols = choose_grid(len(columns))
    plt.figure()
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False)
    i, j = 0, 0
    for n in range(len(columns)):
        axs[i, j].set_title("Histogram for %s" % columns[n])
        axs[i, j].set_xlabel(columns[n])
        axs[i, j].set_ylabel("probability")
        axs[i, j].hist(data[columns[n]].dropna().values, "auto")
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    fig.tight_layout()

    columns = data.select_dtypes(include="number").columns
    rows, cols = choose_grid(len(columns))
    plt.figure()
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False)
    i, j = 0, 0
    for n in range(len(columns)):
        axs[i, j].set_title("Histogram with trend for %s" % columns[n])
        axs[i, j].set_ylabel("probability")
        sns.distplot(
            data[columns[n]].dropna().values,
            norm_hist=True,
            ax=axs[i, j],
            axlabel=columns[n],
        )
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    fig.tight_layout()

    columns = data.select_dtypes(include="number").columns
    rows = len(columns)
    cols = 5
    plt.figure()
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False)
    bins = range(5, 100, 20)
    for i in range(len(columns)):
        for j in range(len(bins)):
            axs[i, j].set_title("Histogram for %s" % columns[i])
            axs[i, j].set_xlabel(columns[i])
            axs[i, j].set_ylabel("probability")
            axs[i, j].hist(data[columns[i]].dropna().values, bins[j])
    fig.tight_layout()

    columns = data.select_dtypes(include="number").columns
    rows, cols = choose_grid(len(columns))
    plt.figure()
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False)
    i, j = 0, 0
    for n in range(len(columns)):
        histogram_with_distributions(axs[i, j], data[columns[n]].dropna(), columns[n])
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    fig.tight_layout()
    plt.show()


def main():
    dataframe = pd.read_csv("{}/data/pd_speech_features.csv".format(ROOT_DIR))
    df = dataframe.copy()
    baseline = [
        "PPE",
        "DFA",
        "RPDE",
        "numPulses",
        "numPeriodsPulses",
        "meanPeriodPulses",
        "stdDevPeriodPulses",
        "locPctJitter",
    ]
    baseline_features = df[baseline]
    plots(baseline_features)


if __name__ == "__main__":
    main()
