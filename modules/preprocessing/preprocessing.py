import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import register_matplotlib_converters

from constants import ROOT_DIR

register_matplotlib_converters()


def df_info(df):
    print(df.shape)
    print(df.head().to_string())
    print(df.columns)
    print(df.dtypes)  # int64 or float64
    print("Columns with null vals: {}".format(find_number_of_null_values(df)))
    print(df.describe().to_string())


def find_number_of_null_values(data):
    mv = {}
    missing_values = []
    for var in data:
        count = data[var].isna().sum()
        mv[var] = count
        if count != 0:
            missing_values.append([var, count])
    return missing_values


def calc_corr(df):
    for i in range(0, 385, 77):
        test_df = df.copy()
        newdf = test_df[df.columns[i : i + 76]]
        data = newdf.copy()
        corr = data.corr()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        fig.colorbar(cax)
        ticks = np.arange(0, len(data.columns), 1)
        ax.set_xticks(ticks)
        plt.xticks(rotation=90)
        ax.set_yticks(ticks)
        ax.set_xticklabels(data.columns)
        ax.set_yticklabels(data.columns)
    plt.show()


def find_corr_var(df, jumps):
    threshold = 0.8
    col_corr = set()
    for i in range(0, 756, jumps):
        test_df = df.copy()
        newdf = test_df[test_df.columns[i : i + jumps - 1]]
        corr_matrix = newdf.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if (
                    (corr_matrix.iloc[i, j] >= threshold)
                    or (corr_matrix.iloc[i, j] <= -threshold)
                ) and (corr_matrix.columns[j] not in col_corr):
                    colname = corr_matrix.columns[i]  # getting the name of column
                    col_corr.add(colname)
                    if colname in test_df.columns:
                        del test_df[colname]  # deleting the column from the dataset
    print(col_corr)
    return col_corr


def remove_cols(df, jumps=12):
    rm_df = df.copy()
    rm_list = find_corr_var(df, jumps)
    for colname in rm_list:
        try:
            print(colname)
            rm_df = rm_df.drop(colname, 1)
        except:
            print("colname: {} -- not in list".format(colname))
    return rm_df


def main():
    dataframe = pd.read_csv("{}/data/pd_speech_features.csv".format(ROOT_DIR))

    df = dataframe.copy()
    print(df.iloc[0])
    print(df.columns)
    groups = df.drop(axis=0, index=0, inplace=False)
    # print(groups)

    """
    df = dataframe.copy()
    df.head(10)
    normalized_df = (df - df.mean()) / df.std()
    rm_df = normalized_df.copy()
    # df_info(df)
    # calc_corr(df)
    # rm_list = find_corr_var(df)
    smaller_df = remove_cols(rm_df)

    # rm2_list = find_corr_var(smaller_df, 8)
    even_smaller = remove_cols(smaller_df, 8)

    print(even_smaller.shape)
    print(rm_df.head().to_string())

    calc_corr(even_smaller)
    """


if __name__ == "__main__":
    main()
