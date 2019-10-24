import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import register_matplotlib_converters
from sklearn.preprocessing import Normalizer
from imblearn.over_sampling import SMOTE, RandomOverSampler

from constants import ROOT_DIR
from modules.functions import multiple_bar_chart

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


def plot_correlation(df):
    data = df.copy()
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


def find_corr_var(df):
    threshold = 0.8
    col_corr = set()
    for i in range(0, 756):
        newdf = df.copy()
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


def remove_cols_with_correlation(df):
    rm_df = df.copy()
    rm_list = find_corr_var(df)
    for colname in rm_list:
        try:
            print(colname)
            rm_df = rm_df.drop(colname, 1)
        except:
            print("colname: {} -- not in list".format(colname))
    return rm_df


def get_start_end_groups(df):
    group_names = list(df)
    out = []
    start = group_names[0]
    previously = group_names[1]
    for i in range(2, len(group_names)):
        if "Unnamed: " not in group_names[i] and "Unnamed: " in previously:
            # group finished
            out.append([start, previously])
            start = group_names[i]
        previously = group_names[i]

    # Split last columns in two
    out.append([start, group_names[538]])
    out.append([group_names[539], group_names[len(group_names) - 2]])
    return out


def smaller_df(dataframe):
    df = dataframe.copy()

    # Returns start and end of each group
    start_end_group = get_start_end_groups(df)
    print(start_end_group)

    # Start with columns that not can be pruned
    df_without_corr = df.copy()
    df_without_corr.columns = df_without_corr.iloc[0]
    df_without_corr = df_without_corr.drop(df_without_corr.index[0])
    df_without_corr = df_without_corr[["id", "gender", "class"]]

    # Checks corr in each group, except the first which is just id and gender
    for elem in start_end_group[1:]:
        group_df = df.loc[:, elem[0] : elem[1]]

        # Sets column names
        group_df.columns = group_df.iloc[0]
        group_df = group_df.drop(group_df.index[0])
        group_df = group_df.apply(
            pd.to_numeric, errors="coerce"
        )  # convert from object to float

        # Update df by removing correlated values
        group_df = remove_cols_with_correlation(group_df)
        print(group_df)
        # Adds correlated values to out df
        df_without_corr = pd.concat([df_without_corr, group_df], axis=1)
        print(df_without_corr)

    print(df_without_corr.head().to_string())
    df_without_corr.to_csv(
        index=True, path_or_buf="{}/data/df_without_corr.csv".format(ROOT_DIR)
    )


def normalize_df(df_nr):
    df_nr = df_nr.copy()
    df_not_normalize = df_nr[["id", "gender", "class"]]

    df_nr = df_nr.drop(["id", "gender", "class"], axis=1)
    print(df_nr.head().to_string())
    transf = Normalizer().fit(df_nr)
    df_nr = pd.DataFrame(transf.transform(df_nr, copy=True), columns=df_nr.columns)
    df_nr = pd.concat([df_not_normalize, df_nr], axis=1)
    print(df_nr.head().to_string())
    return df_nr


def class_balance(dataframe):
    df_nr = dataframe.copy()
    unbal = df_nr
    target_count = unbal["class"].value_counts()
    plt.figure()
    plt.title("Class balance")
    plt.bar(target_count.index, target_count.values)

    min_class = target_count.idxmin()
    ind_min_class = target_count.index.get_loc(min_class)

    print("Minority class:", target_count[ind_min_class])
    print("Majority class:", target_count[1 - ind_min_class])
    print(
        "Proportion:",
        round(target_count[ind_min_class] / target_count[1 - ind_min_class], 2),
        ": 1",
    )

    RANDOM_STATE = 42
    values = {
        "Original": [
            target_count.values[ind_min_class],
            target_count.values[1 - ind_min_class],
        ]
    }

    df_class_min = unbal[unbal["class"] == min_class]
    df_class_max = unbal[unbal["class"] != min_class]

    df_under = df_class_max.sample(len(df_class_min))
    values["UnderSample"] = [target_count.values[ind_min_class], len(df_under)]

    df_over = df_class_min.sample(len(df_class_max), replace=True)
    values["OverSample"] = [len(df_over), target_count.values[1 - ind_min_class]]

    smote = SMOTE(ratio="minority", random_state=RANDOM_STATE)
    y = unbal.pop("class").values
    X = unbal.values
    _, smote_y = smote.fit_sample(X, y)
    smote_target_count = pd.Series(smote_y).value_counts()
    values["SMOTE"] = [
        smote_target_count.values[ind_min_class],
        smote_target_count.values[1 - ind_min_class],
    ]

    plt.figure()
    multiple_bar_chart(
        plt.gca(),
        [target_count.index[ind_min_class], target_count.index[1 - ind_min_class]],
        values,
        "Target",
        "frequency",
        "Class balance",
    )
    plt.show()


def main():
    dataframe = pd.read_csv("{}/data/pd_speech_features.csv".format(ROOT_DIR))
    # smaller_df = smaller_df(dataframe)

    # Result from smaller_df()
    # dataframe = pd.read_csv("{}/data/df_without_corr.csv".format(ROOT_DIR))

    # Plots class balance
    # class_balance(dataframe)


if __name__ == "__main__":
    main()
