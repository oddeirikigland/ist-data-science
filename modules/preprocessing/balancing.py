import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from matplotlib import pyplot as plt

from modules.classification.knn import knn_model
from modules.classification.naive_bayes import naive
from modules.functions import multiple_bar_chart, calculte_models_auc_score


def balance_plots(values):
    plot_dict = {}
    for key, value in values.items():
        plot_dict[key] = list(value.values())

    plt.figure()
    multiple_bar_chart(
        plt.gca(),
        list(values["Original"].keys()),
        plot_dict,
        "Comparison of balancing techniques",
        "Target value",
        "Number of records",
    )
    plt.show()


def compare_balanced_scores(scores):
    plt.figure()
    multiple_bar_chart(
        plt.gca(),
        ["Naive Bayes", "KNN"],
        scores,
        "Comparing balancing techniques",
        "Classifiers",
        "AUC Score",
    )
    plt.show()


def get_sample_df(
    unbalanced_df, output_df, output_class, values, key, value, max_class
):
    if key != output_class:
        df_sample = unbalanced_df.loc[unbalanced_df["Outcome"] == key]
        if not max_class and value > values["Original"][output_class]:
            df_sample = df_sample.sample(values["Original"][output_class])
        elif max_class and value < values["Original"][output_class]:
            df_sample = df_sample.sample(values["Original"][output_class], replace=True)
        output_df = output_df.append(df_sample)
    return output_df


def balancing_training_dataset(trnX, trnY):
    unbal = pd.DataFrame.from_records(trnX)
    unbal["Outcome"] = trnY
    target_count = unbal["Outcome"].value_counts()

    y_values = {}
    for y_value in target_count.index.values:
        y_values[y_value] = target_count[y_value]

    min_class = min(y_values, key=y_values.get)
    max_class = max(y_values, key=y_values.get)

    print(
        "Minority class: {} with {} values".format(min_class, target_count[min_class])
    )
    print(
        "Majority class: {} with {} values".format(max_class, target_count[max_class])
    )
    print("Proportion:", round(y_values[min_class] / y_values[max_class], 2), ": 1")

    values = {"Original": y_values}
    df_under_sample = unbal[unbal["Outcome"] == min_class]
    df_over_sample = unbal[unbal["Outcome"] == max_class]

    for key, value in values["Original"].items():
        df_under_sample = get_sample_df(
            unbalanced_df=unbal,
            output_df=df_under_sample,
            output_class=min_class,
            values=values,
            key=key,
            value=value,
            max_class=False,
        )

        df_over_sample = get_sample_df(
            unbalanced_df=unbal,
            output_df=df_over_sample,
            output_class=max_class,
            values=values,
            key=key,
            value=value,
            max_class=True,
        )

    values["UnderSample"] = dict(df_under_sample["Outcome"].value_counts())
    values["OverSample"] = dict(df_over_sample["Outcome"].value_counts())

    RANDOM_STATE = 42
    smote = SMOTE(ratio="minority", random_state=RANDOM_STATE)
    y = unbal.pop("Outcome").values
    X = unbal.values
    smote_x, smote_y = smote.fit_sample(X, y)
    smote_target_count = pd.Series(smote_y).value_counts()
    values["SMOTE"] = dict(smote_target_count)

    under_sample_y = df_under_sample.pop("Outcome").values
    under_sample_x = df_under_sample.values

    over_sample_y = df_over_sample.pop("Outcome").values
    over_sample_x = df_over_sample.values

    df_diff_balancing = {
        "original_x": trnX,
        "original_y": trnY,
        "under_sample_x": under_sample_x,
        "under_sample_y": under_sample_y,
        "over_sample_x": over_sample_x,
        "over_sample_y": over_sample_y,
        "smote_x": smote_x,
        "smote_y": smote_y,
    }

    return (df_diff_balancing, values)


def finds_best_data_set_balance(trnX, tstX, trnY, tstY, multi_class):
    scores = {}
    df_diff_balancing, values = balancing_training_dataset(trnX, trnY)

    unbalanced_naive_bayes = calculte_models_auc_score(
        naive, trnX, trnY, tstX, tstY, multi_class
    )
    unbalanced_knn = calculte_models_auc_score(
        knn_model, trnX, trnY, tstX, tstY, multi_class
    )
    scores["original"] = [unbalanced_naive_bayes, unbalanced_knn]

    under_sample_naive_bayes = calculte_models_auc_score(
        naive,
        df_diff_balancing["under_sample_x"],
        df_diff_balancing["under_sample_y"],
        tstX,
        tstY,
        multi_class,
    )
    under_sample_knn = calculte_models_auc_score(
        knn_model,
        df_diff_balancing["under_sample_x"],
        df_diff_balancing["under_sample_y"],
        tstX,
        tstY,
        multi_class,
    )
    scores["under_sample"] = [under_sample_naive_bayes, under_sample_knn]

    over_sample_naive_bayes = calculte_models_auc_score(
        naive,
        df_diff_balancing["over_sample_x"],
        df_diff_balancing["over_sample_y"],
        tstX,
        tstY,
        multi_class,
    )
    over_sample_knn = calculte_models_auc_score(
        knn_model,
        df_diff_balancing["over_sample_x"],
        df_diff_balancing["over_sample_y"],
        tstX,
        tstY,
        multi_class,
    )
    scores["over_sample"] = [over_sample_naive_bayes, over_sample_knn]

    smote_naive_bayes = calculte_models_auc_score(
        naive,
        df_diff_balancing["smote_x"],
        df_diff_balancing["smote_y"],
        tstX,
        tstY,
        multi_class,
    )
    smote_knn = calculte_models_auc_score(
        knn_model,
        df_diff_balancing["smote_x"],
        df_diff_balancing["smote_y"],
        tstX,
        tstY,
        multi_class,
    )
    scores["smote"] = [smote_naive_bayes, smote_knn]
    best_technique, best_technique_scores = get_best_balancing_score_and_df(scores)
    best_df_x, best_df_y = (
        df_diff_balancing["{}_x".format(best_technique)],
        df_diff_balancing["{}_y".format(best_technique)],
    )
    return best_technique, best_technique_scores, scores, values, best_df_x, best_df_y


def get_best_balancing_score_and_df(scores):
    max_score_key = max(scores, key=lambda k: sum(scores[k]))
    return max_score_key, scores[max_score_key]


if __name__ == "__main__":
    from constants import ROOT_DIR
    from modules.classification.all_models import split_dataset

    data = pd.read_csv("{}/data/df_without_corr.csv".format(ROOT_DIR))
    df = data.copy()
    trnX, tstX, trnY, tstY, labels = split_dataset(df, y_column_name="class")
    best_technique, best_technique_scores, scores, values, best_df_x, best_df_y = finds_best_data_set_balance(
        trnX, tstX, trnY, tstY, multi_class=False
    )
    balance_plots(values)
    print(scores)
    compare_balanced_scores(scores)

    data = pd.read_csv("{}/data/covtype.data".format(ROOT_DIR))
    df = data.copy()
    trnX, tstX, trnY, tstY, labels = split_dataset(df, y_column_name="Cover_Type")
    best_technique, best_technique_scores, scores, values, best_df_x, best_df_y = finds_best_data_set_balance(
        trnX, tstX, trnY, tstY, multi_class=True
    )
    balance_plots(values)
    print(scores)
    compare_balanced_scores(scores)
