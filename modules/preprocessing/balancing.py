import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from matplotlib import pyplot as plt

from modules.functions import multiple_bar_chart


def balance_plots(target_count, ind_min_class, values):
    plt.figure()
    plt.title("Class balance")
    plt.bar(target_count.index, target_count.values)

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


def balancing_training_dataset(trnX, trnY):
    unbal = pd.DataFrame.from_records(trnX)
    unbal["Outcome"] = trnY
    target_count = unbal["Outcome"].value_counts()
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

    df_class_min = unbal[unbal["Outcome"] == min_class]
    df_class_max = unbal[unbal["Outcome"] != min_class]

    df_under = df_class_max.sample(len(df_class_min))
    values["UnderSample"] = [target_count.values[ind_min_class], len(df_under)]
    under_sample_df = df_under.add(df_class_min, fill_value=0)

    df_over = df_class_min.sample(len(df_class_max), replace=True)
    values["OverSample"] = [len(df_over), target_count.values[1 - ind_min_class]]
    over_sample_df = df_over.add(df_class_max, fill_value=0)

    smote = SMOTE(ratio="minority", random_state=RANDOM_STATE)
    y = unbal.pop("Outcome").values
    X = unbal.values
    smote_x, smote_y = smote.fit_sample(X, y)
    smote_target_count = pd.Series(smote_y).value_counts()
    values["SMOTE"] = [
        smote_target_count.values[ind_min_class],
        smote_target_count.values[1 - ind_min_class],
    ]

    # balance_plots(target_count, ind_min_class, values)

    under_sample_y = under_sample_df.pop("Outcome").values
    under_sample_x = under_sample_df.values

    over_sample_y = over_sample_df.pop("Outcome").values
    over_sample_x = over_sample_df.values

    return (
        under_sample_y,
        under_sample_x,
        over_sample_y,
        over_sample_x,
        smote_x,
        smote_y,
    )
