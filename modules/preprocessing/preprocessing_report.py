import pandas as pd

from modules.classification.all_models import split_dataset
from modules.preprocessing.balancing import (
    balancing_training_dataset,
    finds_best_data_set_balance,
)
from modules.preprocessing.feature_selection import reduce_df_feature_selection
from modules.preprocessing.preprocessing import normalize_df


def preprocessing_report(data, source):
    df = data.copy()

    print("1. Applies preprocessing:")
    if source == "PD":
        target_name = "class"
        print(" 1.1 Normalization")
        df[["class", "id", "gender"]] = df[["class", "id", "gender"]].apply(
            pd.to_numeric
        )
        df = normalize_df(df, columns_not_to_normalize=["id", "gender", "class"])

        print(" 1.2 Balancing")
        trnX, tstX, trnY, tstY, labels = split_dataset(df)
        scores, values = finds_best_data_set_balance(
            trnX, tstX, trnY, tstY, multi_class=False
        )
        # TODO: print scores from balancing comparison more informative
        print(scores)

        print(" 1.3 Feature selection")
        best_score, best_number_features, df = reduce_df_feature_selection(
            df, y_column_name=target_name
        )
        print(
            "   a) Best number of features is {}, with an AUC score of {:.2f}".format(
                best_number_features, best_score
            )
        )
        print(
            "   b) Dataframe reduced from {} columns to {} columns".format(
                len(data.columns), len(df.columns)
            )
        )
    else:
        print(" 1.1 Normalization")
        df = normalize_df(df, columns_not_to_normalize=["Cover_Type"])

        print(" 1.2 Balancing")
        trnX, tstX, trnY, tstY, labels = split_dataset(df, y_column_name="Cover_Type")
        scores, values = finds_best_data_set_balance(
            trnX, tstX, trnY, tstY, multi_class=True
        )
        # TODO: print scores from balancing comparison more informative
        print(scores)

    return df
