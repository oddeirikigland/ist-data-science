import pandas as pd

from modules.preprocessing.feature_selection import reduce_df_feature_selection
from modules.preprocessing.preprocessing import normalize_df


def preprocessing_report(data, source):
    df = data.copy()

    print("1. Applies preprocessing:")
    df[["class", "id", "gender"]] = df[["class", "id", "gender"]].apply(pd.to_numeric)
    df = normalize_df(df)
    print(" 1.1 Normalization")
    print(" 1.2 Feature selection")
    best_score, best_number_features, df = reduce_df_feature_selection(df)
    print(
        "     a) Best number of features is {}, with an AUC score of {:.2f}".format(
            best_number_features, best_score
        )
    )
    print(
        "     b) Dataframe reduced from {} columns to {} columns".format(
            len(data.columns), len(df.columns)
        )
    )
    return df
