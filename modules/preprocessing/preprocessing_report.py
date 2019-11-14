import pandas as pd

from modules.classification.all_models import split_dataset
from modules.preprocessing.balancing import finds_best_data_set_balance
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
        print("Normalization completed")

        print(" 1.2 Feature selection")
        # TODO: reduce step size before submit project
        best_score, best_number_features, df = reduce_df_feature_selection(
            df, y_column_name=target_name, step_size=700
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

        print(" 1.3 Balancing")
        trnX, tstX, trnY, tstY, labels = split_dataset(df)
        best_technique, best_technique_scores, scores, values, best_df_x, best_df_y = finds_best_data_set_balance(
            trnX, tstX, trnY, tstY, multi_class=False
        )
        print("Best balancing technique is {}".format(best_technique))
        print("Continuing using the data set balanced by {}".format(best_technique))
        trnX = best_df_x.copy()
        trnY = best_df_y.copy()

    else:
        print(" 1.1 Normalization")
        df = normalize_df(df, columns_not_to_normalize=["Cover_Type"])
        print("Normalization completed")

        print(" 1.2 Balancing")
        trnX, tstX, trnY, tstY, labels = split_dataset(df, y_column_name="Cover_Type")
        best_technique, best_technique_scores, scores, values, best_df_x, best_df_y = finds_best_data_set_balance(
            trnX, tstX, trnY, tstY, multi_class=True
        )
        print("Best balancing technique is {}".format(best_technique))
        print("Continuing using the data set balanced by {}".format(best_technique))
        trnX = best_df_x.copy()
        trnY = best_df_y.copy()

    print("Preprocessing completed")
    return df, trnX, tstX, trnY, tstY
