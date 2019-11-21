import pandas as pd
from sklearn.model_selection import KFold

from modules.classification.all_models import split_dataset
from modules.preprocessing.balancing import finds_best_data_set_balance, balancing_training_dataset
from modules.preprocessing.feature_selection import reduce_df_feature_selection
from modules.preprocessing.preprocessing import normalize_df


def preprocessing_report(data, source):
    df = data.copy()
    fold = []

    print("1. Applies preprocessing:")
    if source == "PD":
        target_name = "class"

        print(" 1.1 Normalization")
        df[["class", "id", "gender"]] = df[["class", "id", "gender"]].apply(
            pd.to_numeric
        )
        df = normalize_df(df, columns_not_to_normalize=["id", "gender", "class"])

        print(" 1.2 Use median of values for same person")
        df = df.groupby(by="id").median().reset_index()

        print(" 1.3 Feature selection")
        # TODO: reduce step size before submit project
        best_score, best_number_features, df = reduce_df_feature_selection(
            df, y_column_name=target_name, step_size=20
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

        print(" 1.4 Balancing")
        df_fold = df.copy()
        y = df_fold.pop("class").values
        X = df_fold.values
        labels = pd.unique(y)
        kf = KFold(n_splits=10, shuffle=True)
        for train_index, test_index in kf.split(X):
            trnX, tstX = X[train_index], X[test_index]
            trnY, tstY = y[train_index], y[test_index]
            best_technique, best_technique_scores, scores, values, best_df_x, best_df_y = finds_best_data_set_balance(
                trnX, tstX, trnY, tstY, multi_class=False, print_stuff=False
            )
            trnX = best_df_x.copy()
            trnY = best_df_y.copy()
            fold.append([trnX, tstX, trnY, tstY])

        trnX, tstX, trnY, tstY, labels = split_dataset(df)
        best_technique, best_technique_scores, scores, values, best_df_x, best_df_y = finds_best_data_set_balance(
            trnX, tstX, trnY, tstY, multi_class=False
        )
        trnX = best_df_x.copy()
        trnY = best_df_y.copy()
        print("Best balancing technique is {}".format(best_technique))
        print("Continuing using the data set balanced by {}".format(best_technique))

    else:
        print(" 1.1 Normalization")
        df = normalize_df(df, columns_not_to_normalize=["Cover_Type"])
        print("Normalization completed")

        print(" 1.2 Balancing")
        df_fold = df.copy()
        y = df_fold.pop("Cover_Type").values
        X = df_fold.values
        labels = pd.unique(y)
        kf = KFold(n_splits=10, shuffle=True)
        for train_index, test_index in kf.split(X):
            trnX, tstX = X[train_index], X[test_index]
            trnY, tstY = y[train_index], y[test_index]

            # best_technique, best_technique_scores, scores, values, best_df_x, best_df_y =
            df_diff_balancing, values = balancing_training_dataset(trnX, trnY, print_stuff=False)
            trnX = df_diff_balancing["under_sample_x"].copy()
            trnY = df_diff_balancing["under_sample_y"].copy()
            fold.append([trnX, tstX, trnY, tstY])

        trnX, tstX, trnY, tstY, labels = split_dataset(df, y_column_name="Cover_Type")
        best_technique, best_technique_scores, scores, values, best_df_x, best_df_y = finds_best_data_set_balance(
            trnX, tstX, trnY, tstY, multi_class=True
        )
        trnX = best_df_x.copy()
        trnY = best_df_y.copy()
        print("Best balancing technique is {}".format(best_technique))
        print("Continuing using the data set balanced by {}".format(best_technique))

    print("Preprocessing completed")
    return df, trnX, tstX, trnY, tstY, labels, fold
