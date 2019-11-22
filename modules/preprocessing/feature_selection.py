import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2, RFE, SelectFromModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from constants import ROOT_DIR
from modules.functions import (
    write_to_json,
    read_from_json,
    bar_chart,
    line_chart,
    calculte_models_auc_score,
)
from modules.classification.all_models import split_dataset
from modules.classification.naive_bayes import naive
from modules.classification.knn import knn_model
from modules.classification.decision_tree import decision_tree


def cor_selector(X, y, num_feats, feature_name):
    cor_list = []
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:, np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature


def chi_squared(X, X_norm, y, num_feats):
    chi_selector = SelectKBest(chi2, k=num_feats)
    chi_selector.fit(X_norm, y)
    chi_support = chi_selector.get_support()
    chi_feature = X.loc[:, chi_support].columns.tolist()
    return chi_support, chi_feature


def recursive_feature_elimination(X, X_norm, y, num_feats):
    rfe_selector = RFE(
        estimator=LogisticRegression(solver="lbfgs"),
        n_features_to_select=num_feats,
        step=1000,
        verbose=0,
    )
    rfe_selector.fit(X_norm, y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = X.loc[:, rfe_support].columns.tolist()
    return rfe_support, rfe_feature


def lasso_select_from_model(X, X_norm, y, num_feats):
    embeded_lr_selector = SelectFromModel(
        LogisticRegression(solver="lbfgs"), max_features=num_feats
    )
    embeded_lr_selector.fit(X_norm, y)
    embeded_lr_support = embeded_lr_selector.get_support()
    embeded_lr_feature = X.loc[:, embeded_lr_support].columns.tolist()
    return embeded_lr_support, embeded_lr_feature


def tree_based_select_from_model(X, y, num_feats):
    embeded_rf_selector = SelectFromModel(
        RandomForestClassifier(n_estimators=100), max_features=num_feats
    )
    embeded_rf_selector.fit(X, y)

    embeded_rf_support = embeded_rf_selector.get_support()
    embeded_rf_feature = X.loc[:, embeded_rf_support].columns.tolist()
    return embeded_rf_support, embeded_rf_feature


def sorts_df_by_features(data, y_column_name, number_features):
    df = data.copy()
    y_data = df[y_column_name]
    x_data = df.drop([y_column_name], axis=1)

    x_norm = MinMaxScaler().fit_transform(x_data)
    feature_name = x_data.columns.tolist()

    cor_support, cor_feature = cor_selector(
        x_data, y_data, number_features, feature_name
    )
    chi_support, chi_feature = chi_squared(x_data, x_norm, y_data, number_features)
    rfe_support, rfe_feature = recursive_feature_elimination(
        x_data, x_norm, y_data, number_features
    )
    embedded_lr_support, embedded_lr_feature = lasso_select_from_model(
        x_data, x_norm, y_data, number_features
    )
    embedded_rf_support, embedded_rf_feature = tree_based_select_from_model(
        x_data, y_data, number_features
    )

    # put all selection together
    feature_selection_df = pd.DataFrame(
        {
            "Feature": feature_name,
            "Pearson": cor_support,
            "Chi-2": chi_support,
            "RFE": rfe_support,
            "Logistics": embedded_lr_support,
            "Random Forest": embedded_rf_support,
        }
    )
    # count the selected times for each feature
    feature_selection_df["Total"] = np.sum(feature_selection_df, axis=1)
    # sorts df
    feature_selection_df = feature_selection_df.sort_values(
        ["Total", "Feature"], ascending=False
    )
    feature_selection_df.index = range(1, len(feature_selection_df) + 1)
    return feature_selection_df["Feature"].head(number_features).tolist()


def find_best_feature_sub_sets(
    data, y_column_name, step_size=10, verbose=False, save_to_file=False
):
    df = data.copy()
    feature_sets = {}
    for i in range(1, len(df.columns), step_size):
        feature_sets[str(i)] = sorts_df_by_features(
            df, y_column_name, number_features=i
        )
        if verbose:
            print(i)
    if save_to_file:
        write_to_json(feature_sets, "feature_sets")
    return feature_sets


def get_scores_from_feature_sub_sets(
    df, feature_sets, y_column_name, classifier=naive, save_to_file=False
):
    trnX, tstX, trnY, tstY, labels = split_dataset(df)
    best_score = 0.51
    best_number_features = 0
    feature_sub_set_scores = {}
    df_without_class = df.drop([y_column_name], axis=1)
    trn_x_df = pd.DataFrame.from_records(trnX, columns=df_without_class.columns)
    tst_x_df = pd.DataFrame.from_records(tstX, columns=df_without_class.columns)

    for key, value in feature_sets.items():
        trn_x_feature_select = trn_x_df[value].to_numpy()
        tst_x_feature_select = tst_x_df[value].to_numpy()

        model = classifier(trn_x_feature_select, trnY)
        score = calculte_models_auc_score(model, tst_x_feature_select, tstY)
        feature_sub_set_scores[key] = score
        if score > best_score:
            best_score = score
            best_number_features = key
    if save_to_file:
        write_to_json(feature_sub_set_scores, "feature_sub_set_scores")
    return best_score, best_number_features, feature_sets[best_number_features]


def plots_feature_sub_set_scores(feature_sub_set_scores):
    plt.figure()
    line_chart(
        ax=plt.gca(),
        xvalues=list(feature_sub_set_scores.keys()),
        yvalues=list(feature_sub_set_scores.values()),
        title="AUC score based on number of features used",
        xlabel="Number of features",
        ylabel="AUC score",
        percentage=False,
    )
    plt.show()


def reduce_df_feature_selection(data, y_column_name, step_size=10):
    df = data.copy()
    features_in_sub_sets = find_best_feature_sub_sets(
        df, y_column_name, step_size=step_size
    )
    best_score, best_number_features, best_feature_sets = get_scores_from_feature_sub_sets(
        df, features_in_sub_sets, y_column_name
    )
    return best_score, best_number_features, data[best_feature_sets + [y_column_name]]


if __name__ == "__main__":
    data: pd.DataFrame = pd.read_csv("{}/data/covtype.data".format(ROOT_DIR))

    find_best_feature_sub_sets(
        data, y_column_name="Cover_Type", step_size=10, verbose=True, save_to_file=True
    )

    feature_sets = read_from_json("feature_sets")
    print(
        get_scores_from_feature_sub_sets(
            data,
            feature_sets=feature_sets,
            y_column_name="class",
            classifier=knn_model,
            save_to_file=True,
        )
    )

    feature_sub_set_scores = read_from_json("feature_sub_set_scores")
    plots_feature_sub_set_scores(feature_sub_set_scores)
