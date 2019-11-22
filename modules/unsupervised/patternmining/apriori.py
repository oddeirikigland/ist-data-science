from typing import List, Any, Union

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import matplotlib.pyplot as plt, math
from constants import ROOT_DIR


# Samlet funksjon for Covertype
def apriori_ct(df):
    df = df.copy()
    df = df.drop(["Cover_Type"], axis=1)
    select_df=compute_Importance(df,10)

    newdf = select_df.copy()
    for col in newdf:
        if col not in ["Cover_Type"]:
            newdf[col] = pd.cut(newdf[col], 8, labels=['1', '2', '3','4','5','6','7','8'])
    return newdf


def apriori_ct2(disc_newdf):
    dummylist = []
    for att in disc_newdf:
        if att in ["Cover_Type"]:
            disc_newdf[att] = disc_newdf[att].astype("category")
        dummylist.append(pd.get_dummies(disc_newdf[[att]]))
    dummified_df = pd.concat(dummylist, axis=1)

    frequent_itemsets = {}
    minpaterns = 300
    minsup = 1.0
    minconf = 0.9
    while minsup > 0:
        minsup = minsup * 0.95
        frequent_itemsets = apriori(dummified_df, min_support=minsup, use_colnames=True)
        if len(frequent_itemsets) >= minpaterns:
            print("Covertype:")
            print("Minimum support:", minsup)
            break
    print("Number of patterns:", len(frequent_itemsets))
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=minconf)
    if len(rules) < 1:
        print("No rules found with given threshold")
    sortednewrules = rules.sort_values("lift",ascending=False)
    score_lift = mean_lift(sortednewrules)
    score_confidence= mean_confidence(sortednewrules)
    print("Mean lift:", score_lift)
    print("Mean confidence:", score_confidence)
    return sortednewrules


def apriori_ct_with_class(df):
    df = df.copy()
    y = df[["Cover_Type"]]
    df = df.drop(["Cover_Type"], axis=1)
    selector = SelectKBest(f_classif, k=20)
    select_df = selector.fit_transform(df, y)
    mask = selector.get_support()
    new_features = []
    for bool, feature in zip(mask, list(df.columns.values)):
        if bool:
            new_features.append(feature)
    select_df = pd.DataFrame(select_df, columns=new_features)

    newdf = select_df.copy()
    for col in newdf:
        if col not in ["Cover_Type"]:
            newdf[col] = pd.cut(newdf[col], 4, labels=['1', '2', '3', '4'])
    return newdf


# samlet funksjon for parkinson
def apriori_pd(df):
    df = df.copy()
    y = df[["class"]]
    df = df.drop(["class"], axis=1)
    select_df = compute_Importance(df, 35)

    newdf = select_df.copy()
    for col in newdf:
        if col not in ["class", "id", "gender"]:
            newdf[col] = pd.qcut(newdf[col], 7, labels=['1', '2', '3', '4', '5', '6','7'])
    return newdf


def apriori_pd2(disc_newdf):
    dummylist = []
    for att in disc_newdf:
        if att in ["class", "id", "gender"]:
            disc_newdf[att] = disc_newdf[att].astype("category")
        dummylist.append(pd.get_dummies(disc_newdf[[att]]))
    dummified_df = pd.concat(dummylist, axis=1)

    frequent_itemsets = {}
    minpaterns = 300
    minsup = 1.0
    minconf = 0.9
    while minsup > 0:
        minsup = minsup * 0.95
        frequent_itemsets = apriori(dummified_df, min_support=minsup, use_colnames=True)
        if len(frequent_itemsets) >= minpaterns:
            print("Parkinson:")
            print("Minimum support:", minsup)
            break
    print("Number of patterns:", len(frequent_itemsets))
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=minconf)
    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    newrules = rules[(rules['antecedent_len'] >= 3)]
    if len(newrules) < 1:
        print("No rules found with given threshold")
    sortednewrules = newrules.sort_values("lift", ascending=False)
    score_lift = mean_lift(sortednewrules)
    score_confidence = mean_confidence(sortednewrules)
    print("Mean lift:", score_lift)
    print("Mean confidence:", score_confidence)
    return sortednewrules

def apriori_pd_with_class(df):
    df = df.copy()
    y = df[["class"]]
    df = df.drop(["class"], axis=1)
    selector = SelectKBest(f_classif, k=50)
    select_df = selector.fit_transform(df, y)
    mask = selector.get_support()
    new_features = []
    for bool, feature in zip(mask, list(df.columns.values)):
        if bool:
            new_features.append(feature)
    select_df = pd.DataFrame(select_df, columns=new_features)

    newdf = select_df.copy()
    for col in newdf:
        if col not in ["class", "id", "gender"]:
            newdf[col] = pd.qcut(newdf[col], 6, labels=['1', '2', '3', '4', '5', '6'])
    return newdf

def compute_Importance(X, k):

        pca = PCA(svd_solver="auto")
        pca.fit(X)

        T = pca.transform(X)

        # 1 scale principal components
        xvector = pca.components_[0] * max(T[:, 0])
        yvector = pca.components_[1] * max(T[:, 1])

        # 2 compute column importance and sort
        columns = X.columns.values
        impt_features = {
            columns[i]: math.sqrt(xvector[i] ** 2 + yvector[i] ** 2)
            for i in range(len(columns))
        }

        # print("Features by importance:", sorted(zip(impt_features.values(), impt_features.keys()), reverse=True))
        sortedList = sorted(zip(impt_features.values(), impt_features.keys()), reverse=True)
        doneList = sortedList[:k]
        # print("")
        features = []
        for m in range(0, len(doneList)):
            (value, feature) = doneList[m]
            features.append(feature)
        # print(features)
        X_selected_collums = X.loc[:, features]
        # print(X_selected_collums)

        return X_selected_collums

def mean_lift(result):
    metric = "lift"
    measure = result.loc[:, metric]
    measure_sum = 0
    for m in measure:
        measure_sum = measure_sum + m
    if len(measure) == 0:
        mean_measure = 0
    else:
        mean_measure = measure_sum / len(measure)
    return mean_measure


def mean_confidence(result):
    metric = "confidence"
    measure = result.loc[:, metric]
    measure_sum = 0
    for m in measure:
        measure_sum = measure_sum + m
    if len(measure) == 0:
        mean_measure = 0
    else:
        mean_measure = measure_sum / len(measure)
    return mean_measure


# for importering til Rapporten
def patternmining_pd(df):
    a= apriori_pd2(apriori_pd(df))
    return a


def patternmining_ct(df):
    b= apriori_ct2(apriori_ct(df))
    return b


#y-akse numberns of patterns, avg quality rules, avg quality top n rules
#x-akse min support



def top_3rules(rules):
    top_df=rules.head(3)
    return top_df


def main():
    df_pd = pd.read_csv("{}/data/df_without_corr.csv".format(ROOT_DIR))
    df_ct = pd.read_csv("{}/data/covtype.csv".format(ROOT_DIR))

    #plot_number_buckets_PD(df_pd)

    p=patternmining_pd((df_pd))
    c=patternmining_ct(df_ct)
    print(top_3rules(p))
    print(top_3rules(c))
    #print(p)
    #print(c)
    #print(visualize(dummify(select_variables())))


if __name__ == "__main__":
    main()


