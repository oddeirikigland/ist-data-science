import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from constants import ROOT_DIR
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import matplotlib.pyplot as plt, math

from modules.unsupervised.patternmining.apriori import compute_Importance, mean_lift, mean_confidence


def apriori_pd_bin(df):
    df = df.copy()
    df = df.drop(["class"], axis=1)
    select_df = compute_Importance(df, 30)
    newdf = select_df.copy()

    bins = [2, 3, 4, 5, 6, 7, 8]
    scores_depth = []
    scores_width = []
    metric_scores = {}
    for bin in bins:
        print(bin)
        df_cutted = df_cut_dp(newdf, bin)
        df_qcutted = df_qcut_dp(newdf, bin)
        scores_wide = get_score(df_cutted)
        scores_width.append(scores_wide)
        metric_scores.update({'width': scores_width})
        scores_deep = get_score(df_qcutted)
        scores_depth.append(scores_deep)
        metric_scores.update({'depth': scores_depth})

        print(newdf)


def apriori_ct_bin(df):
    df = df.copy()
    df = df.drop(["Cover_Type"], axis=1)
    select_df=compute_Importance(df, 30)
    newdf = select_df.copy()

    bins=[2,3,4,5,6,7,8]
    scores_depth = []
    scores_width = []
    metric_scores = {}
    for bin in bins:
        print(bin)
        df_cutted=df_cut(newdf, bin)
        df_qcutted=df_qcut(newdf, bin)
        scores_wide= get_score(df_cutted)
        scores_width.append(scores_wide)
        metric_scores.update({'width': scores_width})
        scores_deep = get_score(df_qcutted)
        scores_depth.append(scores_deep)
        metric_scores.update({'depth': scores_depth})

        print(newdf)


def df_cut(newdf,bin):
    label= range(1,bin,1)
    for col in newdf:
        if len(newdf[col].unique()) < bin:
            binary_bin = len(newdf[col].unique())
            print(label, binary_bin)
            newdf[col] = pd.cut(newdf[col], binary_bin, labels=range(1,bin), duplicates="drop")
        else:
            print(label, bin)
            newdf[col] = pd.cut(newdf[col], bin, labels=range(1,bin), duplicates="drop")
    return newdf


def df_qcut(newdf,bin):
    for col in newdf:
        if len(newdf[col].unique()) < bin:
            binary_bin = len(newdf[col].unique())
            newdf[col] = pd.qcut(newdf[col], binary_bin, labels=range(1,bin), duplicates="drop")
        else:
            newdf[col] = pd.qcut(newdf[col], bin, labels=range(1,bin), duplicates="drop")
    return newdf


def df_cut_dp(newdf,bin):

    for col in newdf:
        if col not in ["class", "id", "gender"]:
            if len(newdf[col].unique()) < bin:
                binary_bin = len(newdf[col].unique())
                print( binary_bin)
                newdf[col] = pd.cut(newdf[col], binary_bin, labels=range(1,bin), duplicates="drop")
            else:
                print( bin)
                newdf[col] = pd.cut(newdf[col], bin, labels=range(1,bin), duplicates="drop")
    return newdf


def df_qcut_dp(newdf,bin):
    label = range(1, bin, 1)
    for col in newdf:
        if col not in ["class", "id", "gender"]:
            if len(newdf[col].unique()) < bin:
                binary_bin = len(newdf[col].unique())
                print(label, binary_bin)
                newdf[col] = pd.qcut(newdf[col], binary_bin, labels=range(1,bin), duplicates="drop")
            else:
                print(label, bin)
                newdf[col] = pd.qcut(newdf[col], bin, labels=range(1,bin), duplicates="drop")
    return newdf


def get_score(disc_newdf):
    dummylist = []
    for att in disc_newdf:
        dummylist.append(pd.get_dummies(disc_newdf[[att]]))
    dummified_df = pd.concat(dummylist, axis=1)

    frequent_itemsets = {}
    minpaterns = 400
    minsup = 1.0
    minconf = 0.9
    while minsup > 0:
        minsup = minsup * 0.95
        frequent_itemsets = apriori(dummified_df, min_support=minsup, use_colnames=True)
        if len(frequent_itemsets) >= minpaterns:
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
    return score_lift


# samlet funksjon for parkinson

def apriori_pd(df):
    df = df.copy()
    df = df.drop(["class"], axis=1)
    select_df = compute_Importance(df, 40)

    newdf = select_df.copy()
    for col in newdf:
        if col not in ["class", "id", "gender"]:
            newdf[col] = pd.qcut(newdf[col], 6, labels=['1', '2', '3', '4', '5', '6'])
    dummylist = []
    for att in newdf:
        if att in ["class", "id", "gender"]:
            newdf[att] = newdf[att].astype("category")
        dummylist.append(pd.get_dummies(newdf[[att]]))
    dummified_df = pd.concat(dummylist, axis=1)

    frequent_itemsets = {}
    minpaterns = 300
    minsup = 1.0
    minconf = 0.9
    patterns_list={}
    avg_lift_list={}
    avg_toplift_list={}
    while minsup > 0:
        minsup = minsup * 0.90
        frequent_itemsets = apriori(dummified_df, min_support=minsup, use_colnames=True)
        if len(frequent_itemsets) >= minpaterns:
            print("Minimum support:", minsup)
            break
        patterns_list[minsup]=len(frequent_itemsets)
        while len(frequent_itemsets)>0:
            rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=minconf)
            if len(rules) < 1:
                print("No rules found with given threshold")
            sortednewrules= rules.sort_values("lift", ascending=False)
            avg_lift_list[minsup]=mean_lift(sortednewrules)
            avg_toplift_list[minsup]= mean_lift(sortednewrules.head(5))
            break

    return patterns_list, avg_lift_list, avg_toplift_list


def main():
    df_p= pd.read_csv("{}/data/df_without_corr.csv".format(ROOT_DIR))
    df_c= pd.read_csv("{}/data/covtype.csv".format(ROOT_DIR))

    #apriori_ct(df_c)
    apriori_pd(df_p)


if __name__ == '__main__':
    main()