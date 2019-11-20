import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import numpy as np
from constants import ROOT_DIR


def select_variables(df):
    df = df.copy()
    y = df[["class"]]
    df = df.drop(["class"], axis=1)
    selector = SelectKBest(f_classif, k=25)
    select_df = selector.fit_transform(df, y)
    mask = selector.get_support()
    new_features = []
    for bool, feature in zip(mask, list(df.columns.values)):
        if bool:
            new_features.append(feature)

    return pd.DataFrame(select_df, columns=new_features)


def discretize_cut(select_df, labels):
    newdf = select_df.copy()
    for col in newdf:
        if col not in ["class", "id", "gender"]:
            newdf[col] = pd.cut(newdf[col], labels.length, labels=labels)
    return newdf


def discretize_qcut(select_df):
    newdf = select_df.copy()
    for col in newdf:
        if col not in ["class", "id", "gender"]:
            newdf[col] = pd.qcut(newdf[col], 8, labels=['low','lowmid','mid','midhigh','high','veryhigh','highest',8])
    return newdf


def dummify(df):
    dummylist = []
    for att in df:
        if att in ["class", "id", "gender"]:
            df[att] = df[att].astype("category")
        dummylist.append(pd.get_dummies(df[[att]]))
    dummified_df = pd.concat(dummylist, axis=1)
    return dummified_df


def pat_match(dummified_df):
    frequent_itemsets = {}
    minpaterns = 300
    minsup = 1.0
    minconf = 0.9
    while minsup > 0:
        minsup = minsup * 0.9
        frequent_itemsets = apriori(dummified_df, min_support=minsup, use_colnames=True)
        if len(frequent_itemsets) >= minpaterns:
            print("Minimum support:", minsup)
            break
    print("Number of found patterns:", len(frequent_itemsets))
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=minconf)
    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    newrules= rules[(rules['antecedent_len'] >= 3)]
    if len(newrules) < 1:
        print("No rules found with given threshold")
    sortednewrules= newrules.sort_values("lift")
    return sortednewrules


def visualize_patternmatch(dummified_df):
    map = {}
    for col in list(dummified_df):
        map[col] = (dummified_df[col] == 1).sum()
    keys = list(map.keys())
    freqs = list(map.values())
    plt.bar(np.arange(len(freqs)), freqs)
    plt.xticks(np.arange(len(freqs)), keys, rotation="vertical")
    plt.show()


def apriori_cut(df):
    dum_cut = dummify(discretize_cut(select_variables(df)))
    frequent_itemsets1 = pat_match(dum_cut)
    return frequent_itemsets1


def apriori_qcut(df):
    dum_qcut = dummify(discretize_qcut(select_variables(df)))
    frequent_itemsets2 = pat_match(dum_qcut)
    return frequent_itemsets2


def main():
    df = pd.read_csv("{}/data/df_without_corr.csv".format(ROOT_DIR))
    resdepth=apriori_qcut(df)
    print(resdepth)
    print(visualize_patternmatch(apriori_qcut(df)))



if __name__ == "__main__":
    main()
