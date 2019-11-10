import pandas as pd
from mlxtend.frequent_patterns import apriori
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import numpy as np
from constants import ROOT_DIR


def select_variables(df):
    df=df.copy()
    y=df[['class']]
    df=df.drop(['class'],axis=1)
    selector = SelectKBest(f_classif, k=10)
    select_df = selector.fit_transform(df, y)
    mask=selector.get_support()
    new_features=[]
    for bool, feature in zip(mask, list(df.columns.values)):
        if bool:
            new_features.append(feature)
    return pd.DataFrame(select_df, columns=new_features)


def discretize_cut(select_df):
    newdf = select_df.copy()
    for col in newdf:
        if col not in ['class', 'id', 'gender']:
            newdf[col] = pd.cut(newdf[col], 3, labels=['small', 'mid', 'large'])
    return newdf


def discretize_qcut(select_df):
    newdf = select_df.copy()
    for col in newdf:
        if col not in ['class', 'id', 'gender']:
            newdf[col] = pd.qcut(newdf[col], 3, labels=['small', 'mid', 'large'])
    return newdf


def dummify(df):
    dummylist = []
    for att in df:
        if att in ['class','id', 'gender']: df[att] = df[att].astype('category')
        dummylist.append(pd.get_dummies(df[[att]]))
    dummified_df = pd.concat(dummylist, axis=1)
    return dummified_df


def pat_match(dummified_df):
    frequent_itemsets = {}
    minpaterns = 30
    minsup = 1.0
    while minsup > 0:
        minsup = minsup * 0.9
        frequent_itemsets = apriori(dummified_df, min_support=minsup, use_colnames=True)
        if len(frequent_itemsets) >= minpaterns:
            print("Minimum support:", minsup)
            break
    print("Number of found patterns:", len(frequent_itemsets))
    return frequent_itemsets


def visualize_patternmatch(dummified_df):
    map = {}
    for col in list(dummified_df): map[col] = (dummified_df[col] == 1).sum()
    keys = list(map.keys())
    freqs = list(map.values())
    plt.bar(np.arange(len(freqs)), freqs)
    plt.xticks(np.arange(len(freqs)), keys, rotation='vertical')
    plt.plot()

def main():
    df = pd.read_csv("{}/data/df_without_corr.csv".format(ROOT_DIR))
    select_df=select_variables(df)
    dis_c=discretize_cut(select_df)
    print(dis_c)
    #dis_q=discretize_cut(select_df)
    dum=dummify(dis_c)
    frequent_itemsets=pat_match(dum)
    print(frequent_itemsets)
    print(visualize_patternmatch(dum))


if __name__ == "__main__":
    main()