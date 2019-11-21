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
    select_df=compute_Importance(df,25)

    newdf = select_df.copy()
    for col in newdf:
        if col not in ["Cover_Type"]:
            newdf[col] = pd.cut(newdf[col], 4, labels=['1', '2', '3', '4'])
    return newdf


def apriori_ct2(disc_newdf):
    dummylist = []
    for att in disc_newdf:
        if att in ["Cover_Type"]:
            disc_newdf[att] = disc_newdf[att].astype("category")
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
            print("Covertype:")
            print("Minimum support:", minsup)
            break
    print("Number of patterns:", len(frequent_itemsets))
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=minconf)
    rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
    newrules = rules[(rules['antecedent_len'] >= 3)]
    if len(newrules) < 1:
        print("No rules found with given threshold")
    sortednewrules = newrules.sort_values("lift",ascending=False)
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
            newdf[col] = pd.qcut(newdf[col], 6, labels=['1', '2', '3', '4', '5', '6'])
    return newdf


def apriori_pd2(disc_newdf):
    dummylist = []
    for att in disc_newdf:
        if att in ["class", "id", "gender"]:
            disc_newdf[att] = disc_newdf[att].astype("category")
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

def visualize(dummified_df):
    map = {}
    for col in list(dummified_df): map[col] = (dummified_df[col] == 1).sum()
    keys = list(map.keys())
    freqs = list(map.values())
    plt.bar(np.arange(len(freqs)), freqs)
    plt.xticks(np.arange(len(freqs)), keys, rotation='vertical')
    plt.show()



def plot_number_buckets_PD(df):
    df = df.copy()
    y = df[["class"]]
    df = df.drop(["class"], axis=1)
    selector = SelectKBest(f_classif, k=30)
    select_df = selector.fit_transform(df, y)
    mask = selector.get_support()
    new_features = []
    for bool, feature in zip(mask, list(df.columns.values)):
        if bool:
            new_features.append(feature)
    select_df = pd.DataFrame(select_df, columns=new_features)

    bin_numbers = [2, 3, 4, 5, 6, 7, 8]
    metric_scores_depth=[]
    metric_scores_width=[]
    metric_scores = {}
    df_width=select_df.copy()
    df_depth=select_df.copy()
    for bin_nr in bin_numbers:
        #lables = range(0, bin_nr,1)
        print(bin_nr, "kjør ny bin-->")
        for col in df_depth:
            if col not in ["class", "id", "gender"]:
                    df_depth[col] = pd.qcut(df_depth[col], bin_nr, duplicates="drop")
        bin_df1=df_depth
        print(bin_nr)
        print("depth_df:", bin_df1)
        print(df_depth)
        rules, score_depth = apriori_pd2(bin_df1)
        metric_scores_depth.append(score_depth)
        metric_scores.update({'depth': metric_scores_depth})
        print(metric_scores)
        for col in df_width:
            if col not in ["class", "id", "gender"]:
                df_width[col] = pd.cut(df_width[col], bin_nr, duplicates="drop")
        bin_df2=df_width
        print("width coming up:")
        print(bin_df2)
        rules, score_width = apriori_pd2(bin_df2)
        metric_scores_width.append(score_width)
        metric_scores.update({'width': metric_scores_width})
        print(metric_scores)

def plot_number_buckets_CT(df):
    df = df.copy()
    y = df[["Cover_Type"]]
    df = df.drop(["Cover_Type"], axis=1)
    selector = SelectKBest(f_classif, k=10)
    select_df = selector.fit_transform(df, y)
    mask = selector.get_support()
    new_features = []
    for bool, feature in zip(mask, list(df.columns.values)):
        if bool:
            new_features.append(feature)
    select_df = pd.DataFrame(select_df, columns=new_features)

    bin_numbers = [2, 3, 4, 5, 6, 7, 8]
    metric_scores_depth=[]
    metric_scores_width=[]
    metric_scores={}
    df_width=select_df.copy()
    df_depth=select_df.copy()
    for bin_nr in bin_numbers:
            #lables = range(0, bin_nr)
            for col in df_depth:
                if col not in ["Cover_Type"]:
                    if len(df_depth[col].unique()) < bin_nr:
                        bin_nr = len(df_depth.df_depth[col].unique())
                        #labels = range(1, nr_of_labels, 1)
                        df_depth[col] = pd.qcut(df_depth[col], bin_nr, duplicates='drop')
            bin_df1=df_depth
            print(bin_nr)
            print("depth:", bin_df1)
            print(df_depth)
            rules, score_depth = apriori_ct2(bin_df1)
            metric_scores_depth.append(score_depth)
            metric_scores.update({'depth': metric_scores_depth})
            print(metric_scores)
            for col in df_width:
                if col not in ["Cover_Type"]:
                    df_width[col] = pd.cut(df_width[col], bin_nr, duplicates='drop')
            print("width coming up:")
            rules, score_width = apriori_ct2(df_width)
            metric_scores_width.append(score_width)
            metric_scores.update({'width': metric_scores_width})
            print(metric_scores)

#y-akse numberns of patterns, avg quality rules, avg quality top n rules
#x-akse min support

'''
def plot_numbPattern_avg_rules_top_rules(data_type, metrics):
    if data_type == 'PD':
        df = pd.read_csv("{}/data/df_without_corr.csv".format(ROOT_DIR))
        y = df[["class"]]
        df = df.drop(["class"], axis=1)
        selector = SelectKBest(f_classif, 50)
        select_df = selector.fit_transform(df, y)
        mask = selector.get_support()
        new_features = []
        for bool, feature in zip(mask, list(df.columns.values)):
            if bool:
                new_features.append(feature)

        return pd.DataFrame(select_df, columns=new_features)

    elif data_type=='CT':
        df= pd.read_csv("{}/data/covtype.csv".format(ROOT_DIR))
        y = df[["Cover_Type"]]
        df = df.drop(["Cover_Type"], axis=1)
        selector = SelectKBest(f_classif, 10)
        select_df = selector.fit_transform(df, y)
        mask = selector.get_support()
        new_features = []
        for bool, feature in zip(mask, list(df.columns.values)):
            if bool:
                new_features.append(feature)

        return pd.DataFrame(select_df, columns=new_features)

    bin_numbers = [2, 3, 4, 5, 6, 7]
    minpaterns = 300
    for i in range(len(metrics)):
        cuttype_metric_dict = {}
        metric_scores_depth = []
        metric_scores_width = []

        for bin_nr in bin_numbers:
            lables = range(0, bin_nr)
            rules, score_depth = create_associations(df, data_type, 'depth', bin_nr, lables, metrics[i], minpaterns)
            metric_scores_depth.append(score_depth)
            rules, score_width = create_associations(df, data_type, 'width', bin_nr, lables, metrics[i], minpaterns)
            metric_scores_width.append(score_width)

        cuttype_metric_dict.update({'depth': metric_scores_depth, 'width': metric_scores_width})
        print(cuttype_metric_dict)




    else:
        print('Select data_type = mixed or real')
    bin_numbers = [2, 3, 4, 5, 6]
    minpaterns = 300
    for i in range(len(metrics)):
        cuttype_metric_dict = {}
        metric_scores_depth = []
        metric_scores_width = []


        def create_associations(df, data_type, cut_type, nr_of_labels, labels, metric, minpaterns):
            discretized_df = discretize(df, cut_type, nr_of_labels, labels, data_type)
            dummified_df = dummify(discretized_df, data_type)
            rules = get_frequent_itemsets(dummified_df, metric, minpaterns)
            score = mean_score(rules, metric)
            return rules, score


        for bin_nr in bin_numbers:
            lables = range(0, bin_nr)
            rules, score_depth = create_associations(df, data_type, 'depth', bin_nr, lables, metrics[i], minpaterns)
            metric_scores_depth.append(score_depth)
            rules, score_width = create_associations(df, data_type, 'width', bin_nr, lables, metrics[i], minpaterns)
            metric_scores_width.append(score_width)

        cuttype_metric_dict.update({'depth': metric_scores_depth,'width': metric_scores_width})
        print(cuttype_metric_dict)
        if data_type == 'CT':
            title = metrics[i] + ' for dataset: CT'
        else:
            title = metrics[i] + ' for dataset: PD'
        multiple_line_chart(plt.gca(), bin_numbers, cuttype_metric_dict, title, 'number of bins', 'scores')
        plt.show()
'''


def top_3rules(rules):
    top_df=rules.head(3)
    return top_df


def main():
    df_pd = pd.read_csv("{}/data/df_without_corr.csv".format(ROOT_DIR))
    df_ct = pd.read_csv("{}/data/covtype.csv".format(ROOT_DIR))

    #plot_number_buckets_PD(df_pd)
    #plot_number_buckets_CT(df_ct)
    p=patternmining_pd((df_pd))
    c=patternmining_ct(df_ct)
    print(top_3rules(p))
    print(top_3rules(c))
    #print(p)
    #print(c)
    #print(visualize(dummify(select_variables())))


if __name__ == "__main__":
    main()
'''

def try_different_minpaterns(data_type, metric):
    scores = {}
    best_score = 0
    best_minpattern = None
    for i in range(100, 800, 100):
        score = run_association_rule_mining_on_data(data_type, 50, metric, i)[0]
        scores.update({i: score})
        if score > best_score:
            best_minpattern = i
    print('Number of min-patterns with their respectively ',metric,'-score: ', scores)
    return best_minpattern


# For each metric plots a curve of the score associated with the number of bins
def plot_different_nr_of_bins(data_type, metrics):
    if data_type == 'real':
        data = pd.read_csv('/Users/test/PycharmProjects/Metoderi_I_AI/datascience/real_project/files/pd_speech_features.csv',
                           skiprows=1, parse_dates=True, infer_datetime_format=True)
        #data = remove_correlated('files/pd_speech_features.csv', 0.95)
        X, y, df = data_preparation(data, data_type)
        df = select_k_best(X, y, df, 50)
    elif data_type == 'mixed':
        data = set_index_row_for_covtype()
        X, y, prepared_df = data_preparation(data, data_type)
        df = select_k_best(X, y, prepared_df, 10)
    else:
        print('Select data_type = mixed or real')
    bin_numbers = [2, 3, 4, 5, 6]
    minpaterns = 300
    for i in range(len(metrics)):
        cuttype_metric_dict = {}
        metric_scores_depth = []
        metric_scores_width = []

        for bin_nr in bin_numbers:
            lables = range(0, bin_nr)
            rules, score_depth = create_associations(df, data_type, 'depth', bin_nr, lables, metrics[i], minpaterns)
            metric_scores_depth.append(score_depth)
            rules, score_width = create_associations(df, data_type, 'width', bin_nr, lables, metrics[i], minpaterns)
            metric_scores_width.append(score_width)

        cuttype_metric_dict.update({'depth': metric_scores_depth,'width': metric_scores_width})
        print(cuttype_metric_dict)
        if data_type == 'mixed':
            title = metrics[i] + ' for dataset: CT'
        else:
            title = metrics[i] + ' for dataset: PD'
        multiple_line_chart(plt.gca(), bin_numbers, cuttype_metric_dict, title, 'number of bins', 'scores')
        plt.show()

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from mlxtend.frequent_patterns import apriori, association_rules  # for ARM
import matplotlib.pyplot as plt
import numpy as np
#from project.lab2 import remove_correlated
from lab0 import multiple_line_chart
from real_project.src.preprocessing.set_index_row_for_dataframe import set_index_row_for_covtype
from real_project.src.unsupervised.clustering import remove_class_covertype, remove_class_pd

#Prepare X, Y and data frame
def data_preparation(df, data_type):
    if data_type == 'real':
        newdf, y = remove_class_pd(df)
    elif data_type == 'mixed':
        newdf, y = remove_class_covertype(df)
    X = newdf.values
    return X, y, df



# Select 10 best --> returns DF
def select_k_best(X, y, df, k):
    selector = SelectKBest(f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    mask = selector.get_support()
    attributes = df.columns[mask]
    df = pd.DataFrame(X_new, columns=attributes)
    return df


#Discretize DF --> returns DF
def discretize(df, cut_type, nr_of_labels, labels, data_type):
    df = df.copy()
    for col in df:
        if col not in ['class', 'id']:
            if cut_type == 'width':
                # equally divided intervals from min to max.
                # Problem: all observations can end up in one interval. Unbalance. High support
                df[col] = pd.cut(df[col], nr_of_labels, labels)
                # Report: make a chart or stats on how the average pattern suppurt for each alphabet/number of bins differ
            elif cut_type == 'depth':
                # Fixed number of frequencies in each bin. lower support than cut,
                if data_type == 'mixed':
                    df[col] = pd.qcut(df[col], nr_of_labels, duplicates='drop')
                else:
                    df[col] = pd.qcut(df[col], nr_of_labels, labels)
            else:
                print('Select cut type')
    return df


#Dummify DF --> returns DF
def dummify(df, data_type):
    dummylist = []
    for att in df:
        if data_type == 'mixed':
            if att in ['class']:
                df.fillna('2', inplace=True)
            dummylist.append(pd.get_dummies(df[[att]]))
            dummified_df = pd.concat(dummylist, axis=1)
        elif data_type == 'real':
            dummylist.append(pd.get_dummies(df[[att]]))
            dummified_df = pd.concat(dummylist, axis=1)
        else:
            dummified_df = df
            print('No dummifying')
    return dummified_df


# Vizualize the dummified values --> return nothing
def visualize(dummified_df):
    map = {}
    for col in list(dummified_df): map[col] = (dummified_df[col] == 1).sum()
    keys = list(map.keys())
    freqs = list(map.values())
    plt.bar(np.arange(len(freqs)), freqs)
    plt.xticks(np.arange(len(freqs)), keys, rotation='vertical')
    plt.show()


# Return association rules
def get_frequent_itemsets(dummified_df, metric, minpaterns):
    frequent_itemsets = {}
    minsup = 1.0
    while minsup > 0:
        minsup = minsup * 0.95
        frequent_itemsets = apriori(dummified_df, min_support=minsup, use_colnames=True)
        if len(frequent_itemsets) >= minpaterns:
            break
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=0.5)
    return rules

# A gathered method for the whole rule mining process from data preparation to
def run_association_rule_mining_on_data(data_type, top_k, metric, minpaterns):
    # PARKINSON:
    if data_type == 'real':
        data =pd.read_csv('/Users/test/PycharmProjects/Metoderi_I_AI/datascience/real_project/files/pd_speech_features.csv',
                    skiprows=1, parse_dates=True, infer_datetime_format=True)
        #data = remove_correlated('datasets/pd_speech_features.csv', 0.95)
        X, y, df = data_preparation(data, data_type)
        df = select_k_best(X, y, df, top_k)
        # cut_type = 'depth' since qcut is best for PD, and nr_of bins = 6 since  lift metric is best at 6
        rules, score = create_associations(df, data_type, 'depth', 6, [0, 1, 2, 3, 4, 5], metric, minpaterns)
        most_significant = most_signinficant_rules(rules, metric)
        print('Most significant association rules for PD-dataset based on the metric: ',metric, ' :\n')
        print(most_significant, '\n')
        print('Mean lift: ',mean_score(most_significant, 'lift'))
        print('Mean support: ', mean_score(most_significant, 'support'))
        print('Mean confidence: ', mean_score(most_significant, 'confidence'))
        print('Mean conviction: ', mean_score(most_significant, 'conviction'))

    # COVTYPE:
    elif data_type == 'mixed':
        data = set_index_row_for_covtype()
        X, y, df = data_preparation(data, data_type)
        df = select_k_best(X, y, df, top_k)

        # cut_type = 'width' since cut is best for CT, and nr_of bins = 6 since lift metric is best at 6
        rules, score = create_associations(df, data_type, 'width', 4, [0, 1, 2, 3], metric, minpaterns)
        most_significant = most_signinficant_rules(rules, metric)
        print('\nThe 5 most significant association rules for CT-dataset based on ',metric, ':\n', most_significant)
        print('Mean lift: ', mean_score(most_significant, 'lift'))
        print('Mean support: ', mean_score(most_significant, 'support'))
        print('Mean confidence: ', mean_score(most_significant, 'confidence'))
        print('Mean conviction: ', mean_score(most_significant, 'conviction'))

    else:
        print('Select data_type = mixed or real')
        rules = None
        score = 0
        most_significant = None
    return score, most_significant


def main():
    #To be able to show the whole dataframe:
    pd.set_option('display.max_rows', 1000)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)

    #try_different_minpaterns('real', 'lift')
    #try_different_minpaterns('mixed', 'lift')
    #best_minpatern = try_different_minpaterns('real', 'lift')

    #run_association_rule_mining_on_data('real', 50, 'lift', 300)
    #run_association_rule_mining_on_data('mixed', 20, 'lift', 300)

    #print(plot_different_nr_of_bins('mixed', ['lift', 'support', 'leverage', 'confidence']))
    plot_different_nr_of_bins('real', ['lift', 'support', 'leverage', 'confidence'])


if __name__ == '__main__':
    main()'''
