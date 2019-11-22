import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier

from modules.functions import multiple_line_chart, save_model, calculte_models_auc_score


def random_forest(trnX, trnY, n=300, d=50, f="log2", save_file=False):
    rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f)
    rf.fit(trnX, trnY)
    if save_file:
        save_model(rf, "random_forest")
    return rf


def rf_test_different_params(trnX, tstX, trnY, tstY, multi_class, plot=False):
    best_numb_estimator = 0
    best_depth = 0
    best_feature = ""
    best_score = 0
    best_model = None
    n_estimators = [300]
    max_depths = [25]
    max_features = ["sqrt"]

    if plot:
        plt.figure()
        fig, axs = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
    for k in range(len(max_features)):
        f = max_features[k]
        values = {}
        for d in max_depths:
            yvalues = []
            for n in n_estimators:
                rf = random_forest(trnX, trnY, n, d, f)
                score = calculte_models_auc_score(rf, tstX, tstY, multi_class)
                if score > best_score:
                    best_score = score
                    best_depth = d
                    best_numb_estimator = n
                    best_feature = f
                    best_model = rf
                yvalues.append(score)
            values[d] = yvalues
        if plot:
            multiple_line_chart(
                axs[0, k],
                n_estimators,
                values,
                "RF with %s features" % f,
                "nr estimators",
                "Sensitivity",
                percentage=True,
            )
    if plot:
        plt.show()
    return best_model, best_score, best_numb_estimator, best_depth, best_feature


if __name__ == '__main__':
    import pandas as pd
    from modules.classification.all_models import split_dataset
    from matplotlib import pyplot as plt
    from constants import ROOT_DIR
    from modules.functions import read_from_json
    from modules.preprocessing.preprocessing import normalize_df


    data: pd.DataFrame = pd.read_csv("{}/data/pd_speech_features1.csv".format(ROOT_DIR))
    df = data.copy()
    df[["class", "id", "gender"]] = df[["class", "id", "gender"]].apply(
        pd.to_numeric
    )
    df = normalize_df(df, columns_not_to_normalize=["id", "gender", "class"])
    df = df.groupby(by="id").median().reset_index()

    trnX, tstX, trnY, tstY, labels = split_dataset(df)
    columns = list(df)
    columns.remove("class")
    trnX_df = pd.DataFrame.from_records(trnX, columns=columns)
    tstX_df = pd.DataFrame.from_records(tstX, columns=columns)
    feature_sets = read_from_json("feature_sets")
    scores = {}
    for key, value in feature_sets.items():
        print(key)
        print(value)
        train_df = trnX_df.copy()
        test_df = tstX_df.copy()
        train_x = train_df[value]
        test_x = test_df[value]
        model = random_forest(train_x, trnY, n=300, d=25, f="log2", save_file=False)
        scores[key] = calculte_models_auc_score(model, test_x, tstY, multi_class=False)
        print(scores[key])

    percentage = False
    plt.figure()
    ax=plt.gca()
    ax.set_title("Random forest performance based on number of features")
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("Sensitivity Score")
    if percentage:
        ax.set_ylim(0.0, 1.0)
    lists = sorted(scores.items(), key=lambda x: int(x[0]))  # sorted by key, return a list of tuples

    x, y = zip(*lists)  # unpack a list of pairs into two tuples

    plt.plot(x, y)
    plt.show()
    plt.show()
