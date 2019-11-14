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
    n_estimators = [5, 10, 25, 50, 75, 100, 150, 200, 250, 300]
    max_depths = [5, 10, 25, 50]
    max_features = ["sqrt", "log2"]

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
                "Random Forests with %s features" % f,
                "nr estimators",
                "accuracy",
                percentage=True,
            )
    if plot:
        plt.show()
    return best_model, best_score, best_numb_estimator, best_depth, best_feature
