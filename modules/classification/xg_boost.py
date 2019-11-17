import xgboost as xgb
import matplotlib.pyplot as plt

from modules.functions import calculte_models_auc_score, multiple_line_chart


def xg_boost(trnX, tstX, trnY, tstY, multi_class, plot=False):
    best_score = 0
    best_model = None
    best_learning_rate = 0
    best_depth = 0
    best_estimator = 0
    learning_rates = [0.1, 0.05, 0.01]
    max_depths = [5, 25]
    n_estimators = [10, 50, 100, 200, 300, 400, 500]
    if plot:
        plt.figure()
        fig, axs = plt.subplots(1, len(max_depths), figsize=(16, 4), squeeze=False)
    for k in range(len(max_depths)):
        max_depth = max_depths[k]
        values = {}
        for learning_rate in learning_rates:
            yvalues = []
            for n in n_estimators:
                model = xgb.XGBClassifier(
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    n_estimators=n,
                    early_stopping_rounds=10,
                )
                model.fit(trnX, trnY)
                score = calculte_models_auc_score(model, tstX, tstY, multi_class)
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_learning_rate = learning_rate
                    best_depth = max_depth
                    best_estimator = n
                yvalues.append(score)
            values[learning_rate] = yvalues
            if plot:
                multiple_line_chart(
                    axs[0, k],
                    n_estimators,
                    values,
                    "XG Boost with %s depth" % max_depth,
                    "Number of estimators",
                    "accuracy",
                    percentage=True,
                )
    if plot:
        plt.show()
    return best_model, best_score, best_learning_rate, best_depth, best_estimator
