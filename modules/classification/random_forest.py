import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier

from modules.functions import multiple_line_chart, save_model


def random_forest(trnX, tstX, trnY, tstY, n=300, d=50, f="log2"):
    rf = RandomForestClassifier(n_estimators=n, max_depth=d, max_features=f)
    rf.fit(trnX, trnY)
    save_model(rf, "random_forest")
    return rf


def test_different_params(trnX, tstX, trnY, tstY):
    n_estimators = [5, 10, 25, 50, 75, 100, 150, 200, 250, 300]
    max_depths = [5, 10, 25, 50]
    max_features = ["sqrt", "log2"]

    plt.figure()
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
    for k in range(len(max_features)):
        f = max_features[k]
        values = {}
        for d in max_depths:
            yvalues = []
            for n in n_estimators:
                rf = random_forest(trnX, tstX, trnY, tstY, n, d, f)
                prdY = rf.predict(tstX)
                yvalues.append(metrics.accuracy_score(tstY, prdY))
            values[d] = yvalues
        multiple_line_chart(
            axs[0, k],
            n_estimators,
            values,
            "Random Forests with %s features" % f,
            "nr estimators",
            "accuracy",
            percentage=True,
        )
    plt.show()
