from modules.classification.all_models import (
    create_classifier_models,
    get_accuracy_models,
    get_sensitivity_models,
)
from modules.functions import print_table


def classification_report(trnX, tstX, trnY, tstY, source):
    nb, knn, dt, rf = create_classifier_models(trnX, trnY)

    print("2. Classifiers:")
    print(" 2.1 NB")
    print("     a) Suggested parametrization:")
    print("     b) Confusion matrix:")
    print(" 2.2 KNN")
    print("     a) Suggested parametrization:")
    print("     b) Confusion matrix:")
    print("3. Comparative performance:")
    accuracies = get_accuracy_models(tstX, tstY, nb, knn, dt, rf)
    sensitivities = get_sensitivity_models(
        tstX, tstY, nb, knn, dt, rf, multi_class=True if source == "CT" else False
    )

    scores_to_print = [
        ["", "NB", "KNN", "DT", "RF"],
        [
            "3.1 Accuracy",
            round(accuracies["nb"], 2),
            round(accuracies["knn"], 2),
            round(accuracies["dt"], 2),
            round(accuracies["rf"], 2),
        ],
        [
            "3.2 Sensitivity",
            round(sensitivities["nb"], 2),
            round(sensitivities["knn"], 2),
            round(sensitivities["dt"], 2),
            round(sensitivities["rf"], 2),
        ],
    ]
    print_table(scores_to_print)
