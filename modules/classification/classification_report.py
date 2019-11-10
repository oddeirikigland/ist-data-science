from modules.classification.all_models import (
    split_dataset,
    create_classifier_models,
    get_accuracy_models,
)


def classification_report(data, source):
    df = data.copy()

    trnX, tstX, trnY, tstY, labels = split_dataset(df)
    create_classifier_models(trnX, trnY)

    print("2. Classifiers:")
    print(" 2.1 NB")
    print("     a) Suggested parametrization:")
    print("     b) Confusion matrix:")
    print(" 2.2 KNN")
    print("         a) Suggested parametrization:")
    print("         b) Confusion matrix:")
    print("3. Comparative performance: NB | KNN | DT | RF")

    accuracies = get_accuracy_models(tstX, tstY)
    print(
        " 3.1 Accuracy: {:.2f} | {:.2f} | {:.2f} | {:.2f}".format(
            accuracies["nb"], accuracies["knn"], accuracies["dt"], accuracies["rf"]
        )
    )
    print(" 3.2 Sensitivity: ")
