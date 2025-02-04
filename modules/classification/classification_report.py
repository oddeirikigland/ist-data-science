from modules.classification.all_models import (
    get_accuracy_models,
    get_sensitivity_models,
)
from modules.classification.decision_tree import dt_plot_accuracy
from modules.classification.knn import knn_test_several_params
from modules.classification.naive_bayes import naive_test_different_params
from modules.classification.random_forest import rf_test_different_params
from modules.functions import print_table, print_confusion_matrix
from modules.classification.xg_boost import xg_boost
from modules.classification.compare_fold_scores import print_accuracies_sensitivities

PLOT = False


def classification_report(trnX, tstX, trnY, tstY, labels, source, fold):
    multi_class = True if source == "CT" else False

    accuracies_folds = []
    sensitivities_folds = []
    count = 1
    for data_set in fold:
        print("Report from {}/{} folds".format(count, len(fold)))
        count += 1
        trnX, tstX, trnY, tstY = data_set

        print("2. Classifiers:")
        print(" 2.1 NB")
        nb, best_score, best_estimator = naive_test_different_params(
            trnX, tstX, trnY, tstY, multi_class, PLOT
        )
        print(
            "     a) Suggested parametrization: {} values, which gives an AUC score of {:.2f}".format(
                best_estimator, best_score
            )
        )
        print("     b) Confusion matrix:")
        print_confusion_matrix(nb, tstX, tstY, labels)

        print(" 2.2 KNN")
        knn, best_score, best_dist, best_n_value = knn_test_several_params(
            trnX, tstX, trnY, tstY, multi_class, PLOT
        )
        print(
            "     a) Suggested parametrization: {} distance with {} values, which gives an AUC score of {:.2f}".format(
                best_dist, best_n_value, best_score
            )
        )
        print("     b) Confusion matrix:")
        print_confusion_matrix(knn, tstX, tstY, labels)

        print(" 2.3 Decision Tree")
        dt, best_score, best_samples_leaf, best_depth, best_criteria = dt_plot_accuracy(
            trnX, tstX, trnY, tstY, multi_class, PLOT
        )
        print(
            "     a) Suggested parametrization: {} samples leaf with {} depth and {} criteria, which gives an AUC score of {:.2f}".format(
                best_samples_leaf, best_depth, best_criteria, best_score
            )
        )
        print("     b) Confusion matrix:")
        print_confusion_matrix(dt, tstX, tstY, labels)

        print(" 2.4 Random Forest")
        rf, best_score, best_numb_estimator, best_depth, best_feature = rf_test_different_params(
            trnX, tstX, trnY, tstY, multi_class, PLOT
        )
        print(
            "     a) Suggested parametrization: {} estimators with {} depth and {} feature, which gives an AUC score of {:.2f}".format(
                best_numb_estimator, best_depth, best_feature, best_score
            )
        )
        print("     b) Confusion matrix:")
        print_confusion_matrix(rf, tstX, tstY, labels)

        print(" 2.5 XG Boost")
        xg, best_score, best_learning_rate, best_depth, best_estimator = xg_boost(
            trnX, tstX, trnY, tstY, multi_class, PLOT
        )
        print(
            "     a) Suggested parametrization: {} estimators with {} depth and {} learning rate, which gives an AUC score of {:.2f}".format(
                best_estimator, best_depth, best_learning_rate, best_score
            )
        )
        print("     b) Confusion matrix:")
        print_confusion_matrix(xg, tstX, tstY, labels)

        print("3. Comparative performance:")
        accuracies = get_accuracy_models(tstX, tstY, nb, knn, dt, rf, xg)
        sensitivities = get_sensitivity_models(
            tstX, tstY, nb, knn, dt, rf, xg, multi_class
        )

        accuracies_folds.append(accuracies)
        sensitivities_folds.append(sensitivities)

        scores_to_print = [
            ["", "NB", "KNN", "DT", "RF", "XG"],
            [
                "3.1 Accuracy",
                round(accuracies["nb"], 2),
                round(accuracies["knn"], 2),
                round(accuracies["dt"], 2),
                round(accuracies["rf"], 2),
                round(accuracies["xg"], 2),
            ],
            [
                "3.2 Sensitivity",
                round(sensitivities["nb"], 2),
                round(sensitivities["knn"], 2),
                round(sensitivities["dt"], 2),
                round(sensitivities["rf"], 2),
                round(sensitivities["xg"], 2),
            ],
        ]
        print_table(scores_to_print)
    print_accuracies_sensitivities(accuracies_folds, sensitivities_folds)
