from modules.functions import print_table


def get_list_value(seq):
    return {
        "min": round(min(seq), 2),
        "max": round(max(seq), 2),
        "avg": round(sum(seq) / len(seq), 2),
    }


def get_scores_from_classifier(accuricy_list, sensitivity_list, classifier):
    accuracy_values_classifier = [x[classifier] for x in accuricy_list]
    senility_values_classifier = [x[classifier] for x in sensitivity_list]

    classifier_accuracy = get_list_value(accuracy_values_classifier)
    classifier_senility = get_list_value(senility_values_classifier)
    return classifier_accuracy, classifier_senility


def print_accuracies_sensitivities(accuricy_list, sensitivity_list):
    scores = {"accuricy": {}, "sensitivity": {}}
    for classifier in accuricy_list[0]:
        scores["accuricy"][classifier], scores["sensitivity"][
            classifier
        ] = get_scores_from_classifier(accuricy_list, sensitivity_list, classifier)

    print("\nComparinson performance from K fold")
    scores_to_print = [
        ["", "", "NB", "KNN", "DT", "RF", "XG"],
        [
            "Accuracy",
            "Avg",
            scores["accuricy"]["nb"]["avg"],
            scores["accuricy"]["knn"]["avg"],
            scores["accuricy"]["dt"]["avg"],
            scores["accuricy"]["rf"]["avg"],
            scores["accuricy"]["xg"]["avg"],
        ],
        [
            "",
            "Min",
            scores["accuricy"]["nb"]["min"],
            scores["accuricy"]["knn"]["min"],
            scores["accuricy"]["dt"]["min"],
            scores["accuricy"]["rf"]["min"],
            scores["accuricy"]["xg"]["min"],
        ],
        [
            "",
            "Max",
            scores["accuricy"]["nb"]["max"],
            scores["accuricy"]["knn"]["max"],
            scores["accuricy"]["dt"]["max"],
            scores["accuricy"]["rf"]["max"],
            scores["accuricy"]["xg"]["max"],
        ],
        [
            "Sensitivity",
            "Avg",
            scores["sensitivity"]["nb"]["avg"],
            scores["sensitivity"]["knn"]["avg"],
            scores["sensitivity"]["dt"]["avg"],
            scores["sensitivity"]["rf"]["avg"],
            scores["sensitivity"]["xg"]["avg"],
        ],
        [
            "",
            "Min",
            scores["sensitivity"]["nb"]["min"],
            scores["sensitivity"]["knn"]["min"],
            scores["sensitivity"]["dt"]["min"],
            scores["sensitivity"]["rf"]["min"],
            scores["sensitivity"]["xg"]["min"],
        ],
        [
            "",
            "Max",
            scores["sensitivity"]["nb"]["max"],
            scores["sensitivity"]["knn"]["max"],
            scores["sensitivity"]["dt"]["max"],
            scores["sensitivity"]["rf"]["max"],
            scores["sensitivity"]["xg"]["max"],
        ],
    ]
    print_table(scores_to_print)


if __name__ == "__main__":
    accuracies_pd = [
        {"nb": 0.8, "knn": 0.8, "dt": 0.3, "rf": 0.3, "xg": 0.9},
        {"nb": 0.6, "knn": 1.0, "dt": 0.4, "rf": 0.4, "xg": 0.7},
        {"nb": 0.8, "knn": 0.8, "dt": 0.3, "rf": 0.7, "xg": 0.7},
        {"nb": 0.6, "knn": 0.7, "dt": 0.7, "rf": 0.7, "xg": 0.6},
        {"nb": 0.7, "knn": 0.7, "dt": 0.6, "rf": 0.4, "xg": 0.5},
        {"nb": 0.8, "knn": 0.8, "dt": 0.4, "rf": 0.6, "xg": 0.9},
        {"nb": 0.6, "knn": 0.8, "dt": 0.5, "rf": 0.5, "xg": 0.7},
        {"nb": 0.8, "knn": 0.9, "dt": 0.2, "rf": 0.8, "xg": 0.7},
        {"nb": 0.7, "knn": 0.8, "dt": 0.2, "rf": 0.2, "xg": 0.7},
        {"nb": 0.9, "knn": 0.7, "dt": 0.6, "rf": 0.5, "xg": 0.7},
    ]

    sensitivity_pd = [
        {
            "nb": 0.7619047619047619,
            "knn": 0.8571428571428572,
            "dt": 0.5,
            "rf": 0.5,
            "xg": 0.8333333333333333,
        },
        {"nb": 0.6666666666666667, "knn": 1.0, "dt": 0.5, "rf": 0.5, "xg": 0.75},
        {
            "nb": 0.8571428571428572,
            "knn": 0.8571428571428572,
            "dt": 0.5,
            "rf": 0.5,
            "xg": 0.6904761904761906,
        },
        {
            "nb": 0.7142857142857143,
            "knn": 0.6904761904761905,
            "dt": 0.5,
            "rf": 0.5,
            "xg": 0.7142857142857143,
        },
        {"nb": 0.75, "knn": 0.7083333333333334, "dt": 0.5, "rf": 0.5, "xg": 0.5},
        {
            "nb": 0.8333333333333334,
            "knn": 0.8333333333333334,
            "dt": 0.5,
            "rf": 0.5,
            "xg": 0.875,
        },
        {
            "nb": 0.6000000000000001,
            "knn": 0.8,
            "dt": 0.5,
            "rf": 0.5,
            "xg": 0.7000000000000001,
        },
        {"nb": 0.875, "knn": 0.9375, "dt": 0.5, "rf": 0.5, "xg": 0.8125},
        {"nb": 0.625, "knn": 0.6875, "dt": 0.5, "rf": 0.5, "xg": 0.625},
        {
            "nb": 0.9166666666666667,
            "knn": 0.7083333333333334,
            "dt": 0.5,
            "rf": 0.5833333333333333,
            "xg": 0.7083333333333334,
        },
    ]

    print_accuracies_sensitivities(accuracies_pd, sensitivity_pd)
