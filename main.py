import sys
import pandas as pd

from modules.preprocessing.preprocessing import normalize_df
from modules.classification.all_models import (
    split_dataset,
    create_classifier_models,
    get_accuracy_models,
)


def report(source, dataframe, task):
    out = ""
    df = dataframe.copy()

    if source == "PD":
        df = normalize_df(df)
        out += "1. Applied preprocessing: Normalization\n"

        if task == "preprocessing":
            return out
        if task == "classification":
            trnX, tstX, trnY, tstY, labels = split_dataset(df)
            create_classifier_models(trnX, trnY)

            out += "2. Classifiers:\n"
            out += "2.1 NB\n"
            out += "a) Suggested parametrization: \n"
            out += "b) Confusion matrix: \n"
            out += "2.2 KNN\n"
            out += "a) Suggested parametrization: \n"
            out += "b) Confusion matrix: \n"
            out += "3. Comparative performance: NB | KNN | DT | RF\n"

            accuracies = get_accuracy_models(tstX, tstY)
            out += "3.1 Accuracy: {:.2f} | {:.2f} | {:.2f} | {:.2f}\n".format(
                accuracies["nb"], accuracies["knn"], accuracies["dt"], accuracies["rf"]
            )
            out += "3.2 Sensitivity: \n"

    elif source == "CT":
        pass
    else:
        return "Invalid source"
    return out


if __name__ == "__main__":
    """A: read arguments"""
    args = sys.stdin.readline().rstrip("\n").split(" ")
    n, source, task = int(args[0]), args[1], args[2]
    task = task.replace("\r", "")

    """B: read dataset"""
    data, header = [], sys.stdin.readline().rstrip("\n").rstrip("\r").split(",")
    for i in range(n - 1):
        data.append(sys.stdin.readline().rstrip("\n").rstrip("\r").split(","))
    dataframe = pd.DataFrame(data, columns=header)

    """C: output results"""
    print(report(source, dataframe, task))
