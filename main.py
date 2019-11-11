import sys
import pandas as pd
import warnings

from modules.preprocessing.preprocessing_report import preprocessing_report
from modules.classification.classification_report import classification_report
from modules.unsupervised.unsupervised_report import unsupervised_report

warnings.filterwarnings("ignore")


def report(source, dataframe, task):
    if source != "PD" and source != "CT":
        return "Invalid source"

    df = dataframe.copy()
    df = preprocessing_report(data=df, source=source)

    if task == "preprocessing":
        return ""
    elif task == "classification":
        classification_report(data=df, source=source)
    elif task == "unsupervised":
        unsupervised_report(data=df, source=source)
    return ""


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
