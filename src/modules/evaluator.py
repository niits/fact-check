import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

from .datasets.base import LABELS


def evaluate_file(
    input_file: str,
    labels: list[str] | None = None,
) -> dict:
    """Evaluate predictions and return report data.

    Returns a dict with keys:
        - cls_report_text: str (the printed classification report)
        - cls_report_dict: dict (classification_report with output_dict=True)
        - confusion_matrix: ndarray
        - df: DataFrame of the input file
    """
    if labels is None:
        labels = LABELS
    df = pd.read_csv(input_file)

    y_true = df["label"].tolist()
    y_pred = df["pred"].tolist()

    cls_report_text = classification_report(y_true, y_pred, labels=labels)
    cls_report_dict = classification_report(
        y_true, y_pred, labels=labels, output_dict=True
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    print(cls_report_text)
    print(matrix)

    return {
        "cls_report_text": cls_report_text,
        "cls_report_dict": cls_report_dict,
        "confusion_matrix": matrix,
        "df": df,
    }
