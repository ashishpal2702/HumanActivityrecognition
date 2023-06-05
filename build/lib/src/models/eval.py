import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    ConfusionMatrixDisplay,
    r2_score,
    roc_curve,
    recall_score,
    precision_score,
)
import shap


class Eval:

    def eval_metrics(self, y_pred, y_actual):
        cm = confusion_matrix(y_actual, y_pred, labels=y_actual.unique())
        roc_score = roc_auc_score(y_actual, y_pred)
        f1 = f1_score(y_actual, y_pred)
        accuracy = accuracy_score(y_actual, y_pred)
        precision = precision_score(y_actual, y_pred)
        recall = recall_score(y_actual, y_pred)
        return accuracy, roc_score, f1, precision, recall

    def interpret_model_results(self, model, x):
        explainer = shap.Explainer(model)
        shap_values = explainer(x)
        return shap_values
