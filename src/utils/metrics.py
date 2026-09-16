"""Common metrics utilities wrapping sklearn metrics.

These functions return numpy arrays/dicts and do not print; printing
and reporting is handled by the evaluation script.
"""
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import numpy as np


def classification_report_dict(y_true, y_pred, target_names=None):
    """Return sklearn classification report as a dict."""
    labels = None
    if target_names is not None:
        labels = list(range(len(target_names)))
    return classification_report(y_true, y_pred, labels=labels, target_names=target_names, output_dict=True)


def confusion_matrix_array(y_true, y_pred):
    """Return confusion matrix as numpy array."""
    return confusion_matrix(y_true, y_pred)


def accuracy(y_true, y_pred):
    return float(accuracy_score(y_true, y_pred))


def precision_recall_f1(y_true, y_pred, average='macro'):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    return {'precision': float(p), 'recall': float(r), 'f1': float(f1)}


def pair_overlap_count(cm, i, j):
    """Return symmetric count of confusions between classes i and j."""
    cm = np.asarray(cm)
    return int(cm[i, j] + cm[j, i])

