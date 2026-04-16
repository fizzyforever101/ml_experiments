import numpy as np
from sklearn.metrics import roc_curve

def select_threshold(y_true, y_prob, grid=None):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    best_idx = np.argmax(tpr - fpr)
    
    return float(thresholds[best_idx])

def compute_group_thresholds(y_true, y_prob, protected, group_col, grid=None, default=0.5):
    thresholds = {}
    for group in protected[group_col].unique():
        idx = protected[group_col] == group
        if idx.sum() == 0:
            thresholds[group] = default
        else:
            thresholds[group] = select_threshold(y_true[idx], y_prob[idx], grid=grid)
    return thresholds

def apply_group_thresholds(y_prob, protected, group_col, thresholds, default=0.5):
    y_prob = np.asarray(y_prob)
    preds = np.zeros_like(y_prob, dtype=int)
    for i, group in enumerate(protected[group_col]):
        t = thresholds.get(group, default)
        preds[i] = 1 if y_prob[i] > t else 0
    return preds
