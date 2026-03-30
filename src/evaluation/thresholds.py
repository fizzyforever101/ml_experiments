import numpy as np

def select_threshold(y_true, y_prob, grid=None):
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)

    best_t = 0.5
    best_score = -np.inf

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    for t in grid:
        y_pred = (y_prob > t).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        youden = tpr - fpr

        if youden > best_score:
            best_score = youden
            best_t = t

    return float(best_t)

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
