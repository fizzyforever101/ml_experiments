import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1

    ece = 0.0
    for i in range(n_bins):
        idx = bin_ids == i
        if np.any(idx):
            acc = y_true[idx].mean()
            conf = y_prob[idx].mean()
            ece += np.abs(acc - conf) * (idx.mean())
    return float(ece)

def compute_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    try:
        auroc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auroc = np.nan

    try:
        auprc = average_precision_score(y_true, y_prob)
    except ValueError:
        auprc = np.nan

    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "ECE": expected_calibration_error(y_true, y_prob),
        "FNR": fn / (fn + tp) if (fn + tp) > 0 else 0.0,
        "FPR": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "TPR" : tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "TNR" : tn / (tn + fp) if (tn + fp) > 0 else 0.0
    }
