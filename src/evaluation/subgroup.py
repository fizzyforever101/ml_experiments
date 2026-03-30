import numpy as np

from src.evaluation.metrics import compute_metrics

def _predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1 / (1 + np.exp(-scores))
    raise ValueError("Model must implement predict_proba or decision_function.")

def subgroup_analysis(model, X, y, protected, group_col, threshold=0.5):
    results = {}

    for group in protected[group_col].unique():
        idx = protected[group_col] == group
        X_g = X[idx]
        y_g = y[idx]

        y_prob = _predict_proba(model, X_g)
        y_pred = (y_prob > threshold).astype(int)
        results[group] = compute_metrics(y_g, y_pred, y_prob)

    return results

def subgroup_analysis_from_probs(y_true, y_prob, protected, group_col, threshold=0.5):
    results = {}
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    for group in protected[group_col].unique():
        idx = protected[group_col] == group
        y_g = y_true[idx]
        y_prob_g = y_prob[idx]
        y_pred_g = (y_prob_g > threshold).astype(int)
        results[group] = compute_metrics(y_g, y_pred_g, y_prob_g)

    return results
