import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def plot_fnr_gap(subgroup_results, save_path, title):
    ensure_dir(os.path.dirname(save_path))

    groups = list(subgroup_results.keys())
    fnrs = [subgroup_results[g]["FNR"] for g in groups]

    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(groups, fnrs)
    plt.xticks(rotation=45)
    plt.title(title)
    plt.ylabel("False Negative Rate")

    # When all FNR values are near 1.0, zooming avoids visually flattening groups.
    fmin, fmax = min(fnrs), max(fnrs)
    if fmin > 0.9:
        lower = max(0.0, fmin - 0.03)
        upper = min(1.0, max(fmax + 0.005, lower + 0.01))
        plt.ylim(lower, upper)
    else:
        plt.ylim(0, 1)

    for bar, val in zip(bars, fnrs):
        plt.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_calibration(prob_true, prob_pred, save_path, label):
    ensure_dir(os.path.dirname(save_path))

    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker='o', label=label)
    plt.plot([0,1],[0,1], linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Calibration Curve")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_metric_comparison(before, after, save_path, metric="FNR"):
    ensure_dir(os.path.dirname(save_path))

    groups = list(before.keys())
    before_vals = [before[g][metric] for g in groups]
    after_vals = [after[g][metric] for g in groups]

    x = np.arange(len(groups))

    plt.figure(figsize=(9, 4.8))
    bars_before = plt.bar(x - 0.2, before_vals, width=0.4, label="Before")
    bars_after = plt.bar(x + 0.2, after_vals, width=0.4, label="After")

    plt.xticks(x, groups, rotation=45)
    plt.ylabel(metric)
    plt.title(f"{metric} Before vs After")

    vals = before_vals + after_vals
    vmin, vmax = min(vals), max(vals)
    if vmin > 0.9 and metric.upper() == "FNR":
        lower = max(0.0, vmin - 0.03)
        upper = min(1.0, max(vmax + 0.005, lower + 0.01))
        plt.ylim(lower, upper)
    elif metric.upper() == "ECE" and vmax < 0.05:
        plt.ylim(0, max(0.03, vmax * 1.25))
    else:
        plt.ylim(0, max(1.0, vmax * 1.1))

    for bar, val in zip(bars_before, before_vals):
        plt.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars_after, after_vals):
        plt.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_threshold_tradeoff(y_true, y_prob, save_path, thresholds=None):
    ensure_dir(os.path.dirname(save_path))

    if thresholds is None:
        thresholds = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]

    recalls, precisions, fprs = [], [], []
    for t in thresholds:
        y_pred = (y_prob > t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        recalls.append(rec)
        precisions.append(ppv)
        fprs.append(fpr)

    plt.figure(figsize=(8.2, 4.8))
    plt.plot(thresholds, recalls, marker="o", label="Recall")
    plt.plot(thresholds, fprs, marker="o", label="FPR")
    plt.plot(thresholds, precisions, marker="o", label="Precision")
    plt.gca().invert_xaxis()
    plt.ylim(0, 1)
    plt.xlabel("Threshold (decreasing left to right)")
    plt.ylabel("Metric Value")
    plt.title("Operating-Point Tradeoff Across Thresholds")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()