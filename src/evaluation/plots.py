import os
import matplotlib.pyplot as plt
import numpy as np

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def plot_fnr_gap(subgroup_results, save_path, title):
    ensure_dir(os.path.dirname(save_path))

    groups = list(subgroup_results.keys())
    fnrs = [subgroup_results[g]["FNR"] for g in groups]

    plt.figure()
    plt.bar(groups, fnrs)
    plt.xticks(rotation=45)
    plt.title(title)
    plt.ylabel("False Negative Rate")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_calibration(prob_true, prob_pred, save_path, label):
    ensure_dir(os.path.dirname(save_path))

    plt.figure()
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

    plt.figure()
    plt.bar(x - 0.2, before_vals, width=0.4, label="Before")
    plt.bar(x + 0.2, after_vals, width=0.4, label="After")

    plt.xticks(x, groups, rotation=45)
    plt.ylabel(metric)
    plt.title(f"{metric} Before vs After")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()