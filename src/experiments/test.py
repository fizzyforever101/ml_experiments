import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


JSON_PATH = "./results/tables/olives_fairness_avg.json"
SAVE_DIR = "./results/plots"


def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


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
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

    for bar, val in zip(bars_after, after_vals):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def load_results(path):
    with open(path, "r") as f:
        return json.load(f)


def add_overall_group(results_for_model, subgroup_type, metric):
    out = {
        "Overall": {
            metric: results_for_model["overall"][metric]
        }
    }

    for group_name, metric_dict in results_for_model["subgroup"][subgroup_type].items():
        out[group_name] = {
            metric: metric_dict[metric]
        }

    return out


def main():
    data = load_results(JSON_PATH)

    plots = [
        ("baseline", "reweighted", "race", "FNR", "baseline_vs_reweighted_race_fnr.png"),
        ("baseline", "thresholded", "race", "FNR", "baseline_vs_thresholded_race_fnr.png"),
        ("baseline", "reweighted", "gender", "FNR", "baseline_vs_reweighted_gender_fnr.png"),
        ("baseline", "thresholded", "gender", "FNR", "baseline_vs_thresholded_gender_fnr.png"),
        ("baseline", "reweighted", "race", "FPR", "baseline_vs_reweighted_race_fpr.png"),
        ("baseline", "thresholded", "race", "FPR", "baseline_vs_thresholded_race_fpr.png"),
        ("baseline", "reweighted", "gender", "FPR", "baseline_vs_reweighted_gender_fpr.png"),
        ("baseline", "thresholded", "gender", "FPR", "baseline_vs_thresholded_gender_fpr.png"),
    ]

    for before_model, after_model, subgroup_type, metric, filename in plots:
        before = add_overall_group(data[before_model], subgroup_type, metric)
        after = add_overall_group(data[after_model], subgroup_type, metric)

        save_path = os.path.join(SAVE_DIR, filename)
        print(f"Saving {save_path}")

        plot_metric_comparison(
            before=before,
            after=after,
            save_path=save_path,
            metric=metric
        )

    print("Done.")


if __name__ == "__main__":
    main()