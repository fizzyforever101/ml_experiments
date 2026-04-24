import json
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# Load JSON
# -----------------------------
with open("./results/tables/olives_fairness_avg.json", "r") as f:
    data = json.load(f)


# -----------------------------
# Config: CHOOSE WHAT TO PLOT
# -----------------------------
PLOT_OVERALL = True
PLOT_SUBGROUP = True

# Choose which metrics to plot
METRICS = ["FNR", "FPR"]

# Choose which subgroup dimensions
SUBGROUP_TYPES = ["race", "ethnicity", "gender"]

# Optional: limit to specific models
MODELS = ["baseline", "reweighted", "thresholded"]


# -----------------------------
# Helper: Overall DataFrame
# -----------------------------
def get_overall_df(data):
    rows = []
    for model in MODELS:
        metrics = data[model]["overall"]
        row = {"model": model}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------
# Helper: Subgroup DataFrame
# -----------------------------
def get_subgroup_df(data, subgroup_type):
    rows = []
    for model in MODELS:
        subgroups = data[model]["subgroup"][subgroup_type]
        for group_name, metrics in subgroups.items():
            row = {
                "model": model,
                "group": group_name
            }
            row.update(metrics)
            rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------
# Plot: Overall metrics
# -----------------------------
def plot_overall(df):
    for metric in METRICS:
        plt.figure()
        pivot = df.set_index("model")[metric]
        pivot.plot(kind="bar")
        plt.title(f"Overall {metric}")
        plt.ylabel(metric)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()


# -----------------------------
# Plot: Subgroup metrics
# -----------------------------
def plot_subgroups(df, subgroup_type):
    for metric in METRICS:
        plt.figure()
        pivot = df.pivot(index="group", columns="model", values=metric)
        pivot.plot(kind="bar")
        plt.title(f"{subgroup_type.capitalize()} - {metric}")
        plt.ylabel(metric)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()


# -----------------------------
# MAIN
# -----------------------------
# if PLOT_OVERALL:
#     overall_df = get_overall_df(data)
#     plot_overall(overall_df)

# if PLOT_SUBGROUP:
#     for subgroup in SUBGROUP_TYPES:
#         df = get_subgroup_df(data, subgroup)
#         plot_subgroups(df, subgroup)

# -----------------------------
# Overall + Race FNR on SAME plot (no loop)

f"{config['output']['dir']}/plots/mimic_{reweight_attr}_fairness_ece.png"

# -----------------------------
race_df = get_subgroup_df(data, "race")

overall_df_plot = pd.DataFrame({
    "model": MODELS,
    "group": ["Overall"] * len(MODELS),
    "FNR": [data[m]["overall"]["FNR"] for m in MODELS]
})

combined_df = pd.concat([
    overall_df_plot,
    race_df[["model", "group", "FNR"]]
])

plt.figure()
plt.title("FNR: Overall + Race Subgroups")

pivot = combined_df.pivot(index="group", columns="model", values="FNR")
pivot.plot(kind="bar")

plt.ylabel("False Negative Rate")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("overall_race_fnr.png", dpi=300, bbox_inches="tight")

# -----------------------------
# Overall + Gender FNR on SAME plot (no loop)
# -----------------------------
race_df = get_subgroup_df(data, "gender")

overall_df_plot = pd.DataFrame({
    "model": MODELS,
    "group": ["Overall"] * len(MODELS),
    "FNR": [data[m]["overall"]["FNR"] for m in MODELS]
})

combined_df = pd.concat([
    overall_df_plot,
    race_df[["model", "group", "FNR"]]
])

plt.figure()
plt.title("FNR: Overall + Gender Subgroups")

pivot = combined_df.pivot(index="group", columns="model", values="FNR")
pivot.plot(kind="bar")

plt.ylabel("False Negative Rate")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("overall_gender_fnr.png", dpi=300, bbox_inches="tight")

# -----------------------------
# Overall + Race FPR on SAME plot (no loop)
# -----------------------------
race_df = get_subgroup_df(data, "race")

overall_df_plot = pd.DataFrame({
    "model": MODELS,
    "group": ["Overall"] * len(MODELS),
    "FPR": [data[m]["overall"]["FPR"] for m in MODELS]
})

combined_df = pd.concat([
    overall_df_plot,
    race_df[["model", "group", "FPR"]]
])

plt.figure()
plt.title("FPR: Overall + Race Subgroups")

pivot = combined_df.pivot(index="group", columns="model", values="FPR")
pivot.plot(kind="bar")

plt.ylabel("False Positive Rate")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("overall_race_fpr.png", dpi=300, bbox_inches="tight")

# -----------------------------
# Overall + Gender FPR on SAME plot (no loop)
# -----------------------------
race_df = get_subgroup_df(data, "gender")

overall_df_plot = pd.DataFrame({
    "model": MODELS,
    "group": ["Overall"] * len(MODELS),
    "FPR": [data[m]["overall"]["FPR"] for m in MODELS]
})

combined_df = pd.concat([
    overall_df_plot,
    race_df[["model", "group", "FPR"]]
])

plt.figure()
plt.title("FPR: Overall + Gender Subgroups")

pivot = combined_df.pivot(index="group", columns="model", values="FPR")
pivot.plot(kind="bar")

plt.ylabel("False Positive Rate")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("overall_gender_fpr.png", dpi=300, bbox_inches="tight")
plt.close()

