import json
from copy import deepcopy


# -----------------------------
# Load multiple JSON files
# -----------------------------
file_paths = [
    "results/tables/olives_fairness_2542.json",
    "results/tables/olives_fairness_2543.json",
    "results/tables/olives_fairness_2544.json",
    "results/tables/olives_fairness_2545.json",
    "results/tables/olives_fairness_2546.json",
]

json_list = []
for path in file_paths:
    with open(path, "r") as f:
        json_list.append(json.load(f))


# -----------------------------
# Recursive averaging function
# -----------------------------
def average_dicts(dicts):
    """
    Recursively average a list of dictionaries with identical structure.
    """
    if isinstance(dicts[0], dict):
        result = {}
        for key in dicts[0]:
            values = [d[key] for d in dicts]
            result[key] = average_dicts(values)
        return result

    elif isinstance(dicts[0], (int, float)):
        return sum(dicts) / len(dicts)

    else:
        # For non-numeric fields (e.g., strings), just take the first
        return dicts[0]


# -----------------------------
# Compute average
# -----------------------------
avg_data = average_dicts(json_list)


# -----------------------------
# Save result
# -----------------------------
with open("results/tables/olives_fairness_avg.json", "w") as f:
    json.dump(avg_data, f, indent=4)

print("Saved averaged metrics")