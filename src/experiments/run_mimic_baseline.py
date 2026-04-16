import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.mimic_loader import load_mimic
from src.models.train_tabular import get_model
from src.evaluation.subgroup import subgroup_analysis
from src.evaluation.plots import plot_fnr_gap, plot_calibration, plot_threshold_tradeoff
from src.evaluation.calibration import get_calibration
from src.evaluation.metrics import compute_metrics
from src.utils.io import save_json
from src.utils.seed import set_seed

config = yaml.safe_load(open("config/config.yaml"))
set_seed(config["training"]["seed"])

X, y, protected = load_mimic(
    config["data"]["mimic_path"],
    config["fairness"]["mimic_protected_attributes"]
)

max_rows = config.get("training", {}).get("mimic_max_rows")
if max_rows and len(y) > max_rows:
    original_rows = len(y)
    sampled_idx = y.sample(n=max_rows, random_state=config["training"]["seed"]).index
    X = X.loc[sampled_idx]
    y = y.loc[sampled_idx]
    protected = protected.loc[sampled_idx]
    print(f"Using sampled MIMIC rows: {len(y)} (from {original_rows})")

X_train, X_test, y_train, y_test, p_train, p_test = train_test_split(
    X, y, protected, test_size=0.2, stratify=y
)

threshold = config["fairness"].get("threshold", 0.5)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = get_model(config["model"]["tabular"]["type"])
model.fit(X_train, y_train)

metrics = {}

for attr in config["fairness"]["mimic_protected_attributes"]:
    res = subgroup_analysis(model, X_test, y_test, p_test, attr, threshold=threshold)

    print(f"\n{attr}:", res)

    plot_fnr_gap(
        res,
        f"{config['output']['dir']}/plots/mimic_{attr}_fnr.png",
        f"MIMIC FNR by {attr} (threshold={threshold})"
    )
    metrics[attr] = res

# overall calibration + metrics
y_prob = model.predict_proba(X_test)[:, 1]
y_pred = (y_prob > config["fairness"]["threshold"]).astype(int)
overall = compute_metrics(y_test, y_pred, y_prob)
metrics["overall"] = overall

prob_true, prob_pred = get_calibration(y_test, y_prob)
plot_calibration(
    prob_true,
    prob_pred,
    f"{config['output']['dir']}/plots/mimic_calibration.png",
    "MIMIC Overall"
)
plot_threshold_tradeoff(
    y_test,
    y_prob,
    f"{config['output']['dir']}/plots/mimic_threshold_tradeoff.png"
)

save_json(
    metrics,
    f"{config['output']['dir']}/tables/mimic_baseline_metrics.json"
)
