import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.mimic_loader import load_mimic
from src.models.train_tabular import get_model
from src.fairness.reweighing import compute_group_weights
from src.evaluation.subgroup import subgroup_analysis
from src.evaluation.plots import plot_metric_comparison
from src.utils.io import save_json
from src.utils.seed import set_seed

config = yaml.safe_load(open("config/config.yaml"))
set_seed(config["training"]["seed"])

protected_attrs = config["fairness"]["mimic_protected_attributes"]
reweight_attr = config["fairness"]["mimic_reweight_attribute"]

X, y, protected = load_mimic(
    config["data"]["mimic_path"],
    protected_attrs
)

X_train, X_test, y_train, y_test, p_train, p_test = train_test_split(
    X, y, protected, test_size=0.2, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

baseline_model = get_model(config["model"]["tabular"]["type"])
baseline_model.fit(X_train, y_train)

baseline = subgroup_analysis(baseline_model, X_test, y_test, p_test, reweight_attr)

weights = compute_group_weights(p_train, reweight_attr)
reweighted_model = get_model(config["model"]["tabular"]["type"])
reweighted_model.fit(X_train, y_train, sample_weight=weights)

after = subgroup_analysis(reweighted_model, X_test, y_test, p_test, reweight_attr)

plot_metric_comparison(
    baseline,
    after,
    f"{config['output']['dir']}/plots/mimic_{reweight_attr}_fairness_fnr.png",
    metric="FNR"
)
plot_metric_comparison(
    baseline,
    after,
    f"{config['output']['dir']}/plots/mimic_{reweight_attr}_fairness_ece.png",
    metric="ECE"
)

save_json(
    {"baseline": baseline, "reweighted": after},
    f"{config['output']['dir']}/tables/mimic_fairness_{reweight_attr}.json"
)
