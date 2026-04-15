import yaml
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.utils.data.dataloader import default_collate
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
import pandas as pd

from src.data.olives_loader import OlivesDataset
from src.models.train_image import get_model
from src.fairness.reweighing import compute_group_weights
from src.evaluation.subgroup import subgroup_analysis_from_probs
from src.evaluation.thresholds import compute_group_thresholds
from src.evaluation.metrics import compute_metrics
from src.evaluation.plots import plot_metric_comparison
from src.utils.io import save_json
from src.utils.seed import set_seed

config = yaml.safe_load(open("config/config.yaml"))
set_seed(config["training"]["seed"])

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

dataset = OlivesDataset(
    config["data"]["olives"]["img_dir"],
    config["data"]["olives"]["labels_csv"],
    transform=transform,
    label_col=config["data"]["olives"]["label_col"],
    id_col=config["data"]["olives"]["id_col"]
)

labels = dataset.df[config["data"]["olives"]["label_col"]].values
train_idx, temp_idx = train_test_split(
    range(len(dataset)),
    test_size=0.3,
    stratify=labels,
    random_state=config["training"]["seed"]
)
val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5,
    stratify=labels[temp_idx],
    random_state=config["training"]["seed"]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(train_loader):
    model = get_model().to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])

    model.train()
    for epoch in range(config["training"]["epochs"]):
        for imgs, labels, meta in train_loader:
            imgs = imgs.to(device)
            labels = labels.float().to(device)
            outputs = model(imgs).squeeze()
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} loss: {loss.item()}")

    return model

def collect_outputs(model, loader):
    model.eval()
    y_true, y_prob, meta_rows = [], [], []
    with torch.no_grad():
        for imgs, labels, meta in loader:
            imgs = imgs.to(device)
            outputs = model(imgs).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()

            y_true.extend(labels.numpy())
            y_prob.extend(probs)
            meta_rows.extend([m.to_dict() for m in meta])

    return (
        pd.Series(y_true),
        pd.Series(y_prob),
        pd.DataFrame(meta_rows)
    )

train_loader = DataLoader(
    Subset(dataset, train_idx),
    batch_size=config["training"]["batch_size"],
    shuffle=True,
    collate_fn=lambda batch: (
        default_collate([b[0] for b in batch]),
        default_collate([b[1] for b in batch]),
        [b[2] for b in batch],
    )
)
val_loader = DataLoader(
    Subset(dataset, val_idx),
    batch_size=config["training"]["batch_size"],
    shuffle=False,
    collate_fn=lambda batch: (
        default_collate([b[0] for b in batch]),
        default_collate([b[1] for b in batch]),
        [b[2] for b in batch],
    )
)
test_loader = DataLoader(
    Subset(dataset, test_idx),
    batch_size=config["training"]["batch_size"],
    shuffle=False,
    collate_fn=lambda batch: (
        default_collate([b[0] for b in batch]),
        default_collate([b[1] for b in batch]),
        [b[2] for b in batch],
    )
)

# Baseline
baseline_model = train_model(train_loader)
val_y_true, val_y_prob, val_meta = collect_outputs(baseline_model, val_loader)
test_y_true, test_y_prob, test_meta = collect_outputs(baseline_model, test_loader)

baseline = {}
for attr in config["fairness"]["olives_protected_attributes"]:
    baseline[attr] = subgroup_analysis_from_probs(
        test_y_true,
        test_y_prob,
        test_meta,
        attr,
        threshold=config["fairness"]["threshold"]
    )

# Reweighting
reweight_attr = config["fairness"]["olives_reweight_attribute"]
train_meta = dataset.df.iloc[list(train_idx)]
weights = compute_group_weights(train_meta, reweight_attr)
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

reweight_loader = DataLoader(
    Subset(dataset, train_idx),
    batch_size=config["training"]["batch_size"],
    sampler=sampler,
    collate_fn=lambda batch: (
        default_collate([b[0] for b in batch]),
        default_collate([b[1] for b in batch]),
        [b[2] for b in batch],
    )
)

reweighted_model = train_model(reweight_loader)
rw_test_y_true, rw_test_y_prob, rw_test_meta = collect_outputs(reweighted_model, test_loader)

reweighted = {}
for attr in config["fairness"]["olives_protected_attributes"]:
    reweighted[attr] = subgroup_analysis_from_probs(
        rw_test_y_true,
        rw_test_y_prob,
        rw_test_meta,
        attr,
        threshold=config["fairness"]["threshold"]
    )

# Group-specific thresholds (computed on val)
threshold_attr = config["fairness"]["olives_threshold_attribute"]
thresholds = compute_group_thresholds(val_y_true, val_y_prob, val_meta, threshold_attr)

thresholded = {}
for group in test_meta[threshold_attr].unique():
    idx = test_meta[threshold_attr] == group
    y_g = test_y_true[idx]
    y_prob_g = test_y_prob[idx]
    y_pred_g = (y_prob_g > thresholds[group]).astype(int)
    thresholded[group] = compute_metrics(y_g, y_pred_g, y_prob_g)

plot_metric_comparison(
    baseline[threshold_attr],
    thresholded,
    f"{config['output']['dir']}/plots/olives_{threshold_attr}_threshold_fnr.png",
    metric="FNR"
)
plot_metric_comparison(
    baseline[threshold_attr],
    reweighted[threshold_attr],
    f"{config['output']['dir']}/plots/olives_{threshold_attr}_reweight_ece.png",
    metric="ECE"
)

save_json(
    {
        "baseline": baseline,
        "reweighted": reweighted,
        "thresholded": thresholded,
        "thresholds": thresholds
    },
    f"{config['output']['dir']}/tables/olives_fairness_{threshold_attr}.json"
)
