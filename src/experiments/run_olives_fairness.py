import yaml
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torch.utils.data.dataloader import default_collate
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt

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

SAVE = True
unq_id = 1

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

dataset = OlivesDataset(
    config["data"]["olives"]["img_dir"],
    config["data"]["olives"]["labels_csv"],
    config["data"]["olives"]["demographics_csv"],
    transform=transform,
    label_col="DRT/ME",
    path_col="Path (Trial/Arm/Folder/Visit/Eye/Image Name)",
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
    epoch_losses = []

    for epoch in range(config["training"]["epochs"]):
        epoch_loss = 0.0
        for imgs, labels, paths, meta in train_loader:
            imgs = imgs.to(device)
            labels = labels.float().to(device)

            outputs = model(imgs).squeeze()
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        epoch_losses.append({
            "epoch": epoch + 1,
            "loss": float(epoch_loss)
        })

        print(f"Epoch {epoch+1} loss: {epoch_loss:.10f}")

    return model, epoch_losses

def collect_outputs(model, loader):
    model.eval()
    y_true, y_prob, meta_rows = [], [], []
    with torch.no_grad():
        for imgs, labels, paths, meta in loader:
            imgs = imgs.to(device)
            outputs = model(imgs).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()

            y_true.extend(labels.numpy())
            y_prob.extend(probs)
            meta_rows.extend(meta)

    return (
        pd.Series(y_true),
        pd.Series(y_prob),
        pd.DataFrame(meta_rows)
    )

def compute_overall_from_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob > threshold).astype(int)
    return compute_metrics(y_true, y_pred, y_prob)

def compute_overall_from_group_thresholds(y_true, y_prob, meta, group_attr, thresholds):
    y_pred = pd.Series(index=y_prob.index, dtype=int)

    for group in meta[group_attr].unique():
        idx = meta[group_attr] == group
        y_pred.loc[idx] = (y_prob.loc[idx] > thresholds[group]).astype(int)

    return compute_metrics(y_true, y_pred.astype(int), y_prob)

train_loader = DataLoader(
    Subset(dataset, train_idx),
    batch_size=config["training"]["batch_size"],
    shuffle=True,
    collate_fn=lambda batch: (
        default_collate([b[0] for b in batch]),
        default_collate([b[1] for b in batch]),
        [b[2] for b in batch],
        [b[3] for b in batch],
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
        [b[3] for b in batch],
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
        [b[3] for b in batch],
    )
)

## Baseline
print("Starting Baseline Training")
baseline_model, losses = train_model(train_loader)
if SAVE: 
    torch.save(baseline_model.state_dict(), f"olives_baseline_{unq_id}.pth")
save_json(
    {"epoch_losses": losses},
    f"{config['output']['dir']}/tables/olives_baseline_losses_{unq_id}.json"
)

print("Starting Baseline Testing")
val_y_true, val_y_prob, val_meta = collect_outputs(baseline_model, val_loader)
test_y_true, test_y_prob, test_meta = collect_outputs(baseline_model, test_loader)

fpr, tpr, roc_thresholds = roc_curve(val_y_true, val_y_prob)
roc_auc = auc(fpr, tpr)

# Plot
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle='--')  # diagonal baseline
best_idx = np.argmax(tpr - fpr)
baseline_threshold = float(roc_thresholds[best_idx])
plt.scatter(fpr[best_idx], tpr[best_idx])
plt.annotate(
    f"Threshold = {roc_thresholds[best_idx]:.6f}",
    (fpr[best_idx], tpr[best_idx]),
    textcoords="offset points",
    xytext=(10, -10),
    ha='left'
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Baseline ROC Curve")
plt.legend()

plt.savefig("baseline_roc.png", dpi=300, bbox_inches="tight")
plt.close()

baseline = {}
for attr in config["fairness"]["olives_protected_attributes"]:
    baseline[attr] = subgroup_analysis_from_probs(
        test_y_true,
        test_y_prob,
        test_meta,
        attr,
        threshold=baseline_threshold
        #threshold=config["fairness"]["threshold"]
    )

baseline_overall = compute_overall_from_threshold(
    test_y_true,
    test_y_prob,
    baseline_threshold
)

## Group-specific thresholds (computed on val)
print("Starting Threshold Calcs")
threshold_attr = config["fairness"]["olives_threshold_attribute"]
thresholds = compute_group_thresholds(val_y_true, val_y_prob, val_meta, threshold_attr)

# Build thresholded predictions for the full test set first
thresholded_test_pred = pd.Series(index=test_y_prob.index, dtype=int)

for group in test_meta[threshold_attr].unique():
    idx = test_meta[threshold_attr] == group
    thresholded_test_pred.loc[idx] = (
        test_y_prob.loc[idx] > thresholds[group]
    ).astype(int)

thresholded_test_pred = thresholded_test_pred.astype(int)

# Match baseline/reweighted format: metrics for every protected attribute
thresholded = {}
for attr in config["fairness"]["olives_protected_attributes"]:
    thresholded[attr] = {}

    for group in test_meta[attr].unique():
        idx = test_meta[attr] == group
        y_g = test_y_true.loc[idx]
        y_prob_g = test_y_prob.loc[idx]
        y_pred_g = thresholded_test_pred.loc[idx]

        thresholded[attr][group] = compute_metrics(y_g, y_pred_g, y_prob_g)


thresholded_overall = compute_metrics(
    test_y_true,
    thresholded_test_pred,
    test_y_prob
)

## Reweighting
print("Starting Reweighting Training")
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
        [b[3] for b in batch],
    )
)

reweighted_model, losses = train_model(reweight_loader)
if SAVE:
    torch.save(reweighted_model.state_dict(), f"olives_reweighted_{unq_id}.pth")
save_json(
    {"epoch_losses": losses},
    f"{config['output']['dir']}/tables/olives_reweighted_losses_{unq_id}.json"
)

print("Starting Reweighting Testing")
val_y_true, val_y_prob, val_meta = collect_outputs(reweighted_model, val_loader)
rw_test_y_true, rw_test_y_prob, rw_test_meta = collect_outputs(reweighted_model, test_loader)

fpr, tpr, roc_thresholds = roc_curve(val_y_true, val_y_prob)
roc_auc = auc(fpr, tpr)

# Plot
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle='--')  # diagonal baseline
best_idx = np.argmax(tpr - fpr)
reweighted_threshold = float(roc_thresholds[best_idx])

plt.scatter(fpr[best_idx], tpr[best_idx])
plt.annotate(
    f"Threshold = {roc_thresholds[best_idx]:.6f}",
    (fpr[best_idx], tpr[best_idx]),
    textcoords="offset points",
    xytext=(10, -10),
    ha='left'
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Reweighted ROC Curve")
plt.legend()

plt.savefig("reweight_roc.png", dpi=300, bbox_inches="tight")
plt.close()

reweighted = {}
for attr in config["fairness"]["olives_protected_attributes"]:
    reweighted[attr] = subgroup_analysis_from_probs(
        rw_test_y_true,
        rw_test_y_prob,
        rw_test_meta,
        attr,
        threshold=reweighted_threshold
        #threshold=config["fairness"]["threshold"]
    )


reweighted_overall = compute_overall_from_threshold(
    rw_test_y_true,
    rw_test_y_prob,
    reweighted_threshold
)


plot_metric_comparison(
    baseline[threshold_attr],
    thresholded[threshold_attr],
    f"{config['output']['dir']}/plots/olives_{threshold_attr}_threshold_fnr.png",
    metric="FNR"
)
plot_metric_comparison(
    baseline[threshold_attr],
    thresholded[threshold_attr],
    f"{config['output']['dir']}/plots/olives_{threshold_attr}_threshold_fpr.png",
    metric="FPR"
)
plot_metric_comparison(
    baseline[threshold_attr],
    reweighted[threshold_attr],
    f"{config['output']['dir']}/plots/olives_{threshold_attr}_reweight_fnr.png",
    metric="FNR"
)
plot_metric_comparison(
    baseline[threshold_attr],
    reweighted[threshold_attr],
    f"{config['output']['dir']}/plots/olives_{threshold_attr}_reweight_fpr.png",
    metric="FPR"
)

save_json(
    {
        "baseline": {
            "subgroup": baseline,
            "overall": baseline_overall,
            "threshold": baseline_threshold
        },
        "reweighted": {
            "subgroup": reweighted,
            "overall": reweighted_overall,
            "threshold": reweighted_threshold
        },
        "thresholded": {
            "subgroup": thresholded,
            "overall": thresholded_overall,
            "thresholds": thresholds
        }
    },
    f"{config['output']['dir']}/tables/olives_fairness_{unq_id}.json"
)