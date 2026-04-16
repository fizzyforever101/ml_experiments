import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataloader import default_collate
import torchvision.transforms as T
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.data.olives_loader import OlivesDataset
from src.models.train_image import get_model
from src.evaluation.subgroup import subgroup_analysis_from_probs
from src.evaluation.plots import plot_fnr_gap
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
    config["data"]["olives"]["demographics_csv"],
    transform=transform,
    label_col="DRT/ME",
    path_col="Path (Trial/Arm/Folder/Visit/Eye/Image Name)",
)

labels = dataset.df[config["data"]["olives"]["label_col"]].values
train_idx, test_idx = train_test_split(
    range(len(dataset)),
    test_size=0.2,
    stratify=labels,
    random_state=config["training"]["seed"]
)

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("GPU available: ", torch.cuda.is_available())

model = get_model().to(device)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])

# img, label, path, demo = dataset[0]
# print(type(img), img.shape)   # feature (image tensor)
# print(type(label), label)     # label
# print(type(demo), demo)       # demographic data

print(f"Total dataset size: {len(dataset)}")
print(f"Training set size: {len(train_idx)}")
print(f"Test set size: {len(test_idx)}")

train_labels = labels[train_idx]
test_labels = labels[test_idx]

print("Train class counts:")
unique, counts = np.unique(train_labels, return_counts=True)
print(dict(zip(unique, counts)))

print("Test class counts:")
unique, counts = np.unique(test_labels, return_counts=True)
print(dict(zip(unique, counts)))

model.load_state_dict(torch.load("olives_model_250.pth", map_location=device, weights_only=True))
model.eval()

y_true, y_prob, meta_rows = [], [], []
with torch.no_grad():
    for imgs, labels, paths, meta in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs).squeeze()
        probs = torch.sigmoid(outputs).cpu().numpy()

        y_true.extend(labels.numpy())
        y_prob.extend(probs)
        meta_rows.extend(meta)

fpr, tpr, thresholds = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

# Plot
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle='--')  # diagonal baseline
best_idx = np.argmax(tpr - fpr)
plt.scatter(fpr[best_idx], tpr[best_idx])
plt.annotate(
    f"Threshold = {thresholds[best_idx]:.2f}",
    (fpr[best_idx], tpr[best_idx]),
    textcoords="offset points",
    xytext=(10, -10),
    ha='left'
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.savefig("roc_curve.png", dpi=300, bbox_inches="tight")
plt.close()

meta_df = pd.DataFrame(meta_rows)
metrics = {}
meta_df["age"] = (meta_df["age"] // 10) * 10

for attr in config["fairness"]["olives_protected_attributes"]:
    res = subgroup_analysis_from_probs(
        y_true,
        y_prob,
        meta_df,
        attr,
        threshold=config["fairness"]["threshold"]
    )
    metrics[attr] = res
    # plot_fnr_gap(
    #     res,
    #     f"{config['output']['dir']}/plots/olives_{attr}_fnr.png",
    #     f"OLIVES FNR by {attr}"
    # )

save_json(
    metrics,
    f"{config['output']['dir']}/tables/olives_baseline_metrics.json"
)
