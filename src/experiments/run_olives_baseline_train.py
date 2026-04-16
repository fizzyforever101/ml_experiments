import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataloader import default_collate
import torchvision.transforms as T
import yaml
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

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

epoch_losses = []

for epoch in range(config["training"]["epochs"]):
    model.train()
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

torch.save(model.state_dict(), "olives_model.pth")

save_json(
    {"epoch_losses": epoch_losses},
    f"{config['output']['dir']}/tables/olives_epoch_losses.json"
)
