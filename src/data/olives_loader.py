import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import os

class OlivesDataset(Dataset):
    def __init__(self, img_dir, labels_csv, transform=None, label_col="label", id_col="image"):
        self.df = pd.read_csv(labels_csv)
        self.img_dir = img_dir
        self.transform = transform
        self.label_col = label_col
        self.id_col = id_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = os.path.join(self.img_dir, row[self.id_col])
        image = Image.open(img_path).convert("RGB")

        label = row[self.label_col]
        if isinstance(label, str):
            label = 1 if label.lower() in ["1", "true", "yes", "positive"] else 0

        if self.transform:
            image = self.transform(image)

        return image, label, row
