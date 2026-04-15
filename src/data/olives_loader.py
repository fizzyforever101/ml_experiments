import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class OlivesDataset(Dataset):
    def __init__(self, img_dir, labels_csv, demographics_csv, transform=None, label_col="label",
                 path_col="Path (Trial/Arm/Folder/Visit/Eye/Image Name)",
                 trial_filter=None):
        self.img_dir = img_dir
        self.transform = transform
        self.label_col = label_col
        self.path_col = path_col

        labels_df = pd.read_csv(labels_csv)
        labels_df = labels_df.rename(columns={
            "Patient_ID": "patient_id"
        })
        labels_df["patient_id"] = labels_df["patient_id"].astype(str).str.strip()

        demo_df = pd.read_csv(demographics_csv)
        demo_df = demo_df.rename(columns={
            "Patient \nID": "patient_id",
            "Age": "age",
            "Gender": "gender",
            "Ethnicity": "ethnicity",
            "Race": "race",
        })
        demo_df["patient_id"] = demo_df["patient_id"].astype(str).str.strip()

        self.df = labels_df.merge(
            demo_df[["patient_id", "age", "gender", "ethnicity", "race"]],
            on="patient_id",
            how="inner",
        )

        valid_indices = []  # filter out TREX rows from the csv
        for i, row in self.df.iterrows():
            rel_path = str(row[self.path_col]).strip().lstrip("/")
            if rel_path.startswith("Prime_FULL/"):
                rel_path = rel_path[len("Prime_FULL/"):]

            img_path = os.path.join(self.img_dir, rel_path)

            if os.path.exists(img_path):
                valid_indices.append(i)

        # Only save PRIME samples
        self.df = self.df.loc[valid_indices].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        rel_path = str(row[self.path_col]).strip().lstrip("/")
        if rel_path.startswith("Prime_FULL/"):
            rel_path = rel_path[len("Prime_FULL/"):]

        img_path = os.path.join(self.img_dir, rel_path)
        image = Image.open(img_path).convert("RGB")

        label = int(row[self.label_col])

        if self.transform:
            image = self.transform(image)

        demographics = {
            "patient_id": row["patient_id"],
            "age": row["age"],
            "gender": row["gender"],
            "ethnicity": row["ethnicity"],
            "race": row["race"],
        }

        return image, label, rel_path, demographics