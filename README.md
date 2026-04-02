# Clinical AI Fairness

## Overview
This project evaluates bias in clinical AI systems using:
- MIMIC-III (EHR)
- OLIVES (medical imaging)

## Features
- Subgroup fairness analysis
- Calibration curves
- Fairness mitigation (reweighting)
- Automatic plotting

## Run

pip install -r requirements.txt  
bash scripts/run_all.sh

## Config
Edit `config/config.yaml` to set dataset paths and protected attributes:
- `data.mimic_path` for the processed MIMIC-III CSV (must include `mortality`)
- `data.olives.*` for OLIVES images + labels
- `fairness.*` to pick protected attributes and which group to reweight/threshold

## Datasets
### OLIVES (OCT)
1. Download the OLIVES archives and extract them locally.
2. Set the config to point to the extracted images folder and labels CSV:
   - `data.olives.img_dir`
   - `data.olives.labels_csv`
   - `data.olives.id_col` (image filename column)
   - `data.olives.label_col` (binary label column)
3. Ensure any protected attributes you want to evaluate (e.g., `race`, `ethnicity`, `gender`, `age_group`) exist as columns in the labels CSV.

Source:
```
https://zenodo.org/records/7105232
```

### MIMIC-III (EHR)
1. Obtain access through PhysioNet and download the raw tables.
2. Use the helper script to build a simple baseline CSV:
```
python scripts/build_mimic_csv.py --mimic_dir /path/to/mimic-iii-csvs --output data/processed/mimic.csv
```
3. The output CSV includes:
   - `mortality` (binary label)
   - `age` (numeric)
   - protected attributes (e.g., `race`, `gender`)
   - any other numeric/categorical features you want to model
4. Update `data.mimic_path` in `config/config.yaml`.

Sources:
```
https://www.physionet.org/content/mimiciii/
https://www.kaggle.com/datasets/ihssanened/mimic-iii-clinical-databaseopen-access/data
```

#### Optional: First-Day Vitals/Labs Features
If you want first-24h ICU vitals/labs, provide an item map CSV and pass it to the script:
```
python scripts/build_mimic_csv.py \
  --mimic_dir /path/to/mimic-iii-csvs \
  --output data/processed/mimic.csv \
  --item_map /path/to/item_map.csv
```

`item_map.csv` format:
- `source`: `chartevents` or `labevents`
- `itemid`: numeric ITEMID from `D_ITEMS.csv` / `D_LABITEMS.csv`
- `name`: short feature name (used to build columns like `chartevents_heartrate_mean`)

Example row (replace ITEMIDs with the ones you want from MIMIC-III dictionaries):
```
source,itemid,name
chartevents,ITEMID_GOES_HERE,heartrate
labevents,ITEMID_GOES_HERE,glucose
```

## Outputs
- results/plots/
- results/tables/


# Test Commit