#!/usr/bin/env python3
import argparse
import os
import re
import pandas as pd
import numpy as np

def _find_file(mimic_dir, filename):
    target = filename.lower()
    for f in os.listdir(mimic_dir):
        if f.lower() == target or f.lower() == f"{target}.gz":
            return os.path.join(mimic_dir, f)
    raise FileNotFoundError(f"Could not find {filename} in {mimic_dir}")

def _map_ethnicity_to_race(eth):
    if pd.isna(eth):
        return "unknown"
    eth = str(eth).upper()
    if "WHITE" in eth:
        return "white"
    if "BLACK" in eth or "AFRICAN" in eth:
        return "black"
    if "HISPANIC" in eth or "LATINO" in eth:
        return "hispanic"
    if "ASIAN" in eth:
        return "asian"
    if "NATIVE" in eth or "ALASKA" in eth or "AMERICAN INDIAN" in eth:
        return "native"
    if "MIDDLE EAST" in eth:
        return "middle_eastern"
    return "other"

def _sanitize_feature_name(name):
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")

def _load_item_map(item_map_path):
    item_map = pd.read_csv(item_map_path)
    required = {"source", "itemid", "name"}
    if not required.issubset(set(item_map.columns)):
        raise ValueError(f"item_map must include columns: {sorted(required)}")
    item_map["source"] = item_map["source"].str.lower()
    item_map["name"] = item_map["name"].apply(_sanitize_feature_name)
    return item_map

def _aggregate_events(
    path,
    source,
    item_map,
    icu_stays,
    hours=24,
    chunksize=500_000
):
    if not path or not os.path.exists(path):
        print(f"Skipping {source}: file not found at {path}")
        return pd.DataFrame()

    source = source.lower()
    itemids = set(item_map["itemid"].unique())

    if source == "chartevents":
        id_col = "icustay_id"
        time_lookup = dict(zip(icu_stays["icustay_id"], icu_stays["intime"]))
        hadm_to_icu = None
    elif source == "labevents":
        id_col = "hadm_id"
        time_lookup = dict(zip(icu_stays["hadm_id"], icu_stays["intime"]))
        hadm_to_icu = dict(zip(icu_stays["hadm_id"], icu_stays["icustay_id"]))
    else:
        raise ValueError("source must be 'chartevents' or 'labevents'")

    agg = {}
    usecols = [id_col, "itemid", "charttime", "valuenum"]

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        chunk = chunk[chunk["itemid"].isin(itemids)]
        if chunk.empty:
            continue

        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
        chunk = chunk.dropna(subset=["charttime", "valuenum"])

        if source == "chartevents":
            chunk["intime"] = chunk[id_col].map(time_lookup)
            chunk["icustay_id"] = chunk[id_col]
        else:
            chunk["intime"] = chunk[id_col].map(time_lookup)
            chunk["icustay_id"] = chunk[id_col].map(hadm_to_icu)

        chunk = chunk.dropna(subset=["intime", "icustay_id"])
        if chunk.empty:
            continue

        hours_from_icu = (chunk["charttime"] - chunk["intime"]).dt.total_seconds() / 3600.0
        chunk = chunk[(hours_from_icu >= 0) & (hours_from_icu <= hours)]
        if chunk.empty:
            continue

        grouped = (
            chunk.groupby(["icustay_id", "itemid"])["valuenum"]
            .agg(["sum", "count", "min", "max"])
            .reset_index()
        )

        for row in grouped.itertuples(index=False):
            key = (int(row.icustay_id), int(row.itemid))
            if key not in agg:
                agg[key] = [row.sum, row.count, row.min, row.max]
            else:
                agg[key][0] += row.sum
                agg[key][1] += row.count
                agg[key][2] = min(agg[key][2], row.min)
                agg[key][3] = max(agg[key][3], row.max)

    if not agg:
        return pd.DataFrame()

    rows = []
    for (icu_id, itemid), (s, c, mn, mx) in agg.items():
        rows.append(
            {
                "icustay_id": icu_id,
                "itemid": itemid,
                "mean": s / c if c > 0 else np.nan,
                "min": mn,
                "max": mx,
            }
        )

    return pd.DataFrame(rows)

def build_mimic_csv(
    mimic_dir,
    output_path,
    include_non_icu=False,
    item_map_path=None,
    chartevents_path=None,
    labevents_path=None,
    hours=24,
    chunksize=500_000
):
    admissions = pd.read_csv(
        _find_file(mimic_dir, "ADMISSIONS.csv"),
        usecols=[
            "SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME",
            "ADMISSION_TYPE", "INSURANCE", "MARITAL_STATUS",
            "ETHNICITY", "HOSPITAL_EXPIRE_FLAG"
        ]
    )
    patients = pd.read_csv(
        _find_file(mimic_dir, "PATIENTS.csv"),
        usecols=["SUBJECT_ID", "GENDER", "DOB"]
    )
    icustays = pd.read_csv(
        _find_file(mimic_dir, "ICUSTAYS.csv"),
        usecols=["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME", "OUTTIME", "FIRST_CAREUNIT"]
    )

    for df in [admissions, patients, icustays]:
        df.columns = [c.lower() for c in df.columns]

    admissions["admittime"] = pd.to_datetime(admissions["admittime"])
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"])
    patients["dob"] = pd.to_datetime(patients["dob"])
    icustays["intime"] = pd.to_datetime(icustays["intime"])
    icustays["outtime"] = pd.to_datetime(icustays["outtime"])

    # First ICU stay per admission
    icustays = icustays.sort_values("intime").drop_duplicates(subset=["hadm_id"])

    # Merge admissions + patients
    merged = admissions.merge(patients, on="subject_id", how="left")
    if include_non_icu:
        merged = merged.merge(icustays, on=["subject_id", "hadm_id"], how="left")
    else:
        merged = merged.merge(icustays, on=["subject_id", "hadm_id"], how="inner")

    # Age at admission (cap de-identified ages > 89)
    age = (merged["admittime"] - merged["dob"]).dt.days / 365.25
    merged["age"] = age.clip(lower=0)
    merged.loc[merged["age"] > 89, "age"] = 90

    # LOS features
    merged["los_hosp_days"] = (merged["dischtime"] - merged["admittime"]).dt.total_seconds() / 86400.0
    merged["los_icu_days"] = (merged["outtime"] - merged["intime"]).dt.total_seconds() / 86400.0

    # Protected attribute: race (mapped from ethnicity)
    merged["race"] = merged["ethnicity"].apply(_map_ethnicity_to_race)
    merged["gender"] = merged["gender"].str.lower()

    # Label
    merged["mortality"] = merged["hospital_expire_flag"].astype(int)

    # Select columns for modeling
    keep_cols = [
        "mortality",
        "age",
        "gender",
        "race",
        "admission_type",
        "insurance",
        "marital_status",
        "first_careunit",
        "los_hosp_days",
        "los_icu_days",
    ]

    df_out = merged[keep_cols].copy()
    df_out = df_out.replace([np.inf, -np.inf], np.nan).dropna(subset=["mortality", "age", "gender", "race"])

    # Optional: aggregate first-24h vitals/labs if item map is provided
    if item_map_path:
        item_map = _load_item_map(item_map_path)
        icu_stays = merged.dropna(subset=["icustay_id"])[["hadm_id", "icustay_id", "intime"]]
        chartevents_path = chartevents_path or _find_file(mimic_dir, "CHARTEVENTS.csv")
        labevents_path = labevents_path or _find_file(mimic_dir, "LABEVENTS.csv")

        feats = []
        if (item_map["source"] == "chartevents").any():
            ce_map = item_map[item_map["source"] == "chartevents"]
            ce_df = _aggregate_events(
                chartevents_path,
                "chartevents",
                ce_map,
                icu_stays,
                hours=hours,
                chunksize=chunksize
            )
            ce_df["source"] = "chartevents"
            feats.append(ce_df)

        if (item_map["source"] == "labevents").any():
            le_map = item_map[item_map["source"] == "labevents"]
            le_df = _aggregate_events(
                labevents_path,
                "labevents",
                le_map,
                icu_stays,
                hours=hours,
                chunksize=chunksize
            )
            le_df["source"] = "labevents"
            feats.append(le_df)

        if feats:
            feat_df = pd.concat(feats, ignore_index=True)
            feat_df = feat_df.merge(item_map, on=["source", "itemid"], how="left")
            feat_df["feature_base"] = feat_df["source"] + "_" + feat_df["name"]

            feat_long = feat_df.melt(
                id_vars=["icustay_id", "feature_base"],
                value_vars=["mean", "min", "max"],
                var_name="stat",
                value_name="value"
            )
            feat_long["feature"] = feat_long["feature_base"] + "_" + feat_long["stat"]
            feat_wide = feat_long.pivot_table(
                index="icustay_id",
                columns="feature",
                values="value",
                aggfunc="first"
            ).reset_index()

            df_out = df_out.merge(
                merged[["icustay_id", "hadm_id"]],
                left_index=True,
                right_index=True,
                how="left"
            )
            df_out = df_out.merge(feat_wide, on="icustay_id", how="left")
            df_out = df_out.drop(columns=["icustay_id", "hadm_id"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"Saved {len(df_out)} rows to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic_dir", required=True, help="Path to raw MIMIC-III CSV folder")
    parser.add_argument("--output", default="data/processed/mimic.csv", help="Output CSV path")
    parser.add_argument("--include_non_icu", action="store_true", help="Keep admissions without ICU stays")
    parser.add_argument("--item_map", default=None, help="CSV mapping: source,itemid,name")
    parser.add_argument("--chartevents", default=None, help="Path to CHARTEVENTS.csv (optional)")
    parser.add_argument("--labevents", default=None, help="Path to LABEVENTS.csv (optional)")
    parser.add_argument("--hours", type=int, default=24, help="Hours from ICU admit to include events")
    parser.add_argument("--chunksize", type=int, default=500000, help="Chunk size for event CSVs")
    args = parser.parse_args()

    build_mimic_csv(
        args.mimic_dir,
        args.output,
        include_non_icu=args.include_non_icu,
        item_map_path=args.item_map,
        chartevents_path=args.chartevents,
        labevents_path=args.labevents,
        hours=args.hours,
        chunksize=args.chunksize
    )
