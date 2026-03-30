import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# ---------------------------
# 1. HANDLE MISSING VALUES
# ---------------------------
def handle_missing(df):
    """
    Fill missing values:
    - Numeric → median
    - Categorical → 'unknown'
    """
    df = df.copy()

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("unknown")

    return df

# ---------------------------
# 2. ENCODE CATEGORICALS
# ---------------------------
def encode_categoricals(df, categorical_cols):
    """
    One-hot encode categorical variables
    """
    return pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# ---------------------------
# 3. CREATE AGE GROUPS
# ---------------------------
def create_age_groups(df):
    """
    Convert raw age → buckets for fairness analysis
    """
    df = df.copy()

    bins = [0, 30, 50, 70, 100, 150]
    labels = ["young", "middle", "senior", "elderly"]

    df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)

    return df

# ---------------------------
# 4. SCALE FEATURES
# ---------------------------
def scale_features(X_train, X_test):
    """
    Standardize features
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler

# ---------------------------
# 5. FULL PIPELINE (MIMIC)
# ---------------------------
def preprocess_mimic(df, protected_cols):
    """
    Full preprocessing pipeline
    """

    # Step 1: clean missing
    df = handle_missing(df)

    # Step 2: create age groups
    if "age" in df.columns:
        df = create_age_groups(df)

    # Step 3: separate protected attributes
    missing = [c for c in protected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing protected columns in MIMIC data: {missing}")
    protected = df[protected_cols].copy()

    # Step 4: encode categorical features (excluding protected)
    if "mortality" not in df.columns:
        raise ValueError("MIMIC data must include a 'mortality' column.")
    feature_cols = df.drop(columns=protected_cols + ["mortality"])
    categorical_cols = feature_cols.select_dtypes(include=["object"]).columns

    X = encode_categoricals(feature_cols, categorical_cols)

    y = df["mortality"]

    return X, y, protected
