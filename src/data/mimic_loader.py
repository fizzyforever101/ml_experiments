import pandas as pd

from src.data.preprocess import preprocess_mimic

def load_mimic(path, protected_cols):
    df = pd.read_csv(path)
    X, y, protected = preprocess_mimic(df, protected_cols)
    return X, y, protected
