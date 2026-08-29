from pathlib import Path
import pandas as pd

def load_data(filepath: str | Path) -> pd.DataFrame:
    """Loads dataset from path."""
    return pd.read_csv(filepath)

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes obvious data corruption errors at the row level.
    Imputation must happen inside pipelines to prevent leakage.
    """
    df_clean = df.copy()
    df_clean = df_clean[df_clean['person_age'] < 100]
    df_clean = df_clean[df_clean['person_emp_length'] <= 60]
    return df_clean
