from pathlib import Path
import pandas as pd

def load_data(filepath: str | Path) -> pd.DataFrame:
    """Loads raw dataset from file path."""
    return pd.read_csv(filepath)

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Removes row-level data entry corruption errors."""
    df_clean = df.copy()
    df_clean = df_clean[df_clean['person_age'] < 100]
    df_clean = df_clean[df_clean['person_emp_length'] <= 60]
    return df_clean
