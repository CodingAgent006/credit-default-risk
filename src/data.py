import pandas as pd

def load_data(filepath: str) -> pd.DataFrame:
    """Loads raw CSV dataset."""
    return pd.read_csv(filepath)

def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Removes invalid data entry errors (e.g., impossible ages/employment lengths)."""
    df_clean = df.copy()
    # Filter logical age and employment length errors
    df_clean = df_clean[df_clean['person_age'] < 100]
    df_clean = df_clean[df_clean['person_emp_length'] <= 60]
    return df_clean