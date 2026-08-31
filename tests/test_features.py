import pandas as pd
from src.features import build_preprocessor

def test_build_preprocessor_execution():
    sample = pd.DataFrame([{
        'person_age': 25, 'person_income': 50000, 'person_emp_length': 3.0,
        'loan_amnt': 10000, 'loan_int_rate': 10.5, 'loan_percent_income': 0.2,
        'cb_person_cred_hist_length': 4, 'person_home_ownership': 'RENT',
        'loan_intent': 'PERSONAL', 'loan_grade': 'A', 'cb_person_default_on_file': 'N'
    }])
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(sample)
    assert transformed.shape[0] == 1
