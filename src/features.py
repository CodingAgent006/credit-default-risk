import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def build_preprocessor() -> ColumnTransformer:
    """
    Creates a scikit-learn ColumnTransformer for preprocessing 
    numerical and categorical columns.
    """
    # 1. Numerical pipeline: Imputation -> Log Transformed Income -> Scaling
    numeric_features = [
        'person_age', 'person_income', 'person_emp_length', 
        'loan_amnt', 'loan_int_rate', 'loan_percent_income', 
        'cb_person_cred_hist_length'
    ]
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 2. Categorical pipeline: One-Hot Encoding
    categorical_features = [
        'person_home_ownership', 'loan_intent', 
        'loan_grade', 'cb_person_default_on_file'
    ]
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    # 3. Combine into ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor