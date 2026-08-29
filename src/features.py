import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

def build_preprocessor() -> ColumnTransformer:
    """Creates preprocessing pipeline without data leakage."""
    
    # 1. Numerical Pipeline: Median Impute -> Log Transform Income -> Scale
    # We use a custom transformer logic for income log-scaling
    log_cols = ['person_income']
    standard_num_cols = [
        'person_age', 'person_emp_length', 'loan_amnt', 
        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length'
    ]

    log_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('log', FunctionTransformer(np.log1p, feature_names_in_="same")),
        ('scaler', StandardScaler())
    ])

    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 2. Categorical Pipeline
    cat_cols = [
        'person_home_ownership', 'loan_intent', 
        'loan_grade', 'cb_person_default_on_file'
    ]
    
    cat_pipeline = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    # 3. Combine Processors
    preprocessor = ColumnTransformer(
        transformers=[
            ('num_log', log_pipeline, log_cols),
            ('num_std', num_pipeline, standard_num_cols),
            ('cat', cat_pipeline, cat_cols)
        ]
    )
    
    return preprocessor
