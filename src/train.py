import sys
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# 1. Dynamically set project paths (Works regardless of current working directory)
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
DATA_PATH = ROOT_DIR / "data" / "credit_risk_dataset.csv"
MODEL_SAVE_PATH = ROOT_DIR / "models" / "best_model.joblib"

# Ensure project root is in sys.path for clean imports
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.data import load_data, clean_raw_data
from src.features import build_preprocessor


def run_model_training_and_selection(
    data_path: Path = DATA_PATH, 
    model_save_path: Path = MODEL_SAVE_PATH
):
    # 2. Load and clean data
    raw_df = load_data(str(data_path))
    df = clean_raw_data(raw_df)

    X = df.drop(columns=['loan_status'])
    y = df['loan_status']

    # 3. Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Setup Stratified 5-Fold Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    preprocessor = build_preprocessor()

    # -------------------------------------------------------------
    # MODEL 1: Logistic Regression (Baseline)
    # -------------------------------------------------------------
    lr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])

    lr_param_grid = {
        'classifier__C': [0.01, 0.1, 1.0, 10.0],
        'classifier__solver': ['lbfgs', 'liblinear']
    }

    print("--- Tuning Baseline: Logistic Regression ---")
    grid_lr = GridSearchCV(
        estimator=lr_pipeline,
        param_grid=lr_param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1
    )
    grid_lr.fit(X_train, y_train)
    print(f"Logistic Regression Best CV ROC-AUC: {grid_lr.best_score_:.4f}")

    # -------------------------------------------------------------
    # MODEL 2: XGBoost Classifier (Challenger)
    # -------------------------------------------------------------
    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

    xgb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss'
        ))
    ])

    xgb_param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [3, 5, 7],
        'classifier__learning_rate': [0.01, 0.1]
    }

    print("\n--- Tuning Challenger: XGBoost Classifier ---")
    grid_xgb = GridSearchCV(
        estimator=xgb_pipeline,
        param_grid=xgb_param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1
    )
    grid_xgb.fit(X_train, y_train)
    print(f"XGBoost Best CV ROC-AUC: {grid_xgb.best_score_:.4f}")

    # -------------------------------------------------------------
    # MODEL SELECTION & ARTIFACT SAVING
    # -------------------------------------------------------------
    if grid_xgb.best_score_ > grid_lr.best_score_:
        best_model = grid_xgb.best_estimator_
        print(f"\n🏆 Champion Model: XGBoost (CV ROC-AUC: {grid_xgb.best_score_:.4f})")
    else:
        best_model = grid_lr.best_estimator_
        print(f"\n🏆 Champion Model: Logistic Regression (CV ROC-AUC: {grid_lr.best_score_:.4f})")

    # Ensure models/ directory exists before saving
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_save_path)
    print(f"Saved winning pipeline artifact to: {model_save_path}")

    return X_test, y_test


if __name__ == "__main__":
    run_model_training_and_selection()