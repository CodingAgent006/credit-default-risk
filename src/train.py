from pathlib import Path
import joblib
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from src.data import load_data, clean_raw_data
from src.features import build_preprocessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "credit_risk_dataset.csv"
MODEL_DIR = PROJECT_ROOT / "models"

def train_pipeline(data_path: Path = DATA_PATH, model_dir: Path = MODEL_DIR):
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and row-clean data
    raw_df = load_data(data_path)
    clean_df = clean_raw_data(raw_df)

    X = clean_df.drop(columns=['loan_status'])
    y = clean_df['loan_status']

    # 2. Stratified Split (Strict isolation of test set)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Save test set for evaluation scripts
    X_test.to_parquet(model_dir / "X_test.parquet")
    y_test.to_frame().to_parquet(model_dir / "y_test.parquet")

    # 4. Construct End-to-End Pipeline
    preprocessor = build_preprocessor()
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        eval_metric='logloss'
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # 5. Fit pipeline on training set
    print("Training XGBoost Pipeline...")
    pipeline.fit(X_train, y_train)

    # 6. Save model pipeline
    model_path = model_dir / "best_model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Pipeline saved to {model_path}")

if __name__ == "__main__":
    train_pipeline()
