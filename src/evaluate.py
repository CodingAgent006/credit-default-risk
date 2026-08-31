from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"

def evaluate_model(model_dir: Path = MODEL_DIR):
    model_path = model_dir / "best_model.joblib"
    X_test_path = model_dir / "X_test.parquet"
    y_test_path = model_dir / "y_test.parquet"

    if not model_path.exists() or not X_test_path.exists():
        raise FileNotFoundError("Artifacts missing! Run 'python src/train.py' first.")

    pipeline = joblib.load(model_path)
    X_test = pd.read_parquet(X_test_path)
    y_test = pd.read_parquet(y_test_path)['loan_status']

    # Performance Evaluation
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\n" + "="*20 + " EVALUATION REPORT " + "="*20)
    print(f"Calibrated ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # SHAP Interpretability
    print("\nGenerating SHAP Summary Report...")
    preprocessor = pipeline.named_steps['preprocessor']
    X_transformed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    # Extract fitted XGBoost model from Calibration wrapper
    calibrated_classifier = pipeline.named_steps['classifier']
    fitted_xgb = calibrated_classifier.calibrated_classifiers_[0].estimator

    explainer = shap.TreeExplainer(fitted_xgb)
    shap_values = explainer(X_transformed)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(model_dir / "shap_summary.png")
    plt.close()
    print(f"SHAP summary saved to {model_dir / 'shap_summary.png'}")

if __name__ == "__main__":
    evaluate_model()
