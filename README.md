# Credit Default Risk Prediction Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red.svg)
![Tests](https://img.shields.io/badge/Tests-Pytest-green.svg)

An end-to-end production-ready Machine Learning system that predicts borrower loan default probability. The pipeline incorporates leak-free feature preprocessing, **probability calibration via Isotonic Regression**, model explainability using **SHAP**, and a high-performance REST API built with **FastAPI**.

---

## 📌 Project Overview

Predicting credit default risk accurately is essential for financial institutions to balance credit approval rates with expected loss mitigation. This repository provides a complete ML engineering workflow—from data cleaning and feature pipeline construction to probability-calibrated model training, SHAP feature interpretation, unit testing, and containerized serving.

### Key Features
- **Leak-Free Preprocessing:** Features (including log-transformations, numerical imputations, and one-hot encoding) are encapsulated within Scikit-learn `ColumnTransformer` pipelines fitted strictly on training splits.
- **Probability Calibration:** Employs `CalibratedClassifierCV` (Isotonic Regression) over XGBoost to output true, reliable risk probabilities.
- **Explainability & Interpretability:** Integrates SHAP (SHapley Additive exPlanations) to interpret global feature importance and support auditability.
- **Production API:** Served via FastAPI with lifespan state management, Pydantic request/response validation, and fail-fast startup checks.
- **Containerized:** Docker multi-stage builds for lean container deployment.

---

## 📁 Repository Structure

```text
credit-default-risk/
├── .gitignore               # Ignored files (virtualenvs, models, data)
├── Dockerfile               # Multi-stage Docker container specification
├── README.md                # Project documentation
├── requirements.txt         # Lightweight production dependencies
├── requirements-dev.txt     # Complete development & testing dependencies
├── api/
│   ├── __init__.py
│   └── main.py              # FastAPI application with lifecycle controls
├── data/
│   └── credit_risk_dataset.csv # Raw dataset (not tracked in Git)
├── models/                  # Generated artifacts directory
│   ├── best_model.joblib    # Serialized calibrated pipeline artifact
│   ├── X_test.parquet       # Held-out evaluation features
│   ├── y_test.parquet       # Held-out evaluation targets
│   └── shap_summary.png     # SHAP feature importance output
├── notebooks/
│   └── eda.ipynb            # Exploratory Data Analysis & data profiling
├── src/
│   ├── __init__.py
│   ├── data.py              # Data loading & row-level validation
│   ├── features.py          # Preprocessing & transformation pipelines
│   ├── train.py             # Training script with probability calibration
│   ├── evaluate.py          # Metrics generation & SHAP summary plotting
│   └── predict.py           # Programmatic inference abstraction
└── tests/
    ├── __init__.py
    ├── test_api.py          # Integration tests for FastAPI endpoints
    └── test_features.py     # Unit tests for feature pipelines
```
## 📦 Data Source

The dataset used in this project is the **[Credit Risk Dataset on Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)**.

To set up the dataset locally:
1. Download `credit_risk_dataset.csv` from Kaggle.
2. Place the file inside the `data/` directory:
   ```text
   credit-default-risk/
   └── data/
       └── credit_risk_dataset.csv
---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+
- Docker (optional, for containerized execution)

### 1. Clone Repository & Create Virtual Environment
```bash
git clone https://github.com/your-username/credit-default-risk.git
cd credit-default-risk

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts ctivate
```

### 2. Install Dependencies
For development and training:
```bash
pip install -r requirements-dev.txt
```

---

## 🚀 Usage Guide

### Step 1: Model Training & Artifact Generation
Run the training script to load raw data, perform stratified train-test splitting, train the calibrated XGBoost model, and persist `.joblib` model artifacts to `models/`:

```bash
python src/train.py
```

### Step 2: Model Evaluation & SHAP Interpretability
Evaluate the saved model on the held-out test split. This generates classification metrics, ROC-AUC score, and saves a SHAP summary plot (`models/shap_summary.png`):

```bash
python src/evaluate.py
```

### Step 3: Run Tests
Execute unit and integration tests using Pytest:

```bash
pytest tests/
```

### Step 4: Launch REST API Locally
Start the local FastAPI server:

```bash
uvicorn api.main:app --reload --port 8000
```
- Interactive API Docs (Swagger): `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/`

---

## 🐳 Docker Deployment

### 1. Build Image
```bash
docker build -t credit-default-api .
```

### 2. Run Container
```bash
docker run -d -p 8000:8000 --name credit-risk-service credit-default-api
```

---

## 🔌 API Endpoint Usage

### `POST /predict`

**Request Body Example:**
```json
{
  "person_age": 28,
  "person_income": 55000,
  "person_home_ownership": "RENT",
  "person_emp_length": 4.0,
  "loan_intent": "PERSONAL",
  "loan_grade": "B",
  "loan_amnt": 8000,
  "loan_int_rate": 11.2,
  "loan_percent_income": 0.15,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 4
}
```

**cURL Example:**
```bash
curl -X 'POST'   'http://localhost:8000/predict?threshold=0.35'   -H 'accept: application/json'   -H 'Content-Type: application/json'   -d '{
  "person_age": 28,
  "person_income": 55000,
  "person_home_ownership": "RENT",
  "person_emp_length": 4.0,
  "loan_intent": "PERSONAL",
  "loan_grade": "B",
  "loan_amnt": 8000,
  "loan_int_rate": 11.2,
  "loan_percent_income": 0.15,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 4
}'
```

**Response Example:**
```json
{
  "default_probability": 0.1245,
  "is_high_risk": false,
  "threshold_used": 0.35
}
```

---

## 📊 Model & Calibration Architecture

1. **Feature Engineering Pipeline:**
   - Log-transforms skewed numeric fields (`person_income`).
   - Imputes missing values (`SimpleImputer(strategy='median')`).
   - Scales numerical variables (`StandardScaler`).
   - One-hot encodes categorical parameters (`OneHotEncoder`).
2. **Probability Calibration:**
   - Raw XGBoost logit outputs are calibrated via `CalibratedClassifierCV(method='isotonic', cv=5)` to ensure predicted probabilities match real-world default frequencies.
