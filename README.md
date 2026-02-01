# Churn Prediction
![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green)

Predict customer churn using a saved Logistic Regression model. Includes a Streamlit demo (app.py) and a training notebook.

Quick links:
- Demo: streamlit run app.py
- Notebook: notebooks/churn.ipynb

---

## Table of contents
- [Demo / Quick Start](#demo--quick-start)
- [Usage](#usage)
  - [Streamlit app (UI)](#streamlit-app-ui)
- [Model artifacts & preprocessing details](#model-artifacts--preprocessing-details)
- [Reproducing training](#reproducing-training)
- [Troubleshooting & notes](#troubleshooting--notes)
- [Limitations & model performance](#limitations--model-performance)
- [Contributing & License](#contributing--license)

---

## Demo / Quick Start

1. Clone and enter repo:
```bash
git clone https://github.com/codewithsam14/Churn-Prediction.git
cd Churn-Prediction
```

2. Create a virtual environment (recommended) and install:
```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

Open the URL printed by Streamlit.

---

## Usage

### Streamlit app (UI)
- Use the web form to enter customer attributes (services, demographics, account).
- Clicking "Predict Churn" runs the same preprocessing as training (binary mapping + get_dummies, feature alignment, scaling) and shows:
  - Predicted class (WILL CHURN / WILL STAY)
  - Confidence & Risk level
  - Probability chart and a short recommendation

If model artifacts are missing, the app shows: "Model files not found. Run the notebook first."

---

## Model artifacts & preprocessing details

Artifacts (models/):
- customer_churn_model.pkl — LogisticRegression (scikit-learn; saved with sklearn 1.6.1)
- churn_scaler.pkl — StandardScaler
- columns.pkl — ordered list of training columns (including dummies)

Preprocessing summary (must match training):
- Map binary fields: gender (Male=1), Partner/Dependents/PhoneService/PaperlessBilling (Yes=1).
- One-hot encode remaining categoricals: InternetService, MultipleLines, Contract, PaymentMethod, and service fields.
- Reindex to columns.pkl (fill missing dummies with 0).
- Scale numeric features with churn_scaler.pkl before model.predict / predict_proba.

Expected numeric features: tenure, MonthlyCharges, TotalCharges.
See models/columns.pkl for exact final feature order.

---

## Reproducing training
- Open notebooks/churn.ipynb (contains EDA, preprocessing, SMOTE balancing, model training, evaluation).
- Dataset included: data/WA_Fn-UseC_-Telco-Customer-Churn.csv
- To reproduce: install requirements, open the notebook in Jupyter and run cells in order.
- Recommended scikit-learn: 1.6.1 (pickles were created with this).

---

## Troubleshooting & notes
- Ensure models/ contains: customer_churn_model.pkl, churn_scaler.pkl, columns.pkl.
- Pickles/joblib: only load trusted files.
- If you get pickle/version errors, try matching package versions in a fresh venv.

---

## Limitations & model performance
- Model: Logistic Regression baseline. Accuracy ~77% (see notebook for full metrics).
- Dataset size: ~7,043 customers (Telco dataset).
- Use outputs as guidance — combine with business rules and human review.
- Retrain if you change encodings or add categories.

---

## Contributing & License
Contributions welcome — open an issue or PR.

License: MIT

---

If you'd like an even shorter README (one-page quick start only) or badges/CI badges added, tell me which details to include.
