# 🚀 Customer Churn Prediction System  

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)  
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)  
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange?logo=scikitlearn)  
![License](https://img.shields.io/badge/License-MIT-green)  
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)

---

## 📌 One-Line Summary  

A Machine Learning powered web application that predicts telecom customer churn using Logistic Regression and SMOTE balancing.

---

## 📌 Table of Contents
- <a href="#overview">Overview</a>
- <a href="#business-problem">Business Problem</a>
- <a href="#dataset">Dataset</a>
- <a href="#tools--technologies">Tools & Technologies</a>
- <a href="#project-structure">Project Structure</a>
- <a href="#data-cleaning--preparation">Data Cleaning & Preparation</a>
- <a href="#exploratory-data-analysis-eda">Exploratory Data Analysis (EDA)</a>
- <a href="#model_training">Model Training</a>
- <a href="#dashboard">Dashboard</a>
- <a href="#how-to-run-this-project">How to Run This Project</a>
- <a href="#final-recommendations">Final Recommendations</a>
- <a href="#author--contact">Author & Contact</a>

---

<h2><a class="anchor" id="overview"></a>Overview</h2>

Customer churn is a major challenge in the telecom industry. This project builds a Machine Learning model to predict whether a customer will churn based on service usage, billing details, and contract information.

The system includes:

- ✔ Data preprocessing  
- ✔ Feature engineering  
- ✔ Class imbalance handling (SMOTE)  
- ✔ Logistic Regression model  
- ✔ Interactive Streamlit dashboard  
- ✔ Real-time churn prediction  

---

<h2><a class="anchor" id="business-problem"></a>Problem Statement</h2>

Telecom companies face revenue loss due to customer churn.  

The objective of this project is to:

- Predict whether a customer will churn  
- Identify key churn-driving factors  
- Help businesses take proactive retention actions  

---

<h2><a class="anchor" id="dataset"></a>Dataset</h2>

- Dataset: Telco Customer Churn Dataset  
- Total Records: 7,043  
- Features: 20+  
- Target Variable: `Churn (Yes/No)`

### Key Features:

- Gender  
- Senior Citizen  
- Tenure  
- Contract Type  
- Internet Service  
- Monthly Charges  
- Payment Method  
- Total Charges  

---

<h2><a class="anchor" id="tools--technologies"></a>Tools & Technologies</h2>

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Imbalanced-learn (SMOTE)  
- Joblib  
- Plotly  
- Streamlit  

---

---

<h2><a class="anchor" id="project-structure"></a>Project Structure</h2>

```
Customer-Churn-Prediction
│
├── data/
│      └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── models/
│      ├── churn_scaler.pkl
│      ├── columns.pkl
│      └── customer_churn_model.pkl
├── notebooks/
│      └── churn.ipynb
├── app.py
├── requirements.txt
└── README.md
```

---
<h2><a class="anchor" id="data-cleaning--preparation"></a>Data Cleaning & Preparation</h2>

- Missing value handling  
- Encoding categorical variables  
- Feature scaling (StandardScaler)  
- SMOTE for class balancing

---

---
<h2><a class="anchor" id="exploratory-data-analysis-eda"></a>Exploratory Data Analysis (EDA)</h2>

### 🔹 Target Distribution
- 73% Non-Churn
- 27% Churn
- Imbalance handled using SMOTE during training

### 🔹 Key Insights

- Customers with low tenure are more likely to churn.
- Month-to-month contract users show highest churn rate.
- Higher monthly charges increase churn probability.
- Fiber optic internet users churn more frequently.
- Long-term contract customers are more stable.

### 🔹 Conclusion from EDA
Tenure, contract type, and monthly charges are strong indicators of churn and were important features in model training.

---

<h2><a class="anchor" id="model_training"></a>Model Training</h2>

- Logistic Regression  
- Train-test split  
- Evaluation using Accuracy, Precision, Recall, F1 Score

### Model Performance  

| Metric | Score |
|--------|--------|
| Accuracy | ~79% |
| F1 Score | ~0.62 |
| Balanced using | SMOTE |

---
<h2><a class="anchor" id="dashboard"></a>Dashboard</h2>

Dashboard Link: https://churn-prediction14.streamlit.app

- Model saved using Joblib  
- Web application built with Streamlit  
- Real-time predictions via UI

## 📸 Application Features  

### 🏠 Home Page
- Customer input form  
- Instant churn prediction  
- Clean UI  

### 📂 Upload Dataset Page
- Upload CSV file  
- Dataset analysis  
- Visualization  
- Model insights  

---

---
<h2><a class="anchor" id="how-to-run-this-project"></a>How to Run This Project</h2>

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2️⃣ Create Virtual Environment (Optional)

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run Streamlit App

```bash
streamlit run app.py
```
---

---

<h2><a class="anchor" id="final-recommendations"></a>Key Insights</h2>

- Month-to-month contract customers are more likely to churn  
- Higher monthly charges increase churn probability  
- Customers with shorter tenure churn more  
- Payment method influences churn behavior  
- Internet service type plays a significant role  

---

## 🏁 Results & Conclusion  

The Logistic Regression model achieves around **79% accuracy** with balanced performance using SMOTE.  

This solution helps telecom companies identify at-risk customers and take preventive retention strategies.



<h2><a class="anchor" id="author--contact"></a>Author & Contact</h2>

**Jay Dhandhukiya**  

- GitHub: https://github.com/codewithsam14 
- Email: jayddd838@gmail.com  

---

## ⭐ Support  

If you like this project, give it a ⭐ on GitHub!
