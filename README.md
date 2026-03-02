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

## 📖 Overview  

Customer churn is a major challenge in the telecom industry. This project builds a Machine Learning model to predict whether a customer will churn based on service usage, billing details, and contract information.

The system includes:

- ✔ Data preprocessing  
- ✔ Feature engineering  
- ✔ Class imbalance handling (SMOTE)  
- ✔ Logistic Regression model  
- ✔ Interactive Streamlit dashboard  
- ✔ Real-time churn prediction  

---

## 🎯 Problem Statement  

Telecom companies face revenue loss due to customer churn.  

The objective of this project is to:

- Predict whether a customer will churn  
- Identify key churn-driving factors  
- Help businesses take proactive retention actions  

---

## 📂 Dataset  

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

## 🛠 Tools & Technologies  

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Imbalanced-learn (SMOTE)  
- Joblib  
- Plotly  
- Streamlit  

---

## ⚙️ Methods  

### 🔹 Data Preprocessing
- Missing value handling  
- Encoding categorical variables  
- Feature scaling (StandardScaler)  
- SMOTE for class balancing  

### 🔹 Model Training
- Logistic Regression  
- Train-test split  
- Evaluation using Accuracy, Precision, Recall, F1 Score  

### 🔹 Deployment
- Model saved using Joblib  
- Web application built with Streamlit  
- Real-time predictions via UI  

---

## 📊 Model Performance  

| Metric | Score |
|--------|--------|
| Accuracy | ~79% |
| F1 Score | ~0.62 |
| Balanced using | SMOTE |

---

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

## ▶️ How to Run the Project  

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

## 📁 Project Structure  

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

## 📈 Key Insights  

- Month-to-month contract customers are more likely to churn  
- Higher monthly charges increase churn probability  
- Customers with shorter tenure churn more  
- Payment method influences churn behavior  
- Internet service type plays a significant role  

---

## 🏁 Results & Conclusion  

The Logistic Regression model achieves around **79% accuracy** with balanced performance using SMOTE.  

This solution helps telecom companies identify at-risk customers and take preventive retention strategies.

---

## 🔮 Future Work  

- Implement SVM / Random Forest  
- Hyperparameter tuning  
- Add SHAP for explainability  
- Deploy on cloud (Streamlit Cloud)  
- Improve UI/UX  

---

## 👨‍💻 Author  

**Jay Dhandhukiya**  

- GitHub: https://github.com/codewithsam14 
- Email: jayddd838@gmail.com  

---

## ⭐ Support  

If you like this project, give it a ⭐ on GitHub!
