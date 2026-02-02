import streamlit as st
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")
import plotly.graph_objects as go

# Page config
st.set_page_config(page_title="Churn Prediction", layout="wide")


# Custom CSS
st.markdown("""
<style>
.stButton>button {
    width: 100%;
    background-color: #FF4B4B;
    color: white;
    font-weight: bold;
    padding: 0.5rem 1rem;
    border-radius: 10px;
}
.prediction-box {
    padding: 1.5rem;
    border-radius: 10px;
    text-align: center;
    font-size: 1.3rem;
    font-weight: bold;
}
.churn { background-color: #ffebee; color: #c62828; }
.no-churn { background-color: #e8f5e9; color: #2e7d32; }
</style>
""", unsafe_allow_html=True)

# Load model 
@st.cache_resource
def load_model():
    try:
        model = joblib.load("models/customer_churn_model.pkl")   
        scaler = joblib.load("models/churn_scaler.pkl")
        columns = joblib.load("models/columns.pkl")
        return model, scaler, columns
    
    except FileNotFoundError:
        st.error("Model files not found. Run the notebook first.")
        return None, None, None

model, scaler, columns = load_model()

# Title
st.title(" Customer Churn Prediction")
st.markdown("Predict customer churn with **Logistic Regression**")
st.markdown("---")

# UI Inputs
col1, col2 = st.columns(2)

with col1:

    st.subheader("Services")
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple = st.selectbox("Multiple Lines", ["No", "Yes"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    security = st.selectbox("Online Security", ["No", "Yes"])
    backup = st.selectbox("Online Backup", ["No", "Yes"])


    st.subheader("Demographics")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

with col2:

    st.subheader(" ")
    device = st.selectbox("Device Protection", ["No", "Yes"])
    tech = st.selectbox("Tech Support", ["No", "Yes"])
    tv = st.selectbox("Streaming TV", ["No", "Yes"])
    movies = st.selectbox("Streaming Movies", ["No", "Yes"])


    st.subheader(" ")
    st.markdown(" ")
    st.subheader("Account")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    monthly = st.number_input("Monthly Charges ($)", 0.0, 150.0, 50.0)
    total = st.number_input("Total Charges ($)", 0.0, 10000.0, monthly * tenure)

st.markdown("---")

# Prediction
if st.button("Predict Churn",type="primary"):

    input_data = {
        'gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone,
        'MultipleLines': multiple,
        'InternetService': internet,
        'OnlineSecurity': security,
        'OnlineBackup': backup,
        'DeviceProtection': device,
        'TechSupport': tech,
        'StreamingTV': tv,
        'StreamingMovies': movies,
        'Contract': contract,
        'PaperlessBilling': paperless,
        'PaymentMethod': payment,
        'MonthlyCharges': monthly,
        'TotalCharges': total
    }

    df = pd.DataFrame([input_data])

    # Binary encoding
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    # One-hot encoding
    df = pd.get_dummies(df)

    # Match training columns
    df = df.reindex(columns=columns, fill_value=0)

    # Scaling
    df_scaled = scaler.transform(df)

    # Prediction
    prediction = model.predict(df_scaled)[0]
    proba = model.predict_proba(df_scaled)[0]

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        if prediction == 1:
            st.markdown('<div class="prediction-box churn">WILL CHURN</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="prediction-box no-churn">WILL STAY</div>', unsafe_allow_html=True)

    with c2:
        st.metric("Confidence", f"{max(proba)*100:.1f}%")

    with c3:
        risk = "High" if proba[1] > 0.7 else "Medium" if proba[1] > 0.4 else "Low"
        st.metric("Risk Level", risk)


    # Probability chart
    fig = go.Figure([
        go.Bar(name="No Churn", x=["Probability"], y=[proba[0]]),
        go.Bar(name="Churn", x=["Probability"], y=[proba[1]])
    ])
    fig.update_layout(title="Prediction Probabilities", barmode="group", height=350)
    st.plotly_chart(fig, use_container_width=True)

    if prediction == 1:
        st.warning("**At Risk:** Offer retention incentives, upgrade to longer contract, provide better support")
    else:
        st.success("**Low Risk:** Continue excellent service, send surveys, offer loyalty benefits")

# Sidebar
with st.sidebar:
    st.header("Model Info")
    st.info("""
    **Logistic Regression**
    - Accuracy: ~77%
    - Dataset: 7,043 customers
    - Features: 20 Attributes
    - Balanced with SMOTE
    - Scaled with StandardScaler
    """)

    st.header("Top Predictors")
    st.markdown("""
    - Contract type  
    - Tenure  
    - Monthly charges  
    - Internet service  
    - Payment method
    """)

    st.markdown("---")
    st.caption("Built with Streamlit & Scikit-learn")
