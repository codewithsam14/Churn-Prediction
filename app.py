import streamlit as st
import pandas as pd
import joblib
import warnings
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# PAGE CONFIG
st.set_page_config(page_title="Customer Churn App", layout="wide")

# CUSTOM CSS 
st.markdown("""
<style>
.main-title {
    font-size: 32px;
    font-weight: bold;
}
.section-title {
    font-size: 20px;
    font-weight: 600;
    margin-top: 20px;
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

.stButton>button {
    width: 100%;
    background-color: #FF4B4B;
    color: white;
    font-weight: bold;
    padding: 0.5rem 1rem;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# SESSION STATE FOR NAVIGATION
if "page" not in st.session_state:
    st.session_state.page = "Predict"

def set_page(page):
    st.session_state.page = page

# SIDEBAR
with st.sidebar:

    st.title("Navigation")
    st.button("Home Page", on_click=set_page, args=("Predict",))
    st.button("Upload Dataset & Analyze", on_click=set_page, args=("Upload",))

    st.markdown("---")

    st.header("Model Info")
    st.info("""
    **Logistic Regression**
    - Accuracy: ~79%
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
    st.caption("Built with ❤️ Streamlit & Scikit-learn")

# LOAD MODEL
@st.cache_resource
def load_model():
    try:
        model = joblib.load("models/customer_churn_model.pkl")
        scaler = joblib.load("models/churn_scaler.pkl")
        columns = joblib.load("models/columns.pkl")
        return model, scaler, columns
    except:
        return None, None, None

model, scaler, columns = load_model()


# PREDICT PAGE
if st.session_state.page == "Predict":

    st.markdown('<div class="main-title">Customer Churn Prediction</div>', unsafe_allow_html=True)
    st.markdown("Predict customer churn using Logistic Regression")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Services</div>', unsafe_allow_html=True)
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple = st.selectbox("Multiple Lines", ["No", "Yes"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        security = st.selectbox("Online Security", ["No", "Yes"])
        backup = st.selectbox("Online Backup", ["No", "Yes"])

        st.markdown('<div class="section-title">Demographics</div>', unsafe_allow_html=True)
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

    with col2:
        st.markdown('<div class="section-title">Account Details</div>', unsafe_allow_html=True)
        device = st.selectbox("Device Protection", ["No", "Yes"])
        tech = st.selectbox("Tech Support", ["No", "Yes"])
        tv = st.selectbox("Streaming TV", ["No", "Yes"])
        movies = st.selectbox("Streaming Movies", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox("Payment Method",
                               ["Electronic check", "Mailed check",
                                "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly = st.number_input("Monthly Charges ($)", 0.0, 150.0, 50.0)
        total = st.number_input("Total Charges ($)", 0.0, 10000.0, monthly * tenure)

    st.markdown("---")

    if st.button("Predict Churn"):

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
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
        for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
            df[col] = df[col].map({'Yes': 1, 'No': 0})

        df = pd.get_dummies(df)
        df = df.reindex(columns=columns, fill_value=0)

        df_scaled = scaler.transform(df)
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

# Upload & Analyze Dataset
elif st.session_state.page == "Upload":
    st.markdown('<div class="main-title">Upload Churn Dataset & Analyze</div>', unsafe_allow_html=True)
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file:

        df = pd.read_csv(uploaded_file)

        st.success("Dataset Uploaded Successfully!")

        # BASIC METRICS
        col1, col2, col3 = st.columns(3)

        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())

        st.markdown("---")

        # DATA PREVIEW
        st.subheader("Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        st.markdown("---")

        # MISSING VALUES VISUALIZATION
        missing = df.isnull().sum()
        missing = missing[missing > 0]

        if len(missing) > 0:
            st.subheader("Missing Values by Column")

            fig = go.Figure([
                go.Bar(x=missing.index, y=missing.values)
            ])
            fig.update_layout(title="Missing Values Distribution")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No Missing Values Found!")

        st.markdown("---")

        # CORRELATION HEATMAP
        numeric_df = df.select_dtypes(include=['int64', 'float64'])

        if numeric_df.shape[1] > 1:
            st.subheader("Correlation Heatmap")

            corr = numeric_df.corr()

            import plotly.express as px
            fig = px.imshow(corr, text_auto=True)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # CHURN ANALYSIS 
        if "Churn" in df.columns:

            st.subheader("Churn Distribution")

            churn_counts = df["Churn"].value_counts()
            churn_rate = (churn_counts / len(df)) * 100

            colA, colB = st.columns(2)
            colA.metric("Churn Customers", churn_counts.get(1, 0))
            colB.metric("Churn Rate (%)", f"{churn_rate.get(1, 0):.2f}%")

            fig = go.Figure([
                go.Bar(
                    x=churn_counts.index.astype(str),
                    y=churn_counts.values
                )
            ])
            fig.update_layout(title="Churn Distribution")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # GENERATE REPORT
        st.subheader("Generate Summary Report")

        report_data = {
            "Metric": [
                "Total Rows",
                "Total Columns",
                "Missing Values"
            ],
            "Value": [
                df.shape[0],
                df.shape[1],
                df.isnull().sum().sum()
            ]
        }

        if "Churn" in df.columns:
            report_data["Metric"].extend([
                "Churn Customers",
                "Churn Rate (%)"
            ])
            report_data["Value"].extend([
                churn_counts.get(1, 0),
                round(churn_rate.get(1, 0), 2)
            ])

        report = pd.DataFrame(report_data)

        st.dataframe(report, use_container_width=True)

        csv_report = report.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Report",
            data=csv_report,
            file_name="churn_analysis_report.csv",
            mime="text/csv"
        )

else:
    st.info("Upload a dataset to generate analysis report.")
