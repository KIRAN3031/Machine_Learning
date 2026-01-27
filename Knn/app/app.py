import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler (train & save these first!)
@st.cache_resource
def load_model():
    knn = joblib.load('knn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('features.pkl')  # ['person_age','person_income','loan_amnt','cb_person_cred_hist_length']
    return knn, scaler, features

knn, scaler, feature_cols = load_model()

st.set_page_config(page_title="Credit Risk KNN Predictor", layout="wide", page_icon="🏦")

st.title("🏦 Instant Credit Risk Assessment")
st.markdown("**K-Nearest Neighbors (K=29)** similarity-based decisions using customer profiles like yours. Low Risk = Approve | High Risk = Review/Decline.")

# Sidebar for inputs (fintech app style)
st.sidebar.header("👤 Enter Customer Details")
person_age = st.sidebar.slider("Age", 18, 80, 30, help="person_age")
person_income = st.sidebar.number_input("Annual Income ($)", 10000, 500000, 60000, help="person_income")
loan_amnt = st.sidebar.number_input("Loan Amount ($)", 1000, 50000, 10000, help="loan_amnt")
cb_person_cred_hist_length = st.sidebar.slider("Credit History Length (months)", 0, 60, 10, help="cb_person_cred_hist_length")

input_df = pd.DataFrame({
    feature_cols[0]: [person_age],
    feature_cols[1]: [person_income],
    feature_cols[2]: [loan_amnt],
    feature_cols[3]: [cb_person_cred_hist_length]
})

if st.sidebar.button("🚀 Predict Risk", type="primary"):
    # Scale & predict
    input_scaled = scaler.transform(input_df)
    prediction = knn.predict(input_scaled)[0]
    prob = knn.predict_proba(input_scaled)[0]  # [low_risk_prob, high_risk_prob] assuming 0=low,1=high
    
    risk_label = "Low Risk ✅" if prediction == 0 else "High Risk ⚠️"
    st.success(f"**Prediction: {risk_label}**")
    st.info(f"Low Risk Prob: {prob[0]:.1%} | High Risk Prob: {prob[1]:.1%}")

    # KNN Explanation: distances to nearest neighbors
    distances, indices = knn.kneighbors(input_scaled, n_neighbors=5)
    st.subheader("🔍 Top 5 Similar Customers")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Avg Distance to Neighbors", f"{distances.mean():.2f}")
    with col2:
        st.metric("Closest Match Distance", f"{distances.min():.2f}")

# Main dashboard
tab1, tab2 = st.tabs(["📊 Prediction", "ℹ️ How It Works"])

with tab1:
    st.dataframe(input_df.T, use_container_width=True)
    
with tab2:
    st.markdown("""
    ### Why KNN for Similarity-Based Risk?
    - **No complex formulas**: Finds K=29 most similar historical customers [your model].
    - **Preprocessing needed**: StandardScaler equalizes scales (income vs age) [web:14].
    - **Non-linear boundaries**: Handles real customer clusters perfectly.
    - **Explainable**: "Approved because 24/29 neighbors succeeded."
    
    **Your Preprocessing Fixes**:
    - Outliers dropped (<6%) → Robust distances.
    - Missing values dropped → Reliable neighbors.
    """)

# Footer
st.markdown("---")
st.markdown("*Built for fintech similarity scoring. Retrain with `joblib.dump(knn, 'knn_model.pkl')` etc.*")
