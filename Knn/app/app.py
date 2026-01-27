import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Credit Risk KNN", layout="wide", page_icon="🏦")

@st.cache_data
def load_and_preprocess_data():
    """Load YOUR dataset with exact preprocessing"""
    df = pd.read_csv('credit_risk_dataset.csv')
    
    # YOUR exact outlier removal (<6%)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_pct = (df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0] / len(df)) * 100
        if outlier_pct < 6:
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    df = df.dropna()
    
    feature_cols = ['person_age','person_income','loan_amnt','cb_person_cred_hist_length']
    X = df[feature_cols]
    y = df['loan_status']
    
    return X, y, feature_cols, df

@st.cache_resource
def get_trained_model():
    """Train KNN on-the-fly (no file saves)"""
    X, y, feature_cols, df = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    knn = KNeighborsClassifier(n_neighbors=29, metric='minkowski')
    knn.fit(X_train_scaled, y_train)
    
    return knn, scaler, feature_cols, len(df)

st.title("🏦 Instant Credit Risk Assessment")
st.markdown("**K-Nearest Neighbors (K=29)** - Similarity-based fintech decisions")

# Load data once
X, y, feature_cols, df = load_and_preprocess_data()
col1, col2 = st.columns(2)
col1.metric("📊 Total Customers", len(df))
col2.metric("⚖️ Low/High Risk", f"{(1-y).mean():.0%} / {y.mean():.0%}")

# Train model
if 'knn' not in st.session_state:
    with st.spinner("🔄 Training KNN model..."):
        st.session_state.knn, st.session_state.scaler, st.session_state.feature_cols, _ = get_trained_model()
        st.success("✅ Model ready!")

knn = st.session_state.knn
scaler = st.session_state.scaler
feature_cols = st.session_state.feature_cols

# Sidebar inputs
st.sidebar.header("👤 New Customer")
person_age = st.sidebar.slider("Age", 18, 80, 35)
person_income = st.sidebar.number_input("Income ($)", 10000, 500000, 60000)
loan_amnt = st.sidebar.number_input("Loan Amount ($)", 1000, 50000, 12000)
cb_person_cred_hist_length = st.sidebar.slider("Credit History (months)", 0, 60, 15)

input_df = pd.DataFrame({
    feature_cols[0]: [person_age],
    feature_cols[1]: [person_income],
    feature_cols[2]: [loan_amnt],
    feature_cols[3]: [cb_person_cred_hist_length]
})

if st.sidebar.button("🚀 Predict Risk", type="primary"):
    input_scaled = scaler.transform(input_df)
    prediction = knn.predict(input_scaled)[0]
    prob = knn.predict_proba(input_scaled)[0]
    
    risk_label = "✅ LOW RISK" if prediction == 0 else "⚠️ HIGH RISK"
    col1, col2, col3 = st.columns(3)
    col1.metric("Decision", risk_label)
    col2.metric("Low Risk Prob", f"{prob[0]:.1%}")
    col3.metric("High Risk Prob", f"{prob[1]:.1%}")
    
    # Top 5 similar customers (distances)
    distances, _ = knn.kneighbors(input_scaled, n_neighbors=5)
    fig = px.bar(x=["Neighbor 1", "2", "3", "4", "5"], y=distances[0], 
                 title="📏 Distance to Top 5 Similar Customers")
    st.plotly_chart(fig, use_container_width=True)

# Dashboard tabs
tab1, tab2 = st.tabs(["📈 Model Performance", "🔍 Dataset Insights"])

with tab1:
    # Feature importance via permutation (KNN style)
    fig = px.scatter(df, x=feature_cols[1], y=feature_cols[2], color='loan_status',
                    title="Income vs Loan Amount by Risk", size=feature_cols[0])
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x=feature_cols[0], color='loan_status', 
                          title="Age Distribution by Risk")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.pie(values=y.value_counts().values, names=['Low Risk', 'High Risk'])
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("*✅ Auto-loads your CSV → Exact preprocessing → Trains KNN → Live predictions | No file saves*")
