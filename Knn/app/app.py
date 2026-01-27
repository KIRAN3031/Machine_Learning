import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Credit Risk KNN", layout="wide", page_icon="🏦")

@st.cache_data
def load_and_preprocess_data(file_path):
    """Load and preprocess dataset exactly like your notebook"""
    df = pd.read_csv(file_path)
    
    # Your outlier handling (<6% removal)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_percentage = (len(outliers) / len(df)) * 100
        if outlier_percentage < 6:
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    df = df.dropna()
    
    feature_cols = ['person_age','person_income','loan_amnt','cb_person_cred_hist_length']
    X = df[feature_cols]
    y = df['loan_status']
    
    return X, y, feature_cols

@st.cache_resource
def train_knn_model(X, y, feature_cols):
    """Train KNN model with scaling"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    knn = KNeighborsClassifier(n_neighbors=29, metric='minkowski')
    knn.fit(X_train_scaled, y_train)
    
    y_pred = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Save for faster reload
    joblib.dump(knn, 'knn_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(feature_cols, 'features.pkl')
    
    return knn, scaler, feature_cols, accuracy

st.title("🏦 Credit Risk KNN Predictor")
st.markdown("**K=29 Nearest Neighbors** - Similarity-based risk scoring for fintech")

# Load data
csv_file = st.file_uploader("Upload credit_risk_dataset.csv", type="csv")
if csv_file is not None:
    df = pd.read_csv(csv_file)
    st.success(f"✅ Loaded {len(df)} rows")
    st.dataframe(df.head())
    
    X, y, feature_cols = load_and_preprocess_data(csv_file)
    st.info(f"After cleaning: {len(X)} samples | Features: {feature_cols}")
    
    # Train model
    if st.button("🚀 Train KNN Model"):
        with st.spinner("Training K=29 KNN..."):
            knn, scaler, feature_cols, acc = train_knn_model(X, y, feature_cols)
            st.success(f"✅ Model trained! Accuracy: {acc:.3f}")
        
        # Model performance plot
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(x=y, color=y, title="Target Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = go.Figure(data=[go.Pie(labels=['Low Risk', 'High Risk'], values=[(1-y).mean()*100, y.mean()*100])])
            fig.update_layout(title="Risk Split")
            st.plotly_chart(fig, use_container_width=True)
    
    # Prediction section
    st.subheader("🔮 Real-time Prediction")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        person_age = st.slider("Age", 18, 80, 35)
    with col2:
        person_income = st.number_input("Income ($)", 10000, 500000, 60000)
    with col3:
        loan_amnt = st.number_input("Loan Amount ($)", 1000, 50000, 12000)
    with col4:
        cb_person_cred_hist_length = st.slider("Credit History (months)", 0, 60, 15)
    
    input_df = pd.DataFrame({
        feature_cols[0]: [person_age],
        feature_cols[1]: [person_income],
        feature_cols[2]: [loan_amnt],
        feature_cols[3]: [cb_person_cred_hist_length]
    })
    
    if st.button("🎯 Predict Risk", type="primary"):
        try:
            knn = joblib.load('knn_model.pkl')
            scaler = joblib.load('scaler.pkl')
            input_scaled = scaler.transform(input_df)
            prediction = knn.predict(input_scaled)[0]
            prob = knn.predict_proba(input_scaled)[0]
            
            risk_label = "✅ LOW RISK (Approve)" if prediction == 0 else "⚠️ HIGH RISK (Decline)"
            col1, col2 = st.columns(2)
            col1.metric("Decision", risk_label)
            col2.metric("Low Risk Prob", f"{prob[0]:.1%}")
            
            # KNN neighbors visualization
            distances, indices = knn.kneighbors(input_scaled, n_neighbors=10)
            fig = px.bar(x=range(1,11), y=distances[0], title="Distance to 10 Nearest Neighbors")
            st.plotly_chart(fig, use_container_width=True)
            
        except FileNotFoundError:
            st.error("❌ Train model first!")
else:
    st.info("👆 Upload your credit_risk_dataset.csv to start")

st.markdown("---")
st.markdown("*Full KNN pipeline: Load → Clean → Train → Predict | No separate files needed*")