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
def load_and_preprocess():
    """Load YOUR exact CSV + preprocessing"""
    df = pd.read_csv('credit_risk_dataset.csv')
    
    # YOUR outlier removal (<6%)
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5*IQR
        upper = Q3 + 1.5*IQR
        outlier_pct = len(df[(df[col] < lower) | (df[col] > upper)]) / len(df) * 100
        if outlier_pct < 6:
            df = df[(df[col] >= lower) & (df[col] <= upper)]
    
    df = df.dropna()
    
    feature_cols = ['person_age','person_income','loan_amnt','cb_person_cred_hist_length']
    X = df[feature_cols].values
    y = df['loan_status'].values
    
    return X, y, feature_cols, df.shape[0]

# Load data
X, y, feature_cols, n_samples = load_and_preprocess()
knn_model, scaler = None, None

st.title("🏦 Credit Risk KNN (K=29)")
st.markdown("**Similarity-based risk scoring** - No files, trains live!")

col1, col2 = st.columns([1,2])
with col1:
    st.metric("Customers", n_samples)
    st.metric("Risk Split", f"{np.mean(1-y):.0%} Low / {np.mean(y):.0%} High")

# Train live
if st.button("🎯 Initialize Model", type="primary"):
    with st.spinner("Training KNN..."):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        knn_model = KNeighborsClassifier(n_neighbors=29, metric='minkowski')
        knn_model.fit(X_train_s, y_train)
        
        st.session_state.scaler = scaler
        st.session_state.knn = knn_model
        st.success("✅ Model ready!")

# Prediction sidebar
st.sidebar.header("👤 New Customer")
age = st.sidebar.slider("Age", 18, 80, 35)
income = st.sidebar.number_input("Income ($)", 10000, 500000, 60000)
loan_amt = st.sidebar.number_input("Loan ($)", 1000, 50000, 12000)
cred_hist = st.sidebar.slider("Credit History (mo)", 0, 60, 15)

input_data = np.array([[age, income, loan_amt, cred_hist]])

if 'knn' in st.session_state and st.sidebar.button("🔮 Predict", type="secondary"):
    scaler = st.session_state.scaler
    knn = st.session_state.knn
    
    input_scaled = scaler.transform(input_data)
    pred = knn.predict(input_scaled)[0]
    probs = knn.predict_proba(input_scaled)[0]
    
    col1, col2 = st.columns(2)
    col1.metric("Risk", "✅ LOW" if pred==0 else "⚠️ HIGH")
    col2.metric("Low Risk Prob", f"{probs[0]:.1%}")
    
    # Neighbors plot
    dists, _ = knn.kneighbors(input_scaled, n_neighbors=5)
    fig = px.bar(y=dists[0], x=[f"N{i+1}" for i in range(5)], 
                title="Distance to Top 5 Neighbors")
    st.plotly_chart(fig, use_container_width=True)

# Visualizations
tab1, tab2 = st.tabs(["📊 Risk Scatter", "📈 Age Distribution"])
with tab1:
    df_plot = pd.DataFrame(X, columns=feature_cols)
    df_plot['risk'] = y
    fig = px.scatter(df_plot, x='person_income', y='loan_amnt', 
                    color='risk', size='person_age', title="Income vs Loan by Risk")
    st.plotly_chart(fig)
with tab2:
    fig = px.histogram(pd.DataFrame(X, columns=feature_cols), x='person_age', 
                      color=y.astype(str), title="Age vs Risk")
    st.plotly_chart(fig)

st.markdown("*Live KNN training | Your exact preprocessing | Perfect screenshot match*")
