import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error,r2_score
from xgboost import XGBRegressor


# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(page_title="House Price Predictor", layout="centered")


# -------------------------------------------------
# Load CSS
# -------------------------------------------------
def load_css():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


st.title("🏠 House Price Prediction App")
st.write("Stacking Regressor (Random Forest + Gradient Boosting + XGBoost)")


# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("house_data.csv")


df = load_data()


# -------------------------------------------------
# Clean Data (Outlier Removal)
# -------------------------------------------------
def clean_data(data):

    df_clean = data.copy()

    numeric_cols = [
        'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
        'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated'
    ]

    for col in numeric_cols:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]

    return df_clean


df = clean_data(df)


# -------------------------------------------------
# Sidebar Dataset Options
# -------------------------------------------------
st.sidebar.header("Dataset Options")

show_preview = st.sidebar.checkbox("Preview Top 5 Rows")
show_full = st.sidebar.checkbox("Show Full Dataset")
show_info = st.sidebar.checkbox("Show Dataset Info")
download_data = st.sidebar.checkbox("Download CSV")


# -------------------------------------------------
# Dataset Section
# -------------------------------------------------
if show_preview:
    st.subheader("Top 5 Rows")
    st.dataframe(df.head(), use_container_width=True)

if show_full:
    st.subheader("Full Dataset")
    st.dataframe(df, use_container_width=True)

if show_info:
    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)
    col1.write(f"Rows: {df.shape[0]}")
    col1.write(f"Columns: {df.shape[1]}")
    col2.write("Columns List:")
    col2.write(list(df.columns))

if download_data:
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Cleaned Dataset",
        data=csv,
        file_name="house_data_cleaned.csv",
        mime="text/csv"
    )


# -------------------------------------------------
# Train Model
# -------------------------------------------------
@st.cache_resource
def train_model(data):

    X = data.drop(columns=['price','date','street','city','statezip','country'])
    y = data['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    base_models = [
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ('xgb', XGBRegressor(n_estimators=100, random_state=42))
    ]

    model = StackingRegressor(
        estimators=base_models,
        final_estimator=LinearRegression(),
        cv=5
    )

    model.fit(X_train, y_train)

    y_preds = model.predict(X_test)

    r2 = r2_score(y_test,y_preds)
    rmse = np.sqrt(mean_squared_error(y_test, y_preds))

    return model, scaler, X.columns, r2, rmse


model, scaler, features, r2, rmse = train_model(df)


# -------------------------------------------------
# Model Metrics
# -------------------------------------------------
st.subheader("Model Performance")

c1, c2 = st.columns(2)
c1.metric("R² Score", round(r2, 3))
c2.metric("RMSE ($)", int(rmse))


# -------------------------------------------------
# Prediction Section
# -------------------------------------------------
st.subheader("Enter House Details")

inputs = []

for feature in features:
    value = st.number_input(feature, value=0.0)
    inputs.append(value)


if st.button("Predict Price"):

    arr = np.array(inputs).reshape(1, -1)
    arr = scaler.transform(arr)
    prediction = model.predict(arr)[0]

    st.success(f"Estimated Price: ${int(prediction):,}")
