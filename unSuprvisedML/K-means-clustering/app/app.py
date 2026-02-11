import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Customer Segmentation", layout="centered")


# -------------------------
# Load CSS
# -------------------------
def load_css():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


# -------------------------
# Title
# -------------------------
st.title("🛒 Wholesale Customer Segmentation")
st.write("KMeans Clustering based on Milk & Grocery Spending")


# -------------------------
# Load Dataset
# -------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Wholesale customers data.csv")

df = load_data()


# -------------------------
# Sidebar Controls
# -------------------------
st.sidebar.header("Controls")

k = st.sidebar.slider(
    "Number of Clusters (K)",
    min_value=2,
    max_value=10,
    value=4
)

show_data = st.sidebar.checkbox("Preview Dataset")
show_elbow = st.sidebar.checkbox("Show Elbow Method")
show_download = st.sidebar.checkbox("Download Clustered Data")


# -------------------------
# Data Preview
# -------------------------
if show_data:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)


# -------------------------
# Feature Selection
# -------------------------
features = ['Milk', 'Grocery']
X = df[features]


# -------------------------
# Data Preparation
# -------------------------
X_log = np.log1p(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_log)


# -------------------------
# Elbow Method
# -------------------------
if show_elbow:

    st.subheader("Elbow Method")

    wcss = []

    for i in range(1, 11):
        km = KMeans(n_clusters=i, random_state=42)
        km.fit(X_scaled)
        wcss.append(km.inertia_)

    fig, ax = plt.subplots()

    ax.plot(range(1, 11), wcss, marker='o')
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("WCSS")
    ax.set_title("Elbow Curve")

    st.pyplot(fig)


# -------------------------
# KMeans Model
# -------------------------
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X_scaled)

df['Cluster'] = labels


# -------------------------
# Cluster Plot
# -------------------------
st.subheader("Customer Clusters")

fig, ax = plt.subplots(figsize=(7, 5))

ax.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=labels
)

ax.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker='X',
    s=300
)

ax.set_xlabel("Milk (log-scaled)")
ax.set_ylabel("Grocery (log-scaled)")
ax.set_title("KMeans Clusters")

st.pyplot(fig)


# -------------------------
# Cluster Profiling
# -------------------------
st.subheader("Cluster Profiling (Average Spending)")

profile = df.groupby("Cluster")[features].mean()
st.dataframe(profile.round(2), use_container_width=True)


# -------------------------
# Download Option
# -------------------------
if show_download:
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Clustered Dataset",
        data=csv,
        file_name="clustered_customers.csv",
        mime="text/csv"
    )