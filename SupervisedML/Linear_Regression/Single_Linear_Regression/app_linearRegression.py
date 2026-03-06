import streamlit as st
import  seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error

# Page configuration

st.set_page_config("Simple Linear Regression", layout="centered")

# load css
def load_css(filename):
    css_path = Path(__file__).parent / filename
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles.css")



# Title

st.markdown("""
    <div class="card">
        <h1>Simple Linear Regression </h1>
        <p>Predict <b> Tip Amount </b> from <b>Total Bill</b> using Linear Regression... </p>   

    </div> 
            """, unsafe_allow_html=True)


# Load dataset


@st.cache_data

def load_data():
    data = sns.load_dataset('tips')
    return data

df = load_data()

# Dataset Preview

st.markdown('<div class = "card">'
            '<h3>Dataset Preview</h2>', unsafe_allow_html=True)
st.dataframe(df.head())

# Prepare data

X, y = df[['total_bill']], df['tip']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Train model

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# Metrics Calculation

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)


# Display Metrics
st.markdown('<div class = "card">'
            '<h3>Model Performance Metrics</h3>', unsafe_allow_html=True)
st.markdown(f"""
            <ul>
                <li> Mean Absolute Error (MAE): {mae:.2f}<li>
                <li> Root Mean Squared Error (RMSE): {rmse:.2f}<li>
                <li> R-squared (R²) Score: {r2:.2f} <li>
            <ul>
""")

# Visualization
st.markdown('<div class = "card">'
            "<h3>Total Bill vs Tip</h3>", unsafe_allow_html=True)
fig,ax = plt.subplots()
ax.scatter(df["total_bill"], df["tip"], alpha=0.6)
ax.plot(df['total_bill'], model.predict(scaler.transform(X)), color='red')
ax.set_xlabel("Total Bill")
ax.set_ylabel("Tip")
st.pyplot(fig)

st.markdown("""
</div>
""", unsafe_allow_html=True)



# Performance

st.markdown("""
    <div class="card">
            """
            '<h3>Model Performance</h3>',
            unsafe_allow_html=True)
c1,c2,c3 = st.columns(3)
c1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
c2.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")
c3.metric("R-squared (R²) Score", f"{r2:.2f}")

st.markdown("""
    </div>
            """, unsafe_allow_html=True)


# Slope(m) and Intercept(c)
st.markdown(f"""
    <div class="card">
            <h3>Model Interception</h3>
            <p><B> co-efficient (m) : </B> {model.coef_[0]:.2f}<br>
            <b>Intercept: </b> {model.intercept_:.3f}</p>
    </div>
            """, unsafe_allow_html=True)


# Predicition

max_bill = float(df.total_bill.max())
min_bill = float(df.total_bill.min())
value = 30.0

st.markdown(
    f"""
    <div class="card">
        <p>Max bill: {max_bill}</p>
        <p>Min bill: {min_bill}</p>
        <p>Value: {value}</p>
    """,
    unsafe_allow_html=True,
)

total_bill = st.slider(
    "Select Total Bill Amount",
    min_value=min_bill,
    max_value=max_bill,
    value=value,
    step=0.1,
)
tip = model.predict(scaler.transform(np.array([[total_bill]])))[0]

st.markdown(
    f'<div class="predicition-box"> Predicted Tip Amount: {tip:.2f} </div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    </div>
    """,
    unsafe_allow_html=True,
)