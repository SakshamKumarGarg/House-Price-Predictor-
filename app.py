# app.py

import streamlit as st  # pyright: ignore[reportMissingImports]
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
import seaborn as sns  # pyright: ignore[reportMissingModuleSource]
from sklearn.datasets import fetch_california_housing

# -----------------------------
# Load trained model
# -----------------------------
model = joblib.load("house_price_model.pkl")

# Load dataset for visualization
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

st.set_page_config(page_title="California House Price Predictor", layout="wide")

# -----------------------------
# Title & description
# -----------------------------
st.title("🏠 California House Price Predictor")
st.write("""
Predict California housing prices based on features like income, rooms, population, and location.
Also includes interactive visualizations for better insights.
""")

# -----------------------------
# Input features
# -----------------------------
st.sidebar.header("Enter House Features")
MedInc = st.sidebar.number_input("Median Income ($10k)", 0.0, 20.0, 5.0)
HouseAge = st.sidebar.number_input("House Age (years)", 1.0, 50.0, 20.0)
AveRooms = st.sidebar.number_input("Average Rooms per Household", 1.0, 10.0, 5.0)
AveBedrms = st.sidebar.number_input("Average Bedrooms per Household", 0.5, 5.0, 1.0)
Population = st.sidebar.number_input("Population", 1.0, 5000.0, 500.0)
AveOccup = st.sidebar.number_input("Average Occupants per Household", 1.0, 10.0, 3.0)
Latitude = st.sidebar.number_input("Latitude", 32.0, 42.0, 35.0)
Longitude = st.sidebar.number_input("Longitude", -125.0, -114.0, -120.0)

# -----------------------------
# Predict Button
# -----------------------------
if st.button("Predict Price"):
    input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                            Population, AveOccup, Latitude, Longitude]])
    prediction = model.predict(input_data)[0]
    st.success(f"🏡 Estimated House Price: ${prediction * 100000:.2f}")

# -----------------------------
# Visualizations
# -----------------------------
st.header("📊 Visualizations")

# Feature correlation heatmap
if st.checkbox("Show Feature Correlation Heatmap"):
    df = pd.concat([X, y.rename("Target")], axis=1)
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Predicted vs Actual Prices
if st.checkbox("Show Predicted vs Actual Prices"):
    y_pred_all = model.predict(X)
    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(y, y_pred_all, alpha=0.6, color='green')
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.set_title("Actual vs Predicted House Prices")
    st.pyplot(fig)

# Feature importance (Linear Regression coefficients)
if st.checkbox("Show Feature Importance"):
    coefficients = pd.Series(model.named_steps['model'].coef_, index=X.columns)
    fig, ax = plt.subplots(figsize=(10,6))
    coefficients.sort_values().plot(kind='barh', color='skyblue', ax=ax)
    ax.set_title("Feature Importance (Linear Regression Coefficients)")
    st.pyplot(fig)
