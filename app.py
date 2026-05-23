import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("model.pkl")

# Flower names
flowers = ["Setosa", "Versicolor", "Virginica"]

# Page config
st.set_page_config(
    page_title="Iris Flower Prediction",
    page_icon="🌸"
)

# Title
st.title("🌸 Iris Flower Prediction App")

st.write(
    "This machine learning app predicts the species of an Iris flower "
    "based on flower measurements."
)

st.divider()

# Input section
st.subheader("Enter Flower Measurements")

sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.0)
sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.0)
petal_length = st.slider("Petal Length", 1.0, 7.0, 4.0)
petal_width = st.slider("Petal Width", 0.1, 2.5, 1.0)

# Prediction button
if st.button("Predict Flower"):

    data = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    prediction = model.predict(data)[0]

    st.success(f"Predicted Flower: {flowers[prediction]}")

st.divider()

st.caption("Built using Streamlit and Scikit-learn")
