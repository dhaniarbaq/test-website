import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="ML Streamlit System", layout="wide")

st.title("End to End Machine Learning System")
st.write("Model Development, Evaluation, and Deployment using Streamlit")

menu = st.sidebar.radio(
    "System Menu",
    ["Upload Data", "Model Training", "Model Evaluation", "Prediction"]
)

# -------------------------
# Upload Data
# -------------------------
if menu == "Upload Data":
    st.header("Upload Dataset")

    file = st.file_uploader("Upload CSV file", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.success("Dataset loaded successfully")
        st.dataframe(df.head())

# -------------------------
# Model Training
# -------------------------
elif menu == "Model Training":
    st.header("Model Development")

    file = st.file_uploader("Upload Training Dataset", type=["csv"])
    target = st.text_input("Enter target column name")

    if st.button("Train Model"):
        if file and target:
            df = pd.read_csv(file)

            X = df.drop(columns=[target])
            y = df[target]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            model = LogisticRegression()
            model.fit(X_train, y_train)

            joblib.dump(model, "model.pkl")
            joblib.dump(scaler, "scaler.pkl")

            st.success("Model trained and saved successfully")

# -------------------------
# Model Evaluation
# -------------------------
elif menu == "Model Evaluation":
    st.header("Model Evaluation")

    file = st.file_uploader("Upload Evaluation Dataset", type=["csv"])
    target = st.text_input("Enter target column name")

    if st.button("Evaluate Model"):
        if file and target:
            df = pd.read_csv(file)

            model = joblib.load("model.pkl")
            scaler = joblib.load("scaler.pkl")

            X = df.drop(columns=[target])
            y = df[target]

            X_scaled = scaler.transform(X)
            y_pred = model.predict(X_scaled)

            accuracy = accuracy_score(y, y_pred)
            cm = confusion_matrix(y, y_pred)

            st.write("Accuracy:", accuracy)

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", ax=ax)
            st.pyplot(fig)

# -------------------------
# Prediction
# -------------------------
elif menu == "Prediction":
    st.header("Model Deployment")

    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")

    st.write("Enter feature values")

    feature_1 = st.number_input("Feature 1")
    feature_2 = st.number_input("Feature 2")

    if st.button("Predict"):
        input_data = scaler.transform([[feature_1, feature_2]])
        prediction = model.predict(input_data)
        st.success(f"Prediction Result: {prediction[0]}")
