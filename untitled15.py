import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# --------------------------------
# Page Configuration
# --------------------------------
st.set_page_config(page_title="ML Streamlit System", layout="wide")

st.title("End to End Machine Learning System")
st.write("Model Development, Evaluation, and Deployment using Streamlit")

# --------------------------------
# Session State Initialization
# --------------------------------
if "data" not in st.session_state:
    st.session_state.data = None

if "model_name" not in st.session_state:
    st.session_state.model_name = None

menu = st.sidebar.radio(
    "System Menu",
    ["Upload Data", "Model Development", "Model Evaluation", "Model Deployment"]
)

# --------------------------------
# Upload Data
# --------------------------------
if menu == "Upload Data":
    st.header("Upload Dataset")

    file = st.file_uploader("Upload CSV file", type=["csv"])
    if file:
        st.session_state.data = pd.read_csv(file)
        st.success("Dataset uploaded and stored successfully")
        st.dataframe(st.session_state.data.head())

# --------------------------------
# Model Development
# --------------------------------
elif menu == "Model Development":
    st.header("Model Development")

    if st.session_state.data is None:
        st.warning("Please upload the dataset first.")
    else:
        df = st.session_state.data

        target = st.selectbox(
            "Select target column",
            df.columns
        )

        model_choice = st.selectbox(
            "Select Machine Learning Model",
            ["Logistic Regression", "Decision Tree", "Random Forest"]
        )

        if st.button("Train Model"):
            X = df.drop(columns=[target])
            y = df[target]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            if model_choice == "Logistic Regression":
                model = LogisticRegression()
            elif model_choice == "Decision Tree":
                model = DecisionTreeClassifier()
            else:
                model = RandomForestClassifier()

            model.fit(X_train, y_train)

            joblib.dump(model, "model.pkl")
            joblib.dump(scaler, "scaler.pkl")

            st.session_state.model_name = model_choice

            st.success(f"{model_choice} trained successfully using '{target}' as target column")

# --------------------------------
# Model Evaluation
# --------------------------------
elif menu == "Model Evaluation":
    st.header("Model Evaluation")

    if st.session_state.data is None:
        st.warning("Please upload the dataset first.")
    elif not os.path.exists("model.pkl"):
        st.warning("Model not found. Please train the model first.")
    else:
        df = st.session_state.data

        target = st.selectbox(
            "Select target column",
            df.columns
        )

        if st.button("Evaluate Model"):
            model = joblib.load("model.pkl")
            scaler = joblib.load("scaler.pkl")

            X = df.drop(columns=[target])
            y = df[target]

            X_scaled = scaler.transform(X)
            y_pred = model.predict(X_scaled)

            accuracy = accuracy_score(y, y_pred)
            cm = confusion_matrix(y, y_pred)

            st.subheader("Evaluation Results")
            st.write("Model Used:", st.session_state.model_name)
            st.write("Accuracy:", accuracy)

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", ax=ax)
            st.pyplot(fig)

# --------------------------------
# Model Deployment
# --------------------------------
elif menu == "Model Deployment":
    st.header("Model Deployment")

    if st.session_state.data is None:
        st.warning("Please upload the dataset first.")
    elif not os.path.exists("model.pkl"):
        st.warning("Model not found. Please train the model first.")
    else:
        df = st.session_state.data
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")

        st.write("Model Used:", st.session_state.model_name)
        st.write("Enter feature values")

        feature_columns = df.drop(columns=[df.columns[-1]]).columns

        input_values = []
        for col in feature_columns:
            value = st.number_input(f"{col}")
            input_values.append(value)

        if st.button("Predict"):
            input_data = scaler.transform([input_values])
            prediction = model.predict(input_data)
            st.success(f"Prediction Result: {prediction[0]}")
