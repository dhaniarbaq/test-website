import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, r2_score

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
if "task_type" not in st.session_state:
    st.session_state.task_type = None

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
        st.success("Dataset uploaded successfully")
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
        target = st.selectbox("Select target column", df.columns)
        
        # Auto-detect task type
        is_categorical = df[target].nunique() < 20 or df[target].dtype == 'object'
        task = "Classification" if is_categorical else "Regression"
        st.info(f"Detected Task Type: **{task}** (based on target values)")

        model_choice = st.selectbox(
            "Select Machine Learning Model",
            ["Logistic/Linear Regression", "Decision Tree", "Random Forest"]
        )

        if st.button("Train Model"):
            # Prepare Features (X) and Target (y)
            # We drop non-numeric columns for X to prevent errors
            X = df.drop(columns=[target]).select_dtypes(include=[np.number])
            y = df[target]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            # Model Logic based on Task Type
            if task == "Classification":
                if model_choice == "Logistic/Linear Regression": model = LogisticRegression()
                elif model_choice == "Decision Tree": model = DecisionTreeClassifier()
                else: model = RandomForestClassifier()
            else:
                if model_choice == "Logistic/Linear Regression": model = LinearRegression()
                elif model_choice == "Decision Tree": model = DecisionTreeRegressor()
                else: model = RandomForestRegressor()

            model.fit(X_train, y_train)

            # Save necessary components
            joblib.dump(model, "model.pkl")
            joblib.dump(scaler, "scaler.pkl")
            joblib.dump(list(X.columns), "features.pkl") # Save column names
            
            st.session_state.model_name = model_choice
            st.session_state.task_type = task
            st.success(f"{task} model trained successfully!")

# --------------------------------
# Model Evaluation
# --------------------------------
elif menu == "Model Evaluation":
    st.header("Model Evaluation")

    if not os.path.exists("model.pkl"):
        st.warning("Please train the model first.")
    else:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        features = joblib.load("features.pkl")
        df = st.session_state.data
        
        # We need the original target name used in training
        # For simplicity in this demo, we assume user selects the same target
        target = st.selectbox("Select target column used for training", df.columns)
        
        if st.button("Evaluate"):
            X = df[features]
            y = df[target]
            X_scaled = scaler.transform(X)
            y_pred = model.predict(X_scaled)

            if st.session_state.task_type == "Classification":
                acc = accuracy_score(y, y_pred)
                st.metric("Accuracy", f"{acc:.2%}")
                cm = confusion_matrix(y, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                st.pyplot(fig)
            else:
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                st.metric("R2 Score", f"{r2:.4f}")
                st.write(f"Mean Squared Error: {mse:.4f}")
                
                # Regression Plot
                fig, ax = plt.subplots()
                plt.scatter(y, y_pred, alpha=0.5)
                plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
                plt.xlabel("Actual")
                plt.ylabel("Predicted")
                st.pyplot(fig)

# --------------------------------
# Model Deployment
# --------------------------------
elif menu == "Model Deployment":
    st.header("Model Deployment")

    if not os.path.exists("model.pkl"):
        st.warning("Please train the model first.")
    else:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        features = joblib.load("features.pkl")

        st.write(f"Predicting using: **{st.session_state.model_name}**")
        
        input_data = []
        cols = st.columns(2) # Split inputs into two columns for better UI
        for i, col_name in enumerate(features):
            with cols[i % 2]:
                val = st.number_input(f"{col_name}", value=0.0)
                input_data.append(val)

        if st.button("Predict"):
            scaled_input = scaler.transform([input_data])
            prediction = model.predict(scaled_input)
            st.success(f"Prediction Result: {prediction[0]:.4f}" if isinstance(prediction[0], float) else f"Prediction Result: {prediction[0]}")
