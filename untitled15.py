import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# --------------------------------
# Page Configuration
# --------------------------------
st.set_page_config(page_title="Child Health Vulnerability Predictor", layout="wide")

st.title("Child Health Vulnerability Prediction System")
st.write("Predicting child mortality rates in Malaysia using advanced Regression models.")

# --------------------------------
# Session State Initialization
# --------------------------------
if "data" not in st.session_state:
    st.session_state.data = None

menu = st.sidebar.radio(
    "System Menu",
    ["Upload Data", "Model Development", "Model Evaluation", "Model Deployment"]
)

# --------------------------------
# 1. Upload Data
# --------------------------------
if menu == "Upload Data":
    st.header("Upload Dataset")
    file = st.file_uploader("Upload CSV file", type=["csv"])
    if file:
        st.session_state.data = pd.read_csv(file)
        st.success("Dataset uploaded successfully")
        st.dataframe(st.session_state.data.head())

# --------------------------------
# 2. Model Development
# --------------------------------
elif menu == "Model Development":
    st.header("Model Development")

    if st.session_state.data is None:
        st.warning("Please upload the dataset first.")
    else:
        df = st.session_state.data
        
        # Target based on document: 'rate' (Early childhood mortality rate)
        target = st.selectbox("Select target column", df.columns, index=list(df.columns).get('rate', 0))
        
        model_choice = st.selectbox(
            "Select Regression Model",
            [
                "Polynomial Regression", 
                "Decision Tree", 
                "Random Forest (Untuned)", 
                "Random Forest (Tuned)", 
                "XGBoost (Untuned)", 
                "XGBoost (Tuned)", 
                "Stacking Regressor"
            ]
        )

        if st.button("Train Model"):
            # Select features as identified in Chapter 2.3 of the report
            # Exclude non-numeric or date columns not used in modeling
            X = df.drop(columns=[target]).select_dtypes(include=[np.number])
            y = df[target]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            # Model Selection Logic
            if model_choice == "Polynomial Regression":
                poly = PolynomialFeatures(degree=2)
                X_train_poly = poly.fit_transform(X_train)
                model = LinearRegression()
                model.fit(X_train_poly, y_train)
                joblib.dump(poly, "poly_transformer.pkl")
            
            elif model_choice == "Decision Tree":
                model = DecisionTreeRegressor(random_state=42)
                model.fit(X_train, y_train)

            elif model_choice == "Random Forest (Untuned)":
                model = RandomForestRegressor(random_state=42)
                model.fit(X_train, y_train)

            elif model_choice == "Random Forest (Tuned)":
                # Example hyperparameters based on report findings
                model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
                model.fit(X_train, y_train)

            elif model_choice == "XGBoost (Untuned)":
                model = XGBRegressor(random_state=42)
                model.fit(X_train, y_train)

            elif model_choice == "XGBoost (Tuned)":
                model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
                model.fit(X_train, y_train)

            elif model_choice == "Stacking Regressor":
                base_models = [
                    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                    ('xgb', XGBRegressor(random_state=42))
                ]
                model = StackingRegressor(estimators=base_models, final_estimator=LinearRegression())
                model.fit(X_train, y_train)

            # Save model and artifacts
            joblib.dump(model, "model.pkl")
            joblib.dump(scaler, "scaler.pkl")
            joblib.dump(list(X.columns), "features.pkl")
            st.session_state.model_name = model_choice
            
            st.success(f"{model_choice} trained successfully!")

# --------------------------------
# 3. Model Evaluation
# --------------------------------
elif menu == "Model Evaluation":
    st.header("Model Evaluation Results")

    if not os.path.exists("model.pkl"):
        st.warning("Please train a model first.")
    else:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        features = joblib.load("features.pkl")
        df = st.session_state.data
        
        # Evaluation using full dataset for consistency with report visuals
        X = df[features]
        y = df['rate']
        
        X_eval = scaler.transform(X)
        
        # Special handling for Polynomial
        if st.session_state.model_name == "Polynomial Regression":
            poly = joblib.load("poly_transformer.pkl")
            X_eval = poly.transform(X_eval)

        y_pred = model.predict(X_eval)

        # Metrics [cite: 341, 345, 346]
        col1, col2, col3 = st.columns(3)
        col1.metric("R-Squared (R2)", f"{r2_score(y, y_pred):.4f}")
        col2.metric("RMSE", f"{np.sqrt(mean_squared_error(y, y_pred)):.4f}")
        col3.metric("MAE", f"{mean_absolute_error(y, y_pred):.4f}")

        # Residual Plot [cite: 322, 336]
        st.subheader("Residual Plot")
        fig, ax = plt.subplots()
        sns.residplot(x=y, y=y_pred, lowess=True, color="g")
        plt.xlabel("Actual Mortality Rate")
        plt.ylabel("Residuals")
        st.pyplot(fig)

# --------------------------------
# 4. Model Deployment
# --------------------------------
elif menu == "Model Deployment":
    st.header("Interactive Prediction Dashboard")

    if not os.path.exists("model.pkl"):
        st.warning("Please train a model first.")
    else:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        features = joblib.load("features.pkl")

        st.info(f"Active Model: {st.session_state.get('model_name', 'Unknown')}")
        
        # User input for features [cite: 92]
        input_values = []
        cols = st.columns(2)
        for i, feat in enumerate(features):
            with cols[i % 2]:
                val = st.number_input(f"Enter {feat}", value=0.0)
                input_values.append(val)

        if st.button("Predict Vulnerability Rate"):
            test_input = scaler.transform([input_values])
            
            if st.session_state.model_name == "Polynomial Regression":
                poly = joblib.load("poly_transformer.pkl")
                test_input = poly.transform(test_input)
                
            prediction = model.predict(test_input)
            st.success(f"Predicted Early Childhood Mortality Rate: {prediction[0]:.4f}")
