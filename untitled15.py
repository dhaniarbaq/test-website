import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Try to import XGBoost, handle error if not installed
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# --------------------------------
# Page Configuration
# --------------------------------
st.set_page_config(page_title="Child Health Vulnerability Predictor", layout="wide")

st.title("Child Health Vulnerability Prediction System")
st.write("Predicting mortality rates in Malaysia based on socio-economic factors.")

# --------------------------------
# Session State Initialization
# --------------------------------
if "data" not in st.session_state:
    # Auto-load the specific dataset if it exists in the repo
    if os.path.exists("mergednew.csv"):
        st.session_state.data = pd.read_csv("mergednew.csv")
    else:
        st.session_state.data = None

menu = st.sidebar.radio(
    "System Menu",
    ["Dataset Overview", "Model Development", "Model Evaluation", "Model Deployment"]
)

# --------------------------------
# 1. Dataset Overview
# --------------------------------
if menu == "Dataset Overview":
    st.header("Dataset Overview")
    if st.session_state.data is not None:
        df = st.session_state.data
        st.write("### Raw Data Preview (mergednew.csv)")
        st.dataframe(df.head())
        
        st.write("### Data Statistics")
        st.write(df.describe())
        
        st.write("### Missing Values")
        st.write(df.isnull().sum())
    else:
        st.warning("Dataset 'mergednew.csv' not found. Please upload it in the sidebar or include it in your GitHub folder.")
        uploaded_file = st.file_uploader("Upload mergednew.csv", type=["csv"])
        if uploaded_file:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.rerun()

# --------------------------------
# 2. Model Development
# --------------------------------
elif menu == "Model Development":
    st.header("Model Development")

    if st.session_state.data is None:
        st.warning("Please ensure the dataset is loaded first.")
    else:
        df = st.session_state.data.copy()
        
        # Preprocessing: The report implies using socio-economic features to predict 'rate'
        target = "rate"
        
        # 1. Drop rows where target is NaN
        df = df.dropna(subset=[target])
        
        # 2. Identify Features
        cat_cols = ['state', 'type', 'sex']
        num_cols = ['piped_water', 'sanitation', 'electricity', 'income_mean', 'gini', 'poverty_absolute', 'cpi']
        
        # 3. Simple Cleaning: Fill numeric NaNs with median
        for col in num_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # 4. Encoding Categorical
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            
        st.info(f"Target: **{target}** | Features: {cat_cols + num_cols}")

        model_list = [
            "Polynomial Regression", 
            "Decision Tree", 
            "Random Forest (Untuned)", 
            "Random Forest (Tuned)", 
            "Stacking Regressor"
        ]
        if XGB_AVAILABLE:
            model_list.extend(["XGBoost (Untuned)", "XGBoost (Tuned)"])
        else:
            st.error("XGBoost is not installed. Please add 'xgboost' to your requirements.txt")

        model_choice = st.selectbox("Select Regression Model", model_list)

        if st.button("Train Model"):
            X = df[cat_cols + num_cols]
            y = df[target]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            # Model Logic
            if model_choice == "Polynomial Regression":
                poly = PolynomialFeatures(degree=2)
                X_train_p = poly.fit_transform(X_train)
                model = LinearRegression()
                model.fit(X_train_p, y_train)
                joblib.dump(poly, "poly_transformer.pkl")
            
            elif model_choice == "Decision Tree":
                model = DecisionTreeRegressor(max_depth=10, random_state=42)
                model.fit(X_train, y_train)

            elif model_choice == "Random Forest (Untuned)":
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)

            elif model_choice == "Random Forest (Tuned)":
                model = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42)
                model.fit(X_train, y_train)

            elif model_choice == "XGBoost (Untuned)":
                model = XGBRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)

            elif model_choice == "XGBoost (Tuned)":
                model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
                model.fit(X_train, y_train)

            elif model_choice == "Stacking Regressor":
                base = [('rf', RandomForestRegressor(n_estimators=50)), ('dt', DecisionTreeRegressor())]
                model = StackingRegressor(estimators=base, final_estimator=LinearRegression())
                model.fit(X_train, y_train)

            # Save
            joblib.dump(model, "model.pkl")
            joblib.dump(scaler, "scaler.pkl")
            joblib.dump(encoders, "encoders.pkl")
            joblib.dump(cat_cols + num_cols, "feature_names.pkl")
            st.session_state.model_name = model_choice
            
            st.success(f"{model_choice} Trained Successfully!")

# --------------------------------
# 3. Model Evaluation
# --------------------------------
elif menu == "Model Evaluation":
    st.header("Evaluation Results")
    if not os.path.exists("model.pkl"):
        st.warning("Please train a model first.")
    else:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        feature_names = joblib.load("feature_names.pkl")
        
        # Evaluate on the processed dataset
        df = st.session_state.data.dropna(subset=['rate'])
        X = df[feature_names].copy()
        
        # Re-apply encoding for evaluation
        encoders = joblib.load("encoders.pkl")
        for col, le in encoders.items():
            X[col] = le.transform(X[col].astype(str))
            
        X_scaled = scaler.transform(X)
        if st.session_state.model_name == "Polynomial Regression":
            poly = joblib.load("poly_transformer.pkl")
            X_scaled = poly.transform(X_scaled)

        y_true = df['rate']
        y_pred = model.predict(X_scaled)

        col1, col2, col3 = st.columns(3)
        col1.metric("R² Score", f"{r2_score(y_true, y_pred):.4f}")
        col2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
        col3.metric("MAE", f"{mean_absolute_error(y_true, y_pred):.4f}")

        # Prediction Plot
        fig, ax = plt.subplots()
        plt.scatter(y_true, y_pred, alpha=0.3, color='blue')
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel("Actual Rate")
        plt.ylabel("Predicted Rate")
        st.pyplot(fig)

# --------------------------------
# 4. Model Deployment
# --------------------------------
elif menu == "Model Deployment":
    st.header("Predict Child Mortality Rate")
    if not os.path.exists("model.pkl"):
        st.warning("Train a model first.")
    else:
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        encoders = joblib.load("encoders.pkl")
        feature_names = joblib.load("feature_names.pkl")

        input_data = []
        cols = st.columns(2)
        for i, feat in enumerate(feature_names):
            with cols[i % 2]:
                if feat in encoders:
                    val = st.selectbox(f"Select {feat}", encoders[feat].classes_)
                    input_data.append(encoders[feat].transform([val])[0])
                else:
                    val = st.number_input(f"Enter {feat}", value=0.0)
                    input_data.append(val)

        if st.button("Predict"):
            final_input = scaler.transform([input_data])
            if st.session_state.get('model_name') == "Polynomial Regression":
                poly = joblib.load("poly_transformer.pkl")
                final_input = poly.transform(final_input)
            
            res = model.predict(final_input)
            st.success(f"The Predicted Mortality Rate is: {res[0]:.4f}")
