import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# XGBoost is required for your best model architecture
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# --------------------------------
# Page Configuration
# --------------------------------
st.set_page_config(page_title="Child Health Vulnerability Analysis System", layout="wide")

# --------------------------------
# Data Loading & Global Preprocessing
# --------------------------------
@st.cache_data
def load_and_clean_data():
    if os.path.exists("mergednew.csv"):
        df = pd.read_csv("mergednew.csv")
        # Target variable 'rate' cleaning
        df = df.dropna(subset=['rate'])
        
        # Numeric Imputation (Median)
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        
        # Categorical Imputation
        cat_cols = df.select_dtypes(include=['object']).columns
        df[cat_cols] = df[cat_cols].fillna('Unknown')
        
        return df
    return None

df_clean = load_and_clean_data()

# --------------------------------
# Sidebar Navigation
# --------------------------------
st.sidebar.title("System Navigation")
menu = st.sidebar.radio(
    "Select Module",
    ["About this System", "Dataset Overview", "Model Training", "Prediction Dashboard"]
)

# --------------------------------
# 1. About this System
# --------------------------------
if menu == "About this System":
    st.header("Predicting Child Health Vulnerabilities in Malaysia")
    
    st.subheader("System Overview")
    st.write("""
    This advanced Machine Learning platform is engineered to predict Early Childhood Mortality Rates 
    across Malaysia. By evaluating critical socio-economic indicators—including household income 
    distribution, poverty absolute levels, and infrastructure accessibility (piped water and sanitation)—the 
    system provides high-precision vulnerability assessments to guide public health interventions.
    """)
    
    st.subheader("Technical Architecture: Tuned Stacking Regressor")
    st.write("""
    The 'Best Model' implemented here is a **Tuned Stacking Regressor**. This ensemble architecture 
    combines two powerful base learners:
    1. **Random Forest Regressor:** Handles high-dimensional data and captures complex feature interactions.
    2. **XGBoost Regressor:** Utilizes gradient boosting to minimize residual errors sequentially.
    
    A **Linear Regression meta-learner** then integrates the predictions from these two models to produce 
    a final, stabilized output. This multi-layered approach ensures the system remains robust against 
    data noise and provides superior accuracy compared to individual algorithms.
    """)
    
    

# --------------------------------
# 2. Dataset Overview
# --------------------------------
elif menu == "Dataset Overview":
    st.header("Dataset Descriptive Analysis")
    if df_clean is not None:
        st.write("Current cleaned records from 'mergednew.csv' used for analytical benchmarking.")
        st.dataframe(df_clean.head())
        
        st.subheader("National Statistical Benchmarks")
        stats_df = pd.DataFrame({
            "Indicator": ["Mean Mortality Rate", "Standard Deviation", "Minimum Recorded", "Maximum Recorded"],
            "Value": [df_clean['rate'].mean(), df_clean['rate'].std(), df_clean['rate'].min(), df_clean['rate'].max()]
        })
        st.table(stats_df)
        
        st.subheader("Feature Correlation Analysis")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df_clean.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.error("System Error: 'mergednew.csv' not detected.")

# --------------------------------
# 3. Model Training
# --------------------------------
elif menu == "Model Training":
    st.header("Model Optimization and Training")
    if not XGB_AVAILABLE:
        st.error("Critical Failure: XGBoost library not found. Please add 'xgboost' to requirements.txt.")
    elif df_clean is not None:
        features = ['state', 'type', 'sex', 'piped_water', 'sanitation', 'electricity', 'income_mean', 'gini', 'poverty_absolute', 'cpi']
        target = 'rate'
        
        st.write("Initializing Training for **Best Model (Stacking: Random Forest + XGBoost)**.")
        
        if st.button("Execute Training Protocol"):
            with st.spinner("Processing ensemble layers..."):
                X = df_clean[features].copy()
                y = df_clean[target]

                # Categorical Encoding
                encoders = {}
                for col in ['state', 'type', 'sex']:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    encoders[col] = le

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

                # Stacking Architecture (RF + XGBoost)
                base_learners = [
                    ('rf_tuned', RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)),
                    ('xgb_tuned', XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42))
                ]
                stacking_model = StackingRegressor(
                    estimators=base_learners, 
                    final_estimator=LinearRegression()
                )

                stacking_model.fit(X_train, y_train)

                # Save artifacts
                joblib.dump(stacking_model, "model.pkl")
                joblib.dump(scaler, "scaler.pkl")
                joblib.dump(encoders, "encoders.pkl")
                joblib.dump(features, "feature_names.pkl")
                
                st.success("Tuned Stacking Model trained and serialized successfully.")
                
                y_pred = stacking_model.predict(X_test)
                st.subheader("Performance Validation Metrics")
                c1, c2, c3 = st.columns(3)
                c1.metric("R-Squared", f"{r2_score(y_test, y_pred):.4f}")
                c2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
                c3.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.4f}")

# --------------------------------
# 4. Prediction Dashboard
# --------------------------------
elif menu == "Prediction Dashboard":
    st.header("Vulnerability Prediction Dashboard")
    if os.path.exists("model.pkl"):
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        encoders = joblib.load("encoders.pkl")
        features = joblib.load("feature_names.pkl")

        st.write("Adjust parameters below to generate a localized health vulnerability prediction.")
        
        input_data = []
        cols = st.columns(2)
        for i, f in enumerate(features):
            with cols[i % 2]:
                if f in encoders:
                    val = st.selectbox(f"{f.replace('_', ' ').capitalize()}", encoders[f].classes_)
                    input_data.append(encoders[f].transform([val])[0])
                else:
                    val = st.number_input(f"{f.replace('_', ' ').capitalize()}", value=float(df_clean[f].median()))
                    input_data.append(val)

        if st.button("Generate Mortality Prediction"):
            final_in = scaler.transform([input_data])
            res = model.predict(final_in)[0]
            
            # Formatted Output Requirements
            st.divider()
            st.subheader("Predictive Result")
            st.success(f"**Predicted Rate: {res:.2f}%**")
            st.write(f"The model predicts a child mortality rate of **{res:.2f} deaths per 1,000 live births** for the given input parameters.")

            # Classification Benchmarking
            avg_rate = df_clean['rate'].mean()
            std_dev = df_clean['rate'].std()

            col_l, col_r = st.columns(2)
            with col_l:
                st.write("**Vulnerability Classification:**")
                if res < (avg_rate - 0.5 * std_dev):
                    st.info("Status: Lower than National Average")
                elif res > (avg_rate + 0.5 * std_dev):
                    st.error("Status: Higher than National Average")
                else:
                    st.warning("Status: Within National Average Range")

            with col_r:
                st.write("**Recommended Strategic Action:**")
                if res > (avg_rate + 0.5 * std_dev):
                    st.write("Initiate localized socio-economic audit and infrastructure resource allocation.")
                else:
                    st.write("Maintain current health surveillance and infrastructure support protocols.")

            # Result Distribution Visual
            fig, ax = plt.subplots(figsize=(10, 3))
            sns.kdeplot(df_clean['rate'], fill=True, color="skyblue", label="National Historical Distribution")
            plt.axvline(res, color="red", linestyle="--", label="User Prediction")
            plt.xlabel("Mortality Rate")
            plt.legend()
            st.pyplot(fig)
    else:
        st.warning("System Status: Prediction inactive. Please complete the Model Training module first.")
