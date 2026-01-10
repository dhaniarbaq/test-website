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
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

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
        # Ensure target variable 'rate' is present and cleaned
        df = df.dropna(subset=['rate'])
        
        # Missing Value Imputation: Numeric (Median)
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        
        # Missing Value Imputation: Categorical (Placeholder)
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
# 1. About this System (New Description Module)
# --------------------------------
if menu == "About this System":
    st.header("Predicting Child Health Vulnerabilities in Malaysia")
    st.subheader("System Description")
    st.write("""
    This interactive Machine Learning system is designed to analyze and predict child health vulnerabilities, 
    specifically focused on Early Childhood Mortality Rates across various states in Malaysia. 
    By integrating socio-economic indicators—such as household income, poverty levels, and infrastructure 
    access—the system provides data-driven insights to support public health policy and intervention strategies.
    """)
    
    st.subheader("Methodology: Tuned Stacking Regressor")
    st.write("""
    The system utilizes a **Tuned Stacking Regressor**, which is an ensemble learning technique. 
    It functions by combining the predictive strengths of multiple 'base' models (Optimized Random Forest 
    and Decision Trees) and using a 'meta-learner' (Linear Regression) to generate a final, highly 
    accurate estimation. This approach minimizes individual model biases and maximizes predictive stability.
    """)
    

# --------------------------------
# 2. Dataset Overview
# --------------------------------
elif menu == "Dataset Overview":
    st.header("Dataset Descriptive Analysis")
    if df_clean is not None:
        st.write("The following table presents the preprocessed socio-economic data used as the foundation for this analysis.")
        st.dataframe(df_clean.head())
        
        st.subheader("National Statistical Benchmarks")
        stats_df = pd.DataFrame({
            "Indicator": ["Mean Mortality Rate", "Standard Deviation", "Minimum Recorded", "Maximum Recorded"],
            "Value": [df_clean['rate'].mean(), df_clean['rate'].std(), df_clean['rate'].min(), df_clean['rate'].max()]
        })
        st.table(stats_df)
        
        st.subheader("Multivariate Correlation Analysis")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df_clean.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.error("System Error: 'mergednew.csv' not detected in root directory.")

# --------------------------------
# 3. Model Training (Stacking Only)
# --------------------------------
elif menu == "Model Training":
    st.header("Model Optimization and Training")
    if df_clean is not None:
        features = ['state', 'type', 'sex', 'piped_water', 'sanitation', 'electricity', 'income_mean', 'gini', 'poverty_absolute', 'cpi']
        target = 'rate'
        
        st.write("The system will now execute the **Tuned Stacking Regressor** training protocol.")
        
        if st.button("Initialize Training"):
            with st.spinner("Optimizing ensemble parameters..."):
                X = df_clean[features].copy()
                y = df_clean[target]

                # Categorical Encoding
                encoders = {}
                for col in ['state', 'type', 'sex']:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    encoders[col] = le

                # Scaling and Splitting
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

                # Stacking Architecture (Optimized)
                base_learners = [
                    ('rf_tuned', RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)),
                    ('dt_optimized', DecisionTreeRegressor(max_depth=10, random_state=42))
                ]
                stacking_model = StackingRegressor(
                    estimators=base_learners, 
                    final_estimator=LinearRegression()
                )

                stacking_model.fit(X_train, y_train)

                # Artifact Serialization
                joblib.dump(stacking_model, "model.pkl")
                joblib.dump(scaler, "scaler.pkl")
                joblib.dump(encoders, "encoders.pkl")
                joblib.dump(features, "feature_names.pkl")
                
                # Internal Evaluation for Verification
                y_pred = stacking_model.predict(X_test)
                st.success("Training successfully completed.")
                
                st.subheader("Performance Metrics (Test Set)")
                c1, c2, c3 = st.columns(3)
                c1.metric("R-Squared", f"{r2_score(y_test, y_pred):.4f}")
                c2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
                c3.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.4f}")
    else:
        st.warning("Action Required: Please ensure dataset is available before training.")

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

        st.write("Enter the following socio-economic parameters to generate a vulnerability assessment.")
        
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

        if st.button("Generate Mortality Analysis"):
            final_in = scaler.transform([input_data])
            res = model.predict(final_in)[0]
            
            # Formatted Results as per User Instructions
            st.divider()
            st.subheader("Analysis Results")
            st.success(f"**Predicted Rate: {res:.2f}%**")
            st.write(f"The model predicts a child mortality rate of **{res:.2f} deaths per 1,000 live births** for the given input parameters.")

            # Strategic Benchmarking
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
                st.write("**Primary Recommendation:**")
                if res > (avg_rate + 0.5 * std_dev):
                    st.write("Prioritize urgent socio-economic interventions and infrastructure auditing.")
                else:
                    st.write("Continue standard health surveillance and infrastructure maintenance.")

            # Distribution Visual
            fig, ax = plt.subplots(figsize=(10, 3))
            sns.kdeplot(df_clean['rate'], fill=True, color="skyblue", label="National Distribution")
            plt.axvline(res, color="red", linestyle="--", label="Current Prediction")
            plt.xlabel("Mortality Rate")
            plt.legend()
            st.pyplot(fig)
    else:
        st.warning("System Status: Prediction requires trained model artifacts. Please proceed to the Training module.")
