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

# XGBoost handling
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# --------------------------------
# Page Configuration
# --------------------------------
st.set_page_config(page_title="Child Health Vulnerability Analysis", layout="wide")

st.title("Child Health Vulnerability Prediction System")
st.write("Predictive modeling of childhood mortality rates in Malaysia based on socio-economic indicators.")

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
menu = st.sidebar.radio(
    "Navigation Menu",
    ["Dataset Overview", "Model Development", "Model Evaluation", "Model Deployment"]
)

# --------------------------------
# 1. Dataset Overview
# --------------------------------
if menu == "Dataset Overview":
    st.header("Dataset Overview")
    if df_clean is not None:
        st.write("The following table displays the preprocessed dataset used for model training.")
        st.dataframe(df_clean.head())
        
        # Statistical Benchmarks
        st.subheader("Statistical Benchmarks")
        stats_df = pd.DataFrame({
            "Metric": ["Mean Mortality Rate", "Standard Deviation", "Minimum Value", "Maximum Value"],
            "Value": [df_clean['rate'].mean(), df_clean['rate'].std(), df_clean['rate'].min(), df_clean['rate'].max()]
        })
        st.table(stats_df)
        
        st.subheader("Feature Correlation Analysis")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df_clean.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.error("Error: 'mergednew.csv' not found in the root directory.")

# --------------------------------
# 2. Model Development
# --------------------------------
elif menu == "Model Development":
    st.header("Model Development and Training")
    if df_clean is not None:
        features = ['state', 'type', 'sex', 'piped_water', 'sanitation', 'electricity', 'income_mean', 'gini', 'poverty_absolute', 'cpi']
        target = 'rate'
        
        st.info(f"Target Variable: {target} | Input Features: {len(features)}")
        
        model_choice = st.selectbox("Select Regression Algorithm", [
            "Polynomial Regression", "Decision Tree", 
            "Random Forest (Untuned)", "Random Forest (Tuned)", 
            "XGBoost (Untuned)", "XGBoost (Tuned)", "Stacking Regressor"
        ])

        if st.button("Execute Training"):
            X = df_clean[features].copy()
            y = df_clean[target]

            # Categorical Feature Encoding
            encoders = {}
            for col in ['state', 'type', 'sex']:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                encoders[col] = le

            # Feature Scaling
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            # Algorithm Initialization
            if model_choice == "Polynomial Regression":
                poly = PolynomialFeatures(degree=2)
                X_train = poly.fit_transform(X_train)
                model = LinearRegression()
                joblib.dump(poly, "poly_transformer.pkl")
            elif model_choice == "Decision Tree":
                model = DecisionTreeRegressor(max_depth=10)
            elif model_choice == "Random Forest (Tuned)":
                model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
            elif model_choice == "XGBoost (Tuned)" and XGB_AVAILABLE:
                model = XGBRegressor(n_estimators=200, learning_rate=0.05)
            elif model_choice == "Stacking Regressor":
                base = [('rf', RandomForestRegressor(n_estimators=100)), ('dt', DecisionTreeRegressor())]
                model = StackingRegressor(estimators=base, final_estimator=LinearRegression())
            else:
                model = RandomForestRegressor(n_estimators=100) if "Random" in model_choice else DecisionTreeRegressor()

            model.fit(X_train, y_train)

            # Serialization of artifacts
            joblib.dump(model, "model.pkl")
            joblib.dump(scaler, "scaler.pkl")
            joblib.dump(encoders, "encoders.pkl")
            joblib.dump(features, "feature_names.pkl")
            st.session_state.model_name = model_choice
            st.success(f"Model Training Complete: {model_choice}")

# --------------------------------
# 3. Model Evaluation
# --------------------------------
elif menu == "Model Evaluation":
    st.header("Predictive Performance Evaluation")
    if os.path.exists("model.pkl"):
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        encoders = joblib.load("encoders.pkl")
        features = joblib.load("feature_names.pkl")

        X_eval = df_clean[features].copy()
        y_true = df_clean['rate']
        for col, le in encoders.items():
            X_eval[col] = le.transform(X_eval[col].astype(str))

        X_scaled = scaler.transform(X_eval)
        if st.session_state.get('model_name') == "Polynomial Regression":
            poly = joblib.load("poly_transformer.pkl")
            X_scaled = poly.transform(X_scaled)

        y_pred = model.predict(X_scaled)
        
        # Evaluation Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("R-Squared", f"{r2_score(y_true, y_pred):.4f}")
        c2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
        c3.metric("MAE", f"{mean_absolute_error(y_true, y_pred):.4f}")

        # Visualization: Regression Plot
        fig, ax = plt.subplots()
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.title("Actual vs. Predicted Mortality Values")
        plt.xlabel("Actual Rate")
        plt.ylabel("Predicted Rate")
        st.pyplot(fig)
    else:
        st.warning("Assessment required: Please train a model to view evaluation results.")

# --------------------------------
# 4. Model Deployment
# --------------------------------
elif menu == "Model Deployment":
    st.header("Deployment: Mortality Rate Prediction")
    if os.path.exists("model.pkl"):
        model = joblib.load("model.pkl")
        scaler = joblib.load("scaler.pkl")
        encoders = joblib.load("encoders.pkl")
        features = joblib.load("feature_names.pkl")

        input_data = []
        cols = st.columns(2)
        for i, f in enumerate(features):
            with cols[i % 2]:
                if f in encoders:
                    val = st.selectbox(f"Select {f.capitalize()}", encoders[f].classes_)
                    input_data.append(encoders[f].transform([val])[0])
                else:
                    val = st.number_input(f"Enter {f.capitalize()}", value=float(df_clean[f].median()))
                    input_data.append(val)

        if st.button("Generate Prediction"):
            final_in = scaler.transform([input_data])
            if st.session_state.get('model_name') == "Polynomial Regression":
                poly = joblib.load("poly_transformer.pkl")
                final_in = poly.transform(final_in)
            
            res = model.predict(final_in)[0]
            
            # Formatted Output
            st.success(f"Predicted Early Childhood Mortality Rate: {res:.2f}")

            # Statistical Benchmarking
            avg_rate = df_clean['rate'].mean()
            std_dev = df_clean['rate'].std()

            st.divider()
            col_left, col_right = st.columns(2)

            with col_left:
                st.subheader("Vulnerability Classification")
                if res < (avg_rate - 0.5 * std_dev):
                    st.info("Classification: Lower than Average Vulnerability")
                    st.write("The predicted value is significantly below historical national benchmarks.")
                elif res > (avg_rate + 0.5 * std_dev):
                    st.error("Classification: Higher than Average Vulnerability")
                    st.write("The predicted value indicates heightened risk levels requiring strategic intervention.")
                else:
                    st.warning("Classification: Within National Average Range")
                    st.write("The predicted value is consistent with standard historical distributions.")

            with col_right:
                st.subheader("Strategic Recommendations")
                if res < (avg_rate - 0.5 * std_dev):
                    st.write("- Maintain current infrastructure and socio-economic support standards.")
                    st.write("- Documentation of local best practices for regional scalability.")
                elif res > (avg_rate + 0.5 * std_dev):
                    st.write("- Priority enhancement of sanitation and potable water infrastructure.")
                    st.write("- Expansion of social safety nets and financial aid for low-income households.")
                else:
                    st.write("- Implementation of preventive health measures and income equality monitoring.")
                    st.write("- Continuous surveillance of cost-of-living indices (CPI).")

            # Result Distribution Visualization
            st.subheader("Statistical Positioning within National Distribution")
            fig, ax = plt.subplots(figsize=(10, 3))
            sns.kdeplot(df_clean['rate'], fill=True, color="skyblue", label="National Historical Distribution")
            plt.axvline(res, color="red", linestyle="--", label="Current Prediction", linewidth=2)
            plt.axvline(avg_rate, color="green", linestyle="-", label="National Average", linewidth=2)
            plt.xlabel("Mortality Rate")
            plt.legend()
            st.pyplot(fig)
    else:
        st.warning("System Status: Model artifacts not found. Deployment requires prior training.")
