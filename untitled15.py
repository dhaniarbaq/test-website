import streamlit as st
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Stacking Regression Prediction System",
    layout="centered"
)

# =========================
# Header Section
# =========================
st.title("Stacking Regression Prediction System")

st.write(
    "This draft web application represents the deployment of a tuned "
    "stacking regression model developed in the accompanying Jupyter Notebook. "
    "The model was trained, tuned, and evaluated prior to deployment."
)

st.divider()

# =========================
# Sidebar Input Section
# =========================
st.sidebar.header("User Input Features")

st.sidebar.write(
    "The following inputs correspond to the features used during model "
    "training in the Jupyter Notebook."
)

feature_1 = st.sidebar.number_input("Feature 1", min_value=0.0, step=0.1)
feature_2 = st.sidebar.number_input("Feature 2", min_value=0.0, step=0.1)
feature_3 = st.sidebar.number_input("Feature 3", min_value=0.0, step=0.1)
feature_4 = st.sidebar.number_input("Feature 4", min_value=0.0, step=0.1)

predict_button = st.sidebar.button("Predict")

# =========================
# Output Section
# =========================
if predict_button:
    # Placeholder logic for draft demonstration
    input_data = np.array([feature_1, feature_2, feature_3, feature_4])
    draft_prediction = input_data.mean()

    st.subheader("Prediction Output")
    st.success(f"Predicted Value: {draft_prediction:.2f}")

    st.subheader("Model Reference")
    st.write("""
    - Model Type: Tuned Stacking Regression  
    - Base Models: Random Forest, Gradient Boosting, Support Vector Regression  
    - Meta-Learner: Linear Regression  
    - Model Development: Conducted in Jupyter Notebook (.ipynb)
    """)

    st.subheader("Output Explanation")
    st.write(
        "The displayed value represents the predicted target variable "
        "based on the user-provided inputs. In the final system, this "
        "prediction would be generated directly by the tuned stacking "
        "regression model developed in the notebook."
    )

else:
    st.info(
        "Please enter feature values in the sidebar and click Predict "
        "to view the system output."
    )
