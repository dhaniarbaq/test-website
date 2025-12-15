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
    "This web-based system demonstrates the layout of a machine learning "
    "application using a stacking regression model for prediction."
)

st.divider()

# =========================
# Sidebar Input Section
# =========================
st.sidebar.header("Input Features")

feature_1 = st.sidebar.number_input("Feature 1", min_value=0.0, step=0.1)
feature_2 = st.sidebar.number_input("Feature 2", min_value=0.0, step=0.1)
feature_3 = st.sidebar.number_input("Feature 3", min_value=0.0, step=0.1)
feature_4 = st.sidebar.number_input("Feature 4", min_value=0.0, step=0.1)

st.sidebar.divider()

predict_btn = st.sidebar.button("Predict")

# =========================
# Output Section
# =========================
if predict_btn:
    # Mock prediction (for layout demonstration only)
    input_data = np.array([feature_1, feature_2, feature_3, feature_4])
    mock_prediction = input_data.mean()

    st.subheader("Prediction Result")
    st.success(f"Predicted Value: {mock_prediction:.2f}")

    st.subheader("Model Information")
    st.write("""
    - **Model Type:** Stacking Regression  
    - **Base Models:** Random Forest, Gradient Boosting, Support Vector Regression  
    - **Meta-Learner:** Linear Regression  
    """)

    st.subheader("System Output Description")
    st.write(
        "The system generates a numerical prediction based on the values "
        "entered by the user. This draft focuses on system layout and user interaction."
    )

else:
    st.info("Please enter feature values in the sidebar and click Predict.")
