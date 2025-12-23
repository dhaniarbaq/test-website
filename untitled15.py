import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="Stacking Regression System", layout="wide")

# ---------- Helpers ----------
def read_any(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(file)
    raise ValueError("Unsupported file type")

def add_date_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df[date_col], errors="coerce")
    df[date_col] = dt
    df["month"] = dt.dt.month
    df["quarter"] = dt.dt.quarter
    df["year_from_date"] = dt.dt.year
    df["dayofyear"] = dt.dt.dayofyear
    return df

def infer_columns(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
    cat_cols = [c for c in X.columns if X[c].dtype == "object"]
    dt_cols = [c for c in X.columns if np.issubdtype(X[c].dtype, np.datetime64)]
    num_cols = [c for c in X.columns if c not in cat_cols + dt_cols]
    return X, cat_cols, num_cols, dt_cols

def make_preprocessor(cat_cols, num_cols):
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, cat_cols),
            ("num", num_pipe, num_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3
    )

def make_stacking_model(selected_models, meta_name):
    estimators = []
    if "Random Forest" in selected_models:
        estimators.append(("rf", RandomForestRegressor(random_state=42)))
    if "Gradient Boosting" in selected_models:
        estimators.append(("gbr", GradientBoostingRegressor(random_state=42)))
    if "Ridge" in selected_models:
        estimators.append(("ridge", Ridge(random_state=42)))
    if "SVR" in selected_models:
        estimators.append(("svr", SVR()))

    if meta_name == "Ridge":
        final_est = Ridge(random_state=42)
    elif meta_name == "Random Forest":
        final_est = RandomForestRegressor(random_state=42)
    else:
        final_est = Ridge(random_state=42)

    model = StackingRegressor(
        estimators=estimators,
        final_estimator=final_est,
        passthrough=False,
        cv=5,
        n_jobs=None
    )
    return model

def tune_pipeline(pipe, mode: str):
    # Smaller search space for Quick, larger for Thorough
    if mode == "Quick":
        n_iter = 15
    elif mode == "Balanced":
        n_iter = 35
    else:
        n_iter = 60

    # Parameter distributions cover common knobs
    # Some params may not exist depending on selected base learners
    param_dist = {
        "model__rf__n_estimators": [200, 400, 600],
        "model__rf__max_depth": [None, 6, 10, 16],
        "model__rf__min_samples_split": [2, 5, 10],
        "model__gbr__n_estimators": [150, 300, 500],
        "model__gbr__learning_rate": [0.03, 0.06, 0.1],
        "model__gbr__max_depth": [2, 3, 4],
        "model__svr__C": [1.0, 3.0, 10.0, 30.0],
        "model__svr__gamma": ["scale", "auto"],
        "model__final_estimator__alpha": [0.1, 1.0, 10.0, 30.0],
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=5,
        random_state=42,
        n_jobs=None,
        refit=True
    )
    return search

def metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

# ---------- Session state ----------
if "df" not in st.session_state:
    st.session_state.df = None
if "target" not in st.session_state:
    st.session_state.target = None
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "feature_cols" not in st.session_state:
    st.session_state.feature_cols = None

# ---------- UI ----------
st.title("Stacking Regression System")
st.caption("Upload data, train a tuned stacking regressor, predict, explain, export.")

page = st.sidebar.radio("Navigation", ["Home", "Data", "Train", "Predict", "Explain"])

if page == "Home":
    st.subheader("Welcome")
    st.write("This app builds a regression model using stacking with hyperparameter tuning.")
    st.write("Go to Data to upload and configure, then Train, then Predict and Explain.")

elif page == "Data":
    st.subheader("Data upload and setup")

    up = st.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx", "xls"])
    if up is not None:
        df = read_any(up)

        # Add date features if date exists
        if "date" in df.columns:
            df = add_date_features(df, "date")

        st.session_state.df = df
        st.write("Preview")
        st.dataframe(df.head(50), use_container_width=True)

        st.write("Shape:", df.shape)

        miss = df.isna().sum().sort_values(ascending=False)
        with st.expander("Missing values"):
            st.dataframe(miss[miss > 0].to_frame("missing"), use_container_width=True)

        target_default = "rate" if "rate" in df.columns else df.columns[-1]
        target = st.selectbox("Select target column", options=df.columns.tolist(), index=df.columns.tolist().index(target_default))
        st.session_state.target = target

        test_size = st.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
        seed = st.number_input("Random seed", value=42, step=1)

        st.session_state.test_size = float(test_size)
        st.session_state.seed = int(seed)

        st.success("Data loaded and configured")

elif page == "Train":
    st.subheader("Train stacking regression")

    if st.session_state.df is None or st.session_state.target is None:
        st.warning("Go to Data first and upload a dataset")
    else:
        df = st.session_state.df.copy()
        target = st.session_state.target

        X, cat_cols, num_cols, dt_cols = infer_columns(df, target)

        # Drop datetime columns after feature extraction
        if len(dt_cols) > 0:
            X = X.drop(columns=dt_cols)

        st.write("Detected categorical columns:", cat_cols)
        st.write("Detected numeric columns:", num_cols)

        selected_models = st.multiselect(
            "Select base learners",
            ["Random Forest", "Gradient Boosting", "Ridge", "SVR"],
            default=["Random Forest", "Gradient Boosting", "Ridge"]
        )
        meta = st.selectbox("Select meta learner", ["Ridge", "Random Forest"], index=0)
        mode = st.selectbox("Tuning mode", ["Quick", "Balanced", "Thorough"], index=1)

        if len(selected_models) < 2:
            st.error("Pick at least 2 base learners for stacking")
        else:
            if st.button("Train model"):
                y = df[target].values
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=st.session_state.test_size,
                    random_state=st.session_state.seed
                )

                pre = make_preprocessor(cat_cols, num_cols)
                model = make_stacking_model(selected_models, meta)
                pipe = Pipeline(steps=[("pre", pre), ("model", model)])

                search = tune_pipeline(pipe, mode)
                with st.spinner("Training and tuning in progress"):
                    search.fit(X_train, y_train)

                best_pipe = search.best_estimator_
                st.session_state.pipeline = best_pipe
                st.session_state.feature_cols = X.columns.tolist()

                y_pred = best_pipe.predict(X_test)
                rmse, mae, r2 = metrics(y_test, y_pred)

                st.success("Training completed")
                st.write("Best parameters")
                st.json(search.best_params_)

                st.write("Metrics")
                st.dataframe(pd.DataFrame({
                    "RMSE": [rmse],
                    "MAE": [mae],
                    "R2": [r2]
                }), use_container_width=True)

                # Predicted vs actual chart
                fig1 = plt.figure()
                plt.scatter(y_test, y_pred, s=12)
                plt.xlabel("Actual")
                plt.ylabel("Predicted")
                st.pyplot(fig1)

                # Residuals
                residuals = y_test - y_pred
                fig2 = plt.figure()
                plt.hist(residuals, bins=30)
                plt.xlabel("Residual")
                plt.ylabel("Count")
                st.pyplot(fig2)

                # Save model
                model_bytes = joblib.dump(best_pipe, "stacking_model.joblib")
                with open("stacking_model.joblib", "rb") as f:
                    st.download_button(
                        "Download trained model",
                        data=f,
                        file_name="stacking_model.joblib",
                        mime="application/octet-stream"
                    )

elif page == "Predict":
    st.subheader("Predict")

    if st.session_state.pipeline is None:
        st.warning("Train a model first")
    else:
        pipe = st.session_state.pipeline
        feat_cols = st.session_state.feature_cols

        tab1, tab2 = st.tabs(["Single prediction", "Batch prediction"])

        with tab1:
            st.write("Enter values for features")
            input_data = {}
            for c in feat_cols:
                input_data[c] = st.text_input(c, value="")

            if st.button("Predict single"):
                one = pd.DataFrame([input_data])

                # Basic type coercion attempt
                for col in one.columns:
                    one[col] = pd.to_numeric(one[col], errors="ignore")

                pred = pipe.predict(one)[0]
                st.success(f"Prediction: {pred}")

        with tab2:
            up2 = st.file_uploader("Upload file for batch prediction", type=["csv", "xlsx", "xls"], key="batch")
            if up2 is not None:
                df2 = read_any(up2)
                if "date" in df2.columns:
                    df2 = add_date_features(df2, "date")
                    # date becomes datetime then we keep only derived numeric features
                    if "date" in df2.columns:
                        df2 = df2.drop(columns=["date"])

                # Keep only required columns
                missing_cols = [c for c in feat_cols if c not in df2.columns]
                if len(missing_cols) > 0:
                    st.error("Missing columns: " + ", ".join(missing_cols))
                else:
                    Xb = df2[feat_cols].copy()
                    preds = pipe.predict(Xb)
                    out = df2.copy()
                    out["prediction"] = preds
                    st.dataframe(out.head(50), use_container_width=True)

                    csv = out.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download predictions CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )

elif page == "Explain":
    st.subheader("Explain model")

    if st.session_state.pipeline is None or st.session_state.df is None:
        st.warning("Train a model first")
    else:
        pipe = st.session_state.pipeline
        df = st.session_state.df.copy()
        target = st.session_state.target
        feat_cols = st.session_state.feature_cols

        X = df[feat_cols].copy()
        y = df[target].values

        st.write("Permutation importance estimates how much each feature affects predictions.")
        if st.button("Compute permutation importance"):
            with st.spinner("Computing importance"):
                r = permutation_importance(
                    pipe, X, y,
                    n_repeats=10,
                    random_state=42,
                    scoring="neg_root_mean_squared_error"
                )
            imp = pd.DataFrame({
                "feature": feat_cols,
                "importance_mean": r.importances_mean
            }).sort_values("importance_mean", ascending=False)

            st.dataframe(imp.head(20), use_container_width=True)

            fig = plt.figure()
            top = imp.head(15)
            plt.barh(top["feature"][::-1], top["importance_mean"][::-1])
            plt.xlabel("Importance")
            plt.ylabel("Feature")
            st.pyplot(fig)
