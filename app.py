import streamlit as st
import pandas as pd
import joblib
import io
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score, classification_report, mean_absolute_error

from src.data_loader import DataLoader
from src.preprocessor import PipelineBuilder
from src.trainer import ModelTrainer
from src.visualizer import Visualizer

st.set_page_config(page_title="Drop & Train V2", layout="wide")
st.title("🧠 Drop & Train: Professional Edition")

# --- Helper Function for Hyperparameters ---
def get_hyperparams(algo_name, problem_type):
    params = {}
    st.sidebar.header("⚙️ Hyperparameters")
    
    if "Random Forest" in algo_name:
        params["n_estimators"] = st.sidebar.slider("n_estimators (Trees)", 10, 500, 100)
        params["max_depth"] = st.sidebar.slider("max_depth", 1, 50, 10)
        
    elif "XGBoost" in algo_name or "Gradient Boosting" in algo_name:
        params["learning_rate"] = st.sidebar.slider("learning_rate", 0.01, 0.5, 0.1)
        params["n_estimators"] = st.sidebar.slider("n_estimators", 10, 500, 100)
        params["max_depth"] = st.sidebar.slider("max_depth", 1, 20, 3)
        
    elif "SVM" in algo_name:
        params["C"] = st.sidebar.slider("C (Regularization)", 0.01, 10.0, 1.0)
        params["kernel"] = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly"])
        
    elif "KNN" in algo_name:
        params["n_neighbors"] = st.sidebar.slider("n_neighbors", 1, 20, 5)
        params["weights"] = st.sidebar.selectbox("weights", ["uniform", "distance"])
        
    elif "Logistic" in algo_name:
        params["C"] = st.sidebar.slider("C (Inverse Reg)", 0.01, 10.0, 1.0)
        
    # Add more as needed...
    return params

# 1. Load Data
uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:
    df = DataLoader.load_data(uploaded_file)
    
    if df is not None:
        # --- NEW FEATURE: Drop Columns ---
        st.write("### 🛠️ Data Setup")
        all_cols = df.columns.tolist()
        
        # Allow dropping columns BEFORE selecting target to avoid errors
        cols_to_drop = st.multiselect("Select columns to DROP (e.g. IDs, Names)", all_cols)
        
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            st.warning(f"Dropped: {cols_to_drop}")
            
        # 2. Configuration
        st.write("---")
        col1, col2, col3 = st.columns(3)
        
        # Check if we have columns left
        if len(df.columns) > 0:
            target = col1.selectbox("Select Target Variable", df.columns)
            
            # Auto-detect problem type
            if df[target].nunique() < 20 and (df[target].dtype == 'object' or df[target].dtype == 'int64'):
                problem_type = "Classification"
            else:
                problem_type = "Regression"
                
            col2.info(f"Detected Type: **{problem_type}**")

            # 3. Data Split
            X = df.drop(columns=[target])
            y = df[target]
            
            test_size = col3.slider("Test Size", 0.1, 0.5, 0.2)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # 4. Model Selection
            if problem_type == "Classification":
                algo_options = ["Random Forest", "XGBoost", "Logistic Regression", "Support Vector Machine (SVM)", "K-Nearest Neighbors (KNN)", "Decision Tree", "Gradient Boosting"]
            else:
                algo_options = ["Random Forest", "XGBoost", "Linear Regression", "Support Vector Machine (SVR)", "K-Nearest Neighbors (KNN)", "Decision Tree", "Gradient Boosting"]
            
            algo = st.selectbox("Select Algorithm", algo_options)
            
            # --- Get Hyperparams from Sidebar ---
            params = get_hyperparams(algo, problem_type)

            # 5. Training Trigger
            if st.button("Train Model"):
                with st.spinner("Training Pipeline..."):
                    try:
                        # A. Build Preprocessor
                        pipeline_builder = PipelineBuilder(X_train)
                        preprocessor = pipeline_builder.build_preprocessor()
                        
                        # B. Train Model (Pass params now!)
                        trainer = ModelTrainer(problem_type)
                        model_pipeline = trainer.train(X_train, y_train, preprocessor, algo, params)
                        
                        # C. Predictions
                        y_pred = model_pipeline.predict(X_test)
                        
                        # --- Results Section ---
                        st.markdown("---")
                        st.subheader("📊 Model Performance")
                        
                        # Metrics
                        metric_col1, metric_col2 = st.columns(2)
                        
                        if problem_type == "Classification":
                            acc = accuracy_score(y_test, y_pred)
                            metric_col1.metric("Accuracy", f"{acc:.2%}")
                            st.text("Classification Report:")
                            st.code(classification_report(y_test, y_pred))
                            
                            st.subheader("Confusion Matrix")
                            Visualizer.plot_confusion_matrix(y_test, y_pred, labels=model_pipeline.classes_)
                        else:
                            r2 = r2_score(y_test, y_pred)
                            mae = mean_absolute_error(y_test, y_pred)
                            metric_col1.metric("R² Score", f"{r2:.4f}")
                            metric_col2.metric("MAE", f"{mae:.4f}")
                            
                            st.subheader("Actual vs Predicted")
                            Visualizer.plot_actual_vs_predicted(y_test, y_pred)

                        # --- Feature Importance ---
                        st.markdown("---")
                        st.subheader("✨ Feature Importance")
                        try:
                            preprocessor_step = model_pipeline.named_steps['preprocessor']
                            feature_names = preprocessor_step.get_feature_names_out()
                            Visualizer.plot_feature_importance(model_pipeline, feature_names)
                        except Exception as e:
                            st.info("Feature importance not available for this model/configuration.")

                        # --- Download Model ---
                        st.markdown("---")
                        st.subheader("💾 Download Trained Pipeline")
                        buffer = io.BytesIO()
                        joblib.dump(model_pipeline, buffer)
                        buffer.seek(0)
                        
                        st.download_button(
                            label="Download .joblib Model",
                            data=buffer,
                            file_name="drop_and_train_model.joblib",
                            mime="application/octet-stream"
                        )
                        
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
        else:
            st.error("You dropped all columns! Please keep at least one feature and one target.")