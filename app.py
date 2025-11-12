# app.py
import io
import json
import logging
import datetime
from typing import Tuple, Dict, Any

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import ElasticNet, SGDClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.exceptions import NotFittedError

# Optional libs - ensure installed in requirements before production
try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    XGBClassifier = XGBRegressor = None

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
except Exception:
    ImbPipeline = SMOTE = None

# ----- CONFIG -----
MAX_FILE_BYTES = 8 * 1024 * 1024      # 8 MB max upload on public host
MAX_ROWS_SAMPLE = 20000               # sample large datasets down to this many rows for training
MAX_N_ITER = 60                       # cap RandomizedSearchCV iterations
MAX_CV_FOLDS = 5                      # cap CV folds
DEFAULT_N_ITER = 40
DEFAULT_CV_FOLDS = 5

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Drop-and-Train ", layout="wide")
st.title("Drop-and-Train")

# ----- Session state initialization -----
if "best_models" not in st.session_state:
    st.session_state["best_models"] = {}
if "df" not in st.session_state:
    st.session_state["df"] = None
if "target" not in st.session_state:
    st.session_state["target"] = None

# ----- Helpers -----
def sanitize_cell(v):
    """Mitigate simple CSV injection by prefixing suspicious leading characters in strings."""
    if isinstance(v, str) and v and v[0] in ("=", "+", "-", "@"):
        return "'" + v
    return v

def safe_read_file(uploaded) -> pd.DataFrame:
    """Read CSV/TXT/XLSX safely with try/except and basic sanitization."""
    uploaded.seek(0, io.SEEK_SET)
    name = uploaded.name.lower()
    try:
        if name.endswith(".txt"):
            sep = st.text_input("Enter the separator (default is ',')", value=",")
            df = pd.read_csv(uploaded, sep=sep, encoding="utf-8", engine="python")
        elif name.endswith(".csv"):
            df = pd.read_csv(uploaded, encoding="utf-8", engine="python")
        elif name.endswith(".xlsx"):
            df = pd.read_excel(uploaded)
        else:
            raise ValueError("Unsupported file type. Allowed: csv, txt, xlsx")
    except Exception as e:
        raise RuntimeError(f"Failed to parse file: {e}")
    # sanitize mostly-string cells
    try:
        df = df.applymap(sanitize_cell)
    except Exception:
        logger.warning("Sanitization applymap failed; skipping per-cell sanitization.")
    return df

def infer_problem_type(y: pd.Series) -> str:
    """Simple heuristic to decide classification vs regression."""
    if y.dtype == "object" or y.dtype.name == "category":
        return "classification"
    if y.nunique() <= 20:
        return "classification"
    if np.issubdtype(y.dtype, np.number):
        return "regression"
    try:
        _ = pd.to_numeric(y.dropna())
        return "regression"
    except Exception:
        return "classification"

def build_preprocessor(X: pd.DataFrame):
    """Return ColumnTransformer and inferred numeric/categorical lists."""
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                             ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preproc = ColumnTransformer([("num", num_pipeline, num_cols), ("cat", cat_pipeline, cat_cols)], remainder="drop")
    return preproc, num_cols, cat_cols

def sample_large_df(df: pd.DataFrame, max_rows: int = MAX_ROWS_SAMPLE) -> Tuple[pd.DataFrame, bool]:
    if df.shape[0] > max_rows:
        df_sampled = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
        return df_sampled, True
    return df, False

def prepare_estimators(problem: str, compare_models: list):
    """
    Build estimator instances and their RandomizedSearchCV param_distributions.
    Only include models that appear in compare_models.
    """
    estimators = {}
    param_distributions = {}

    if problem == "classification":
        if "RandomForest" in compare_models:
            estimators["rf"] = RandomForestClassifier(random_state=42)
            param_distributions["rf"] = {
                "model__n_estimators": [100, 200, 500],
                "model__max_depth": [None, 10, 30],
                "model__min_samples_split": [2, 5, 10]
            }

        if "HistGB" in compare_models:
            estimators["histgb"] = HistGradientBoostingClassifier(random_state=42)
            param_distributions["histgb"] = {"model__max_iter": [100, 300], "model__max_leaf_nodes": [31, 63]}

        if "ExtraTrees" in compare_models:
            estimators["et"] = ExtraTreesClassifier(n_estimators=200, random_state=42)
            param_distributions["et"] = {"model__n_estimators": [100,200,400]}

        if "KNN" in compare_models:
            estimators["knn"] = KNeighborsClassifier()
            param_distributions["knn"] = {"model__n_neighbors": [3,5,7], "model__weights": ["uniform","distance"]}

        if "SVM" in compare_models:
            estimators["svm"] = SVC(probability=True)
            param_distributions["svm"] = {"model__C": [0.1,1,10], "model__kernel": ["rbf","linear"]}

        if "MLP" in compare_models:
            estimators["mlp"] = MLPClassifier(max_iter=500, early_stopping=True, random_state=42)
            param_distributions["mlp"] = {"model__hidden_layer_sizes": [(64,),(128,64)], "model__alpha":[1e-4,1e-3]}

        if "XGBoost" in compare_models and XGBClassifier is not None:
            estimators["xgb"] = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)
            param_distributions["xgb"] = {
                "model__n_estimators": [100, 300, 600],
                "model__max_depth": [3, 6, 10],
                "model__learning_rate": [0.01, 0.05, 0.1]
            }

        if "LogisticRegression" in compare_models:
            estimators["logreg"] = LogisticRegression(max_iter=2000)
            param_distributions["logreg"] = {"model__C": [0.01, 0.1, 1, 10], "model__penalty": ["l2"]}

    else:  # regression
        if "RandomForest" in compare_models:
            estimators["rf"] = RandomForestRegressor(random_state=42)
            param_distributions["rf"] = {"model__n_estimators": [100, 200], "model__max_depth": [None, 10, 30]}

        if "XGBoost" in compare_models and XGBRegressor is not None:
            estimators["xgb"] = XGBRegressor(random_state=42)
            param_distributions["xgb"] = {"model__n_estimators": [100, 300], "model__max_depth": [3, 6], "model__learning_rate": [0.01, 0.1]}

        if "ExtraTrees" in compare_models:
            estimators["et"] = ExtraTreesRegressor(n_estimators=200, random_state=42)
            param_distributions["et"] = {"model__n_estimators": [100,200,400]}

        if "KNN" in compare_models:
            estimators["knn"] = KNeighborsRegressor()
            param_distributions["knn"] = {"model__n_neighbors": [3,5,7], "model__weights": ["uniform","distance"]}

        if "MLP" in compare_models:
            estimators["mlp"] = MLPRegressor(max_iter=500)
            param_distributions["mlp"] = {"model__hidden_layer_sizes": [(64,),(128,64)], "model__alpha":[1e-4,1e-3]}

    return estimators, param_distributions

# ----- UI: Upload & basic checks -----
uploaded = st.file_uploader("Upload CSV / TXT / XLSX", type=["csv", "txt", "xlsx"])
if not uploaded:
    st.info("Upload a dataset to get started.")
    st.stop()

# quick size guard
try:
    uploaded_bytes = uploaded.getbuffer().nbytes
except Exception:
    uploaded.seek(0, io.SEEK_END)
    uploaded_bytes = uploaded.tell()
    uploaded.seek(0)

if uploaded_bytes > MAX_FILE_BYTES:
    st.error(f"File too large ({uploaded_bytes/1e6:.1f} MB). Max allowed is {MAX_FILE_BYTES/1e6:.1f} MB.")
    st.stop()

# safe read
try:
    df = safe_read_file(uploaded)
except RuntimeError as e:
    st.error(str(e))
    st.stop()

st.write(f"Dataset shape: {df.shape}")
st.dataframe(df.head(200))

# allow dropping columns
drop_cols = st.multiselect("Select columns to drop (optional)", options=df.columns.tolist())
if drop_cols:
    df = df.drop(columns=drop_cols)
    st.success(f"Dropped columns: {drop_cols}")

# sample large datasets
df, was_sampled = sample_large_df(df, MAX_ROWS_SAMPLE)
if was_sampled:
    st.warning(f"Dataset sampled to {MAX_ROWS_SAMPLE} rows for safe public training. Use full data offline for best results.")

# store df in session_state (so evaluation and downloads can access it)
st.session_state["df"] = df.copy()

# Target selection and basic validation
target = st.selectbox("Select target column", options=df.columns.tolist())
if target is None:
    st.error("Please select a target column.")
    st.stop()

# store target in session_state
st.session_state["target"] = target

y = df[target]
if y.isnull().all():
    st.error("Selected target column is all null. Pick a different target.")
    st.stop()

problem = infer_problem_type(y)
st.info(f"Problem inferred: **{problem}** — you can override if needed.")
# Optionally allow override
problem_override = st.radio("Problem type", ("auto", "classification", "regression"), index=0)
if problem_override != "auto":
    problem = problem_override

# ----- Model / CV / UI options -----
use_cv = st.checkbox("Use Cross-Validation + Hyperparam tuning (RandomizedSearchCV)", value=True)
use_smote = st.checkbox("Use SMOTE for class imbalance (classification only)", value=False and (ImbPipeline is not None))
n_iter = st.number_input("Random search iterations (capped)", min_value=10, max_value=500, value=DEFAULT_N_ITER)
cv_folds = st.number_input("CV folds (capped)", min_value=2, max_value=10, value=DEFAULT_CV_FOLDS)
compare_models = st.multiselect("Models to try", options=["RandomForest", "XGBoost","MLP", "LogisticRegression","KNN","SVM","HistGB","ExtraTrees"], default=["RandomForest", "XGBoost"])

# enforce caps
n_iter = int(min(n_iter, MAX_N_ITER))
cv_folds = int(min(cv_folds, MAX_CV_FOLDS))

# Prepare X,y
X = df.drop(columns=[target]).copy()
y = df[target].copy()

# Basic checks
if X.shape[0] < 10:
    st.warning("Very few rows (<10). Model results may be unreliable.")

# infer columns & preproc
preproc, num_cols, cat_cols = build_preprocessor(X)
st.write(f"Numeric columns: {num_cols}")
st.write(f"Categorical columns: {cat_cols}")

# Prepare estimators and param grids
estimators, param_distributions = prepare_estimators(problem, compare_models)

# CV strategy
cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42) if problem == "classification" else KFold(n_splits=cv_folds, shuffle=True, random_state=42)

# ----- Training loop (safe) -----
# Reset best_models placeholder for this run (but keep session_state until training succeeds)
current_run_models: Dict[str, Any] = {}

if st.button("Train selected models"):
    if len(estimators) == 0:
        st.error("No models selected or required libraries missing (e.g., xgboost).")
    else:
        with st.spinner("Training... this may take a while (limited by server caps)."):
            try:
                for key, base_model in estimators.items():
                    st.write(f"→ Candidate: {key}")
                    # build pipeline
                    if use_smote and problem == "classification" and ImbPipeline is not None:
                        pipe = ImbPipeline([("preproc", preproc), ("smote", SMOTE()), ("model", base_model)])
                    else:
                        pipe = Pipeline([("preproc", preproc), ("model", base_model)])

                    if use_cv:
                        search = RandomizedSearchCV(pipe,
                                                    param_distributions=param_distributions.get(key, {}),
                                                    n_iter=n_iter,
                                                    cv=cv_strategy,
                                                    n_jobs=1,   # conservative for public hosts
                                                    scoring=None,
                                                    verbose=0,
                                                    random_state=42,
                                                    refit=True)
                        try:
                            search.fit(X, y)
                            best = search.best_estimator_
                            current_run_models[key] = best
                            st.write(f"Best params for {key}: {search.best_params_}")
                        except MemoryError:
                            st.error(f"MemoryError while training {key}. Try reducing dataset size or model complexity.")
                        except Exception as e:
                            st.error(f"Training {key} failed: {e}")
                    else:
                        try:
                            pipe.fit(X, y)
                            current_run_models[key] = pipe
                            st.write(f"Trained {key} without CV")
                        except Exception as e:
                            st.error(f"Training {key} failed: {e}")

                # persist trained models into session_state (merge with any existing)
                if current_run_models:
                    # overwrite previous run models to avoid stale mixes
                    st.session_state["best_models"] = current_run_models
                    st.success("Training complete. Models saved to session_state.")
                else:
                    st.warning("No models trained successfully in this run.")
            except Exception as e:
                st.error(f"Unexpected error during training: {e}")
                logger.exception(e)

# quick inspect models in session state
if st.session_state.get("best_models"):
    st.write("Models available in session:", list(st.session_state["best_models"].keys()))
else:
    st.write("No trained models in session yet. Train models to enable evaluation & download.")

# ----- Evaluation & download (reads from session_state) -----
models_in_state = st.session_state.get("best_models", {})

if not models_in_state:
    st.info("No trained models available for evaluation. Train models first.")
else:
    if st.button("Evaluate models on holdout 20% split"):
        try:
            df_local = st.session_state.get("df")
            target_local = st.session_state.get("target")
            if df_local is None or target_local is None:
                st.error("Missing dataset or target in session state.")
            else:
                X_eval = df_local.drop(columns=[target_local]).copy()
                y_eval = df_local[target_local].copy()
                eval_problem = infer_problem_type(y_eval)
                X_train, X_test, y_train, y_test = train_test_split(
                    X_eval, y_eval, test_size=0.2, random_state=42,
                    stratify=y_eval if eval_problem == "classification" else None
                )
                for key, model in models_in_state.items():
                    st.subheader(f"Model: {key}")
                    try:
                        preds = model.predict(X_test)
                        if eval_problem == "classification":
                            st.text(classification_report(y_test, preds))
                        else:
                            MAE = mean_absolute_error(y_test, preds)
                            r2 = model.score(X_test, y_test)
                            st.write(key, {"MAE": float(MAE), "r2": float(r2)})
                    except Exception as e:
                        st.error(f"Evaluation failed for {key}: {e}")
        except Exception as e:
            st.error(f"Evaluation error: {e}")
            logger.exception(e)

# Allow user to pick a model to download (from session_state)
if models_in_state:
    chosen_key = st.selectbox("Choose a model to download", options=list(models_in_state.keys()))
    if chosen_key:
        chosen_model = models_in_state[chosen_key]
        buf = io.BytesIO()
        try:
            joblib.dump(chosen_model, buf)
            buf.seek(0)
            st.download_button("Download model (joblib)", data=buf, file_name=f"{chosen_key}_model.joblib")
            # metadata
            meta = {
                "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                "n_rows": int(st.session_state["df"].shape[0]) if st.session_state.get("df") is not None else None,
                "n_features": int(st.session_state["df"].shape[1] - 1) if st.session_state.get("df") is not None and st.session_state.get("target") else None,
                "target": str(st.session_state.get("target")),
                "problem": problem,
                "model_key": chosen_key
            }
            st.download_button("Download model metadata (json)", data=json.dumps(meta, indent=2), file_name=f"{chosen_key}_metadata.json")
        except Exception as e:
            st.error(f"Failed to serialize model: {e}")

st.write("---")
st.caption("Notes: Models are stored in Streamlit session state for the current browser session. For production, persist models to a database or object store and move heavy training into background workers.")
