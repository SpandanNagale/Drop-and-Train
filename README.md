# 🧠 Drop & Train  
*A lightweight AutoML-style Streamlit app for quick model building on your own data.*

---

## 🚀 Overview
**Drop & Train** lets you upload a dataset, select your target variable, choose machine learning algorithms, and instantly train and evaluate models — all from your browser.

No setup, no coding — just **drop your data, pick your target, and train**.

The app automatically:
- Handles data preprocessing (missing values, scaling, encoding)
- Supports both classification and regression tasks
- Performs optional hyperparameter tuning with cross-validation
- Handles imbalanced datasets using SMOTE (optional)
- Evaluates models on a holdout set
- Lets you download trained models and metadata

---

## 🧩 Features

| Category | Description |
|-----------|--------------|
| **Upload** | Supports `.csv`, `.txt`, and `.xlsx` files (up to 8 MB by default) |
| **Preprocessing** | Automatic type inference, missing-value imputation, standard scaling, one-hot encoding |
| **Feature Control** | Option to drop unwanted columns before training |
| **Problem Type Detection** | Auto-detects classification vs. regression |
| **Algorithms** | Random Forest, XGBoost, Logistic Regression (extendable) |
| **Advanced Options** | Cross-validation, RandomizedSearchCV, SMOTE for imbalance |
| **Evaluation** | Classification report or regression metrics (MAE, R²) |
| **Model Export** | Download trained model (`.joblib`) + metadata (`.json`) |
| **Safety** | Input sanitization, dataset sampling, resource caps for public deployment |

---

## 🛠️ Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io)
- **Backend ML:** scikit-learn, XGBoost, imbalanced-learn
- **Languages:** Python 3.9+
- **Environment:** Works locally or hosted on Streamlit Cloud / Render / Docker

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/SpandanNagale/Drop-and-Train.git
cd drop-and-train
