# 🧠 Drop & Train: Professional Edition

> A robust, modular AutoML application built with Streamlit and Scikit-Learn.
> **Zero-code machine learning with production-grade pipelines.**

---

## 🚀 Overview

**Drop & Train** is a full-stack data science tool that allows users to upload raw data, configure machine learning experiments, and export deployment-ready models. 

Unlike basic scripts, this project is built on a **modular architecture** that prioritizes:
1.  **Data Leakage Prevention:** Strict separation of training and testing data before preprocessing.
2.  **Pipeline Portability:** Models are exported as full `sklearn.pipeline.Pipeline` objects (Preprocessor + Model bundled).
3.  **Explainability:** Integrated feature importance and performance visualization.

---

## ⚡ Key Features

* **Modular Architecture:** Codebase separated into `data_loader`, `preprocessor`, `trainer`, and `visualizer` for maintainability.
* **Smart Preprocessing:** Automatic handling of missing values (Imputation) and scaling/encoding based on column types.
* **7+ Algorithms:** * *Classification:* Random Forest, XGBoost, Logistic Regression, SVM, KNN, Decision Trees, Gradient Boosting.
    * *Regression:* Random Forest, XGBoost, Linear Regression, SVR, KNN, Decision Trees, Gradient Boosting.
* **Hyperparameter Tuning:** Dynamic sidebar controls to tweak model specific parameters (e.g., `n_estimators`, `C`, `kernel`).
* **Interactive Visualizations:**
    * Confusion Matrices & Classification Reports.
    * Actual vs. Predicted Scatter Plots (Regression).
    * Feature Importance Rankings (for supported models).
* **Data Control:** Capability to drop specific columns directly from the UI.
* **Model Export:** Download the fully trained pipeline as a `.joblib` file.

---

## 📂 Project Structure

```text
Drop-and-Train/
│
├── src/
│   ├── __init__.py          # Package initialization
│   ├── data_loader.py       # Cached data loading logic
│   ├── preprocessor.py      # Pipeline construction (Imputation/Encoding)
│   ├── trainer.py           # Model initialization and training loop
│   └── visualizer.py        # Plotting logic (SHAP, Confusion Matrix, etc.)
│
├── app.py                   # Main Streamlit application entry point
├── requirements.txt         # Project dependencies
└── README.md                # Documentation

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/SpandanNagale/Drop-and-Train.git
cd drop-and-train
pip install -r requirements.txt
streamlit run app.py