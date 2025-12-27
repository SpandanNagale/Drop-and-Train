import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import numpy as np

class Visualizer:
    
    @staticmethod
    def plot_feature_importance(model_pipeline, feature_names):
        model = model_pipeline.named_steps['model']
        importances = None
        
        # 1. Tree-based importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            
        # 2. Linear coefficient importance
        elif hasattr(model, 'coef_'):
            # Handle multi-class logistic regression (coef_ is 2D)
            if model.coef_.ndim > 1:
                importances = np.mean(np.abs(model.coef_), axis=0)
            else:
                importances = np.abs(model.coef_)
                
        if importances is not None:
            if len(feature_names) == len(importances):
                feature_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values(by='Importance', ascending=False).head(10)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=feature_df, ax=ax, palette='viridis')
                ax.set_title("Top 10 Feature Importances")
                st.pyplot(fig)
            else:
                st.warning("Could not align feature names. (Shape mismatch)")
        else:
            st.info(f"The algorithm '{type(model).__name__}' does not provide intrinsic feature importance.")

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, labels):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels)
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)

    @staticmethod
    def plot_actual_vs_predicted(y_true, y_pred):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_true, y_pred, alpha=0.5)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('Actual vs Predicted Values')
        st.pyplot(fig)