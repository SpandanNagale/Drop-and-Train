import pandas as pd
import streamlit as st

class DataLoader:
    @staticmethod
    @st.cache_data # <--- This is the magic line. It caches the result.
    def load_data(uploaded_file):
        """
        Loads data from CSV or Excel. 
        Cached to prevent reloading on every interaction.
        """
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(uploaded_file)
            else:
                return None
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None