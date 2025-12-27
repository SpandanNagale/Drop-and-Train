import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class PipelineBuilder:
    def __init__(self, X_train):
        """
        X_train is required to automatically detect numeric vs categorical columns.
        This ensures we don't apply the wrong transformations.
        """
        self.numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
        self.categorical_features = X_train.select_dtypes(include=['object', 'category']).columns

    def build_preprocessor(self, impute_strategy='mean'):
        """
        Returns a ColumnTransformer that handles:
        1. Numeric: Imputation -> Scaling
        2. Categorical: Imputation -> OneHotEncoding
        """
        
        # Numeric Pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=impute_strategy)),
            ('scaler', StandardScaler())
        ])

        # Categorical Pipeline
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Combine them
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            verbose_feature_names_out=False # Keeps column names clean
        )
        
        return preprocessor