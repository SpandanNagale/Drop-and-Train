from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.pipeline import Pipeline

class ModelTrainer:
    def __init__(self, problem_type):
        self.problem_type = problem_type # 'Classification' or 'Regression'

    def get_model(self, algorithm_name, params=None):
        if params is None:
            params = {}
            
        # --- CLASSIFICATION MODELS ---
        if self.problem_type == 'Classification':
            if algorithm_name == 'Random Forest':
                return RandomForestClassifier(**params)
            elif algorithm_name == 'XGBoost':
                return XGBClassifier(**params)
            elif algorithm_name == 'Logistic Regression':
                return LogisticRegression(**params)
            elif algorithm_name == 'Support Vector Machine (SVM)':
                return SVC(probability=True, **params) # probability=True needed for ROC/AUC later
            elif algorithm_name == 'K-Nearest Neighbors (KNN)':
                return KNeighborsClassifier(**params)
            elif algorithm_name == 'Decision Tree':
                return DecisionTreeClassifier(**params)
            elif algorithm_name == 'Gradient Boosting':
                return GradientBoostingClassifier(**params)
        
        # --- REGRESSION MODELS ---
        elif self.problem_type == 'Regression':
            if algorithm_name == 'Random Forest':
                return RandomForestRegressor(**params)
            elif algorithm_name == 'XGBoost':
                return XGBRegressor(**params)
            elif algorithm_name == 'Linear Regression':
                return LinearRegression(**params)
            elif algorithm_name == 'Support Vector Machine (SVR)':
                return SVR(**params)
            elif algorithm_name == 'K-Nearest Neighbors (KNN)':
                return KNeighborsRegressor(**params)
            elif algorithm_name == 'Decision Tree':
                return DecisionTreeRegressor(**params)
            elif algorithm_name == 'Gradient Boosting':
                return GradientBoostingRegressor(**params)
        
        return None

    def train(self, X_train, y_train, preprocessor, algorithm_name, params=None):
        """
        Now accepts 'params' dictionary to configure the model.
        """
        model = self.get_model(algorithm_name, params)
        
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        full_pipeline.fit(X_train, y_train)
        return full_pipeline
        
        full_pipeline.fit(X_train, y_train)
        return full_pipeline