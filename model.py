import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

def feature_engineering(X):
    """
    Perform feature engineering to improve model performance
    
    Args:
        X: Features DataFrame
    
    Returns:
        Enhanced DataFrame with engineered features
    """
    X_new = X.copy()
    
    # Create interaction features for important medical metrics
    if 'systolic_bp' in X_new.columns and 'diastolic_bp' in X_new.columns:
        X_new['bp_ratio'] = X_new['systolic_bp'] / X_new['diastolic_bp']
        X_new['pulse_pressure'] = X_new['systolic_bp'] - X_new['diastolic_bp']
    
    # Create BMI category feature
    if 'bmi' in X_new.columns:
        X_new['bmi_category'] = pd.cut(X_new['bmi'], 
                                       bins=[0, 18.5, 25, 30, 35, 100], 
                                       labels=[0, 1, 2, 3, 4])
        X_new['bmi_category'] = X_new['bmi_category'].astype(int)
    
    # Create age groups
    if 'age' in X_new.columns:
        X_new['age_group'] = pd.cut(X_new['age'], 
                                    bins=[0, 40, 60, 80, 120], 
                                    labels=[0, 1, 2, 3])
        X_new['age_group'] = X_new['age_group'].astype(int)
    
    # Create cholesterol ratio feature
    if 'ldl' in X_new.columns and 'hdl' in X_new.columns:
        X_new['cholesterol_ratio'] = X_new['ldl'] / X_new['hdl']
    
    # Create composite risk score
    if all(col in X_new.columns for col in ['age', 'systolic_bp', 'glucose', 'hba1c']):
        # Normalize each component
        age_norm = (X_new['age'] - 40) / 40  # Assuming age range of 40-80
        sbp_norm = (X_new['systolic_bp'] - 120) / 40  # Normal ~120, high ~160
        glucose_norm = (X_new['glucose'] - 100) / 100  # Normal ~100, high ~200
        hba1c_norm = (X_new['hba1c'] - 5.7) / 4  # Normal ~5.7, high ~9.7
        
        # Create composite score (weight based on domain knowledge)
        X_new['composite_risk'] = (
            0.3 * age_norm + 
            0.25 * sbp_norm + 
            0.25 * glucose_norm + 
            0.2 * hba1c_norm
        )
    
    # Handle any NaN values created during feature engineering
    X_new = X_new.fillna(X_new.median())
    
    return X_new

def select_important_features(X, y, threshold=0.01):
    """
    Select important features using a base model
    
    Args:
        X: Features DataFrame
        y: Target variable
        threshold: Importance threshold for feature selection
    
    Returns:
        DataFrame with selected features
    """
    # Create and fit a simple model to find important features
    selector_model = RandomForestClassifier(n_estimators=100, random_state=42)
    selector_model.fit(X, y)
    
    # Create a selector based on feature importance
    selector = SelectFromModel(selector_model, threshold=threshold, prefit=True)
    X_important = selector.transform(X)
    
    # Get the selected feature indices
    selected_indices = selector.get_support(indices=True)
    selected_features = X.columns[selected_indices]
    
    print(f"Selected {len(selected_features)} features out of {X.shape[1]}")
    print(f"Selected features: {', '.join(selected_features)}")
    
    return X[selected_features]

def train_model(X, y, model_type='xgboost'):
    """
    Train a model on the given data with advanced techniques
    
    Args:
        X: Features DataFrame
        y: Target variable
        model_type: Type of model to train ('rf'=Random Forest, 'gb'=Gradient Boosting, 
                    'xgboost'=XGBoost, 'lightgbm'=LightGBM, 'nn'=Neural Network, 'ensemble'=Ensemble)
    
    Returns:
        Trained model
    """
    # Print training information
    print(f"Training {model_type} model on {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Feature names: {', '.join(X.columns.tolist())}")
    print(f"Target distribution: {pd.Series(y).value_counts().to_dict()}")
    
    # Apply feature engineering
    X_engineered = feature_engineering(X)
    
    # Select important features (optional - uncomment to use)
    # X_engineered = select_important_features(X_engineered, y)
    
    # Handle class imbalance if present
    class_weights = None
    if len(pd.Series(y).value_counts()) > 1:
        class_ratio = pd.Series(y).value_counts()[0] / pd.Series(y).value_counts()[1]
        if class_ratio > 2 or class_ratio < 0.5:
            print(f"Class imbalance detected (ratio: {class_ratio:.1f}). Adjusting class weights.")
            class_weights = 'balanced'
    
    # Create the model based on the specified type
    if model_type == 'rf':
        # Random Forest with grid search
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        base_model = RandomForestClassifier(random_state=42, class_weight=class_weights)
        model = RandomizedSearchCV(base_model, param_grid, n_iter=10, 
                                  cv=5, scoring='roc_auc', n_jobs=-1, random_state=42)
    
    elif model_type == 'gb':
        # Gradient Boosting with hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        }
        base_model = GradientBoostingClassifier(random_state=42)
        model = RandomizedSearchCV(base_model, param_grid, n_iter=10, 
                                  cv=5, scoring='roc_auc', n_jobs=-1, random_state=42)
    
    elif model_type == 'xgboost':
        # XGBoost with hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'gamma': [0, 0.1, 0.2]
        }
        
        if class_weights == 'balanced':
            # Calculate class weight
            pos_weight = (y == 0).sum() / (y == 1).sum()
            base_model = xgb.XGBClassifier(objective='binary:logistic', 
                                          scale_pos_weight=pos_weight,
                                          random_state=42)
        else:
            base_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
            
        model = RandomizedSearchCV(base_model, param_grid, n_iter=10, 
                                  cv=5, scoring='roc_auc', n_jobs=-1, random_state=42)
    
    elif model_type == 'lightgbm':
        # LightGBM with hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7, -1],
            'num_leaves': [31, 63, 127],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        
        if class_weights == 'balanced':
            # Calculate class weight
            class_weight_dict = {0: 1.0, 1: (y == 0).sum() / (y == 1).sum()}
            base_model = lgb.LGBMClassifier(objective='binary',
                                           class_weight=class_weight_dict,
                                           random_state=42)
        else:
            base_model = lgb.LGBMClassifier(objective='binary', random_state=42)
            
        model = RandomizedSearchCV(base_model, param_grid, n_iter=10, 
                                  cv=5, scoring='roc_auc', n_jobs=-1, random_state=42)
    
    elif model_type == 'nn':
        # Neural Network with preprocessing pipeline
        param_grid = {
            'mlpclassifier__hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100)],
            'mlpclassifier__alpha': [0.0001, 0.001, 0.01],
            'mlpclassifier__learning_rate_init': [0.001, 0.01],
            'mlpclassifier__max_iter': [500, 1000]
        }
        
        nn_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('mlpclassifier', MLPClassifier(random_state=42))
        ])
        
        model = GridSearchCV(nn_pipeline, param_grid, cv=5, 
                            scoring='roc_auc', n_jobs=-1)
    
    elif model_type == 'ensemble':
        # Stacking ensemble of multiple models
        # First level models
        rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, class_weight=class_weights)
        gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
        xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
        lgbm_model = lgb.LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
        
        # Final meta-learner
        meta_learner = LogisticRegression(random_state=42)
        
        # Create stacking ensemble
        model = StackingClassifier(
            estimators=[
                ('rf', rf),
                ('gb', gb),
                ('xgb', xgb_model),
                ('lgbm', lgbm_model)
            ],
            final_estimator=meta_learner,
            cv=5,
            n_jobs=-1
        )
    
    else:
        raise ValueError("Unknown model type. Use 'rf', 'gb', 'xgboost', 'lightgbm', 'nn', or 'ensemble'.")
    
    # Train the model
    print(f"Starting model training with cross-validation...")
    model.fit(X_engineered, y)
    
    # If we used hyperparameter tuning, print the best parameters and score
    if hasattr(model, 'best_params_'):
        print(f"Best parameters: {model.best_params_}")
        print(f"Best CV score: {model.best_score_:.4f}")
        
        # Use the best estimator
        best_model = model.best_estimator_
    else:
        best_model = model
    
    # Evaluate with cross-validation
    cv_scores = cross_val_score(best_model, X_engineered, y, cv=5, scoring='roc_auc')
    print(f"Cross-validation ROC AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # If we're using a model with feature_importances_, print top features
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': X_engineered.columns,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 most important features:")
        print(feature_importance.head(10))
    
    # Create a model wrapper that will handle the feature engineering internally
    return ModelWrapper(best_model, X.columns.tolist())

class ModelWrapper:
    """Wrapper class to bundle model with preprocessing steps"""
    def __init__(self, model, original_features):
        self.model = model
        self.original_features = original_features
    
    def predict(self, X):
        # Ensure X has the expected features
        X_processed = X[self.original_features]
        # Apply feature engineering
        X_processed = feature_engineering(X_processed)
        # Make prediction
        return self.model.predict(X_processed)
    
    def predict_proba(self, X):
        # Ensure X has the expected features
        X_processed = X[self.original_features]
        # Apply feature engineering
        X_processed = feature_engineering(X_processed)
        # Make probability prediction
        return self.model.predict_proba(X_processed)
    
    # For compatibility with SHAP explainers
    def get_booster(self):
        """For XGBoost/LightGBM compatibility with SHAP"""
        if hasattr(self.model, 'get_booster'):
            return self.model.get_booster()
        return self.model
    
    @property
    def estimators_(self):
        """For Random Forest compatibility with SHAP"""
        if hasattr(self.model, 'estimators_'):
            return self.model.estimators_
        return None
    
    @property
    def feature_importances_(self):
        """For model inspection compatibility"""
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        return None
    
    @property
    def classes_(self):
        """For classification model compatibility"""
        if hasattr(self.model, 'classes_'):
            return self.model.classes_
        return None
    
    # For stacked/ensemble models
    @property
    def estimators(self):
        if hasattr(self.model, 'estimators'):
            return self.model.estimators
        return None
    
    @property
    def final_estimator(self):
        if hasattr(self.model, 'final_estimator'):
            return self.model.final_estimator
        return None
    
    # Pass through any attributes not found to the underlying model
    def __getattr__(self, name):
        return getattr(self.model, name)

def evaluate_model(model, X_test, y_test):
    """
    Calculate performance metrics for the model
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
    
    Returns:
        Dictionary of performance metrics
    """
    # Generate predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    f1 = f1_score(y_test, y_pred)
    
    # Calculate accuracy
    accuracy = (y_pred == y_test).mean()
    
    return {
        "AUROC": roc_auc,
        "AUPRC": pr_auc,
        "F1 Score": f1,
        "Accuracy": accuracy
    }

def save_model(model, path, metrics=None):
    """
    Save the trained model to disk
    
    Args:
        model: Trained model
        path: Path to save the model
        metrics: Optional performance metrics to save with the model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save model and metrics
    model_data = {
        "model": model,
        "metrics": metrics,
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    joblib.dump(model_data, path)
    print(f"Model saved to {path}")

def load_model(path):
    """
    Load a saved model from disk
    
    Args:
        path: Path to the saved model
        
    Returns:
        The loaded model
    """
    model_data = joblib.load(path)
    
    if isinstance(model_data, dict) and "model" in model_data:
        return model_data["model"]
    else:
        # For backwards compatibility with old saved models
        return model_data
