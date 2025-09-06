import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
import joblib
from pathlib import Path

class ModelExplainer:
    def __init__(self, model, X_train, feature_names=None):
        self.model = model
        self.feature_names = feature_names
        
        # Extract the underlying model if it's a ModelWrapper
        self.unwrapped_model = model.model if hasattr(model, 'model') else model
        
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train_values = X_train.values
        else:
            X_train_values = X_train
            if feature_names is None:
                raise ValueError("feature_names must be provided when X_train is not a pandas DataFrame")
        
        # Sample data for KernelExplainer if needed
        if X_train_values.shape[0] > 100:
            # Use a smaller background dataset for faster computation
            background_indices = np.random.choice(X_train_values.shape[0], size=100, replace=False)
            background_data = X_train_values[background_indices]
        else:
            background_data = X_train_values
        
        # Create the appropriate explainer based on model type
        try:
            # Try TreeExplainer first for tree-based models
            if hasattr(self.unwrapped_model, 'estimators_') or hasattr(model, 'estimators_'):
                # RandomForest or other tree ensemble
                self.shap_explainer = shap.TreeExplainer(self.unwrapped_model)
            elif hasattr(self.unwrapped_model, 'get_booster') or hasattr(model, 'get_booster'):
                # XGBoost or LightGBM
                self.shap_explainer = shap.TreeExplainer(self.unwrapped_model)
            elif hasattr(model, 'predict_proba'):
                # For other models with probability output
                self.shap_explainer = shap.KernelExplainer(
                    lambda x: model.predict_proba(x)[:,1], 
                    background_data
                )
            else:
                # For regression models
                self.shap_explainer = shap.KernelExplainer(
                    model.predict, 
                    background_data
                )
                
        except Exception as e:
            print(f"Could not create TreeExplainer: {e}. Falling back to KernelExplainer")
            if hasattr(model, 'predict_proba'):
                # For classification models
                self.shap_explainer = shap.KernelExplainer(
                    lambda x: model.predict_proba(x)[:,1], 
                    background_data
                )
            else:
                # For regression models
                self.shap_explainer = shap.KernelExplainer(
                    model.predict, 
                    background_data
                )
        
        # Create explainer type attribute for reference
        self.explainer_type = self.shap_explainer.__class__.__name__
        print(f"Created {self.explainer_type} for model")
            
        # Cache for SHAP values
        self._global_shap_values = None
        self._global_shap_data = None

    def get_global_explanations(self, X: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        # If X is provided or we don't have cached values, compute SHAP values
        if X is not None or self._global_shap_values is None:
            if X is not None:
                shap_values = self.explainer.shap_values(X)
                self._global_shap_data = X
            else:
                raise ValueError("X must be provided for the first call to get_global_explanations")
                
            # For classifier models, SHAP returns a list of arrays (one per class)
            if isinstance(shap_values, list) and len(shap_values) > 1:
                shap_values = shap_values[1]  # Get SHAP values for the positive class
            
            self._global_shap_values = shap_values
        
        # Calculate mean absolute SHAP value for each feature
        feature_importance = {}
        for i, name in enumerate(self.feature_names):
            feature_importance[name] = np.abs(self._global_shap_values[:, i]).mean()
        
        # Sort by importance
        sorted_features = {k: v for k, v in sorted(
            feature_importance.items(), key=lambda item: item[1], reverse=True
        )}
        
        return sorted_features
    
    def get_local_explanation(self, X_row: pd.DataFrame) -> Dict[str, float]:
        # Ensure X_row is properly shaped for SHAP
        if isinstance(X_row, pd.DataFrame) or isinstance(X_row, pd.Series):
            if len(X_row) == 1:
                shap_values = self.explainer.shap_values(X_row)
            else:
                # If more than one row is passed, use just the first row
                shap_values = self.explainer.shap_values(X_row.iloc[[0]])
        else:
            # Handle numpy arrays
            X_array = X_row.reshape(1, -1) if X_row.ndim == 1 else X_row
            shap_values = self.explainer.shap_values(X_array)

        # For classifier models, SHAP returns a list of arrays (one per class)
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_values = shap_values[1]  # Get SHAP values for the positive class

        # Ensure we're dealing with a single prediction's SHAP values
        if len(shap_values.shape) > 1 and shap_values.shape[0] > 0:
            shap_values = shap_values[0]
        
        # Convert to dictionary
        if isinstance(X_row, pd.DataFrame):
            feature_names = X_row.columns.tolist()
        else:
            feature_names = self.feature_names
        
        # Convert numpy values to Python scalars to avoid array comparison issues
        result = {}
        for i, val in enumerate(shap_values):
            if i < len(feature_names):
                # Handle potentially nested arrays by extracting scalar values
                try:
                    if hasattr(val, 'shape') and len(val.shape) > 0:
                        # If val is an array with shape, take the first element
                        scalar_val = val.item() if val.size == 1 else val[0]
                    else:
                        scalar_val = val
                    result[feature_names[i]] = float(scalar_val)
                except (ValueError, TypeError):
                    # Fallback for complex values - just use 0.0
                    result[feature_names[i]] = 0.0
        
        return result

    def get_top_factors(self, X_row: pd.DataFrame, top_n: int = 5) -> Dict[str, Dict[str, Any]]:
        local_exp = self.get_local_explanation(X_row)
        top_factors = sorted(
            local_exp.items(), 
            key=lambda x: abs(float(x[1])), 
            reverse=True
        )[:top_n]
        
        results = {}
        for feature, shap_value in top_factors:
            # Ensure shap_value is a scalar
            shap_value = float(shap_value)
            direction = "increases" if shap_value > 0 else "decreases"
            impact = "high" if abs(shap_value) > 0.5 else "medium" if abs(shap_value) > 0.2 else "low"
            
            # Get feature value
            if isinstance(X_row, pd.DataFrame):
                feature_value = float(X_row[feature].iloc[0]) if len(X_row) > 0 else 0.0
            elif isinstance(X_row, pd.Series):
                feature_value = float(X_row[feature]) if feature in X_row else 0.0
            else:
                # For numpy arrays
                feature_idx = self.feature_names.index(feature)
                feature_value = float(X_row[0, feature_idx]) if X_row.ndim > 1 else float(X_row[feature_idx])
                
            results[feature] = {
                "shap_value": shap_value,
                "feature_value": feature_value,
                "impact": impact,
                "interpretation": f"{feature} = {feature_value:.2f} {direction} risk"
            }
            
        return results

    def plot_shap_summary(self, X: Optional[pd.DataFrame] = None, save_path: Optional[str] = None):
        if X is not None:
            shap_values = self.explainer.shap_values(X)
            plot_data = X
            
            # For classifier models
            if isinstance(shap_values, list) and len(shap_values) > 1:
                shap_values = shap_values[1]  # Get SHAP values for positive class
        elif self._global_shap_values is not None and self._global_shap_data is not None:
            shap_values = self._global_shap_values
            plot_data = self._global_shap_data
        else:
            raise ValueError("X must be provided when no cached values are available")
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, plot_data, feature_names=self.feature_names)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_shap_dependence(self, feature: str, X: Optional[pd.DataFrame] = None, 
                             interaction_feature: Optional[str] = None, save_path: Optional[str] = None):
        if X is not None:
            shap_values = self.explainer.shap_values(X)
            plot_data = X
            
            # For classifier models
            if isinstance(shap_values, list) and len(shap_values) > 1:
                shap_values = shap_values[1]  # Get SHAP values for positive class
        elif self._global_shap_values is not None and self._global_shap_data is not None:
            shap_values = self._global_shap_values
            plot_data = self._global_shap_data
        else:
            raise ValueError("X must be provided when no cached values are available")
        
        plt.figure(figsize=(10, 6))
        
        if isinstance(plot_data, pd.DataFrame):
            if interaction_feature is None:
                shap.dependence_plot(feature, shap_values, plot_data)
            else:
                shap.dependence_plot(feature, shap_values, plot_data, interaction_index=interaction_feature)
        else:
            feature_idx = self.feature_names.index(feature)
            interaction_idx = self.feature_names.index(interaction_feature) if interaction_feature else None
            shap.dependence_plot(feature_idx, shap_values, plot_data, interaction_index=interaction_idx)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def generate_textual_explanation(self, X_row: pd.DataFrame, prediction_prob: float, 
                                     top_n: int = 3) -> Dict[str, Any]:
        top_factors = self.get_top_factors(X_row, top_n=top_n)
        
        # Generate risk level text
        if prediction_prob >= 0.7:
            risk_level = "HIGH"
            risk_text = "The patient is at high risk of deterioration in the next 90 days."
        elif prediction_prob >= 0.4:
            risk_level = "MODERATE"
            risk_text = "The patient is at moderate risk of deterioration in the next 90 days."
        else:
            risk_level = "LOW"
            risk_text = "The patient is at low risk of deterioration in the next 90 days."
        
        # Generate factors text
        factor_texts = []
        recommendations = []
        
        for feature, info in top_factors.items():
            shap_value = info["shap_value"]
            feature_value = info["feature_value"]
            
            # Generate more specific explanations based on feature
            if feature == "med_adherence":
                if shap_value < 0:
                    factor_texts.append(f"Good medication adherence ({feature_value:.0%}) is helping reduce risk.")
                else:
                    factor_texts.append(f"Poor medication adherence ({feature_value:.0%}) is increasing risk.")
                    recommendations.append("Improve medication adherence through reminders and education.")
            
            elif feature == "glucose":
                if shap_value > 0:
                    factor_texts.append(f"High glucose level ({feature_value:.1f}) is a significant risk factor.")
                    recommendations.append("Monitor glucose more frequently and adjust treatment as needed.")
                else:
                    factor_texts.append(f"Well-controlled glucose level ({feature_value:.1f}) is helping reduce risk.")
            
            elif feature == "hba1c":
                if shap_value > 0:
                    factor_texts.append(f"Elevated HbA1c ({feature_value:.1f}%) indicates poor long-term glucose control.")
                    recommendations.append("Review and possibly adjust diabetes management plan.")
                else:
                    factor_texts.append(f"HbA1c level ({feature_value:.1f}%) is helping reduce risk.")
            
            elif feature == "systolic_bp" or feature == "diastolic_bp":
                bp_type = "Systolic" if feature == "systolic_bp" else "Diastolic"
                if shap_value > 0:
                    factor_texts.append(f"Elevated {bp_type} blood pressure ({feature_value:.0f}) is increasing risk.")
                    recommendations.append("Review blood pressure management and medication.")
                else:
                    factor_texts.append(f"{bp_type} blood pressure ({feature_value:.0f}) is under control.")
            
            elif feature == "steps":
                if shap_value < 0:
                    factor_texts.append(f"Good physical activity ({feature_value:.0f} steps/day) is helping reduce risk.")
                else:
                    factor_texts.append(f"Low physical activity ({feature_value:.0f} steps/day) is increasing risk.")
                    recommendations.append("Encourage increased physical activity with specific goals.")
            
            elif feature == "sleep_hours":
                if shap_value < 0:
                    factor_texts.append(f"Healthy sleep pattern ({feature_value:.1f} hours) is helping reduce risk.")
                else:
                    factor_texts.append(f"Poor sleep pattern ({feature_value:.1f} hours) is increasing risk.")
                    recommendations.append("Improve sleep hygiene and establish a regular sleep schedule.")
            
            else:
                # Generic text for other features
                direction = "increasing" if shap_value > 0 else "decreasing"
                factor_texts.append(f"{feature.replace('_', ' ').title()} ({feature_value:.2f}) is {direction} risk.")
        
        # Make recommendations unique
        recommendations = list(set(recommendations))
        
        return {
            "risk_level": risk_level,
            "risk_probability": prediction_prob,
            "explanation_text": risk_text + " " + " ".join(factor_texts),
            "key_factors": factor_texts,
            "recommendations": recommendations,
            "detailed_factors": top_factors
        }
    
    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
    
    @staticmethod
    def load(path: str) -> 'ModelExplainer':
        return joblib.load(path)

def get_global_explanations(model, X):
    """Legacy function for backward compatibility"""
    explainer = ModelExplainer(model, X)
    return explainer.get_global_explanations(X)

def get_local_explanation(model, X_row):
    """Legacy function for backward compatibility"""
    explainer = ModelExplainer(model, X_row)
    return explainer.get_local_explanation(X_row)
