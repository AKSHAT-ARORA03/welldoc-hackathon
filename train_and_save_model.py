import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from data_schema import load_chronic_data, preprocess_chronic_data, load_patient_data, preprocess_data, generate_synthetic_data
from model import train_model, save_model, evaluate_model
from explainability import ModelExplainer

# Configuration
MODEL_PATH = "models/risk_model.pkl"
EXPLAINER_PATH = "models/explainer.pkl"
DATA_PATH = "data/patients.csv"
CHRONIC_DATA_PATH = "data/synthetic_chronic_dataset.csv"

def main():
    # Create directories if they don't exist
    Path(os.path.dirname(MODEL_PATH)).mkdir(parents=True, exist_ok=True)
    
    # Check if chronic dataset exists and use it
    if os.path.exists(CHRONIC_DATA_PATH):
        print(f"Loading chronic dataset from {CHRONIC_DATA_PATH}")
        df = load_chronic_data(CHRONIC_DATA_PATH)
        df = preprocess_chronic_data(df)
        target_col = 'deterioration_90d'  # Renamed during preprocessing
        id_cols = ['patient_id', 'date']  # Include date as an ID column to exclude from features
    # Fall back to original data loading logic
    elif os.path.exists(DATA_PATH):
        print(f"Loading original dataset from {DATA_PATH}")
        df = load_patient_data(DATA_PATH)
        df = preprocess_data(df)
        target_col = 'deterioration_90d'
        id_cols = ['patient_id', 'date']
    else:
        print(f"No data found. Generating synthetic data at {DATA_PATH}")
        Path(os.path.dirname(DATA_PATH)).mkdir(parents=True, exist_ok=True)
        df = generate_synthetic_data(100, 180)  # 100 patients, 180 days
        df.to_csv(DATA_PATH, index=False)
        df = preprocess_data(df)
        target_col = 'deterioration_90d'
        id_cols = ['patient_id', 'date']
    
    # Print dataset info
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Target distribution: {df[target_col].value_counts()}")
    
    # Filter columns that are in the dataset
    valid_id_cols = [col for col in id_cols if col in df.columns]
    drop_cols = valid_id_cols + [target_col]
    
    # Drop ID columns and target from features
    X = df.drop(columns=drop_cols)
    y = df[target_col]
    
    # Print feature columns to verify
    print(f"Feature columns for training: {X.columns.tolist()}")
    
    # Verify that there are no string columns in the features
    for col in X.columns:
        dtype = X[col].dtype
        print(f"Column {col} dtype: {dtype}")
        
        # If column is object or string type, try to convert to numeric or drop it
        if dtype == 'object' or pd.api.types.is_string_dtype(dtype):
            print(f"Converting string column {col} to numeric...")
            try:
                X[col] = pd.to_numeric(X[col])
                print(f"Successfully converted {col} to numeric")
            except (ValueError, TypeError):
                print(f"Cannot convert {col} to numeric, dropping this column")
                X = X.drop(columns=[col])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Try multiple model types to find the best one
    model_types = ['xgboost', 'ensemble']
    best_model = None
    best_score = 0
    best_metrics = None
    
    for model_type in model_types:
        print(f"\n===== Training {model_type} model =====")
        # Train model
        model = train_model(X_train, y_train, model_type=model_type)
        
        # Evaluate model
        print("\nEvaluating model...")
        metrics = evaluate_model(model, X_test, y_test)
        print(f"Model AUROC: {metrics['AUROC']:.3f}")
        print(f"Model AUPRC: {metrics['AUPRC']:.3f}")
        print(f"Model F1 Score: {metrics['F1 Score']:.3f}")
        print(f"Model Accuracy: {metrics['Accuracy']:.3f}")
        
        # Save the best model
        if metrics['AUROC'] > best_score:
            best_score = metrics['AUROC']
            best_model = model
            best_metrics = metrics
            print(f"New best model: {model_type} with AUROC {best_score:.3f}")
    
    # Save best model
    print(f"\nSaving best model to {MODEL_PATH} with metrics:")
    for metric, value in best_metrics.items():
        print(f"- {metric}: {value:.3f}")
    save_model(best_model, MODEL_PATH, best_metrics)
    
    # Create and save explainer with more robust error handling
    print(f"Creating explainer and saving to {EXPLAINER_PATH}")
    try:
        # Extract features without engineering for better explainability
        X_sample = X_train.copy()
        
        # Create explainer
        explainer = ModelExplainer(best_model, X_sample)
        explainer.save(EXPLAINER_PATH)
        print("Explainer created and saved successfully.")
        
    except Exception as e:
        print(f"Error creating explainer: {e}")
        print("Creating alternative explainer...")
        
        try:
            # Try to use the unwrapped model directly if it's a wrapper
            if hasattr(best_model, 'model'):
                # It's a ModelWrapper - use the underlying model
                unwrapped_model = best_model.model
                explainer = ModelExplainer(unwrapped_model, X_sample)
                print("Created explainer using unwrapped model.")
            else:
                # Fall back to a simpler explainer approach
                from explainability import get_global_explanations
                global_importance = get_global_explanations(best_model, X_sample)
                print(f"Created simpler explainer. Top features: {list(global_importance.keys())[:5]}")
                
            # Save the explainer
            explainer.save(EXPLAINER_PATH)
            print("Alternative explainer saved successfully.")
            
        except Exception as inner_error:
            print(f"Failed to create alternative explainer: {inner_error}")
            print("Model will be available but explanations may be limited.")
    
    print("Done! You can now run the dashboard application.")

if __name__ == "__main__":
    main()