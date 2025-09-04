import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import random
from datetime import datetime, timedelta

def load_chronic_data(file_path: str) -> pd.DataFrame:
    """
    Load the synthetic chronic disease dataset from CSV
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Pandas DataFrame with the data
    """
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} patient records from {file_path}")
    return df

def preprocess_chronic_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the chronic dataset
    
    Args:
        df: Raw DataFrame loaded from CSV
        
    Returns:
        Preprocessed DataFrame
    """
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Convert categorical features to numerical using one-hot encoding
    categorical_cols = ['Gender', 'Smoking', 'Alcohol', 'Exercise_Freq', 'Medication_Adherence']
    
    for col in categorical_cols:
        if col in df_processed.columns:
            df_processed = pd.get_dummies(df_processed, columns=[col], drop_first=False)
    
    # Create a numeric medication adherence column
    if 'Medication_Adherence' not in df_processed.columns and any(col.startswith('Medication_Adherence_') for col in df_processed.columns):
        # Create med_adherence from one-hot encoded columns
        if 'Medication_Adherence_High' in df_processed.columns:
            df_processed['med_adherence'] = (
                df_processed['Medication_Adherence_High'] * 0.9 + 
                df_processed['Medication_Adherence_Medium'] * 0.7 + 
                df_processed['Medication_Adherence_Low'] * 0.3
            )
        else:
            df_processed['med_adherence'] = 0.7  # Default value if columns not found
    
    # Rename columns to match expected format in the model
    column_mapping = {
        'Systolic_BP': 'systolic_bp',
        'Diastolic_BP': 'diastolic_bp',
        'Heart_Rate': 'heart_rate',
        'Glucose': 'glucose',
        'HbA1c': 'hba1c',
        'BMI': 'bmi',
        'Age': 'age',
        'Cholesterol': 'cholesterol',
        'LDL': 'ldl',
        'HDL': 'hdl',
        'Triglycerides': 'triglycerides',
        'Creatinine': 'creatinine',
        'eGFR': 'egfr',
        'WBC': 'wbc',
        'RBC': 'rbc',
        'Hemoglobin': 'hemoglobin',
        'Platelets': 'platelets',
        'Hospitalization_PastYear': 'hospitalizations'
    }
    
    # Apply column renaming where the columns exist
    for old_name, new_name in column_mapping.items():
        if old_name in df_processed.columns:
            df_processed.rename(columns={old_name: new_name}, inplace=True)
    
    # Add date column (use current date)
    current_date = pd.Timestamp.now().strftime('%Y-%m-%d')
    df_processed['date'] = current_date
    
    # Rename Patient_ID to patient_id for consistency
    if 'Patient_ID' in df_processed.columns:
        df_processed.rename(columns={'Patient_ID': 'patient_id'}, inplace=True)
    
    # Rename Risk_Label to deterioration_90d for consistency with original code
    if 'Risk_Label' in df_processed.columns:
        df_processed.rename(columns={'Risk_Label': 'deterioration_90d'}, inplace=True)
    
    # Handle missing values - avoid the pandas warning by using a copy
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col in df_processed.columns:
            # Use direct assignment instead of inplace method
            median_value = df_processed[col].median()
            df_processed[col] = df_processed[col].fillna(median_value)
    
    # Add steps and sleep hours if they don't exist (synthetic values)
    if 'steps' not in df_processed.columns:
        df_processed['steps'] = np.random.randint(1000, 10000, size=len(df_processed))
    
    if 'sleep_hours' not in df_processed.columns:
        df_processed['sleep_hours'] = np.random.uniform(4, 9, size=len(df_processed))
    
    return df_processed

# Keep the original functions for backward compatibility
def generate_synthetic_data(num_patients=100, num_days=30):
    """Generate synthetic patient data for testing"""
    # ...existing code...
    pass

def load_patient_data(file_path):
    """Load patient data from CSV file"""
    # ...existing code...
    return load_chronic_data(file_path)

def preprocess_data(df):
    """Preprocess patient data"""
    # For backward compatibility, try to determine if this is chronic data
    if 'Risk_Label' in df.columns or 'Patient_ID' in df.columns:
        return preprocess_chronic_data(df)
    # ...existing code...
    pass

def get_feature_definitions():
    """Get descriptions of each feature for display in the dashboard"""
    return {
        'age': 'Age (years)',
        'gender': 'Gender',
        'bmi': 'Body Mass Index (kg/m²)',
        'systolic_bp': 'Systolic Blood Pressure (mmHg)',
        'diastolic_bp': 'Diastolic Blood Pressure (mmHg)',
        'heart_rate': 'Heart Rate (bpm)',
        'glucose': 'Blood Glucose (mg/dL)',
        'hba1c': 'HbA1c (%)',
        'cholesterol': 'Total Cholesterol (mg/dL)',
        'ldl': 'LDL Cholesterol (mg/dL)',
        'hdl': 'HDL Cholesterol (mg/dL)',
        'triglycerides': 'Triglycerides (mg/dL)',
        'creatinine': 'Creatinine (mg/dL)',
        'egfr': 'eGFR (mL/min/1.73m²)',
        'med_adherence': 'Medication Adherence',
        'steps': 'Physical Activity (steps/day)',
        'sleep_hours': 'Sleep Duration (hours/day)',
        'hospitalizations': 'Hospitalizations (past year)',
        'smoking': 'Smoking Status',
        'alcohol': 'Alcohol Consumption',
        'exercise': 'Exercise Frequency'
    }