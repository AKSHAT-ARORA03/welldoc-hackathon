import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Update the file paths to ensure the dashboard uses the chronic dataset
DATA_PATH = "data/synthetic_chronic_dataset.csv"
MODEL_PATH = "models/risk_model.pkl"
EXPLAINER_PATH = "models/explainer.pkl"

# Page configuration
st.set_page_config(
    page_title="Chronic Care Risk Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define CSS styles
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        font-weight: bold;
        color: #424242;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .risk-high {
        font-size: 1.2rem;
        font-weight: bold;
        color: #D32F2F;
        padding: 0.5rem;
        border-radius: 0.3rem;
        background-color: rgba(211, 47, 47, 0.1);
    }
    .risk-medium {
        font-size: 1.2rem;
        font-weight: bold;
        color: #F57C00;
        padding: 0.5rem;
        border-radius: 0.3rem;
        background-color: rgba(245, 124, 0, 0.1);
    }
    .risk-low {
        font-size: 1.2rem;
        font-weight: bold;
        color: #388E3C;
        padding: 0.5rem;
        border-radius: 0.3rem;
        background-color: rgba(56, 142, 60, 0.1);
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
        margin-bottom: 1rem;
    }
    .recommendation {
        background-color: #e3f2fd;
        border-left: 0.25rem solid #1976D2;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        border-radius: 0 0.3rem 0.3rem 0;
    }
</style>
""", unsafe_allow_html=True)

def format_feature_name(feature_name):
    """Format a feature name for display"""
    return feature_name.replace('_', ' ').title()

def load_data():
    """Load patient data, model, and explainer"""
    if os.path.exists(DATA_PATH):
        # Use the chronic data schema
        from data_schema import load_chronic_data, preprocess_chronic_data
        df = load_chronic_data(DATA_PATH)
        df = preprocess_chronic_data(df)
    else:
        print(f"Data file not found at {DATA_PATH}")
        return None, None, None
    
    # Load the model if it exists
    model = None
    explainer = None
    if os.path.exists(MODEL_PATH):
        try:
            from model import load_model
            model = load_model(MODEL_PATH)
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Model file not found at {MODEL_PATH}")
    
    # Load the explainer if it exists
    if os.path.exists(EXPLAINER_PATH):
        try:
            from explainability import ModelExplainer
            explainer = ModelExplainer.load(EXPLAINER_PATH)
        except Exception as e:
            print(f"Error loading explainer: {e}")
    else:
        print(f"Explainer file not found at {EXPLAINER_PATH}")
    
    return df, model, explainer

def load_or_train_model(df):
    """Load the model or train if it doesn't exist"""
    if os.path.exists(MODEL_PATH):
        from model import load_model
        model = load_model(MODEL_PATH)
        st.sidebar.success("Model loaded successfully.")
    else:
        st.sidebar.warning("Model not found. Please train the model first.")
        model = None
    
    if os.path.exists(EXPLAINER_PATH):
        explainer = joblib.load(EXPLAINER_PATH)
        st.sidebar.success("Explainer loaded successfully.")
    else:
        st.sidebar.warning("Explainer not found. Will be created when needed.")
        explainer = None
    
    return model, explainer

def create_explainer(model, X):
    """Create and save an explainer"""
    from explainability import ModelExplainer
    os.makedirs(os.path.dirname(EXPLAINER_PATH), exist_ok=True)
    explainer = ModelExplainer(model, X)
    explainer.save(EXPLAINER_PATH)
    return explainer

@st.cache_data
def get_patient_list(df):
    """Get a list of unique patient IDs"""
    return df['patient_id'].unique()

def cohort_view(df, model):
    """Display the cohort view with risk scores"""
    st.markdown("<div class='main-header'>Cohort Risk Overview</div>", unsafe_allow_html=True)
    
    # Get last records for each patient to compute current risk
    latest_records = df.sort_values('date').groupby('patient_id').last().reset_index()
    
    # Create a derived gender column from the one-hot encoded columns
    latest_records['gender'] = 'Unknown'
    if 'Gender_Female' in latest_records.columns:
        latest_records.loc[latest_records['Gender_Female'] == 1, 'gender'] = 'Female'
    if 'Gender_Male' in latest_records.columns:
        latest_records.loc[latest_records['Gender_Male'] == 1, 'gender'] = 'Male'
    
    # Prepare features for prediction
    X = latest_records.drop(columns=['patient_id', 'date', 'deterioration_90d', 'gender'])
    
    # Predict risk scores
    latest_records['risk_score'] = model.predict_proba(X)[:,1]
    
    # Add risk categories
    def risk_category(score):
        if score >= 0.7: return "High"
        elif score >= 0.4: return "Medium"
        return "Low"
    
    latest_records['risk_category'] = latest_records['risk_score'].apply(risk_category)
    
    # Filters
    col1, col2 = st.columns([1, 2])
    with col1:
        risk_filter = st.multiselect(
            "Filter by Risk Category",
            options=["High", "Medium", "Low"],
            default=["High", "Medium", "Low"]
        )
    
    with col2:
        search_name = st.text_input("Search by Patient ID")
    
    # Filter the data
    filtered_records = latest_records[latest_records['risk_category'].isin(risk_filter)]
    if search_name:
        filtered_records = filtered_records[filtered_records['patient_id'].str.contains(search_name, case=False)]
    
    # Display summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Patients", len(latest_records))
    with col2:
        high_risk_count = len(latest_records[latest_records['risk_category'] == "High"])
        st.metric("High Risk Patients", high_risk_count, f"{high_risk_count/len(latest_records):.1%}")
    with col3:
        medium_risk_count = len(latest_records[latest_records['risk_category'] == "Medium"])
        st.metric("Medium Risk Patients", medium_risk_count, f"{medium_risk_count/len(latest_records):.1%}")
    
    # Create risk score distribution visualization
    fig = px.histogram(
        latest_records, 
        x="risk_score", 
        color="risk_category",
        color_discrete_map={"High": "#D32F2F", "Medium": "#F57C00", "Low": "#388E3C"},
        labels={"risk_score": "Risk Score", "count": "Number of Patients"},
        title="Distribution of Risk Scores"
    )
    fig.update_layout(xaxis_title="Risk Score", yaxis_title="Number of Patients")
    st.plotly_chart(fig, use_container_width=True)
    
    # Display the patient table
    st.markdown("<div class='subheader'>Patient List</div>", unsafe_allow_html=True)
    
    # Format the display table
    display_cols = ['patient_id', 'age', 'gender', 'risk_score', 'risk_category']
    display_df = filtered_records[display_cols].copy()
    display_df = display_df.rename(columns={
        'patient_id': 'Patient ID',
        'age': 'Age',
        'gender': 'Gender',
        'risk_score': 'Risk Score',
        'risk_category': 'Risk Category'
    })
    
    # Color-code the risk categories
    def highlight_risk(val):
        if val == 'High':
            return 'background-color: rgba(211, 47, 47, 0.2); color: #D32F2F; font-weight: bold'
        elif val == 'Medium':
            return 'background-color: rgba(245, 124, 0, 0.2); color: #F57C00; font-weight: bold'
        elif val == 'Low':
            return 'background-color: rgba(56, 142, 60, 0.2); color: #388E3C; font-weight: bold'
        return ''
    
    st.dataframe(
        display_df.style.format({'Risk Score': '{:.2%}'}).applymap(highlight_risk, subset=['Risk Category']),
        height=400,
        use_container_width=True
    )
    
    return latest_records

def patient_list_view(df):
    """Display a list of patients with risk scores"""
    st.header("Patient List")
    
    # Add filters
    st.subheader("Filters")
    cols = st.columns(3)
    
    # Define risk filter
    with cols[0]:
        risk_filter = st.selectbox(
            "Risk Level", 
            ["All", "High Risk", "Medium Risk", "Low Risk"]
        )
    
    # Determine filter conditions based on risk level
    if risk_filter == "High Risk":
        df_filtered = df[df['deterioration_90d'] >= 0.7]
    elif risk_filter == "Medium Risk":
        df_filtered = df[(df['deterioration_90d'] < 0.7) & (df['deterioration_90d'] >= 0.4)]
    elif risk_filter == "Low Risk":
        df_filtered = df[df['deterioration_90d'] < 0.4]
    else:
        df_filtered = df.copy()
    
    # Create a patient list table
    patient_data = []
    for _, row in df_filtered.iterrows():
        patient_id = row['patient_id']
        age = int(row['age']) if 'age' in row else 'N/A'
        
        # Determine gender from one-hot encoded columns
        gender = 'Unknown'
        if 'Gender_Female' in row and row['Gender_Female'] == 1:
            gender = 'Female'
        elif 'Gender_Male' in row and row['Gender_Male'] == 1:
            gender = 'Male'
        
        # Risk classification
        risk_score = row['deterioration_90d']
        if isinstance(risk_score, (int, float)):
            risk_level = "High" if risk_score == 1 else "Low"
        else:
            risk_level = "Unknown"
        
        # Recent values
        recent_glucose = f"{row['glucose']:.1f}" if 'glucose' in row and not pd.isna(row['glucose']) else "N/A"
        recent_bp = f"{row['systolic_bp']:.0f}/{row['diastolic_bp']:.0f}" if 'systolic_bp' in row and 'diastolic_bp' in row else "N/A"
        
        patient_data.append({
            "Patient ID": patient_id,
            "Age": age,
            "Gender": gender,
            "Risk Level": risk_level,
            "Risk Score": risk_score,
            "Recent Glucose": recent_glucose,
            "Recent BP": recent_bp
        })
    
    # Convert to DataFrame for display
    if patient_data:
        patient_df = pd.DataFrame(patient_data)
        
        # Make Patient ID clickable
        st.write("Click on a patient ID to view details:")
        
        # Use a data editor with pagination for better performance with large datasets
        selected_indices = st.data_editor(
            patient_df,
            hide_index=True,
            use_container_width=True,
            disabled=["Age", "Gender", "Risk Level", "Risk Score", "Recent Glucose", "Recent BP"],
            key="patient_table",
            num_rows="fixed"
        )
        
        # Handle selection (clicking on row)
        if st.session_state.get('patient_table', {}).get('edited_rows'):
            selected_idx = list(st.session_state['patient_table']['edited_rows'].keys())[0]
            selected_patient_id = patient_df.iloc[int(selected_idx)]["Patient ID"]
            st.session_state['selected_patient_id'] = selected_patient_id
            st.session_state['view'] = 'patient_detail'
            st.rerun()
    else:
        st.write("No patients match the selected filters.")

def patient_detail_view(df, patient_id, model, explainer):
    """Display detailed information for a specific patient"""
    # Get patient data
    patient_data = df[df['patient_id'] == patient_id]
    
    if len(patient_data) == 0:
        st.error(f"Patient {patient_id} not found")
        return
    
    # Get the most recent record
    patient = patient_data.iloc[0]
    
    # Display header with patient info
    st.header(f"Patient {patient_id}")
    
    # Basic information
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = 'Unknown'
        if 'Gender_Female' in patient and patient['Gender_Female'] == 1:
            gender = 'Female'
        elif 'Gender_Male' in patient and patient['Gender_Male'] == 1:
            gender = 'Male'
        st.metric("Age", f"{patient['age']:.0f}")
    with col2:
        st.metric("Gender", gender)
    with col3:
        st.metric("BMI", f"{patient['bmi']:.1f}")
    
    # Risk prediction section
    st.subheader("Risk Assessment")
    
    risk_col1, risk_col2 = st.columns([1, 2])
    
    # Calculate risk score if model is available
    risk_score = float(patient['deterioration_90d'])
    risk_label = "High Risk" if risk_score == 1 else "Low Risk"
    risk_color = "red" if risk_score == 1 else "green"
    
    with risk_col1:
        # Show risk score with color
        st.markdown(
            f"""
            <div style="border-radius:10px; padding:10px; text-align:center; 
                       background-color:{'rgba(255,0,0,0.2)' if risk_score == 1 else 'rgba(0,255,0,0.2)'}">
                <h2 style="color:{risk_color}; margin:0">{risk_label}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with risk_col2:
        # Show explanation if explainer is available
        if model and explainer:
            # Prepare features for the model
            X = patient_data.drop(columns=['patient_id', 'date', 'deterioration_90d'])
            
            # Generate explanation
            explanation = explainer.generate_textual_explanation(X, risk_score, top_n=5)
            
            # Display explanation
            st.markdown(f"**Explanation**: {explanation.get('explanation_text', 'No explanation available')}")
            
            # Display recommendations
            if explanation.get('recommendations'):
                st.subheader("Recommendations")
                for rec in explanation.get('recommendations'):
                    st.markdown(f"- {rec}")
    
    # Clinical measurements
    st.subheader("Clinical Measurements")
    
    # Organize metrics into rows for better layout
    row1_cols = st.columns(4)
    with row1_cols[0]:
        st.metric("Blood Pressure", f"{patient['systolic_bp']:.0f}/{patient['diastolic_bp']:.0f} mmHg")
    with row1_cols[1]:
        st.metric("Heart Rate", f"{patient['heart_rate']:.0f} bpm")
    with row1_cols[2]:
        st.metric("Glucose", f"{patient['glucose']:.1f} mg/dL")
    with row1_cols[3]:
        st.metric("HbA1c", f"{patient['hba1c']:.1f}%")
    
    row2_cols = st.columns(4)
    with row2_cols[0]:
        st.metric("Cholesterol", f"{patient['cholesterol']:.0f} mg/dL")
    with row2_cols[1]:
        st.metric("LDL", f"{patient['ldl']:.0f} mg/dL")
    with row2_cols[2]:
        st.metric("HDL", f"{patient['hdl']:.0f} mg/dL")
    with row2_cols[3]:
        st.metric("Triglycerides", f"{patient['triglycerides']:.0f} mg/dL")
    
    # Lifestyle factors
    st.subheader("Lifestyle Factors")
    lifestyle_cols = st.columns(4)
    
    with lifestyle_cols[0]:
        smoking = "Yes" if 'Smoking_Yes' in patient and patient['Smoking_Yes'] == 1 else "No"
        st.metric("Smoking", smoking)
    
    with lifestyle_cols[1]:
        alcohol = "Yes" if 'Alcohol_Yes' in patient and patient['Alcohol_Yes'] == 1 else "No"
        st.metric("Alcohol", alcohol)
    
    with lifestyle_cols[2]:
        if 'Exercise_Freq_High' in patient:
            if patient['Exercise_Freq_High'] == 1:
                exercise = "High"
            elif patient['Exercise_Freq_Moderate'] == 1:
                exercise = "Moderate"
            else:
                exercise = "Low"
        else:
            exercise = "Unknown"
        st.metric("Exercise", exercise)
    
    with lifestyle_cols[3]:
        if 'med_adherence' in patient:
            adherence = f"{patient['med_adherence']:.0%}"
        else:
            adherence = "Unknown"
        st.metric("Medication Adherence", adherence)
    
    # Additional clinical information
    st.subheader("Additional Clinical Information")
    
    row3_cols = st.columns(4)
    with row3_cols[0]:
        st.metric("Creatinine", f"{patient['creatinine']:.2f} mg/dL")
    with row3_cols[1]:
        st.metric("eGFR", f"{patient['egfr']:.0f} mL/min")
    with row3_cols[2]:
        st.metric("Hemoglobin", f"{patient['hemoglobin']:.1f} g/dL")
    with row3_cols[3]:
        st.metric("Hospitalizations", f"{patient['hospitalizations']:.0f}")
    
    # Back button
    if st.button("Back to Patient List"):
        st.session_state['view'] = 'patient_list'
        st.rerun()

def evaluate_model(model, X_test, y_test):
    """Calculate performance metrics for the model"""
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss
    from sklearn.metrics import f1_score, confusion_matrix, roc_curve
    
    # Generate predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auroc = roc_auc_score(y_test, y_pred_proba)
    
    # PR curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    auprc = auc(recall, precision)
    
    # Brier score
    brier = brier_score_loss(y_test, y_pred_proba)
    
    # F1 score
    f1 = f1_score(y_test, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        "AUROC": auroc,
        "AUPRC": auprc,
        "Brier Score": brier,
        "F1 Score": f1,
        "ROC Curve": {"fpr": fpr, "tpr": tpr},
        "PR Curve": {"precision": precision, "recall": recall},
        "Confusion Matrix": cm
    }

def model_performance_view(model, df):
    """Display model performance metrics"""
    st.markdown("<div class='main-header'>Model Performance</div>", unsafe_allow_html=True)
    
    # Split data for evaluation
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=['patient_id', 'date', 'deterioration_90d'])
    y = df['deterioration_90d']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Get metrics
    metrics = evaluate_model(model, X_test, y_test)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("AUROC", f"{metrics['AUROC']:.3f}")
    with col2:
        st.metric("AUPRC", f"{metrics['AUPRC']:.3f}")
    with col3:
        st.metric("Brier Score", f"{metrics['Brier Score']:.3f}")
    with col4:
        st.metric("F1 Score", f"{metrics['F1 Score']:.3f}")
    
    # Display ROC and PR curves
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(
            x=metrics["ROC Curve"]["fpr"], 
            y=metrics["ROC Curve"]["tpr"],
            labels={"x": "False Positive Rate", "y": "True Positive Rate"},
            title=f"ROC Curve (AUROC: {metrics['AUROC']:.3f})"
        )
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=0, y1=1
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.line(
            x=metrics["PR Curve"]["recall"], 
            y=metrics["PR Curve"]["precision"],
            labels={"x": "Recall", "y": "Precision"},
            title=f"Precision-Recall Curve (AUPRC: {metrics['AUPRC']:.3f})"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Display confusion matrix
    st.markdown("<div class='subheader'>Confusion Matrix</div>", unsafe_allow_html=True)
    
    cm = metrics["Confusion Matrix"]
    cm_df = pd.DataFrame(
        cm,
        index=['Actual: No Deterioration', 'Actual: Deterioration'],
        columns=['Predicted: No Deterioration', 'Predicted: Deterioration']
    )
    
    fig = px.imshow(
        cm_df,
        text_auto=True,
        color_continuous_scale='Blues',
        title="Confusion Matrix"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Display feature importance
    if hasattr(model, 'feature_importances_'):
        st.markdown("<div class='subheader'>Feature Importance</div>", unsafe_allow_html=True)
        
        importance = model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title="Feature Importance"
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main function to run the dashboard"""
    # Sidebar
    st.sidebar.title("Navigation")
    
    # Load data and model
    df, model, explainer = load_data()
    
    if model is None:
        st.error("Model not found. Please train a model first.")
        return
    
    # Navigation
    page = st.sidebar.radio(
        "Select View",
        ["Cohort Overview", "Patient Details", "Model Performance"],
        key="navigation"
    )
    
    # Display appropriate view
    if page == "Cohort Overview":
        latest_records = cohort_view(df, model)
        
        # Add a button to select a patient for detailed view
        st.markdown("<div class='subheader'>View Patient Details</div>", unsafe_allow_html=True)
        selected_patient = st.selectbox("Select a patient for detailed view:", latest_records['patient_id'].unique())
        
        if st.button("View Patient Details"):
            st.session_state.page = "Patient Details"
            st.session_state.selected_patient = selected_patient
            st.rerun()
    
    elif page == "Patient Details":
        # Select a patient
        if "selected_patient" in st.session_state:
            default_patient = st.session_state.selected_patient
            del st.session_state.selected_patient
        else:
            patient_list = get_patient_list(df)
            default_patient = patient_list[0] if len(patient_list) > 0 else None
        
        patient_id = st.sidebar.selectbox("Select Patient", get_patient_list(df), index=0 if default_patient is None else list(get_patient_list(df)).index(default_patient))
        
        # Display patient details
        patient_detail_view(df, patient_id, model, explainer)
    
    elif page == "Model Performance":
        model_performance_view(model, df)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("AI-Driven Risk Prediction Engine for Chronic Care Patients")

if __name__ == "__main__":
    main()
