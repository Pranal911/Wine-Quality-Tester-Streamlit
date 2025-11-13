import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

model_files = {
    "Logistic Regression": "ML_Models/Logistic_Regression_pipeline.joblib",
    "Decision Tree Classifier": "ML_Models/Decision_Tree_Classifier_pipeline.joblib",
    "Random Forest Classifier": "ML_Models/Random_Forest_Classifier_pipeline.joblib",
    "KNN Classifier": "ML_Models/KNN_Classifier_pipeline.joblib",
    "SVC": "ML_Models/SVC_pipeline.joblib",
}

loaded_models = {}
for name, path in model_files.items():
    if os.path.exists(path):
        try:
            loaded_models[name] = joblib.load(path)
        except Exception as e:
            st.error(f"Error loading model {name}: {e}")

st.title("🍷 Wine Quality Prediction System")
st.write("Enter the chemical properties of the wine to predict its quality score (typically 3-8).")
st.warning(
    f"⚠️ Disclaimer: This system is developed by Pranal Thapa (student) using datasets from Kaggle, and is for educational use only."
    " The results are based on machine learning algorithms and are NOT a substitute for professional sensory evaluation."
)

st.subheader("Chemical Feature Inputs")

col1, col2, col3 = st.columns(3)

with col1:
    fixed_acidity = st.number_input(
        "Fixed Acidity (g/dm³)", 
        min_value=4.0, max_value=16.0, value=7.4, step=0.1, format="%.2f"
    )
    volatile_acidity = st.number_input(
        "Volatile Acidity (g/dm³)", 
        min_value=0.1, max_value=1.5, value=0.70, step=0.01, format="%.3f"
    )
    citric_acid = st.number_input(
        "Citric Acid (g/dm³)", 
        min_value=0.0, max_value=1.0, value=0.00, step=0.01, format="%.2f"
    )
    
with col2:
    residual_sugar = st.number_input(
        "Residual Sugar (g/dm³)", 
        min_value=1.0, max_value=15.0, value=2.0, step=0.1, format="%.2f"
    )
    chlorides = st.number_input(
        "Chlorides (g/dm³)", 
        min_value=0.01, max_value=0.6, value=0.08, step=0.001, format="%.4f"
    )
    free_sulfur_dioxide = st.number_input(
        "Free Sulfur Dioxide (mg/dm³)", 
        min_value=1.0, max_value=75.0, value=15.0, step=1.0
    )
    
with col3:
    total_sulfur_dioxide = st.number_input(
        "Total Sulfur Dioxide (mg/dm³)", 
        min_value=6.0, max_value=300.0, value=45.0, step=1.0
    )
    density = st.number_input(
        "Density (g/cm³)", 
        min_value=0.990, max_value=1.004, value=0.997, step=0.0001, format="%.4f"
    )
    ph = st.number_input(
        "pH", 
        min_value=2.7, max_value=4.0, value=3.30, step=0.01, format="%.2f"
    )
    sulphates = st.number_input(
        "Sulphates (g/dm³)", 
        min_value=0.3, max_value=2.0, value=0.65, step=0.01, format="%.2f"
    )
    alcohol = st.number_input(
        "Alcohol (% vol.)", 
        min_value=8.0, max_value=15.0, value=10.0, step=0.1, format="%.2f"
    )

input_data = pd.DataFrame([[
    fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
    chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
    ph, sulphates, alcohol
]], columns=[
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol"
])


st.subheader("Prediction Settings")

model_choice = st.selectbox("Select Model", list(loaded_models.keys()))

if st.button("Predict Wine Quality"):
    if model_choice in loaded_models:
        model = loaded_models[model_choice]
        
        prediction = model.predict(input_data)
        predicted_quality = int(prediction[0])
        
        if predicted_quality >= 6:
            st.success(f"Predicted Quality Score: {predicted_quality}. This is considered good quality according to this algorithm. - Pranal Thapa")
        else:
            st.error(f"Predicted Quality Score: {predicted_quality}. This is considered lower quality according to the algorithm - Pranal Thapa")

        st.caption("Calculation by Pranal Thapa's system.")
    else:
        st.error("Selected model not loaded. Check console for file path errors.")