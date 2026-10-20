import streamlit as st
import pandas as pd
import joblib

# Load trained model and selected feature names
model = joblib.load('best_wealth_model.pkl')
selected_features = list(joblib.load('selected_features.pkl'))

# Main Title and Description
st.title("Household Wealth Index Predictor")
st.write("Select household characteristics to predict wealth category.")

# Dynamically extract Subbasin / Location names from saved features
subbasin_cols = [col for col in selected_features if col.startswith('Subbasin_')]
subbasin_names = [col.replace('Subbasin_', '') for col in subbasin_cols]

# --- Main Page Input Form ---

# 1. Subbasin / Geographic Location
subbasin_selected = None
if subbasin_names:
    subbasin_selected = st.selectbox("Subbasin / Location", subbasin_names)

# 2. Residency Type
residency = st.selectbox("Residency Type (hv025)", ["Urban", "Rural"])

# 3. Water Source and Toilet Facility
water_source = st.selectbox(
    "Drinking Water Source (hv201)",
    [
        "Piped to yard/plot", 
        "Piped into dwelling", 
        "Public tap/standpipe", 
        "Tube well or borehole", 
        "Protected well", 
        "Unprotected well", 
        "Surface water", 
        "Other"
    ]
)

toilet_type = st.selectbox(
    "Toilet Type (hv205)",
    [
        "Flush to pit latrine", 
        "Flush to piped sewer system", 
        "Pit latrine with slab", 
        "Pit latrine without slab/open pit", 
        "No facility/bush/field", 
        "Other"
    ]
)

# 4. Household Assets
electricity = st.selectbox("Electricity Available (hv206)", ["No", "Yes"])
radio = st.selectbox("Radio Available (hv207)", ["No", "Yes"])
tv = st.selectbox("Television Available (hv208)", ["No", "Yes"])
refrigerator = st.selectbox("Refrigerator Available (hv209)", ["No", "Yes"])

# Action Button
if st.button("Predict Wealth Category"):
    # Initialize dataframe with 0s matching exact model features
    input_data = pd.DataFrame(0, index=[0], columns=selected_features)

    # Set selected Subbasin / Location
    if subbasin_selected:
        selected_sub_col = f"Subbasin_{subbasin_selected}"
        if selected_sub_col in input_data.columns:
            input_data.at[0, selected_sub_col] = 1

    # Map Binary Features
    binary_map = {
        'hv025': 1 if residency == "Urban" else 0,
        'hv206': 1 if electricity == "Yes" else 0,
        'hv207': 1 if radio == "Yes" else 0,
        'hv208': 1 if tv == "Yes" else 0,
        'hv209': 1 if refrigerator == "Yes" else 0,
    }
    for col, val in binary_map.items():
        
