import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(page_title="DHS Wealth Predictor", page_icon="🇪🇹", layout="wide")

# Load trained model and exact feature columns
@st.cache_resource
def load_model_and_features():
    model = joblib.load('best_wealth_model.pkl')
    selected_features = list(joblib.load('selected_features.pkl'))
    return model, selected_features

model, selected_features = load_model_and_features()

st.title("🇪🇹 DHS Household Wealth Index Predictor")
st.write("An advanced machine learning web application built for socioeconomic and geospatial wealth tier prediction (**Poor**, **Middle**, **Rich**).")

# Initialize default input dataframe with zeros matching all model features
input_data = pd.DataFrame(0, index=[0], columns=selected_features)

# --- Layout Design: Two Columns for Inputs ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Geographic & Location Attributes")
    
    # Extract Subbasins dynamically
    subbasin_cols = [col for col in selected_features if col.startswith('Subbasin_')]
    subbasin_names = [col.replace('Subbasin_', '') for col in subbasin_cols]
    
    selected_subbasin = None
    if subbasin_names:
        selected_subbasin = st.selectbox("Select Subbasin", subbasin_names)
        sub_col = f"Subbasin_{selected_subbasin}"
        if sub_col in input_data.columns:
            input_data.at[0, sub_col] = 1

    # Residency Type (Urban/Rural - e.g., hv025)
    st.subheader("🏠 Residency Type")
    residency = st.selectbox("Residency", ["Urban", "Rural"])
    if 'hv025' in input_data.columns:
        input_data.at[0, 'hv025'] = 1 if residency == "Urban" else 0

    st.subheader("🚰 Water & Sanitation")
    water_options = ["Piped water", "Tube well / borehole", "Protected well", "Surface water", "Other"]
    selected_water = st.selectbox("Drinking Water Source", water_options)
    
    toilet_options = ["Flush toilet", "Pit latrine", "No facility / Bush", "Other"]
    selected_toilet = st.selectbox("Toilet Facility Type", toilet_options)

with col2:
    st.subheader("💡 Household Assets & Utilities")
    
    # Common DHS Asset Features
    asset_mapping = {
        'Electricity Access (hv206)': 'hv206',
        'Radio Ownership (hv207)': 'hv207',
        'Television Ownership (hv208)': 'hv208',
        'Refrigerator Ownership (hv209)': 'hv209'
    }
    
    for label, col_name in asset_mapping.items():
        if col_name in input_data.columns:
            val = st.selectbox(label, ["No", "Yes"], key=col_name)
            input_data.at[0, col_name] = 1 if val == "Yes" else 0

# Map dynamic water/toilet categories if they exist in features
for col in input_data.columns:
    if selected_water.lower().split()[0] in col.lower():
        input_data.at[0, col] = 1
    if selected_toilet.lower().split()[0] in col.lower():
        input_data.at[0, col] = 1

st.markdown("---")

# --- Prediction Section ---
col_center = st.columns([1, 2, 1])
with col_center[1]:
    predict_btn = st.button("🚀 Predict Household Wealth Category", use_container_width=True)

if predict_btn:
    try:
        prediction = model.predict(input_data)
        
        st.markdown("### 📊 Prediction Result")
        pred_class = prediction[0]
        
        if pred_class.lower() in ['rich', 'richer']:
            st.success(f"Predicted Wealth Tier: **{pred_class}** 🟢 (High Socioeconomic Status)")
        elif pred_class.lower() in ['middle']:
            st.warning(f"Predicted Wealth Tier: **{pred_class}** 🟡 (Moderate Socioeconomic Status)")
        else:
            st.error(f"Predicted Wealth Tier: **{pred_class}** 🔴 (Vulnerable / Lower Socioeconomic Status)")
            
    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
