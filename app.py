import streamlit as st
import pandas as pd
import joblib

# Load trained model and selected feature column names
model = joblib.load('best_wealth_model.pkl')
selected_features = list(joblib.load('selected_features.pkl'))

st.title("DHS Household Wealth Index Predictor 🇪🇹")
st.write("Predict the wealth tier (**Poor**, **Middle**, or **Rich**) of a household based on key socioeconomic and housing assets.")

# Identify dynamically encoded subbasins and categories from feature names
subbasin_cols = [col for col in selected_features if col.startswith('Subbasin_')]
subbasin_names = [col.replace('Subbasin_', '') for col in subbasin_cols]

st.sidebar.header("Household & Location Attributes")

# Geographic & Location Inputs
subbasin_selected = None
if subbasin_names:
    subbasin_selected = st.sidebar.selectbox("Select Subbasin Location", subbasin_names)

residency = st.sidebar.selectbox("Residency Type", ["Urban", "Rural"])

# Household Asset Inputs
st.sidebar.subheader("Assets & Services")
electricity = st.sidebar.selectbox("Electricity Access", ["Yes", "No"])
radio = st.sidebar.selectbox("Radio Ownership", ["Yes", "No"])
tv = st.sidebar.selectbox("Television Ownership", ["Yes", "No"])
refrigerator = st.sidebar.selectbox("Refrigerator Ownership", ["Yes", "No"])

# Drinking Water and Toilet Facility
water_source = st.sidebar.selectbox(
    "Drinking Water Source",
    ["Piped water", "Tube well / borehole", "Protected well", "Unprotected well/spring", "Surface water", "Other"]
)
toilet_type = st.sidebar.selectbox(
    "Toilet Facility",
    ["Flush toilet", "Pit latrine", "No facility / Bush", "Other"]
)

# Predict Button
if st.button("Predict Wealth Category"):
    # Create empty dataframe with 0s matching exact model features
    input_data = pd.DataFrame(0, index=[0], columns=selected_features)

    # Set Selected Subbasin to 1 if applicable
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
        if col in input_data.columns:
            input_data.at[0, col] = val

    # Match any dynamic One-Hot features (Water, Toilet, etc.)
    for col in input_data.columns:
        if water_source.lower() in col.lower():
            input_data.at[0, col] = 1
        if toilet_type.lower() in col.lower():
            input_data.at[0, col] = 1

    # Execute Prediction
    try:
        prediction = model.predict(input_data)
        st.success(f"Predicted Household Wealth Category: **{prediction[0]}**")
    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")
