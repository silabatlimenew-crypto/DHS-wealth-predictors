import streamlit as st
import pandas as pd
import joblib

# Load trained model and selected feature column names
model = joblib.load('best_wealth_model.pkl')
selected_features = joblib.load('selected_features.pkl')

st.title("DHS Household Wealth Index Predictor 🇪🇹")
st.write("Predict the wealth tier (**Poor**, **Middle**, or **Rich**) of a household based on key socioeconomic and housing assets.")

# --- Sidebar Controls ---
st.sidebar.header("Household Characteristics")

# 1. Geographic residency
residency = st.sidebar.selectbox("Residency Type", ["Urban", "Rural"])

# 2. Key Assets
electricity = st.sidebar.selectbox("Electricity Access", ["Yes", "No"])
radio = st.sidebar.selectbox("Radio Ownership", ["Yes", "No"])
tv = st.sidebar.selectbox("Television Ownership", ["Yes", "No"])
refrigerator = st.sidebar.selectbox("Refrigerator Ownership", ["Yes", "No"])

# 3. Water & Sanitation
water_source = st.sidebar.selectbox(
    "Drinking Water Source",
    ["Piped water", "Tube well / borehole", "Protected well", "Unprotected well/spring", "Surface water", "Other"]
)
toilet_type = st.sidebar.selectbox(
    "Toilet Facility",
    ["Flush toilet", "Pit latrine", "No facility / Bush", "Other"]
)

# --- Prediction Action ---
if st.button("Predict Wealth Category"):
    # Initialize a dataframe with zeros using exact feature columns
    input_data = pd.DataFrame(0, index=[0], columns=selected_features)
    
    # Map Binary Features dynamically if present in selected_features
    binary_mappings = {
        'hv025': 1 if residency == "Urban" else 0,
        'hv206': 1 if electricity == "Yes" else 0,
        'hv207': 1 if radio == "Yes" else 0,
        'hv208': 1 if tv == "Yes" else 0,
        'hv209': 1 if refrigerator == "Yes" else 0,
    }
    
    for col, val in binary_mappings.items():
        if col in input_data.columns:
            input_data.at[0, col] = val

    # Map One-Hot Categorical Features if present
    water_col = f"hv201_{water_source}"
    if water_col in input_data.columns:
        input_data.at[0, water_col] = 1

    toilet_col = f"hv205_{toilet_type}"
    if toilet_col in input_data.columns:
        input_data.at[0, toilet_col] = 1

    # Execute Prediction
    try:
        prediction = model.predict(input_data)
        st.success(f"Predicted Household Wealth Category: **{prediction[0]}**")
    except Exception as e:
        st.error(f"Prediction Error: {str(e)}")

