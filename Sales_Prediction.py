import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import sklearn as sk

# ---------------------------------------------------
# Streamlit Page Config + Background Color
# ---------------------------------------------------
st.set_page_config(page_title="Store Sales Predictor", layout="centered")

page_bg = """
<style>
body {
    background-color: #F2F4F7;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ---------------------------------------------------
# Load Model, Encoder, and Feature List
# ---------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = pickle.load(open("simple_xgb_model.pkl", "rb"))
    encoder = pickle.load(open("simple_encoder.pkl", "rb"))
    feature_names = pickle.load(open("simple_feature_names.pkl", "rb"))
    return model, encoder, feature_names

model, encoder, feature_names = load_artifacts()

# ---------------------------------------------------
# UI Header
# ---------------------------------------------------
st.title("📈 Store Sales Prediction App")
st.write("Fill in the store details below to predict future sales.")
st.markdown("---")

# ---------------------------------------------------
# User Inputs with sliders and dropdowns
# ---------------------------------------------------
store_id = st.slider("Store ID", min_value=1, max_value=365, value=10)

store_type = st.selectbox("Store Type", ["S1", "S2", "S3", "S4"])
location_type = st.selectbox("Location Type", ["L1", "L2", "L3"])
region_code = st.selectbox("Region Code", ["R1", "R2", "R3", "R4"])

holiday = st.radio("Holiday?", ["Yes", "No"], horizontal=True)
discount = st.radio("Discount Applied?", ["Yes", "No"], horizontal=True)

date_input = st.date_input("Select Date")

st.markdown("---")

# ---------------------------------------------------
# Prepare Data for Prediction
# ---------------------------------------------------
if st.button("🔮 Predict Sales"):

    year = date_input.year
    month = date_input.month
    day = date_input.day
    day_of_week = date_input.weekday()

    holiday_flag = 1 if holiday == "Yes" else 0
    discount_flag = 1 if discount == "Yes" else 0

    # Base Input
    input_df = pd.DataFrame({
        "Store_id": [store_id],
        "Store_Type": [store_type],
        "Location_Type": [location_type],
        "Region_Code": [region_code],
        "Holiday": [holiday_flag],
        "Discount": [discount_flag],
        "Year": [year],
        "Month": [month],
        "Day": [day],
        "DayOfWeek": [day_of_week]
    })

    # Apply OneHotEncoder
    cat_cols = ["Store_Type", "Location_Type", "Region_Code"]
    encoded_array = encoder.transform(input_df[cat_cols])
    encoded_df = pd.DataFrame(encoded_array,
        columns=encoder.get_feature_names_out(cat_cols))

    # Combine with numeric features
    final_df = pd.concat([
        input_df.drop(columns=cat_cols).reset_index(drop=True),
        encoded_df.reset_index(drop=True)
    ], axis=1)

    # Add missing columns (if any)
    for col in feature_names:
        if col not in final_df.columns:
            final_df[col] = 0

    final_df = final_df[feature_names]

    # Predict
    prediction = model.predict(final_df)[0]

    # ---------------------------------------------------
    # Output Result
    # ---------------------------------------------------
    st.success(f"### 🎯 Predicted Sales: **₹{prediction:,.2f}**")
    st.balloons()


