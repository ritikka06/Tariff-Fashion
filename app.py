import streamlit as st
import pandas as pd
from utils import load_model, predict_price

st.set_page_config(page_title="Tariff-Fashion ML", layout="centered")

st.title("Tariff-Fashion Price Predictor")
st.markdown("Predict fashion product price and suggest tariff bracket based on features.")

model = load_model("model/price_predictor.pkl")

# User Input Form
with st.form("prediction_form"):
    st.subheader("Enter Product Details")

    category = st.selectbox("Category", ['T-shirt', 'Jeans', 'Saree', 'Jacket', 'Kurta'])
    brand = st.selectbox("Brand", ['BrandA', 'BrandB', 'BrandC', 'BrandD'])
    material = st.selectbox("Material", ['Cotton', 'Silk', 'Denim', 'Polyester'])
    region = st.selectbox("Region", ['Asia', 'Europe', 'North America'])
    base_cost = st.number_input("Base Cost (in ₹)", min_value=100, max_value=5000, value=1000)
    weight = st.slider("Weight (kg)", min_value=0.1, max_value=3.0, step=0.1, value=1.0)
    rating = st.slider("Rating (1-5)", min_value=1.0, max_value=5.0, step=0.1, value=4.0)

    submitted = st.form_submit_button("Predict Price")

if submitted:
    input_data = pd.DataFrame([{
        "Category": category,
        "Brand": brand,
        "Material": material,
        "Region": region,
        "BaseCost": base_cost,
        "Weight": weight,
        "Rating": rating
    }])

    predicted_price = predict_price(model, input_data)
    st.success(f"Predicted Price: ₹{int(predicted_price)}")

    # Suggest Tariff Bracket
    if predicted_price < 700:
        tariff = "0–5%"
    elif predicted_price < 1200:
        tariff = "5–10%"
    else:
        tariff = "10–18%"

    st.info(f"Suggested Tariff Bracket: **{tariff}**")
