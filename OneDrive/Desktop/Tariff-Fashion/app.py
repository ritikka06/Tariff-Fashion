import streamlit as st
import pandas as pd
from utils import load_model, predict_price

st.set_page_config(page_title="👗 Tariff-Fashion ML", layout="centered")

st.title("👗 Tariff-Fashion Price Predictor")
st.markdown("""
<style>
    .main-title { font-size:28px; font-weight:bold; }
    .subheader { color: #6c757d; }
    .stButton > button {
        background-color: #6c63ff;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<p class='subheader'>📊 Predict fashion product prices and get suggested tariff brackets based on attributes.</p>", unsafe_allow_html=True)

model = load_model("model/price_predictor.pkl")

with st.form("prediction_form"):
    st.markdown("### 📝 Enter Product Details")

    col1, col2 = st.columns(2)

    with col1:
        category = st.selectbox("👕 Category", ['T-shirt', 'Jeans', 'Saree', 'Jacket', 'Kurta'])
        brand = st.selectbox("🏷️ Brand", ['BrandA', 'BrandB', 'BrandC', 'BrandD'])
        region = st.selectbox("🌍 Region", ['Asia', 'Europe', 'North America'])
        base_cost = st.number_input("💰 Base Cost (₹)", min_value=100, max_value=5000, value=1000,
                                    help="Enter the base cost of manufacturing the product in INR")

    with col2:
        material = st.selectbox("🧵 Material", ['Cotton', 'Silk', 'Denim', 'Polyester'])
        weight = st.slider("⚖️ Weight (kg)", min_value=0.1, max_value=3.0, step=0.1, value=1.0)
        rating = st.slider("⭐ Rating", min_value=1.0, max_value=5.0, step=0.1, value=4.0,
                           help="Average user rating (1 to 5)")

    submitted = st.form_submit_button("🔍 Predict Price")

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

    if predicted_price < 700:
        tariff = "0–5%"
    elif predicted_price < 1200:
        tariff = "5–10%"
    else:
        tariff = "10–18%"

    st.markdown("---")
    st.markdown("## 🧾 Prediction Results")

    col1, col2 = st.columns(2)
    col1.metric("💸 Predicted Price", f"₹{int(predicted_price)}")
    col2.metric("📦 Suggested Tariff", tariff)

    st.success("Prediction successful! Adjust your pricing accordingly.")
