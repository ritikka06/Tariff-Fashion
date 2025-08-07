
import streamlit as st
import pandas as pd
from utils import load_model, predict_price  # Make sure these exist and work

# Cache dataset loading
@st.cache_data
def load_tariff_data():
    df = pd.read_csv("Tariff_fashion_cleaned.csv")
    return df

df = load_tariff_data()

# Prepare UI dropdown options
category_options = sorted(df["Product Type"].unique())
brand_options = sorted(df["Brand Name"].str.replace("&amp;", "&").unique())
cost_min = int(df["Price Before Tariff"].min())
cost_max = int(df["Price Before Tariff"].max())

# Precompute tariff boundaries (if you want a reference)
tariff_boundaries = df["Price After Tariff"].quantile([0.33, 0.66])

# Function: Determine tariff bracket based on tariff percentage
tariff_brackets = [
    (0, 5, "0–5%"),
    (5, 10, "5–10%"),
    (10, 18, "10–18%"),
    (18, 25, "18–25%"),
    (25, 35, "25–35%"),
    (35, float('inf'), "35%+")
]

def get_tariff_bracket(tariff_pct: float) -> str:
    for min_val, max_val, label in tariff_brackets:
        if min_val <= tariff_pct < max_val:
            return label
    return "Unknown"

# Streamlit page config
st.set_page_config(page_title="👗 Tariff-Fashion Price Predictor", layout="centered")

# Header with style
st.markdown("""
    <div style='background-color:#e4eaf9; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
        <h1 style='color:#232946; text-align:center;'>👗 Tariff-Fashion Price Predictor</h1>
        <p style='color:#232946; text-align:center; font-size:1.1rem;'>
            Predict post-tariff prices for your fashion product using Machine Learning.
        </p>
    </div>
""", unsafe_allow_html=True)

# Load the ML model
model = load_model("model/predictor.pkl")  # Adjust path if needed

# Input Form
with st.form("prediction_form"):
    st.markdown("#### 📝 Enter Product Details")
    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox(
            "👕 Category",
            category_options,
            help="Select the fashion product category"
        )
        brand = st.selectbox(
            "🏷️ Brand",
            brand_options,
            help="Choose the product brand"
        )
    with col2:
        base_cost = st.number_input(
            "💰 Base Cost (before tariff in rupees)",
            min_value=cost_min,
            max_value=cost_max,
            value=cost_min,
            help=f"Min {cost_min} to Max {cost_max} (from dataset)"
        )
    submitted = st.form_submit_button("🔍 Predict Price", use_container_width=True)

if submitted:
    # Prepare input DataFrame exactly as per model's training columns
    input_data = pd.DataFrame([{
        "Category": category,
        "Brand": brand,
        "BaseCost": base_cost
    }])

    # Predict post-tariff price
    predicted_price = predict_price(model, input_data)

    # Calculate tariff percentage increase
    tariff_pct = ((predicted_price - base_cost) / base_cost) * 100

    # Determine tariff bracket
    bracket_label = get_tariff_bracket(tariff_pct)

    # Display prediction results
    st.markdown("---")
    st.markdown("<h3 style='color:#232946;'>🧾 Prediction Results</h3>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("💸 Predicted Price After Tariff", f"₹{int(predicted_price)}")
    c2.metric("🧮 Tariff Increase (%)", f"{tariff_pct:.2f}%")
    c3.metric("📦 Tariff Bracket", bracket_label)

    st.success("Prediction successful! Adjust your pricing accordingly.")

# Sidebar with dataset insights
with st.sidebar:
    st.markdown("#### 📊 Dataset Snapshot")
    st.metric("Median Price After Tariff", f"₹{int(df['Price After Tariff'].median())}")
    st.metric("Most Common Category", df["Product Type"].mode()[0] if not df["Product Type"].mode().empty else "N/A")
    st.metric("Most Common Brand", df["Brand Name"].mode()[0] if not df["Brand Name"].mode().empty else "N/A")
    st.markdown("---")
   