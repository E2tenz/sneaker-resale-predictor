

import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Sneaker Resale Predictor", page_icon="👟", layout="centered")

st.title("👟 Sneaker Resale Premium Predictor")
st.markdown("Estimate how much above retail a sneaker will sell for, based on real StockX data.")
st.divider()


@st.cache_resource
def load_model():
    with open("models/best_model.pkl", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    st.warning(" No trained model found. Run `python analysis.py` first.")
    model_loaded = False

st.subheader("Enter Sneaker Details")

col1, col2 = st.columns(2)

with col1:
    retail_price      = st.number_input("Retail Price ($)", min_value=50, max_value=1000, value=160, step=10)
    days_since_release = st.slider("Days Since Release", 0, 730, 30)
    shoe_size         = st.selectbox("Shoe Size (US)", options=[float(x) for x in range(6, 16)])

with col2:
    brand             = st.selectbox("Brand", ["Jordan", "Nike", "Adidas", "Yeezy", "New Balance", "Other"])
    buyer_region      = st.selectbox("Buyer Region", ["New York", "California", "Texas", "Oregon", "Florida", "Other"])
    sale_month        = st.slider("Month of Sale", 1, 12, 6)
    is_limited        = st.checkbox("Limited Release?", value=True)

size_demand_index = 1.0 if 9 <= shoe_size <= 11 else (0.7 if 7 <= shoe_size < 9 or 11 < shoe_size <= 13 else 0.4)
season_map = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
              6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall"}
sale_season = season_map[sale_month]

if st.button(" Predict Resale Premium", use_container_width=True, type="primary"):
    if model_loaded:
        
        brand_encoded  = ["Adidas", "Jordan", "New Balance", "Nike", "Other", "Yeezy"].index(brand) if brand in ["Adidas", "Jordan", "New Balance", "Nike", "Other", "Yeezy"] else 0
        region_encoded = 0
        season_encoded = ["Fall", "Spring", "Summer", "Winter"].index(sale_season)

        features = pd.DataFrame([{
            "retail_price":        retail_price,
            "days_since_release":  days_since_release,
            "shoe_size":           shoe_size,
            "size_demand_index":   size_demand_index,
            "is_limited":          int(is_limited),
            "sale_month":          sale_month,
            "brand":               brand_encoded,
            "buyer_region":        region_encoded,
            "sale_season":         season_encoded,
        }])

        premium = model.predict(features)[0]
        resale_price = retail_price * (1 + premium / 100)

        st.divider()
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Predicted Premium", f"{premium:.1f}%")
        col_b.metric("Est. Resale Price", f"${resale_price:.0f}")
        col_c.metric("Est. Profit", f"${resale_price - retail_price:.0f}")

        if premium > 100:
            st.success(" High demand — strong resale potential")
        elif premium > 40:
            st.info(" Moderate resale premium — worth copping")
        else:
            st.warning(" Low premium — may not be worth the effort")
    else:
        st.error("Train the model first by running `python analysis.py`.")

st.divider()
st.caption("Built by Khalid | StockX x UoA Data Science Portfolio")
