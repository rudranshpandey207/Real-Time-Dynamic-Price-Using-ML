import streamlit as st
import pandas as pd
import pickle
from scraper import get_competitor_price
from utils import simulate_inventory_and_demand

st.set_page_config(page_title="Dynamic Pricing Engine", layout="wide")
st.title("🛒 Real‑Time Dynamic Pricing Engine")

# Load both models
@st.cache_resource
def load_models():
    # Load original model (for live data)
    try:
        with open('model/pricing_model.pkl', 'rb') as f:
            model_original = pickle.load(f)
    except:
        model_original = None
    
    # Load synthetic model (for manual input)
    with open('model/pricing_model_syn.pkl', 'rb') as f:
        model_syn = pickle.load(f)
    
    return model_original, model_syn

model_original, model_syn = load_models()

# Sidebar with model info
with st.sidebar:
    st.header("📊 Model Information")
    st.write("**Synthetic Model (Manual):**")
    st.write("- R² Score: 0.9384")
    st.write("- MAE: ₹87")
    if model_original:
        st.write("\n**Original Model (Live):**")
        st.write("- For real-time data")

# Main input section
st.header("Enter Product Details")

col1, col2 = st.columns(2)

with col1:
    product = st.text_input("Product name:", value="wireless mouse")
    inventory_input = st.number_input("Inventory on hand:", min_value=0, max_value=10000, value=150)

with col2:
    # Option to use live data or manual input
    use_live_data = st.checkbox("Fetch live competitor price", value=False)
    
    if not use_live_data:
        manual_comp_price = st.number_input("Competitor Price (₹):", min_value=0.0, value=3000.0)
        manual_demand = st.number_input("Forecast Demand (units):", min_value=0.0, value=100.0)

# Predict button
if st.button("🎯 Suggest Optimal Price", type="primary"):
    try:
        with st.spinner("Analyzing..."):
            
            if use_live_data:
                # Use ORIGINAL model with live data
                if model_original is None:
                    st.error("Original model not found. Please train it first.")
                else:
                    comp_price = get_competitor_price(product)
                    inventory, demand = simulate_inventory_and_demand()
                    
                    # Use inventory input if provided
                    inventory = inventory_input if inventory_input else inventory
                    
                    X_live = pd.DataFrame({
                        'competitor_price': [comp_price],
                        'forecast_demand': [demand],
                        'inventory': [inventory]
                    })
                    
                    suggested_price = model_original.predict(X_live)[0]
                    st.info("Using: **Original Model** (Live Data)")
            
            else:
                # Use SYNTHETIC model with manual input
                comp_price = manual_comp_price
                demand = manual_demand
                inventory = inventory_input
                
                X_live = pd.DataFrame({
                    'competitor_price': [comp_price],
                    'forecast_demand': [demand],
                    'inventory_on_hand': [inventory]
                })
                
                suggested_price = model_syn.predict(X_live)[0]
                st.info("Using: **Synthetic Model** (Manual Input)")
            
            # Display results
            st.success(f"## 💰 Suggested Optimal Price: ₹{suggested_price:.2f}")
            
            # Show breakdown
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Competitor Price", f"₹{comp_price:.2f}")
            with col2:
                st.metric("Forecast Demand", f"{demand:.0f} units")
            with col3:
                st.metric("Inventory", f"{inventory} units")
            
            # Price comparison
            price_diff = suggested_price - comp_price
            price_diff_pct = (price_diff / comp_price) * 100
            
            if price_diff > 0:
                st.info(f"💡 Price ₹{price_diff:.2f} ({price_diff_pct:.1f}%) **above** competitor")
            else:
                st.warning(f"💡 Price ₹{abs(price_diff):.2f} ({abs(price_diff_pct):.1f}%) **below** competitor")
            
            # Inventory warning
            if inventory < demand:
                st.error(f"⚠️ Low Stock! Inventory ({inventory}) < Demand ({demand:.0f})")
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")