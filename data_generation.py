import pandas as pd
import numpy as np
from pathlib import Path

def generate_realistic_pricing_data(n=1000, seed=42):
    """
    Generate synthetic pricing data with REALISTIC relationships AND minimal data quality issues
    Keeps key features clean for high accuracy
    """
    np.random.seed(seed)
    
    categories = ['Electronics', 'Fashion', 'Home', 'Sports', 'Books', 'Toys']
    brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE']
    
    # Category base prices
    cat_base_cost = {
        'Electronics': 3000,
        'Fashion': 800,
        'Home': 1500,
        'Sports': 1200,
        'Books': 400,
        'Toys': 600
    }
    
    data = []
    
    for i in range(n):
        # Basic info
        product_id = f'PROD{i:04d}'
        product_name = f'Product {i}'
        category = np.random.choice(categories)
        brand = np.random.choice(brands)
        
        # Cost price (base cost with variation)
        base = cat_base_cost[category]
        cost_price = base * np.random.uniform(0.7, 1.3)
        
        # Competitor price (cost + margin 20-50%)
        comp_margin = np.random.uniform(0.20, 0.50)
        competitor_price = cost_price * (1 + comp_margin)
        
        # Our historical price
        our_historical_price = competitor_price * np.random.uniform(0.95, 1.05)
        
        # Discount
        discount_pct = np.random.uniform(0, 25)
        
        # Rating
        rating = np.clip(np.random.normal(4.0, 0.8), 1, 5)
        
        # Demand metrics
        page_views = int(np.random.uniform(100, 10000))
        conversion_rate = np.clip(np.random.normal(0.08, 0.04), 0.01, 0.25)
        units_sold_last_30d = int(page_views * conversion_rate * np.random.uniform(0.8, 1.2))
        
        # Forecast demand
        forecast_demand = units_sold_last_30d * np.random.uniform(0.9, 1.1)
        
        # Inventory
        inventory_on_hand = int(forecast_demand * np.random.uniform(0.5, 2.5))
        
        # Shipping cost
        shipping_cost = cost_price * 0.05 * np.random.uniform(0.8, 1.2)
        
        # ========================================
        # OPTIMAL PRICE FORMULA (REALISTIC!)
        # ========================================
        
        # 1. Start with competitor price
        optimal_price = competitor_price
        
        # 2. Demand factor
        demand_ratio = forecast_demand / (page_views * conversion_rate + 1)
        demand_adjustment = (demand_ratio - 1) * 0.15
        optimal_price *= (1 + demand_adjustment)
        
        # 3. Inventory factor
        inventory_ratio = inventory_on_hand / (forecast_demand + 1)
        if inventory_ratio < 1:
            scarcity_premium = (1 - inventory_ratio) * 0.20
            optimal_price *= (1 + scarcity_premium)
        else:
            excess_discount = min((inventory_ratio - 1) * 0.10, 0.15)
            optimal_price *= (1 - excess_discount)
        
        # 4. Quality factor
        rating_premium = (rating - 3.0) * 0.05
        optimal_price *= (1 + rating_premium)
        
        # 5. Ensure profit margin
        min_price = cost_price * 1.10
        optimal_price = max(optimal_price, min_price)
        
        # 6. Don't price too high
        max_price = competitor_price * 1.30
        optimal_price = min(optimal_price, max_price)
        
        # Add noise
        optimal_price *= np.random.uniform(0.98, 1.02)
        
        # Store row with CLEAN values first
        data.append({
            'product_id': product_id,
            'product_name': product_name,
            'category': category,
            'brand': brand,
            'cost_price': round(cost_price, 2),
            'competitor_price': round(competitor_price, 2),
            'our_historical_price': round(our_historical_price, 2),
            'discount_pct': round(discount_pct, 2),
            'rating': round(rating, 1),
            'page_views': page_views,
            'conversion_rate': round(conversion_rate, 4),
            'units_sold_last_30d': units_sold_last_30d,
            'forecast_demand': round(forecast_demand, 2),
            'inventory_on_hand': inventory_on_hand,
            'shipping_cost': round(shipping_cost, 2),
            'optimal_price': round(optimal_price, 2)
        })
    
    df = pd.DataFrame(data)
    
    # ========================================
    # ADD DATA QUALITY ISSUES (MINIMAL ON KEY FEATURES!)
    # ========================================
    
    # 1. Add 10% missing brand (not a model feature - safe)
    missing_brand_idx = np.random.choice(df.index, size=int(len(df)*0.10), replace=False)
    df.loc[missing_brand_idx, 'brand'] = None
    
    # 2. Add 6% missing cost_price (not a model feature - safe)
    missing_cost_idx = np.random.choice(df.index, size=int(len(df)*0.06), replace=False)
    df.loc[missing_cost_idx, 'cost_price'] = None
    
    # 3. Add 5% negative cost_price (not a model feature - safe)
    neg_cost_idx = np.random.choice(df.index, size=int(len(df)*0.05), replace=False)
    df.loc[neg_cost_idx, 'cost_price'] = -abs(df.loc[neg_cost_idx, 'cost_price'])
    
    # 4. Add 8% rupee symbol to competitor_price (KEY FEATURE - add formats but keep values)
    rupee_idx = np.random.choice(df.index, size=int(len(df)*0.08), replace=False)
    for idx in rupee_idx:
        df.loc[idx, 'competitor_price'] = f"₹{int(df.loc[idx, 'competitor_price'])}"
    
    # 5. Add ONLY 2% text/missing to competitor_price (keep it mostly clean)
    text_comp_idx = np.random.choice([i for i in df.index if i not in rupee_idx], size=int(len(df)*0.02), replace=False)
    for idx in text_comp_idx:
        df.loc[idx, 'competitor_price'] = np.random.choice(['N/A', 'TBD', None])
    
    # 6. Add ONLY 5% missing forecast_demand (KEY FEATURE - keep mostly clean)
    missing_forecast_idx = np.random.choice(df.index, size=int(len(df)*0.05), replace=False)
    df.loc[missing_forecast_idx, 'forecast_demand'] = None
    
    # 7. Add 3% negative forecast_demand
    neg_forecast_idx = np.random.choice([i for i in df.index if i not in missing_forecast_idx], size=int(len(df)*0.03), replace=False)
    df.loc[neg_forecast_idx, 'forecast_demand'] = -abs(df.loc[neg_forecast_idx, 'forecast_demand'])
    
    # 8. Add ONLY 10% missing inventory (KEY FEATURE - keep mostly clean)
    missing_inv_idx = np.random.choice(df.index, size=int(len(df)*0.10), replace=False)
    df.loc[missing_inv_idx, 'inventory_on_hand'] = None
    
    # 9. Add 10% rupee symbol to shipping_cost (not a model feature - safe)
    rupee_ship_idx = np.random.choice(df.index, size=int(len(df)*0.10), replace=False)
    for idx in rupee_ship_idx:
        df.loc[idx, 'shipping_cost'] = f"₹{int(df.loc[idx, 'shipping_cost'])}"
    
    # 10. Add 5% text to shipping_cost (not a model feature - safe)
    text_ship_idx = np.random.choice([i for i in df.index if i not in rupee_ship_idx], size=int(len(df)*0.05), replace=False)
    for idx in text_ship_idx:
        df.loc[idx, 'shipping_cost'] = np.random.choice(['Free', 'TBD', None])
    
    # 11. Add 15% missing discount_pct (not a model feature - safe)
    missing_discount_idx = np.random.choice(df.index, size=int(len(df)*0.15), replace=False)
    df.loc[missing_discount_idx, 'discount_pct'] = None
    
    # 12. Add 10% invalid discount (not a model feature - safe)
    invalid_discount_idx = np.random.choice([i for i in df.index if i not in missing_discount_idx], size=int(len(df)*0.10), replace=False)
    df.loc[invalid_discount_idx, 'discount_pct'] = np.random.uniform(-20, 120, size=len(invalid_discount_idx))
    
    # 13. Add 7% missing page_views (not a model feature - safe)
    missing_views_idx = np.random.choice(df.index, size=int(len(df)*0.07), replace=False)
    df.loc[missing_views_idx, 'page_views'] = None
    
    # 14. Add 5% negative units_sold (not a model feature - safe)
    neg_units_idx = np.random.choice(df.index, size=int(len(df)*0.05), replace=False)
    df.loc[neg_units_idx, 'units_sold_last_30d'] = -abs(df.loc[neg_units_idx, 'units_sold_last_30d'])
    
    # 15. Add 7% missing conversion_rate (not a model feature - safe)
    missing_conv_idx = np.random.choice(df.index, size=int(len(df)*0.07), replace=False)
    df.loc[missing_conv_idx, 'conversion_rate'] = None
    
    # 16. Add 10% invalid conversion_rate (not a model feature - safe)
    invalid_conv_idx = np.random.choice([i for i in df.index if i not in missing_conv_idx], size=int(len(df)*0.10), replace=False)
    df.loc[invalid_conv_idx, 'conversion_rate'] = np.random.uniform(-0.05, 1.2, size=len(invalid_conv_idx))
    
    # 17. Add 15% out-of-range ratings (not a model feature - safe)
    invalid_rating_idx = np.random.choice(df.index, size=int(len(df)*0.15), replace=False)
    df.loc[invalid_rating_idx, 'rating'] = np.random.uniform(0.5, 6.0, size=len(invalid_rating_idx))
    
    # 18. Add ONLY 4% missing optimal_price (target variable - keep clean!)
    missing_optimal_idx = np.random.choice(df.index, size=int(len(df)*0.04), replace=False)
    df.loc[missing_optimal_idx, 'optimal_price'] = None
    
    # 19. Add 5% duplicate rows
    num_duplicates = int(n * 0.05)
    duplicate_indices = np.random.choice(df.index, num_duplicates, replace=True)
    df_duplicates = df.loc[duplicate_indices].copy()
    df = pd.concat([df, df_duplicates], ignore_index=True)
    
    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    print("Generating realistic pricing dataset with controlled data quality issues...")
    df = generate_realistic_pricing_data(n=1000, seed=42)
    
    # Save
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "realistic_pricing_dataset.csv"
    df.to_csv(out_path, index=False)
    
    print(f"✅ Generated {len(df)} rows (includes ~5% duplicates)")
    print(f"✅ Saved to: {out_path}")
    print(f"\n📊 Data Quality Issues (Controlled):")
    print(f"\nKEY MODEL FEATURES (kept mostly clean):")
    print(f"  - competitor_price: ~8% with ₹, ~2% missing/text")
    print(f"  - forecast_demand: ~5% missing, ~3% negative")
    print(f"  - inventory_on_hand: ~10% missing")
    print(f"\nOTHER COLUMNS (more issues added):")
    print(f"  - brand: ~10% missing")
    print(f"  - cost_price: ~6% missing, ~5% negative")
    print(f"  - discount_pct: ~15% missing, ~10% invalid")
    print(f"  - rating: ~15% out of range")
    print(f"  - shipping_cost: ~10% with ₹, ~5% text")
    print(f"  - conversion_rate: ~7% missing, ~10% invalid")
    print(f"\nVerification:")
    print(f"  ₹ symbols in competitor_price: {df['competitor_price'].astype(str).str.contains('₹').sum()}")
    print(f"  ₹ symbols in shipping_cost: {df['shipping_cost'].astype(str).str.contains('₹').sum()}")
    print(f"\nMissing values per column:")
    print(df.isnull().sum())