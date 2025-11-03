import pandas as pd
import numpy as np
from pathlib import Path

def generate_realistic_pricing_data(n=1000, seed=42):
    """
    Generate synthetic pricing data with REALISTIC relationships
    """
    np.random.seed(seed)
    
    categories = ['Electronics', 'Fashion', 'Home', 'Sports', 'Books', 'Toys']
    brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE']
    
    # Category base prices and elasticity
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
        
        # Rating (affects premium)
        rating = np.clip(np.random.normal(4.0, 0.8), 1, 5)
        
        # Demand metrics
        page_views = int(np.random.uniform(100, 10000))
        conversion_rate = np.clip(np.random.normal(0.08, 0.04), 0.01, 0.25)
        units_sold_last_30d = int(page_views * conversion_rate * np.random.uniform(0.8, 1.2))
        
        # Forecast demand (similar to units sold with trend)
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
        
        # 2. Demand factor: high demand → increase price
        demand_ratio = forecast_demand / (page_views * conversion_rate + 1)
        demand_adjustment = (demand_ratio - 1) * 0.15  # ±15% based on demand
        optimal_price *= (1 + demand_adjustment)
        
        # 3. Inventory factor: low inventory → increase price (scarcity)
        inventory_ratio = inventory_on_hand / (forecast_demand + 1)
        if inventory_ratio < 1:  # Running out of stock
            scarcity_premium = (1 - inventory_ratio) * 0.20  # Up to 20% premium
            optimal_price *= (1 + scarcity_premium)
        else:  # Excess inventory
            excess_discount = min((inventory_ratio - 1) * 0.10, 0.15)  # Up to 15% discount
            optimal_price *= (1 - excess_discount)
        
        # 4. Quality factor: high rating → premium pricing
        rating_premium = (rating - 3.0) * 0.05  # ±10% based on rating
        optimal_price *= (1 + rating_premium)
        
        # 5. Ensure profit margin (must be above cost)
        min_price = cost_price * 1.10  # At least 10% profit
        optimal_price = max(optimal_price, min_price)
        
        # 6. Don't price too high above competitor
        max_price = competitor_price * 1.30
        optimal_price = min(optimal_price, max_price)
        
        # Add some noise
        optimal_price *= np.random.uniform(0.98, 1.02)
        
        # Store row
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
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    print("Generating realistic pricing dataset...")
    df = generate_realistic_pricing_data(n=1000, seed=42)
    
    # Save
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "realistic_pricing_dataset.csv"
    df.to_csv(out_path, index=False)
    
    print(f"✅ Generated {len(df)} rows")
    print(f"✅ Saved to: {out_path}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nSummary statistics:")
    print(df[['competitor_price', 'forecast_demand', 'inventory_on_hand', 'optimal_price']].describe())