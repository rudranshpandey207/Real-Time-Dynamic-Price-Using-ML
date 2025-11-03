# utils.py here we make and save the model
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from scraper import get_competitor_price
from forecast import generate_sales_data, forecast_sales

MODEL_PATH = "model/pricing_model.pkl"

def simulate_inventory_and_demand():
    inventory = np.random.randint(10, 200)
    # Use simulated historical sales to forecast one day ahead
    hist = generate_sales_data(days=90)
    fc = forecast_sales(hist, future_days=1)
    demand = int(fc["yhat"].iloc[-1])
    return inventory, demand

def train_and_save_model():
    # ---- Prepare synthetic training data ----
    rng = np.random.RandomState(42)
    n = 200
    competitor = rng.uniform(200, 800, size=n)
    demand = rng.uniform(20, 100, size=n)
    inventory = rng.uniform(10, 300, size=n)
    # Assume optimal price = competitor * (1 + small random margin) adjusted for demand/inventory
    optimal = competitor * (1 + (demand / (inventory + demand)) * 0.1)
    df = pd.DataFrame({
        "competitor_price": competitor,
        "forecast_demand": demand,
        "inventory": inventory,
        "optimal_price": optimal
    })

    # ---- Train model ----
    X = df[["competitor_price", "forecast_demand", "inventory"]]
    y = df["optimal_price"]
    model = LinearRegression()
    model.fit(X, y)

    # ---- Save model ----
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model trained & saved to {MODEL_PATH}")

def load_model():
    import pickle
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    train_and_save_model()
