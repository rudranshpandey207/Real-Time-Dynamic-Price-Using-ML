# forecast.py
import pandas as pd
import numpy as np
try:
    from prophet import Prophet
except Exception:
    try:
        from prophet import Prophet
    except Exception as e:
        raise ImportError(
            "Prophet package is not installed; install it with 'pip install prophet' "
            "or for older environments 'pip install fbprophet'."
        ) from e

def generate_sales_data(days: int = 180) -> pd.DataFrame:
    date_range = pd.date_range(end=pd.Timestamp.today(), periods=days)
    sales = (
        20
        + 10 * np.sin(np.linspace(0, 3 * np.pi, days))
        + np.random.randn(days) * 3
    ).clip(min=0)
    return pd.DataFrame({"ds": date_range, "y": sales})

def forecast_sales(df: pd.DataFrame, future_days: int = 30) -> pd.DataFrame:
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=future_days)
    forecast = m.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(future_days)

if __name__ == "__main__":
    df_hist = generate_sales_data()
    fc = forecast_sales(df_hist)
    print("Next 5 days forecast:")
    print(fc.head())
