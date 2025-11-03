import os
import re
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from serpapi import GoogleSearch
import time
import random

# Load .env for SERPAPI_API_KEY
# Load .env for SERPAPI_API_KEY
# --- Robust path to .env file ---
from pathlib import Path
this_dir = Path(__file__).parent
env_path = this_dir / ".env"
load_dotenv(dotenv_path=env_path)
# --- End of fix ---

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# --- Add this line for debugging ---
if SERPAPI_API_KEY is None:
    print(" ERROR: SERPAPI_API_KEY not found. Check your .env file and path.")
else:
    print("SERPAPI_API_KEY loaded successfully.")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/115.0.0.0 Safari/537.36"
    )
}

def get_price_from_serpapi(product_name: str) -> float:
    params = {
        "engine": "google",
        "q": product_name + " site:flipkart.com OR site:amazon.in",
        "hl": "en",
        "gl": "in",
        "api_key": SERPAPI_API_KEY
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    prices = []

    # Check shopping_results
    for result in results.get("shopping_results", []):
        price = result.get("price")
        if price:
            price_digits = ''.join(c for c in price if c.isdigit() or c == '.')
            try:
                prices.append(float(price_digits))
            except:
                continue

    # Check organic results if no shopping results
    for result in results.get("organic_results", []):
        snippet = result.get("snippet", "")
        match = re.search(r"₹\s?([\d,]+)", snippet)
        if match:
            price_text = match.group(1).replace(",", "")
            prices.append(float(price_text))

    if prices:
        return round(sum(prices[:10]) / len(prices[:10]), 2)
    else:
        raise ValueError("No prices found via SerpAPI")

def get_price_from_flipkart(product_name: str) -> float:
    query = product_name.strip().replace(" ", "+")
    url = f"https://www.flipkart.com/search?q={query}"

    time.sleep(random.uniform(1, 2.5))
    response = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")
    prices = []

    # Class for product prices
    for tag in soup.find_all("div", class_="_30jeq3"):
        price_text = tag.get_text()
        price_digits = re.sub(r"[^\d]", "", price_text)
        if price_digits.isdigit():
            prices.append(float(price_digits))

    if prices:
        return round(sum(prices[:10]) / len(prices[:10]), 2)
    else:
        raise ValueError("No prices found on Flipkart")

def get_competitor_price(product_name: str) -> float:
    try:
        print(f" Trying SerpAPI for: {product_name}")
        return get_price_from_serpapi(product_name)
    except Exception as e1:
        print(f"⚠SerpAPI failed: {e1}")
        try:
            print(f" Trying Flipkart fallback for: {product_name}")
            return get_price_from_flipkart(product_name)
        except Exception as e2:
            print(f"️ Flipkart fallback failed: {e2}")
            print(f" Returning mock price for: {product_name}")
            return get_mock_price(product_name)

def get_mock_price(product_name: str) -> float:
    mock_prices = {
        "mouse": 599,
        "keyboard": 899,
        "laptop": 45000,
        "headphones": 1499,
        "bag": 499,
    }
    return mock_prices.get(product_name.lower(), 999.0)

# Test it
if __name__ == "__main__":
    test_products = ["mouse", "keyboard", "wireless earbuds", "backpack", "random item"]
    for product in test_products:
        try:
            price = get_competitor_price(product)
            print(f"✅ '{product}': ₹{price}")
        except Exception as e:
            print(f"❌ '{product}': {e}")
