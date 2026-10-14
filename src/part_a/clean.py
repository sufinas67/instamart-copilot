"""src/part_a/clean.py -- Cleaning pipeline for raw customers, products, and orders data."""
import os
import sqlite3
import pandas as pd
import numpy as np

def clean_data():
    RAW = os.path.join("data", "raw")
    CLEAN = os.path.join("data", "clean")
    os.makedirs(CLEAN, exist_ok=True)
    
    # 1. Load raw data
    cust = pd.read_csv(os.path.join(RAW, "customers.csv"))
    prod = pd.read_csv(os.path.join(RAW, "products.csv"))
    orders = pd.read_csv(os.path.join(RAW, "orders.csv"))
    
    # 2. Clean customers
    # (a) Standardise city: strip leading/trailing spaces and apply title case
    cust["city"] = cust["city"].astype(str).str.strip().str.title()
    
    # (b) Map gender spelling to male, female, or unknown
    def map_gender(val):
        if pd.isna(val) or str(val).strip() == "":
            return "unknown"
        g = str(val).strip().lower()
        if g in ["m", "male"]:
            return "male"
        if g in ["f", "female"]:
            return "female"
        return "unknown"
    cust["gender"] = cust["gender"].apply(map_gender)
    
    # (c) Treat ages outside 10-100 as missing, then impute with city median age
    cust["age"] = pd.to_numeric(cust["age"], errors="coerce")
    cust.loc[(cust["age"] < 10) | (cust["age"] > 100), "age"] = np.nan
    cust["age"] = cust.groupby("city")["age"].transform(lambda s: s.fillna(s.median())).astype(int)
    
    # (d) Convert signup_date to datetime
    cust["signup_date"] = pd.to_datetime(cust["signup_date"]).dt.strftime("%Y-%m-%d")
    
    # Ensure correct column ordering
    cust_clean = cust[["customer_id", "city", "signup_date", "age", "gender"]].copy()
    
    # 3. Clean products (copy with types)
    prod_clean = prod[["product_id", "product_name", "category", "price_inr"]].copy()
    
    # 4. Clean orders in exact specified order:
    # Rule 1: Convert order_ts to a datetime
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    
    # Rule 2: Remove exact duplicate rows
    orders = orders.drop_duplicates()
    
    # Rule 3: For each order_id keep only the record with the latest order_ts
    orders = orders.sort_values("order_ts").groupby("order_id", as_index=False).last()
    
    # Rule 4: Convert order_amount to numeric, impute unparseable from catalog price
    orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
    orders = orders.merge(prod_clean[["product_id", "price_inr", "category"]], on="product_id", how="left")
    missing_amt_mask = orders["order_amount"].isna()
    imputed_amounts = (
        orders.loc[missing_amt_mask, "price_inr"] * 
        orders.loc[missing_amt_mask, "quantity"] * 
        (1.0 - orders.loc[missing_amt_mask, "discount_pct"] / 100.0)
    ).round(2)
    orders.loc[missing_amt_mask, "order_amount"] = imputed_amounts
    
    # Rule 5: Set negative delivery_minutes to missing.
    orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = np.nan
    # Impute missing delivery_minutes for completed orders with category median
    completed_mask = orders["status"] == "completed"
    cat_median_delivery = orders[completed_mask].groupby("category")["delivery_minutes"].transform("median")
    orders.loc[completed_mask, "delivery_minutes"] = orders.loc[completed_mask, "delivery_minutes"].fillna(cat_median_delivery)
    # Cancelled orders remain NaN for delivery_minutes
    
    # Convert order_ts to standard text format YYYY-MM-DD HH:MM:SS
    orders["order_ts"] = orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    orders_clean = orders[[
        "order_id", "customer_id", "product_id", "quantity", "order_ts",
        "status", "discount_pct", "order_amount", "delivery_minutes"
    ]].copy()
    
    # 5. Save clean tables as CSV files
    cust_clean.to_csv(os.path.join(CLEAN, "customers.csv"), index=False)
    prod_clean.to_csv(os.path.join(CLEAN, "products.csv"), index=False)
    orders_clean.to_csv(os.path.join(CLEAN, "orders.csv"), index=False)
    
    # 6. Save to SQLite database data/instamart.db
    db_path = os.path.join("data", "instamart.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    cust_clean.to_sql("customers", conn, if_exists="replace", index=False)
    prod_clean.to_sql("products", conn, if_exists="replace", index=False)
    orders_clean.to_sql("orders", conn, if_exists="replace", index=False)
    conn.close()
    
    print("Clean datasets saved to data/clean/ and data/instamart.db successfully.")

if __name__ == "__main__":
    clean_data()
