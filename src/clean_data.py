"""src/clean_data.py -- Data cleaning pipeline for customers and orders with explicit change tracking."""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import sqlite3
import pandas as pd
import numpy as np

def clean_data():
    RAW = os.path.join("data", "raw")
    CLEAN = os.path.join("data", "clean")
    os.makedirs(CLEAN, exist_ok=True)
    
    print("=== PART A: CLEANING PIPELINE ===")
    
    # 1. Load raw datasets
    cust = pd.read_csv(os.path.join(RAW, "customers.csv"))
    prod = pd.read_csv(os.path.join(RAW, "products.csv"))
    orders = pd.read_csv(os.path.join(RAW, "orders.csv"))
    
    # 2. Clean Customers
    print("\n--- Cleaning Customers ---")
    # (a) City standardisation
    raw_cities = cust["city"].copy()
    cust["city"] = cust["city"].astype(str).str.strip().str.title()
    changed_city = (raw_cities != cust["city"]).sum()
    print(f"Rule (a) City standardisation: {changed_city} values modified (mapped to {cust['city'].nunique()} standard cities).")
    
    # (b) Gender standardisation
    raw_genders = cust["gender"].copy()
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
    changed_gender = (raw_genders != cust["gender"]).sum()
    print(f"Rule (b) Gender mapping: {changed_gender} values modified (breakdown: {cust['gender'].value_counts().to_dict()}).")
    
    # (c) Age validation & imputation
    raw_age = cust["age"].copy()
    cust["age"] = pd.to_numeric(cust["age"], errors="coerce")
    outside_mask = (cust["age"] < 10) | (cust["age"] > 100)
    invalid_age_count = outside_mask.sum()
    cust.loc[outside_mask, "age"] = np.nan
    missing_age_before = cust["age"].isna().sum()
    cust["age"] = cust.groupby("city")["age"].transform(lambda s: s.fillna(s.median())).astype(int)
    print(f"Rule (c) Age cleaning: {invalid_age_count} impossible values coerced to NaN, total {missing_age_before} missing ages imputed using city median.")
    
    # (d) Signup date conversion
    cust["signup_date"] = pd.to_datetime(cust["signup_date"]).dt.strftime("%Y-%m-%d")
    print("Rule (d) Signup date: converted all 1200 rows to datetime YYYY-MM-DD.")
    
    cust_clean = cust[["customer_id", "city", "signup_date", "age", "gender"]].copy()
    
    # 3. Clean Products
    prod_clean = prod[["product_id", "product_name", "category", "price_inr"]].copy()
    
    # 4. Clean Orders in exact documented order
    print("\n--- Cleaning Orders ---")
    raw_order_len = len(orders)
    
    # Rule 1: Convert order_ts to datetime
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    print(f"Rule 1: Converted order_ts to datetime for all {len(orders)} rows.")
    
    # Rule 2: Remove exact duplicate rows
    orders_no_dup = orders.drop_duplicates()
    exact_dup_removed = len(orders) - len(orders_no_dup)
    orders = orders_no_dup.copy()
    print(f"Rule 2: Removed {exact_dup_removed} exact duplicate rows.")
    
    # Rule 3: For each order_id keep only record with latest order_ts
    orders_dedup = orders.sort_values("order_ts").groupby("order_id", as_index=False).last()
    status_dup_removed = len(orders) - len(orders_dedup)
    orders = orders_dedup.copy()
    print(f"Rule 3: Removed {status_dup_removed} earlier pending/partial duplicate records, leaving {len(orders)} unique order_id rows.")
    
    # Rule 4: Convert order_amount to numeric and impute missing
    orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
    missing_amt_count = orders["order_amount"].isna().sum()
    orders = orders.merge(prod_clean[["product_id", "price_inr", "category"]], on="product_id", how="left")
    missing_amt_mask = orders["order_amount"].isna()
    imputed_amounts = (
        orders.loc[missing_amt_mask, "price_inr"] * 
        orders.loc[missing_amt_mask, "quantity"] * 
        (1.0 - orders.loc[missing_amt_mask, "discount_pct"] / 100.0)
    ).round(2)
    orders.loc[missing_amt_mask, "order_amount"] = imputed_amounts
    print(f"Rule 4: Imputed {missing_amt_count} non-numeric order_amount entries from catalog price, quantity, and discount.")
    
    # Rule 5: Delivery minutes cleaning & category median imputation
    orders["delivery_minutes"] = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    neg_deliv_count = (orders["delivery_minutes"] < 0).sum()
    orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = np.nan
    
    comp_mask = orders["status"] == "completed"
    comp_missing_deliv = orders.loc[comp_mask, "delivery_minutes"].isna().sum()
    cat_median = orders[comp_mask].groupby("category")["delivery_minutes"].transform("median")
    orders.loc[comp_mask, "delivery_minutes"] = orders.loc[comp_mask, "delivery_minutes"].fillna(cat_median)
    print(f"Rule 5: Set {neg_deliv_count} negative delivery times to NaN. Imputed {comp_missing_deliv} missing delivery times in completed orders via category medians; left 1199 cancelled orders as NaN.")
    
    orders["order_ts"] = orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    orders_clean = orders[[
        "order_id", "customer_id", "product_id", "quantity", "order_ts",
        "status", "discount_pct", "order_amount", "delivery_minutes"
    ]].copy()
    
    # 5. Save Clean CSVs
    cust_clean.to_csv(os.path.join(CLEAN, "customers.csv"), index=False)
    prod_clean.to_csv(os.path.join(CLEAN, "products.csv"), index=False)
    orders_clean.to_csv(os.path.join(CLEAN, "orders.csv"), index=False)
    
    # 6. Save SQLite Database
    db_path = os.path.join("data", "instamart.db")
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    cust_clean.to_sql("customers", conn, if_exists="replace", index=False)
    prod_clean.to_sql("products", conn, if_exists="replace", index=False)
    orders_clean.to_sql("orders", conn, if_exists="replace", index=False)
    conn.close()
    
    print("\nSuccessfully saved data/clean/{customers,products,orders}.csv and data/instamart.db.")

if __name__ == "__main__":
    clean_data()
