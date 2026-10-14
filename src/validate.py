"""src/part_a/validate.py -- Validates clean dataset checkpoints from tasks 3 and 4."""
import os
import sqlite3
import pandas as pd
import numpy as np

def validate_checkpoints():
    CLEAN = os.path.join("data", "clean")
    cust = pd.read_csv(os.path.join(CLEAN, "customers.csv"))
    orders = pd.read_csv(os.path.join(CLEAN, "orders.csv"))
    prod = pd.read_csv(os.path.join(CLEAN, "products.csv"))
    
    # Checkpoint 1: Customers city distribution
    expected_cities = {"Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}
    assert cust["city"].value_counts().to_dict() == expected_cities, f"City distribution mismatch: {cust['city'].value_counts().to_dict()}"
    
    # Checkpoint 2: Customers gender distribution
    expected_genders = {"female": 610, "male": 560, "unknown": 30}
    assert cust["gender"].value_counts().to_dict() == expected_genders, f"Gender distribution mismatch: {cust['gender'].value_counts().to_dict()}"
    
    # Checkpoint 3: Customers age validity
    assert cust["age"].isna().sum() == 0, "Missing ages found!"
    assert cust["age"].between(18, 60).all(), f"Ages outside 18-60 found: min={cust['age'].min()}, max={cust['age'].max()}"
    
    # Checkpoint 4: Total clean orders row count and uniqueness
    assert len(orders) == 12309, f"Expected 12309 order rows, got {len(orders)}"
    assert orders["order_id"].nunique() == 12309, f"Duplicate order_ids exist!"
    
    # Checkpoint 5: Order statuses
    assert (orders["status"] == "pending").sum() == 0, "Found pending orders!"
    assert (orders["status"] == "completed").sum() == 11110, f"Expected 11110 completed orders, got {(orders['status'] == 'completed').sum()}"
    assert (orders["status"] == "cancelled").sum() == 1199, f"Expected 1199 cancelled orders, got {(orders['status'] == 'cancelled').sum()}"
    
    # Checkpoint 6: Missing order_amount
    assert orders["order_amount"].isna().sum() == 0, f"Missing order_amount: {orders['order_amount'].isna().sum()}"
    
    # Checkpoint 7: Missing delivery_minutes partitioned by status
    comp_orders = orders[orders["status"] == "completed"]
    canc_orders = orders[orders["status"] == "cancelled"]
    assert comp_orders["delivery_minutes"].isna().sum() == 0, f"Missing delivery_minutes in completed: {comp_orders['delivery_minutes'].isna().sum()}"
    assert canc_orders["delivery_minutes"].isna().sum() == 1199, f"Expected 1199 missing delivery_minutes in cancelled, got {canc_orders['delivery_minutes'].isna().sum()}"
    
    # Checkpoint 8: Sum of completed order_amount
    comp_sum = round(float(comp_orders["order_amount"].sum()), 2)
    assert comp_sum == 2998668.80, f"Expected completed order sum INR 2998668.80, got {comp_sum}"
    
    # Checkpoint 9: Verify database tables
    db_path = os.path.join("data", "instamart.db")
    assert os.path.exists(db_path), "Database data/instamart.db does not exist!"
    conn = sqlite3.connect(db_path)
    db_cust = pd.read_sql_query("SELECT COUNT(*) AS c FROM customers", conn).iloc[0]["c"]
    db_prod = pd.read_sql_query("SELECT COUNT(*) AS c FROM products", conn).iloc[0]["c"]
    db_ord = pd.read_sql_query("SELECT COUNT(*) AS c FROM orders", conn).iloc[0]["c"]
    conn.close()
    assert db_cust == 1200, f"DB customers count: {db_cust}"
    assert db_prod == 40, f"DB products count: {db_prod}"
    assert db_ord == 12309, f"DB orders count: {db_ord}"
    
    print("ALL CHECKPOINTS PASSED")

if __name__ == "__main__":
    validate_checkpoints()
