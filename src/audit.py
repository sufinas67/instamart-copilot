"""src/part_a/audit.py -- Audits raw data files against project specifications."""
import os
import pandas as pd
import numpy as np

def run_audit():
    RAW = os.path.join("data", "raw")
    
    customers_df = pd.read_csv(os.path.join(RAW, "customers.csv"))
    products_df = pd.read_csv(os.path.join(RAW, "products.csv"))
    orders_df = pd.read_csv(os.path.join(RAW, "orders.csv"))
    tickets_df = pd.read_csv(os.path.join(RAW, "tickets.csv"), keep_default_na=False)
    
    print("=== RAW DATA SHAPES & DTYPES ===")
    for name, df in [("customers", customers_df), ("products", products_df), 
                     ("orders", orders_df), ("tickets", tickets_df)]:
        print(f"\nTable: {name}, Shape: {df.shape}")
        missing = df.isna().sum()
        pct = (missing / len(df)) * 100
        summary = pd.DataFrame({"dtype": df.dtypes, "missing_count": missing, "missing_pct": pct})
        print(summary)

    # Specific audit checkpoints
    raw_orders_rows = len(orders_df)
    exact_duplicates = orders_df.duplicated().sum()
    pending_rows = (orders_df["status"] == "pending").sum()
    distinct_order_ids = orders_df["order_id"].nunique()
    
    # In raw orders.csv, 120 rows have order_amount as "error" (67) or "N/A" (53)
    numeric_order_amt = pd.to_numeric(orders_df["order_amount"], errors="coerce")
    non_numeric_amounts = int(numeric_order_amt.isna().sum())
    
    numeric_deliv = pd.to_numeric(orders_df["delivery_minutes"], errors="coerce")
    negative_delivery = int((numeric_deliv < 0).sum())
    blank_deliv_count = int(orders_df["delivery_minutes"].isna().sum())
    
    # customers.csv
    blank_age = int(customers_df["age"].isna().sum())
    numeric_age = pd.to_numeric(customers_df["age"], errors="coerce")
    outside_10_100 = int(((numeric_age < 10) | (numeric_age > 100)).sum())
    
    distinct_city_spellings = int(customers_df["city"].nunique())
    
    non_blank_gender = customers_df["gender"].dropna()
    non_blank_gender = non_blank_gender[non_blank_gender.astype(str).str.strip() != ""]
    distinct_gender_spellings = int(non_blank_gender.nunique())
    blank_gender = int(len(customers_df) - len(non_blank_gender))
    
    print("\n=== SPECIFIC AUDIT FIGURES ===")
    metrics = [
        ("raw orders.csv rows", raw_orders_rows, 13084),
        ("number of exact duplicate rows", exact_duplicates, 119),
        ("number of rows whose status is pending", pending_rows, 656),
        ("number of distinct order_id values", distinct_order_ids, 12309),
        ("number of order_amount entries that cannot be converted to a number", non_numeric_amounts, 120),
        ("number of negative delivery_minutes", negative_delivery, 106),
        ("number of blank delivery_minutes", blank_deliv_count, 2541),
        ("in customers.csv: number of blank age", blank_age, 61),
        ("number of age values outside 10-100", outside_10_100, 6),
        ("number of distinct raw city spellings", distinct_city_spellings, 24),
        ("number of distinct non-blank raw gender spellings", distinct_gender_spellings, 10),
        ("number of blank gender", blank_gender, 30)
    ]
    
    audit_table = pd.DataFrame(metrics, columns=["Audit Metric", "Measured Value", "Reference Value"])
    audit_table["Match"] = audit_table["Measured Value"] == audit_table["Reference Value"]
    print(audit_table.to_string(index=False))
    
    assert audit_table["Match"].all(), "Some audit checkpoints did not match reference values!"
    print("\nALL 12 AUDIT CHECKPOINTS MATCH EXACTLY!")

if __name__ == "__main__":
    run_audit()
