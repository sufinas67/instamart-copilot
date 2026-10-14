"""src/part_a/analyze.py -- SQL queries execution and pandas exploratory data analysis."""
import os
import re
import sqlite3
import pandas as pd
import numpy as np

def run_analysis():
    db_path = os.path.join("data", "instamart.db")
    conn = sqlite3.connect(db_path)
    
    with open(os.path.join("sql", "queries.sql"), "r", encoding="utf-8-sig") as f:
        sql_text = f.read()
    
    # Strip comments first before splitting by semicolon without using dollar signs
    clean_lines = [line.split("--")[0].rstrip() for line in sql_text.splitlines()]
    clean_text = "\n".join(clean_lines)
    
    queries = [q.strip() for q in clean_text.split(";") if q.strip()]
            
    print(f"=== EXECUTING {len(queries)} SQL QUERIES ===")
    for i, q in enumerate(queries, 1):
        df = pd.read_sql_query(q, conn)
        print(f"\n--- Q{i} Results ({len(df)} rows) ---")
        if i == 1:
            print(f"Customer count: {len(df)}")
        elif i == 2:
            print(f"Revenue sum: INR {df['revenue'].sum():.2f}")
            max_r = df.loc[df['revenue'].idxmax()]
            print(f"Largest cell: {max_r['city']} / {max_r['category']} = INR {max_r['revenue']:.2f}")
        elif i == 3:
            print(f"Products above average: {len(df)}")
        elif i == 4:
            print(f"Customers with 0 completed orders: {len(df)}")
        elif i == 5:
            print(f"Customers spend > average: {len(df)}")
        elif i == 6:
            diff = df[df['rank_val'] != df['row_num']]
            print(f"Rows where RANK() != ROW_NUMBER(): {len(diff)}")
            top = df[df['rank_val'] == 1].sort_values('city')
            print("Top customer per city:")
            print(top[['city', 'customer_id', 'total_spend']].to_string(index=False))
        elif i == 7:
            print(f"Daily rows count: {len(df)}")
            print(f"Final cumulative revenue: INR {df.iloc[-1]['cumulative_revenue']:.2f}")

    conn.close()
    
    print("\n=== PANDAS CROSS-CHECK AND EDA ===")
    cust = pd.read_csv("data/clean/customers.csv")
    prod = pd.read_csv("data/clean/products.csv")
    orders = pd.read_csv("data/clean/orders.csv")
    comp = orders[orders["status"] == "completed"].merge(cust, on="customer_id").merge(prod, on="product_id")
    
    # (a) Pivot table cross-check with Q2
    pt = pd.pivot_table(comp, values="order_amount", index="city", columns="category", aggfunc="sum", margins=True)
    print(f"Pivot table grand total: INR {pt.loc['All', 'All']:.2f}")
    
    # City aggregation
    city_agg = comp.groupby("city").agg(
        sum_spend=("order_amount", "sum"),
        mean_spend=("order_amount", "mean"),
        order_count=("order_amount", "count")
    ).round(2)
    print("\nCity Aggregation:")
    print(city_agg)
    
    # (b) Order amount distribution stats & outlier fences
    amt = comp["order_amount"]
    q1 = amt.quantile(0.25)
    q3 = amt.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    
    print("\nOrder Amount Distribution:")
    print(f"Mean: {amt.mean():.2f}, Median: {amt.median():.2f}, Skewness: {amt.skew():.2f}")
    print(f"Q1: {q1:.2f}, Q3: {q3:.2f}, IQR: {iqr:.2f}")
    print(f"Lower Fence: {lower_fence:.2f}, Upper Fence: {upper_fence:.2f}")
    print(f"Outliers Below Lower Fence: {(amt < lower_fence).sum()}")
    print(f"Outliers Above Upper Fence: {(amt > upper_fence).sum()}")
    
    corr = comp[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr().round(3)
    print("\nCorrelation Matrix:")
    print(corr)

if __name__ == "__main__":
    run_analysis()
