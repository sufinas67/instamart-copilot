"""src/run_sql.py -- Executes analytical SQL queries Q1 to Q7 from sql/queries.sql against data/instamart.db."""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import re
import sqlite3
import pandas as pd

def run_queries():
    db_path = os.path.join("data", "instamart.db")
    conn = sqlite3.connect(db_path)
    
    with open(os.path.join("sql", "queries.sql"), "r", encoding="utf-8-sig") as f:
        sql_text = f.read()
        
    lines = [line for line in sql_text.splitlines() if not line.strip().startswith("--")]
    clean_text = "\n".join(lines)
    clean_text = re.sub(r"--.*$", "", clean_text, flags=re.MULTILINE)
    
    queries = [q.strip() for q in clean_text.split(";") if q.strip()]
    
    print("=== EXECUTING SQL QUERIES (Q1 to Q7) ===")
    for i, q in enumerate(queries, 1):
        df = pd.read_sql_query(q, conn)
        print(f"\n--- Q{i} Output ({len(df)} rows) ---")
        if i == 1:
            print(f"Qualifying customer count: {len(df)}")
            assert len(df) == 30
        elif i == 2:
            rev_sum = round(df["revenue"].sum(), 2)
            max_r = df.loc[df["revenue"].idxmax()]
            print(f"Total revenue: INR {rev_sum:.2f}")
            print(f"Top category cell: {max_r['city']} / {max_r['category']} = INR {max_r['revenue']:.2f}")
            assert len(df) == 30 and rev_sum == 2998668.80
            assert max_r["city"] == "Bengaluru" and max_r["category"] == "Electronics" and max_r["revenue"] == 233909.45
        elif i == 3:
            print(f"Products priced above catalogue average: {len(df)}")
            assert len(df) == 11
        elif i == 4:
            print(f"Customers without any completed orders: {len(df)}")
            assert len(df) == 25
        elif i == 5:
            avg_val = pd.read_sql_query("WITH s AS (SELECT customer_id, SUM(order_amount) as t FROM orders WHERE status='completed' GROUP BY customer_id) SELECT AVG(t) as a FROM s", conn).iloc[0]["a"]
            print(f"Customers with spend above average ({avg_val:.2f}): {len(df)}")
            assert len(df) == 550 and round(avg_val, 2) == 2552.06
        elif i == 6:
            diff = df[df["rank_val"] != df["row_num"]]
            print(f"Rows where RANK() != ROW_NUMBER(): {len(diff)}")
            top = df[df["rank_val"] == 1].sort_values("city")
            print("Top spender per city:")
            print(top[["city", "customer_id", "total_spend"]].to_string(index=False))
            assert len(diff) == 8
        elif i == 7:
            cum_rev = round(df.iloc[-1]["cumulative_revenue"], 2)
            print(f"Total days: {len(df)}, Final cumulative revenue: INR {cum_rev:.2f}")
            assert len(df) == 183 and cum_rev == 2998668.80
            
    conn.close()
    print("\nALL SQL QUERIES MATCHED EXACT REFERENCE CHECKPOINTS!")

if __name__ == "__main__":
    run_queries()
