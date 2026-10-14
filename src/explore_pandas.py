"""src/explore_pandas.py -- Exploratory data analysis, pivot table cross-check, outlier detection, and correlation analysis."""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import pandas as pd
import numpy as np

def run_exploration():
    cust = pd.read_csv("data/clean/customers.csv")
    prod = pd.read_csv("data/clean/products.csv")
    orders = pd.read_csv("data/clean/orders.csv")
    
    comp = orders[orders["status"] == "completed"].merge(cust, on="customer_id").merge(prod, on="product_id")
    
    print("=== (a) PIVOT TABLE CROSS-CHECK WITH SQL Q2 ===")
    pt = pd.pivot_table(comp, values="order_amount", index="city", columns="category", aggfunc="sum", margins=True)
    grand_total = round(pt.loc["All", "All"], 2)
    print(f"Pivot table grand total: INR {grand_total:.2f}")
    assert grand_total == 2998668.80
    
    print("\n--- Per-City Completed Spend Named Aggregation ---")
    city_agg = comp.groupby("city").agg(
        sum_spend=("order_amount", "sum"),
        mean_spend=("order_amount", "mean"),
        order_count=("order_amount", "count")
    ).round(2)
    print(city_agg)
    assert city_agg.loc["Bengaluru", "sum_spend"] == 523643.40
    assert city_agg.loc["Bengaluru", "mean_spend"] == 276.47
    assert city_agg.loc["Bengaluru", "order_count"] == 1894
    assert city_agg.loc["Mumbai", "sum_spend"] == 479058.50
    assert city_agg.loc["Mumbai", "mean_spend"] == 297.00
    assert city_agg.loc["Mumbai", "order_count"] == 1613
    
    print("\n=== (b) ORDER AMOUNT DISTRIBUTION AND OUTLIERS ===")
    amt = comp["order_amount"]
    mean_val = amt.mean()
    med_val = amt.median()
    skew_val = amt.skew()
    q1 = amt.quantile(0.25)
    q3 = amt.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outliers_low = (amt < lower_fence).sum()
    outliers_high = (amt > upper_fence).sum()
    
    print(f"Mean: INR {mean_val:.2f}, Median: INR {med_val:.2f}, Skewness: {skew_val:.2f}")
    print(f"Q1: INR {q1:.2f}, Q3: INR {q3:.2f}, IQR: INR {iqr:.2f}")
    print(f"Lower Fence: INR {lower_fence:.2f}, Upper Fence: INR {upper_fence:.2f}")
    print(f"Outliers Below Lower Fence: {outliers_low}")
    print(f"Outliers Above Upper Fence: {outliers_high}")
    
    assert round(mean_val, 2) == 269.91
    assert round(med_val, 2) == 135.00
    assert round(skew_val, 2) == 4.61
    assert round(q1, 2) == 64.80
    assert round(q3, 2) == 297.00
    assert round(upper_fence, 2) == 645.30
    assert outliers_low == 0
    assert outliers_high == 897
    
    print("\n=== CORRELATION MATRIX (COMPLETED ORDERS) ===")
    corr = comp[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr().round(3)
    print(corr)
    
    print("\n=== FOUR-QUESTION ANALYSIS (<= 200 WORDS) ===")
    analysis_text = (
        "1. Data Domain: This dataset captures transactional grocery and electronics deliveries across six Indian metro branches, tracking item quantities, basket spend, discounts, and delivery speeds.\n"
        "2. Distributions & Statistics: Completed order spend has a mean of INR 269.91 versus a median of INR 135.00, demonstrating strong right-skewness (skewness = 4.61). The bulk of transactions represent everyday fast-moving staples priced under INR 300, while occasional electronics purchases extend the tail.\n"
        "3. Outliers & Anomalies: Setting 1.5xIQR fences (Q1: INR 64.80, Q3: INR 297.00) yields an upper threshold of INR 645.30. Exactly 897 completed orders lie above this fence, representing premium gadget orders rather than erroneous data entry; zero negative outliers exist.\n"
        "4. Variable Relationships: Across completed orders, linear associations are weak. The largest correlation in absolute value is between order_amount and discount_pct at -0.095, followed by order_amount and quantity at +0.087, while delivery_minutes is essentially uncorrelated (-0.003). Crucially, empirical correlation of this small magnitude does not establish causation."
    )
    print(analysis_text)
    word_count = len(analysis_text.split())
    print(f"\nWord count: {word_count} words (limit: <= 200 words).")

if __name__ == "__main__":
    run_exploration()
