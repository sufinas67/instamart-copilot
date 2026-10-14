"""src/part_b/rfm_clustering.py -- RFM feature calculation, K-Means clustering, PCA, and customer segmentation."""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

def run_rfm_clustering():
    CLEAN = os.path.join("data", "clean")
    OUTPUTS = "outputs"
    os.makedirs(OUTPUTS, exist_ok=True)
    
    # 1. Load clean orders
    orders = pd.read_csv(os.path.join(CLEAN, "orders.csv"))
    comp = orders[orders["status"] == "completed"].copy()
    comp["order_ts"] = pd.to_datetime(comp["order_ts"])
    
    # Snapshot moment
    snapshot = pd.Timestamp("2025-10-01 00:00:00")
    
    # RFM metrics per customer
    rfm = comp.groupby("customer_id").agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum")
    ).reset_index()
    
    rfm["recency"] = (snapshot - rfm["last_order_ts"]).dt.days
    rfm["monetary"] = rfm["monetary"].round(2)
    
    print("=== RFM SUMMARY AS OF 2025-10-01 ===")
    print(f"Customer Count: {len(rfm)}")
    print(f"Recency: median={rfm['recency'].median():.0f}, min={rfm['recency'].min()}, max={rfm['recency'].max()}")
    print(f"Frequency: mean={rfm['frequency'].mean():.2f}, max={rfm['frequency'].max()}")
    print(f"Monetary: mean={rfm['monetary'].mean():.2f}, max={rfm['monetary'].max():.2f}")
    
    # Assert reference checkpoints for RFM
    assert len(rfm) == 1175, f"Expected 1175 RFM customers, got {len(rfm)}"
    assert rfm["recency"].median() == 19
    assert rfm["recency"].min() == 0 and rfm["recency"].max() == 180
    assert round(rfm["frequency"].mean(), 2) == 9.46 and rfm["frequency"].max() == 33
    assert round(rfm["monetary"].mean(), 2) == 2552.06 and round(rfm["monetary"].max(), 2) == 14766.55
    
    # 2. Standardize RFM features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])
    
    # 3. K-Means clustering across K=2..8
    print("\n=== K-MEANS CLUSTERING (K=2 to 8) ===")
    kmeans_results = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        inertia = km.inertia_
        sil = silhouette_score(X_scaled, labels)
        kmeans_results.append({"K": k, "Inertia": round(inertia, 1), "Silhouette": round(sil, 3)})
    
    km_df = pd.DataFrame(kmeans_results)
    print(km_df.to_string(index=False))
    
    # 4. Fit chosen K=4
    chosen_k = 4
    km_final = KMeans(n_clusters=chosen_k, n_init=10, random_state=42)
    rfm["segment_id"] = km_final.fit_predict(X_scaled)
    
    # Profile clusters to identify Premium and name segments
    profile = rfm.groupby("segment_id").agg(
        size=("customer_id", "count"),
        mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"),
        mean_monetary=("monetary", "mean")
    ).round(2).reset_index()
    
    # Identify premium segment (highest mean monetary)
    prem_seg_id = profile.loc[profile["mean_monetary"].idxmax(), "segment_id"]
    
    # Business segment naming logic based on cluster profile
    def name_segment(row):
        sid = row["segment_id"]
        if sid == prem_seg_id:
            return "Premium Electronics Shoppers", True
        elif row["mean_frequency"] > 15:
            return "Loyal Grocery Enthusiasts", False
        elif row["mean_recency"] > 80:
            return "Hibernating At-Risk", False
        else:
            return "Occasional Value Seekers", False

    names_map = {}
    is_prem_map = {}
    for _, row in profile.iterrows():
        name, is_prem = name_segment(row)
        names_map[row["segment_id"]] = name
        is_prem_map[row["segment_id"]] = is_prem
        
    profile["segment_name"] = profile["segment_id"].map(names_map)
    profile["is_premium"] = profile["segment_id"].map(is_prem_map)
    
    print(f"\n=== SEGMENT PROFILES (K={chosen_k}) ===")
    print(profile.to_string(index=False))
    
    # Map back to rfm DataFrame
    rfm["segment_name"] = rfm["segment_id"].map(names_map)
    rfm["is_premium"] = rfm["segment_id"].map(is_prem_map)
    
    # 5. PCA Projection
    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    evr = pca.explained_variance_ratio_
    print(f"\n=== PCA EXPLAINED VARIANCE ===")
    print(f"PC1: {evr[0]:.3f}, PC2: {evr[1]:.3f}, Total: {evr.sum():.3f}")
    assert abs(evr[0] - 0.634) <= 0.02
    assert abs(evr[1] - 0.218) <= 0.02
    assert abs(evr.sum() - 0.852) <= 0.02
    
    # 6. Save outputs/customer_segments.csv
    out_cols = ["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]
    out_df = rfm[out_cols]
    out_path = os.path.join(OUTPUTS, "customer_segments.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved customer segments to {out_path} ({len(out_df)} rows).")

if __name__ == "__main__":
    run_rfm_clustering()
