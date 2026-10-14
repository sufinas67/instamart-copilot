"""src/part_b/lapse_modeling.py -- Leakage-free lapse prediction modeling, tuning, evaluation, and retention list."""
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

def run_lapse_modeling():
    CLEAN = os.path.join("data", "clean")
    OUTPUTS = "outputs"
    os.makedirs(OUTPUTS, exist_ok=True)
    
    # 1. Load clean datasets
    orders = pd.read_csv(os.path.join(CLEAN, "orders.csv"))
    cust = pd.read_csv(os.path.join(CLEAN, "customers.csv"))
    prod = pd.read_csv(os.path.join(CLEAN, "products.csv"))
    segments = pd.read_csv(os.path.join(OUTPUTS, "customer_segments.csv"))
    
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    cust["signup_date"] = pd.to_datetime(cust["signup_date"])
    cutoff = pd.Timestamp("2025-09-01 00:00:00")
    
    # 2. Partition orders strictly before and after cutoff
    before = orders[orders["order_ts"] < cutoff].copy()
    after = orders[(orders["order_ts"] >= cutoff) & (orders["order_ts"] < pd.Timestamp("2025-10-01 00:00:00"))].copy()
    
    comp_before = before[before["status"] == "completed"].copy()
    comp_after = after[after["status"] == "completed"].copy()
    
    # Cohort: customers with at least one completed order before cutoff
    cohort_cids = sorted(comp_before["customer_id"].unique())
    assert len(cohort_cids) == 1156, f"Expected 1156 customers in cohort, got {len(cohort_cids)}"
    
    # Target label: lapsed = 1 if no completed order in September 2025, else 0
    cids_in_sept = set(comp_after["customer_id"].unique())
    labels = {cid: (0 if cid in cids_in_sept else 1) for cid in cohort_cids}
    y_series = pd.Series([labels[cid] for cid in cohort_cids], index=cohort_cids)
    
    lapsed_count = int((y_series == 1).sum())
    lapsed_pct = (lapsed_count / len(y_series)) * 100
    print("=== COHORT & TARGET SUMMARY ===")
    print(f"Cohort Size: {len(cohort_cids)}")
    print(f"Lapsed Count: {lapsed_count} ({lapsed_pct:.2f}%)")
    assert lapsed_count == 443, f"Expected 443 lapsed customers, got {lapsed_count}"
    
    # 3. Feature engineering strictly before cutoff
    comp_before = comp_before.merge(prod[["product_id", "category"]], on="product_id", how="left")
    
    rfm_before = comp_before.groupby("customer_id").agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
        avg_discount_pct=("discount_pct", "mean"),
        avg_delivery_minutes=("delivery_minutes", "mean")
    ).reset_index()
    rfm_before["recency"] = (cutoff - rfm_before["last_order_ts"]).dt.days
    
    # Cancellation rate
    all_before = before[before["status"].isin(["completed", "cancelled"])]
    canc = all_before.groupby("customer_id")["status"].agg(
        total_orders="count",
        cancelled_orders=lambda s: (s == "cancelled").sum()
    ).reset_index()
    canc["cancellation_rate"] = (canc["cancelled_orders"] / canc["total_orders"]).fillna(0.0)
    
    # Favourite category
    fav_cat = comp_before.groupby(["customer_id", "category"]).size().reset_index(name="count")
    fav_cat = fav_cat.sort_values(["customer_id", "count"], ascending=[True, False]).groupby("customer_id").first().reset_index()[["customer_id", "category"]].rename(columns={"category": "fav_category"})
    
    # Assemble feature table
    feat = pd.DataFrame({"customer_id": cohort_cids})
    feat = feat.merge(cust, on="customer_id", how="left")
    feat = feat.merge(rfm_before, on="customer_id", how="left")
    feat = feat.merge(canc[["customer_id", "cancellation_rate"]], on="customer_id", how="left")
    feat = feat.merge(fav_cat, on="customer_id", how="left")
    feat["tenure_days"] = (cutoff - feat["signup_date"]).dt.days
    feat["cancellation_rate"] = feat["cancellation_rate"].fillna(0.0)
    feat["lapsed"] = feat["customer_id"].map(labels)
    
    # Confirm in code that no feature column contains missing values
    missing_counts = feat.isna().sum().to_dict()
    assert all(v == 0 for v in missing_counts.values()), f"Missing values found in features: {missing_counts}"
    
    num_cols = ["recency", "frequency", "monetary", "avg_discount_pct", "cancellation_rate", "avg_delivery_minutes", "age", "tenure_days"]
    cat_cols = ["city", "gender", "fav_category"]
    
    X = feat[["customer_id"] + num_cols + cat_cols]
    y = feat["lapsed"]
    
    # 4. Train / Test Split (80 / 20 stratified)
    X_train_full, X_test_full, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    train_cids = X_train_full["customer_id"].values
    test_cids = X_test_full["customer_id"].values
    
    X_train = X_train_full[num_cols + cat_cols]
    X_test = X_test_full[num_cols + cat_cols]
    
    print("\n=== TRAIN / TEST SPLIT ===")
    print(f"Train Rows: {len(X_train)}, Test Rows: {len(X_test)}")
    assert len(X_train) == 924 and len(X_test) == 232
    
    # 5. Leakage check on frequency feature
    freq_scaler = StandardScaler()
    freq_scaler.fit(X_train[["frequency"]])
    learned_mean_freq = freq_scaler.mean_[0]
    full_mean_freq = X["frequency"].mean()
    print(f"Learned mean frequency (train-only scaler): {learned_mean_freq:.4f}")
    print(f"Full-cohort mean frequency: {full_mean_freq:.4f}")
    print("Explanation: The learned mean and full-cohort mean differ because the training set is a random 80% sample; computing statistics strictly on training data prevents test distribution information from leaking into model transformations.")
    
    # 6. Baseline model evaluation
    y_pred_base = np.zeros(len(y_test))
    base_acc = accuracy_score(y_test, y_pred_base)
    base_f1 = f1_score(y_test, y_pred_base, zero_division=0)
    print(f"\nBaseline Model: Accuracy = {base_acc*100:.1f}%, Lapsed F1 = {base_f1:.3f}")
    
    # 7. Preprocessing Pipeline (fitted on train only)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), cat_cols)
        ]
    )
    
    # 8. Tune Logistic Regression
    pipe_lr = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])
    param_grid_lr = {"clf__C": [0.01, 0.1, 1, 10, 100]}
    grid_lr = GridSearchCV(pipe_lr, param_grid_lr, cv=5, scoring="f1")
    grid_lr.fit(X_train, y_train)
    best_lr = grid_lr.best_estimator_
    y_pred_lr = best_lr.predict(X_test)
    
    print("\n=== LOGISTIC REGRESSION RESULTS ===")
    print(f"Best Parameters: {grid_lr.best_params_}, CV F1: {grid_lr.best_score_:.3f}")
    print(f"Test Precision: {precision_score(y_test, y_pred_lr):.3f}")
    print(f"Test Recall: {recall_score(y_test, y_pred_lr):.3f}")
    print(f"Test F1: {f1_score(y_test, y_pred_lr):.3f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
    
    # 9. Tune Random Forest
    pipe_rf = Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    param_grid_rf = {
        "clf__max_depth": [3, 5, 8],
        "clf__min_samples_leaf": [5, 10, 20]
    }
    grid_rf = GridSearchCV(pipe_rf, param_grid_rf, cv=5, scoring="f1")
    grid_rf.fit(X_train, y_train)
    best_rf = grid_rf.best_estimator_
    y_pred_rf = best_rf.predict(X_test)
    
    print("\n=== RANDOM FOREST RESULTS ===")
    print(f"Best Parameters: {grid_rf.best_params_}, CV F1: {grid_rf.best_score_:.3f}")
    print(f"Test Precision: {precision_score(y_test, y_pred_rf):.3f}")
    print(f"Test Recall: {recall_score(y_test, y_pred_rf):.3f}")
    print(f"Test F1: {f1_score(y_test, y_pred_rf):.3f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
    assert f1_score(y_test, y_pred_rf) >= 0.55, "Selected model test F1 must be >= 0.55"
    
    # 10. Class-Balance Experiment
    train_full_df = pd.concat([X_train, y_train], axis=1)
    lapsed_rows = train_full_df[train_full_df["lapsed"] == 1]
    non_lapsed_rows = train_full_df[train_full_df["lapsed"] == 0]
    n_oversample = len(non_lapsed_rows) - len(lapsed_rows)
    oversampled_lapsed = lapsed_rows.sample(n=n_oversample, replace=True, random_state=42)
    balanced_train_df = pd.concat([train_full_df, oversampled_lapsed], axis=0)
    
    X_train_bal = balanced_train_df[num_cols + cat_cols]
    y_train_bal = balanced_train_df["lapsed"]
    print(f"\n=== CLASS-BALANCE EXPERIMENT (Oversampled Train) ===")
    print(f"Balanced Train Counts: {y_train_bal.value_counts().to_dict()}")
    
    pipe_lr_bal = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(C=grid_lr.best_params_["clf__C"], max_iter=1000, random_state=42))
    ])
    pipe_lr_bal.fit(X_train_bal, y_train_bal)
    y_pred_lr_bal = pipe_lr_bal.predict(X_test)
    print(f"Balanced LR Test Precision: {precision_score(y_test, y_pred_lr_bal):.3f}")
    print(f"Balanced LR Test Recall: {recall_score(y_test, y_pred_lr_bal):.3f}")
    print(f"Balanced LR Test F1: {f1_score(y_test, y_pred_lr_bal):.3f}")
    
    # 11. Interpretability
    # Extract feature names from fitted preprocessor
    ohe_cols = list(best_rf.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(cat_cols))
    all_feature_names = num_cols + ohe_cols
    
    rf_importances = best_rf.named_steps["clf"].feature_importances_
    feat_imp = pd.DataFrame({"feature": all_feature_names, "importance": rf_importances}).sort_values("importance", ascending=False)
    print("\n=== TOP 5 RANDOM FOREST FEATURES ===")
    print(feat_imp.head(5).to_string(index=False))
    
    lr_coefs = best_lr.named_steps["clf"].coef_[0]
    feat_coef = pd.DataFrame({"feature": all_feature_names, "coefficient": lr_coefs})
    print("\n=== LOGISTIC REGRESSION COEFFICIENTS (SELECTED) ===")
    for f in ["recency", "frequency", "monetary"]:
        val = feat_coef.loc[feat_coef["feature"] == f, "coefficient"].values[0]
        sign_str = "Positive (+)" if val > 0 else "Negative (-)"
        print(f"Feature: {f:<12} Coef: {val:+.4f} ({sign_str})")
        
    # 12. Save Selected Model (Random Forest) and verify reproduction
    model_save_path = os.path.join(OUTPUTS, "lapse_model.joblib")
    joblib.dump(best_rf, model_save_path)
    reloaded_model = joblib.load(model_save_path)
    reloaded_pred = reloaded_model.predict(X_test)
    assert np.array_equal(y_pred_rf, reloaded_pred), "Reloaded model failed to reproduce exact test predictions!"
    print(f"\nModel saved to {model_save_path} and verified with joblib.load.")
    
    # 13. Retention List: Top 10 Test-Set Premium Customers
    test_summary = pd.DataFrame({
        "customer_id": test_cids,
        "actually_lapsed": y_test.values,
        "lapse_probability": best_rf.predict_proba(X_test)[:, 1].round(4)
    })
    test_summary = test_summary.merge(segments[["customer_id", "is_premium"]], on="customer_id", how="left")
    
    prem_test = test_summary[test_summary["is_premium"] == True].copy()
    retention_top10 = prem_test.sort_values("lapse_probability", ascending=False).head(10)
    
    retention_list_path = os.path.join(OUTPUTS, "retention_list.csv")
    retention_top10[["customer_id", "lapse_probability", "actually_lapsed"]].to_csv(retention_list_path, index=False)
    
    print("\n=== RETENTION LIST (TOP 10 TEST PREMIUM CUSTOMERS) ===")
    print(retention_top10[["customer_id", "lapse_probability", "actually_lapsed"]].to_string(index=False))
    truly_lapsed_count = retention_top10["actually_lapsed"].sum()
    print(f"Truly lapsed among top 10: {truly_lapsed_count} / 10")
    print("NOTE: The retention list is illustrative and not a leakage-free evaluation because the Premium flag is a post-hoc whole-window label.")

if __name__ == "__main__":
    run_lapse_modeling()
