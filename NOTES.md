# Capstone Measured Numerical Results & Checkpoints Log (NOTES.md)

## Part A: Data Foundation
### Raw Data Audit Checkpoints
- Raw orders.csv rows: 13,084 (Ref: 13,084) - PASS
- Exact duplicate rows: 119 (Ref: 119) - PASS
- Pending status rows: 656 (Ref: 656) - PASS
- Distinct order_id values: 12,309 (Ref: 12,309) - PASS
- Non-numeric order_amount: 120 (Ref: 120) - PASS
- Negative delivery_minutes: 106 (Ref: 106) - PASS
- Blank delivery_minutes: 2,541 (Ref: 2,541) - PASS
- Blank age in customers.csv: 61 (Ref: 61) - PASS
- Age values outside 10-100: 6 (Ref: 6) - PASS
- Distinct raw city spellings: 24 (Ref: 24) - PASS
- Distinct non-blank raw gender spellings: 10 (Ref: 10) - PASS
- Blank gender in customers.csv: 30 (Ref: 30) - PASS

### Customers Clean Checkpoints
- Cities: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184 - PASS
- Genders: female 610, male 560, unknown 30 - PASS
- Missing age: 0, Min age: 18, Max age: 60 - PASS

### Orders Clean Checkpoints
- Total clean order rows: 12,309 (one per order_id) - PASS
- Completed orders: 11,110; Cancelled orders: 1,199; Pending orders: 0 - PASS
- Missing order_amount: 0 - PASS
- Missing delivery_minutes in completed orders: 0 - PASS
- Missing delivery_minutes in cancelled orders: 1,199 - PASS
- Completed order spend sum: INR 2,998,668.80 (Ref: 2,998,668.80) - PASS

### SQL Queries Checkpoints
- Q1 (WHERE > 500 then HAVING > 8000): 30 customers - PASS
- Q2 (Three-table join): 30 rows, Revenue sum INR 2,998,668.80, Largest cell: Bengaluru / Electronics = INR 233,909.45 - PASS
- Q3 (Scalar subquery above catalog avg): 11 products - PASS
- Q4 (LEFT JOIN no completed order): 25 customers - PASS
- Q5 (CTE spend > avg 2,552.06): 550 customers - PASS
- Q6 (Window functions): Top per city: Bengaluru C0082 (12,856.20), Chennai C0930 (10,151.40), Delhi C0291 (13,742.60), Hyderabad C0743 (12,730.05), Mumbai C0693 (14,766.55), Pune C0426 (10,377.10); Exactly 8 rows have RANK() != ROW_NUMBER() - PASS
- Q7 (Running total): 183 daily rows, Final cumulative revenue: INR 2,998,668.80 - PASS

### Pandas Exploratory Analysis
- Pivot table grand total: INR 2,998,668.80
- City aggregations:
 - Bengaluru: sum INR 523,643.40, mean INR 276.47, count 1,894
 - Chennai: sum INR 463,629.15, mean INR 272.72, count 1,700
 - Delhi: sum INR 502,829.80, mean INR 243.62, count 2,064
 - Hyderabad: sum INR 528,589.90, mean INR 284.95, count 1,855
 - Mumbai: sum INR 479,058.50, mean INR 297.00, count 1,613
 - Pune: sum INR 500,918.05, mean INR 252.48, count 1,984
- Distribution of completed order_amount:
 - Mean: INR 269.91, Median: INR 135.00, Skewness: 4.61 (right-skewed)
 - Q1: INR 64.80, Q3: INR 297.00, IQR: INR 232.20
 - Lower fence: INR -283.50 (0 outliers below), Upper fence: INR 645.30 (897 outliers above)
- Correlation matrix:
 - order_amount vs discount_pct: -0.095 (largest in absolute value)
 - order_amount vs quantity: +0.087
 - delivery_minutes vs order_amount: -0.003
 - Correlation does not imply causation.

---

## Part B: Customer Segments and Lapse Prediction
### RFM Metrics (as of 2025-10-01)
- Customer count: 1,175
- Recency: median 19 days, min 0 days, max 180 days
- Frequency: mean 9.46, max 33
- Monetary: mean INR 2,552.06, max INR 14,766.55

### K-Means Clustering (K=2..8)
- K=2: Inertia = 1820.8, Silhouette = 0.465
- K=3: Inertia = 1268.2, Silhouette = 0.445
- K=4: Inertia = 743.1, Silhouette = 0.526 (Optimal elbow and peak silhouette score)
- K=5: Inertia = 617.4, Silhouette = 0.480
- K=6: Inertia = 503.2, Silhouette = 0.482
- K=7: Inertia = 408.5, Silhouette = 0.416
- K=8: Inertia = 366.5, Silhouette = 0.412
- Note: K=8 always achieves lower inertia than K=2 because adding cluster centroids mathematically minimizes sum of squared distances to cluster centers; however, choosing K based on inertia alone leads to overfitting (at K=N, inertia=0). K=4 provides the optimal balance of cohesion, separation, and business actionability.

### Segment Profiles (Chosen K=4)
- Segment 0 (Size 436): Mean R = 28.86d, Mean F = 3.77, Mean M = INR 1,057.62 | Name: Occasional Value Seekers | Action: Re-engage with category-specific discount coupons | is_premium = False
- Segment 1 (Size 138): Mean R = 24.12d, Mean F = 6.77, Mean M = INR 7,129.70 | Name: Premium Electronics Shoppers | Action: Dedicated priority support and early access to flagship launches | is_premium = True (Highest Mean Monetary)
- Segment 2 (Size 419): Mean R = 8.77d, Mean F = 19.40, Mean M = INR 3,410.02 | Name: Loyal Grocery Enthusiasts | Action: Subscription bundle savings / VIP fast delivery | is_premium = False
- Segment 3 (Size 182): Mean R = 109.04d, Mean F = 2.20, Mean M = INR 685.99 | Name: Hibernating At-Risk | Action: Win-back campaign with high-discount voucher | is_premium = False
- PCA Explained Variance Ratios: PC1 = 0.634, PC2 = 0.218, Total = 0.852 (Ref: 0.634, 0.218, 0.852)

### Lapse Modeling (Cutoff 2025-09-01)
- Cohort: 1,156 customers with completed order before cutoff
- Target: 443 lapsed (38.32%), 713 not lapsed (61.68%)
- Train / Test Split: 924 train / 232 test (stratified)
- Learned mean frequency (train-only scaler): 8.1147 vs Full-cohort mean frequency: 8.0285 (differs due to sample variance of the 80% split; fitting strictly on training avoids data leakage).
- Baseline: Always predict not lapsed -> Accuracy 61.6%, Lapsed F1 = 0.000
- Logistic Regression: Best C = 100, CV F1 = 0.651, Test Precision = 0.644, Test Recall = 0.652, Test F1 = 0.648
- Random Forest: Best max_depth = 8, min_samples_leaf = 10, CV F1 = 0.680, Test Precision = 0.659, Test Recall = 0.652, Test F1 = 0.655 (Selected Model)
- Class-Balance Experiment: Oversampled training set to 570 each class; refit Logistic Regression on balanced data -> Test Recall jumped from 0.652 to 0.854 (+20.2 pp), while Test Precision adjusted to 0.589 (F1 = 0.697). Recall moved the most, allowing the retention team to capture ~85% of churners.
- Top 5 Random Forest Features:
  1. frequency (0.2797)
  2. monetary (0.2714)
  3. recency (0.1152)
  4. avg_delivery_minutes (0.0612)
  5. cancellation_rate (0.0523)
- Logistic Regression Coefficients:
 - recency: +0.0396 (positive: higher days since last order increases lapse probability)
 - frequency: -1.6982 (negative: higher order volume strongly decreases lapse risk)
 - monetary: +0.2846 (positive: higher spend conditional on frequency indicates sporadic big-ticket buyers at higher lapse risk)
- Retention list: Top 10 test Premium customers ranked by lapse probability saved to outputs/retention_list.csv (truly lapsed count: 1 / 10).
- CAUTION: The Premium flag is a post-hoc whole-window label, so the retention list is illustrative and not a leakage-free evaluation.

---

## Part C: Support Ticket Triage with Guardrails
- Input Guardrails: 4 empty tickets (`skipped_empty`), 4 long tickets (`flagged_long`, 734 tokens each).
- PII Redaction Substitutions: Email = 55, Phone = 91, Order ID = 132, Total = 278 (0 leftover PII instances).
- Resilience & Retries: 19 `TransientLLMError` failures simulated across 156 calls; 100% recovered on retry 0 with exponential backoff and jitter (0.045s - 0.057s wait).
- Status Breakdown across 156 eligible tickets:
 - classified_json: 135
 - classified_keyword: 12
 - flagged_invalid_output: 9
- Performance Comparison:
 - Zero-shot: 147 classified, Accuracy = 85.7% (126/147), Total tokens = 10,492, Tokens/ticket = 71.4. Errors: 11 fraud -> billing, 10 technical -> billing.
 - Few-shot: 147 classified, Accuracy = 100.0% (147/147), Total tokens = 20,320, Tokens/ticket = 138.2.
- Token Budget Demo: Budget = 6000 exhausted on ticket 98, total 66 tickets ended `budget_exhausted` (saved to outputs/triage_budget_demo.csv).
- Final Routing Results (Few-shot, outputs/triage_results.csv, 164 rows):
 - support_queue: 102 (NORMAL)
 - security_team: 34 (HIGH)
 - human_review: 17 (NORMAL)
 - premium_desk: 11 (HIGH)

---

## Part D: Policy RAG Assistant and Refund Agent
- Chunk count (overlap=10): 15 chunks total (refund 4, delivery 3, escalation 3, payment 3, privacy 2).
- Retrieval query test ('refund policy late delivery'): Top 3 chunks all from `refund_policy.txt`.
- Overlap = 10 Evaluation (14 questions):
 - Answerable Doc Hits: 12 / 12 (100%)
 - Answerable Fact Hits: 12 / 12 (100%)
 - Unanswerable Fallbacks: 2 / 2 (100%)
- Overlap = 0 Evaluation:
 - Doc Hits: 12 / 12 (100%)
 - Fact Hits: 10 / 12 (83.3%)
 - Failing Question: "What delivery time makes a completed order eligible for a full refund?" (expected: 'more than 45 minutes', returned: 'receives a full refund of the order amount.' because the sentence was split across the 40-word boundary).
- ReAct Agent 7 Scenarios:
 - S1: REFUND INR 128.25 | calls=5 | tokens=616 | ans=REFUND: INR 128.25 issued.
 - S2: REFUND INR 135.00 | calls=5 | tokens=616 | ans=REFUND: INR 135.0 issued.
 - S3: ESCALATE INR 1999.00 | calls=5 | tokens=603 | ans=ESCALATE: refund above INR 1000 sent to a human lead.
 - S4: DENIED | refund=0.00 | calls=4 | tokens=419 | ans=DENIED: delivery was within 45 minutes.
 - S5: DENIED | refund=0.00 | calls=3 | tokens=246 | ans=DENIED: only completed orders can be refunded.
 - S6: NOT_FOUND | refund=0.00 | calls=2 | tokens=76 | ans=NOT_FOUND: no such order.
 - S7: FALLBACK_MAX_ITERATIONS | refund=0.00 | calls=6 | tokens=1557 | ans=Maximum iteration limit reached.
- Safe Stop Demos:
 - S1 Token Limit Demo (token_limit=200): `FALLBACK_TOKEN_LIMIT`
 - S1 Timeout Demo (time_limit_seconds=0.0): `FALLBACK_TIMEOUT`
- Crash and Resume Demo on S1:
 - Tools executed before simulated crash (after step 2): `['get_order', 'search_policy']`
 - Tools executed across crash and resume: `['get_order', 'search_policy', 'calculate_refund', 'issue_refund']` (exactly 4 unique tools, 0 repeated)
 - Final Outcome after Resume: `REFUND` with INR 128.25.
