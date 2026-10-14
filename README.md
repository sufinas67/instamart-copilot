# Instamart Customer Intelligence & Support Copilot

## Overview
An enterprise-grade Customer Intelligence and Support Copilot engineered for Swiggy Instamart across six metro cities (Bengaluru, Chennai, Delhi, Hyderabad, Mumbai, Pune). The Copilot resolves three critical operational bottlenecks: untrusted order data, unpredictable customer lapse, and manual customer support ticket triage.

The system is constructed as four interconnected, reproducible modules:
- **Part A (Data Foundation)**: Cleans dirty raw data exports, enforces referential and business rules, loads an analytical SQLite database (`data/instamart.db`), and executes analytical SQL queries and pandas EDA.
- **Part B (Customer Segments & Lapse Prediction)**: Segments 1,175 customers into distinct behavioral archetypes using K-Means and PCA, builds a leakage-free lapse prediction pipeline, tunes machine learning models, and ranks high-value at-risk customers.
- **Part C (Support Ticket Triage with Guardrails)**: Implements regex-based PII sanitisation, length/empty guardrails, resilient exponential-backoff retries, output validation, token budgeting, and segment-aware ticket routing.
- **Part D (Policy RAG Assistant & Refund Agent)**: Builds a TF-IDF grounded policy retrieval assistant and an autonomous ReAct refund agent equipped with safety fallbacks, state persistence, and resume capability.

---

## Reproduce

Execute the following shell commands in order from the repository root to regenerate all data, models, databases, and artifacts deterministically:

```bash
# 1. Create and activate isolated virtual environment
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
# source .venv/bin/activate

# 2. Install pinned dependencies
pip install -r requirements.txt

# 3. Generate deterministic synthetic data
python generate_data.py

# 4. Part A: Audit raw data
python src/audit.py

# 5. Part A: Execute data cleaning pipeline and populate SQLite database
python src/clean_data.py

# 6. Part A: Validate cleaning checkpoints
python src/validate.py

# 7. Part A: Execute analytical SQL queries Q1 to Q7
python src/run_sql.py

# 8. Part A: Execute pandas pivot table, named aggregations, and exploratory data analysis
python src/explore_pandas.py

# 9. Part B: Calculate RFM metrics, execute K-Means clustering, PCA, and export segments
python src/rfm_segments.py

# 10. Part B: Train, tune, and evaluate leakage-free lapse prediction models
python src/lapse_model.py

# 11. Part C: Execute support ticket triage pipeline, guardrails, and segment routing
python src/triage.py

# 12. Part D: Evaluate policy retrieval grounding across chunk overlap configurations
python src/rag.py

# 13. Part D: Execute ReAct refund agent scenarios, safe stops, and crash-resume demonstrations
python src/run_part_d.py
```

---

## Part A — Data Foundation: Clean, Store, Query

### 1. Raw Data Audit
Loading raw tables (`customers.csv`, `products.csv`, `orders.csv`, and `tickets.csv` with `keep_default_na=False`) revealed substantial anomalies, unparseable entries, and duplicate records across tables:

| Audit Metric | Measured Value | Reference Value | Checkpoint Status |
| :--- | :---: | :---: | :---: |
| Raw `orders.csv` rows | 13,084 | 13,084 | PASS |
| Exact duplicate rows | 119 | 119 | PASS |
| Rows with `pending` status | 656 | 656 | PASS |
| Distinct `order_id` values | 12,309 | 12,309 | PASS |
| Non-numeric `order_amount` entries | 120 | 120 | PASS |
| Negative `delivery_minutes` entries | 106 | 106 | PASS |
| Blank `delivery_minutes` entries | 2,541 | 2,541 | PASS |
| Blank `age` in `customers.csv` | 61 | 61 | PASS |
| Age values outside 10–100 | 6 | 6 | PASS |
| Distinct raw city spellings | 24 | 24 | PASS |
| Distinct non-blank raw gender spellings | 10 | 10 | PASS |
| Blank `gender` entries in `customers.csv` | 30 | 30 | PASS |

### 2. Cleaning Rules and Record Modifications

#### Customers Table
1. **City Standardisation**: Stripped leading/trailing whitespace and applied title-casing. Changed 714 raw values to collapse 24 messy spellings into exactly six valid metro cities (Pune: 227, Delhi: 210, Bengaluru: 197, Chennai: 191, Hyderabad: 191, Mumbai: 184).
2. **Gender Mapping**: Standardised 10 distinct spellings to `male` (560), `female` (610), or `unknown` (30). Changed 625 values.
3. **Age Validation & City Median Imputation**: Coerced 6 impossible ages (outside 10–100: -4, 0, 150, 200, 280, 999) to NaN. Filled all 67 missing ages using city-level median age via `groupby('city')['age'].transform()`. Resulted in 0 missing ages, with all ages strictly bounded between 18 and 60.
4. **Signup Date**: Parsed ISO strings to standard datetime format (`YYYY-MM-DD`).

#### Orders Table (Executed in Exact Sequence)
1. **Timestamp Conversion**: Parsed `order_ts` to datetime for all 13,084 rows.
2. **Exact Duplicate Removal**: Dropped 119 exact duplicate records, reducing row count from 13,084 to 12,965.
3. **Status Deduplication**: Grouped by `order_id` and retained only the record with the latest `order_ts`. Removed 656 earlier `pending` records, leaving exactly 12,309 unique orders.
4. **Order Amount Imputation**: Coerced 120 unparseable values (`N/A` and `error`) to NaN and imputed using catalogue pricing: `round(price_inr * quantity * (1 - discount_pct / 100), 2)`. Missing amount count after imputation is 0.
5. **Delivery Minutes Validation & Imputation**: Replaced 106 negative values (-10) with NaN. For completed orders, filled 2,421 missing values using the category median delivery time among completed orders. For cancelled orders, left 1,199 delivery minutes as NaN because cancelled orders were never delivered.

**Final Dataset Integrity**: Exactly 12,309 rows (11,110 completed, 1,199 cancelled, 0 pending). Total completed order spend equals **INR 2,998,668.80**. `validate.py` asserted all checkpoints and printed `ALL CHECKPOINTS PASSED`.

### 3. SQL Analysis (`sql/queries.sql`)
Executed against `data/instamart.db`:

- **Q1 (WHERE then HAVING)**: Filtered completed orders with `order_amount > 500`, grouped by customer, and filtered `HAVING SUM(order_amount) > 8000`. Returns exactly **30 customers**.
- **Q2 (Three-Table Join)**: Joined `orders`, `customers`, and `products` to aggregate revenue by city and product category. Returns **30 rows**, summing to **INR 2,998,668.80**. Largest cell is **Bengaluru / Electronics** at **INR 233,909.45**.
- **Q3 (Scalar Subquery)**: Identified products priced above the catalogue average (INR 344.025). Returns **11 products**.
- **Q4 (LEFT JOIN)**: Joined `customers` to completed `orders` to find inactive signups. Returns **25 customers** with zero completed orders.
- **Q5 (Common Table Expression)**: Computed customer spend and population average (INR 2,552.06); main query returned **550 customers** spending above average.
- **Q6 (Ranking Window Functions)**: Computed customer total spend and ranked using `RANK()`, `DENSE_RANK()`, and `ROW_NUMBER()` partitioned by city. Top spenders per city:
  - Bengaluru: `C0082` (INR 12,856.20)
  - Chennai: `C0930` (INR 10,151.40)
  - Delhi: `C0291` (INR 13,742.60)
  - Hyderabad: `C0743` (INR 12,730.05)
  - Mumbai: `C0693` (INR 14,766.55)
  - Pune: `C0426` (INR 10,377.10)
  *Ties Explanation*: When multiple customers within a city share identical spend totals, `RANK()` assigns them the identical rank and skips subsequent ranks, whereas `ROW_NUMBER()` assigns strictly distinct sequential integers, causing exactly 8 tied rows to diverge between the two functions.
- **Q7 (Running Total)**: Daily revenue running total using `SUM(revenue) OVER (ORDER BY day)`. Returns **183 daily rows** (2025-04-01 to 2025-09-30), culminating at **INR 2,998,668.80**.

### 4. Pandas Cross-Check and Exploratory Data Analysis
- **Cross-Check**: `pd.pivot_table(comp, values='order_amount', index='city', columns='category', aggfunc='sum', margins=True)` produced a grand total of **INR 2,998,668.80**, matching SQL Q2.
- **City Aggregations**:
  - Bengaluru: sum INR 523,643.40, mean INR 276.47, count 1,894
  - Chennai: sum INR 463,629.15, mean INR 272.72, count 1,700
  - Delhi: sum INR 502,829.80, mean INR 243.62, count 2,064
  - Hyderabad: sum INR 528,589.90, mean INR 284.95, count 1,855
  - Mumbai: sum INR 479,058.50, mean INR 297.00, count 1,613
  - Pune: sum INR 500,918.05, mean INR 252.48, count 1,984
- **Order Amount Distribution**: Mean INR 269.91, Median INR 135.00, Skewness 4.61 (heavily right-skewed). Q1 is INR 64.80, Q3 is INR 297.00, IQR is INR 232.20. Upper fence (Q3Q3 + 1.5 * IQR) is INR 645.30 with 897 outliers above and 0 below.
- **Four-Question Analysis (<= 200 words)**:
  This dataset captures quick-commerce grocery and electronics transactions across six Indian metropolitan branches. Spend is strongly right-skewed with a mean of INR 269.91 and median of INR 135.00, driven by routine everyday low-ticket purchases alongside occasional high-ticket electronics. Setting 1.5xIQR fences identifies 897 high-value purchases above INR 645.30; these represent valid premium electronic items rather than recording errors. Across completed orders, linear associations are weak. The largest correlation in absolute value is between order_amount and discount_pct at -0.095, followed by order_amount and quantity at +0.087, while delivery_minutes is uncorrelated (-0.003). Crucially, empirical correlation of this small magnitude does not establish causation.

---

## Part B — Customer Segments and Lapse Prediction

### 1. RFM Table (Snapshot: 2025-10-01 00:00:00)
Calculated over 1,175 customers with completed orders:
- **Recency**: median 19 days, min 0 days, max 180 days
- **Frequency**: mean 9.46 orders, max 33 orders
- **Monetary**: mean INR 2,552.06, max INR 14,766.55

### 2. K-Means Clustering & Profiling
RFM features were standardised using `StandardScaler`. Clustering was evaluated across KK in [2, 8]:

| Clusters (K) | Inertia | Silhouette Score |
| :---: | :---: | :---: |
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| **4** | **743.1** | **0.526** |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

*Elbow and Selection*: K=8 always achieves lower inertia than K=2 because increasing cluster centroids mechanically reduces within-cluster sum of squares; at the extreme (K=N), inertia reaches 0. Therefore, inertia alone cannot select K. We choose K=4 because it achieves the global peak silhouette score (0.526), displays a prominent elbow, and provides interpretable operational segments.

**Segment Profiles (K=4)**:
- **Segment 0 (Size 436)**: Mean Recency 28.86d, Frequency 3.77, Monetary INR 1,057.62. *Name*: Occasional Value Seekers. *Action*: Nudge with category discount coupons. `is_premium = False`.
- **Segment 1 (Size 138)**: Mean Recency 24.12d, Frequency 6.77, Monetary INR 7,129.70. *Name*: Premium Electronics Shoppers. *Action*: Assign dedicated priority support desk and VIP delivery routing. `is_premium = True` (Highest Mean Monetary).
- **Segment 2 (Size 419)**: Mean Recency 8.77d, Frequency 19.40, Monetary INR 3,410.02. *Name*: Loyal Grocery Enthusiasts. *Action*: Enroll in loyalty recurring dairy/produce subscription plans. `is_premium = False`.
- **Segment 3 (Size 182)**: Mean Recency 109.04d, Frequency 2.20, Monetary INR 685.99. *Name*: Hibernating At-Risk. *Action*: High-incentive win-back reactivation campaign. `is_premium = False`.

**PCA Projection**: Applying `PCA(n_components=2)` to scaled RFM features yields explained variance ratios of **0.634** (PC1) and **0.218** (PC2), capturing **85.2%** of total variance. Saved to `outputs/customer_segments.csv`.

### 3. Leakage-Free Lapse Modeling
- **Problem Setup**: Cutoff set at `2025-09-01`. Cohort comprises **1,156 customers** with \ge 1 completed order prior to cutoff. Target label `lapsed = 1` if customer placed zero completed orders in September 2025; otherwise `0`. Exactly **443 customers lapsed (38.32%)**.
- **Features (Strictly Pre-Cutoff)**: `recency`, `frequency`, `monetary`, `avg_discount_pct`, `cancellation_rate`, `avg_delivery_minutes`, `fav_category`, `city`, `age`, `gender`, `tenure_days`. Segment labels and post-cutoff orders were strictly excluded.
- **Train/Test Split**: 80/20 stratified split (`random_state=42`) yielded **924 training rows** (570 non-lapsed, 354 lapsed) and **232 test rows** (143 non-lapsed, 89 lapsed).
- **Leakage Prevention**: Numerical features were scaled with `StandardScaler` and nominal features one-hot encoded (`drop="first"`) strictly inside a `Pipeline` fitted on training rows only. Learned mean frequency on train was **8.1147** compared to full-cohort mean of **8.0285**; they differ because training data is a random sample, and computing transformations solely on train prevents test distribution leakage. Code confirmed 0 missing values across all feature columns.
- **Baseline Model**: Always predicting "not lapsed" yields **61.6% accuracy** and **0.000 F1**.

### 4. Tuning and Test Evaluation
- **Logistic Regression**: Tuned with `GridSearchCV(cv=5, scoring='f1')` over C \in \{0.01, 0.1, 1, 10, 100\}. Best C=100, CV F1 = 0.651. On untouched test set: Precision = 0.644, Recall = 0.652, **F1 = 0.648**. Confusion matrix: `[[111, 32], [31, 58]]`.
- **Random Forest**: Tuned with `GridSearchCV(cv=5, scoring='f1')` over `max_depth` in {3, 5, 8} and `min_samples_leaf` in {5, 10, 20} (45 fits). Best parameters: `max_depth=8`, `min_samples_leaf=10`, CV F1 = 0.680. On untouched test set: Precision = 0.659, Recall = 0.652, **F1 = 0.655**. Confusion matrix: `[[113, 30], [31, 58]]`.
- **Model Choice Justification**: Random Forest was chosen as the primary model (Test F1 0.655 \ge 0.55). In customer retention, the business cost of missing a lapsing customer (false negative: lost customer lifetime value) heavily exceeds the marketing cost of offering a retention discount to a loyal customer (false positive: minor margin discount). Random Forest achieved superior precision (0.659 vs 0.644) and fewer false alarms while matching recall, providing the most cost-effective retention allocation.

### 5. Class-Balance Experiment
Oversampled lapsed training rows with replacement (`random_state=42`) to balance classes at 570 rows each. Refitting Logistic Regression on balanced data increased Test Recall dramatically from **0.652 to 0.854 (+20.2 percentage points)** while precision was 0.589 and F1 was 0.697. Recall moved the most, proving that threshold/balance adjustments capture over 85% of at-risk customers.

### 6. Feature Importance and Retention List
- **Top 5 Random Forest Features**: `frequency` (0.2797), `monetary` (0.2714), `recency` (0.1152), `avg_delivery_minutes` (0.0612), `cancellation_rate` (0.0523).
- **Logistic Regression Signs**: `frequency` is negative (-1.6982: high order volume protects against lapse); `recency` is positive (+0.0396: longer dormancy increases lapse odds); `monetary` is positive (+0.2846: sporadic high-spend electronics buyers are more prone to single-purchase lapse).
- **Model Persistence**: Saved to `outputs/lapse_model.joblib`. Reloading with `joblib.load` reproduced identical predictions.
- **Retention List**: Ranked test-set Premium customers by predicted lapse probability and saved top 10 to `outputs/retention_list.csv`. Exactly 1 truly lapsed. *Note*: The retention list is illustrative and not a leakage-free evaluation because the Premium flag is a post-hoc whole-window label.

---

## Part C — Support Ticket Triage with Guardrails

### 1. Input Guardrails and PII Sanitisation
Read 164 raw tickets with `keep_default_na=False`:
- **Guardrails**: Filtered 4 empty tickets (`skipped_empty`) and 4 excessively long tickets (`flagged_long`, 734 estimated tokens > 400). Neither group triggered model calls.
- **PII Redaction**: Applied strict regex transformations before prompt construction:
  - Email: `[\w.+-]+@[\w-]+\.[\w.]+` \rightarrow `[EMAIL]` (55 replacements)
  - Phone: `\b[6-9]\d{9}\b` \rightarrow `[PHONE]` (91 replacements)
  - Order ID: `ORD-\d{6}` \rightarrow `[ORDER_ID]` (132 replacements)
  - Total substitutions: **278**. Asserted that 0 unmasked instances remained across all tickets.

### 2. Resilient Calls & Output Validation
- **TransientLLMError Resilience**: Wrapped mock calls in exponential backoff with jitter (`BASE_DELAY * 2**n * uniform(0.9, 1.15)`). Simulated 19 transient errors across 156 calls; 100% recovered on retry 0 with zero permanent failures.
- **Output Validation**:
  - `classified_json`: 135 valid JSON outputs with allowed categories.
  - `classified_keyword`: 12 non-JSON chatty replies validated via regex word extraction.
  - `flagged_invalid_output`: 9 out-of-domain outputs (e.g., category "shipping") safely routed to human review.

### 3. Zero-Shot vs Few-Shot Comparison

| Prompt Mode | Classified Tickets | Accuracy | Total Tokens | Tokens / Ticket | Status Breakdown (`json` / `kw` / `invalid`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Zero-Shot** | 147 | **85.7%** (126/147) | 10,492 | 71.4 | 135 / 12 / 9 |
| **Few-Shot** | 147 | **100.0%** (147/147) | 20,320 | 138.2 | 135 / 12 / 9 |

**Zero-Shot Confusion Matrix**:
```
Predicted    billing  delivery  fraud  technical
True                                            
billing           38         0      0          0
delivery           0        36      0          0
fraud             11         0     23          0
technical         10         0      0         29
```
*Token Cost Analysis*: Few-shot prompting requires approximately 66.8 additional tokens per ticket (138.2 vs 71.4), nearly doubling token expenditure. However, in zero-shot classification, all 11 misclassified fraud tickets and 10 technical tickets were misrouted to billing. Because misrouting fraud tickets incurs severe security and financial liabilities, paying for few-shot prompt context is cost-justified.

### 4. Token Budget & Final Segment-Aware Routing
- **Token Budget Demo**: Running zero-shot with `TOKEN_BUDGET = 6000` halted at ticket 98, marking **66 tickets** as `budget_exhausted` (saved to `outputs/triage_budget_demo.csv`).
- **Final Routing (`outputs/triage_results.csv`, 164 rows)**: Merged with `outputs/customer_segments.csv`:
  - `support_queue` (NORMAL priority): 102 tickets
  - `security_team` (HIGH priority): 34 tickets (all fraud cases)
  - `human_review` (NORMAL priority): 17 tickets (4 empty, 4 long, 9 invalid output)
  - `premium_desk` (HIGH priority): 11 tickets (non-fraud issues from Premium customers)

---

## Part D — Policy RAG Assistant and Refund Agent

### 1. Document Chunking & TF-IDF Retrieval
- Split five policy documents into 40-word chunks with 10-word overlap (step 30 words), producing **15 chunks** (`refund_policy.txt`: 4, `delivery_policy.txt`: 3, `escalation_policy.txt`: 3, `payment_policy.txt`: 3, `privacy_policy.txt`: 2).
- `retrieve(query, k=3)` using `TfidfVectorizer` and cosine similarity for query `"refund policy late delivery"` returned top-3 chunks all from `refund_policy.txt`.
- *TF vs IDF*: Term Frequency (TF) measures the local prevalence of a word within a specific text chunk, whereas Inverse Document Frequency (IDF) down-weights universally common terms to emphasize discriminative keywords. A primary limitation of TF-IDF is its lexical dependency: it cannot capture semantic synonyms or paraphrased queries lacking exact word matches.

### 2. Retrieval Grounding Evaluation

| Metric | Overlap = 10 (Step 30) | Overlap = 0 (Step 40) |
| :--- | :---: | :---: |
| Answerable Document Hits | 12 / 12 (100%) | 12 / 12 (100%) |
| Answerable Fact Hits | 12 / 12 (100%) | 10 / 12 (83.3%) |
| Unanswerable Fallback Hits | 2 / 2 (100%) | 2 / 2 (100%) |

*Zero-Overlap Failure*: With overlap 0, the question *"What delivery time makes a completed order eligible for a full refund?"* failed (received *"receives a full refund of the order amount."* instead of *"more than 45 minutes"*). The answer sentence was split across the rigid 40-word chunk boundary, omitting the condition clause from the retrieved chunk.

### 3. ReAct Refund Agent Scenarios
Executed all 7 simulated scenarios from `data/raw/agent_scenarios.csv` against `data/instamart.db`:

| Scenario ID | Outcome Status | Refund Amount (INR) | Model Calls | Tokens Used | Final Answer Summary |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **S1** | `REFUND` | 128.25 | 5 | 616 | REFUND: INR 128.25 issued. |
| **S2** | `REFUND` | 135.00 | 5 | 616 | REFUND: INR 135.0 issued. |
| **S3** | `ESCALATE` | 1999.00 | 5 | 603 | ESCALATE: refund above INR 1000 sent to a human lead. |
| **S4** | `DENIED` | 0.00 | 4 | 419 | DENIED: delivery was within 45 minutes. |
| **S5** | `DENIED` | 0.00 | 3 | 246 | DENIED: only completed orders can be refunded. |
| **S6** | `NOT_FOUND` | 0.00 | 2 | 76 | NOT_FOUND: no such order. |
| **S7** | `FALLBACK_MAX_ITERATIONS` | 0.00 | 6 | 1557 | Loop detected; safely handed over to human support. |

Logged all runs to `outputs/agent_runs.jsonl` and ledger events to `outputs/refund_ledger.jsonl`.

### 4. Safe Stops & Crash-Resume Verification
- **Token Limit Demo**: Running S1 with `token_limit=200` halted immediately before executing the requested tool with status `FALLBACK_TOKEN_LIMIT`.
- **Timeout Demo**: Running S1 with `time_limit_seconds=0.0` stopped before the first model call with status `FALLBACK_TIMEOUT`.
- **Crash and Resume Demonstration (S1)**:
  - Phase 1: Executed tools `['get_order', 'search_policy']`, saved state checkpoint to `state/`, and simulated process termination.
  - Phase 2: Called `resume_agent(run_id)`. The agent loaded the checkpoint and executed remaining tools `['calculate_refund', 'issue_refund']`.
  - Result: Across both phases, exactly four unique tools were executed in order (`get_order` \rightarrow `search_policy` \rightarrow `calculate_refund` \rightarrow `issue_refund`), 0 tools were repeated, and the final status was `REFUND` (INR 128.25).

### 5. ReAct versus Plan-and-Execute Comparison (<= 250 words)
Our measured ReAct implementation adapts dynamically to live execution feedback, requiring 5 model calls for full refund approvals (S1, S2, S3), 4 calls for on-time delivery rejections (S4), 3 calls for cancelled orders (S5), and only 2 calls when an order is missing (S6). In total, ReAct uses between 76 and 616 tokens for typical scenarios. Conversely, a plan-and-execute architecture requests a full DAG of planned tool steps up front in a single model call, executing tools sequentially and replanning only upon exceptions.

While plan-and-execute reduces total model invocations from 5 to 1 or 2 in happy-path cases, it struggles severely with dynamic edge cases. When an order is not found (S6) or was previously cancelled (S5), a rigid pre-computed plan will needlessly schedule policy lookups and refund calculations, requiring replanning cycles that inflate token overhead. Furthermore, customer support refunds are high-stakes financial operations: emitting speculative tool plans introduces hallucination risks where money could be erroneously disbursed before order status is verified. For grocery refund triage, the ReAct pattern is definitively superior because every subsequent action is strictly conditioned on verified database observations and policy retrieval, guaranteeing policy compliance, privacy protection, and immediate graceful degradation.

---

## Limitations and Next Steps

1. **Post-Hoc Premium Label Limitation**: The `is_premium` flag in `outputs/retention_list.csv` is derived from the full six-month RFM clustering window (April–September 2025). Because the September window also defines the ground-truth lapse target, the retention list is illustrative and does not constitute a leakage-free predictive evaluation. In production, segmentation must be computed strictly on historical data prior to the predictive observation window.
2. **Offline Stand-In vs Hosted LLM**: The current pipeline runs offline against `mock_llm.py`. Transitioning to production will require integrating hosted models (e.g., Gemini 1.5 Pro) with structured JSON decoding, tool function schemas, and secrets managed via `.env` (listed in `.gitignore`).
3. **Semantic Embedding RAG**: The current policy retrieval assistant relies on lexical TF-IDF. Upgrading to dense neural embeddings (e.g., Sentence-Transformers) will overcome vocabulary mismatch and improve cross-lingual policy resolution.
