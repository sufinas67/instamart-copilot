import sys, os
sys.path.insert(0, os.path.abspath('.'))
"""src/part_c/triage.py -- Support ticket triage pipeline with guardrails, resilient retries, prompt comparison, budget control, and segment-aware routing."""
import os
import re
import json
import random
import time
import pandas as pd
import numpy as np

import mock_llm
from src.part_c.guardrails import check_input_guardrails, redact_pii, verify_no_pii_remains

ALLOWED_CATEGORIES = ["billing", "delivery", "technical", "fraud"]
BASE_DELAY = 0.05

ZERO_SHOT_PROMPT = """CLASSIFY the customer support ticket into exactly one of these categories:
- billing
- delivery
- technical
- fraud

Respond strictly in valid JSON format:
{"category": "<category>", "reason": "<short explanation>"}"""

FEW_SHOT_PROMPT = """CLASSIFY the customer support ticket into exactly one of these categories:
- billing
- delivery
- technical
- fraud

Respond strictly in valid JSON format:
{"category": "<category>", "reason": "<short explanation>"}

EXAMPLES:
Ticket: There is an unauthorized transaction of INR 1200 on my card linked to order ORD-100001.
Output: {"category": "fraud", "reason": "unauthorized card transaction"}

Ticket: The delivery rider marked ORD-100002 as delivered but I never received the groceries.
Output: {"category": "delivery", "reason": "order marked delivered but not received"}"""

def call_mock_with_retry(messages, ticket_id, retry_log=None):
    """Executes mock_llm.chat with up to 3 retries and exponential backoff with jitter."""
    for n in range(3):
        try:
            resp = mock_llm.chat(messages)
            return resp
        except mock_llm.TransientLLMError as err:
            wait = BASE_DELAY * (2 ** n) * random.uniform(0.9, 1.15)
            log_line = f"Retry {n} for ticket {ticket_id}: wait={wait:.4f}s ({err})"
            if retry_log is not None:
                retry_log.append(log_line)
            print(log_line)
            time.sleep(wait)
    raise RuntimeError(f"Exceeded max retries (3) for ticket {ticket_id}")

def validate_response(content: str):
    """
    Validates model output:
    1. Valid JSON with category in allowed list -> classified_json
    2. Invalid JSON, but lowercase word extraction yields exactly one allowed category -> classified_keyword
    3. Else -> flagged_invalid_output
    """
    if not content:
        return "flagged_invalid_output", None
        
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            cat = str(data.get("category", "")).strip().lower()
            if cat in ALLOWED_CATEGORIES:
                return "classified_json", cat
            else:
                return "flagged_invalid_output", None
    except Exception:
        pass
        
    words = re.findall(r"[a-z]+", content.lower())
    matches = [w for w in words if w in ALLOWED_CATEGORIES]
    if len(matches) == 1:
        return "classified_keyword", matches[0]
        
    return "flagged_invalid_output", None

def run_triage_pipeline(prompt_type="few-shot", token_budget=None, log_retries=False):
    """Runs support ticket triage pipeline over data/raw/tickets.csv."""
    tickets = pd.read_csv(os.path.join("data", "raw", "tickets.csv"), keep_default_na=False)
    system_prompt = FEW_SHOT_PROMPT if prompt_type == "few-shot" else ZERO_SHOT_PROMPT
    
    mock_llm.reset()
    random.seed(42)
    
    running_tokens = 0
    records = []
    retry_log = [] if log_retries else None
    
    for _, row in tickets.iterrows():
        tid = row["ticket_id"]
        cid = row["customer_id"]
        oid = row["order_id"]
        raw_text = row["text"]
        true_cat = row["true_category"]
        
        # 1. Input Guardrails
        guard_status = check_input_guardrails(raw_text)
        if guard_status in ["skipped_empty", "flagged_long"]:
            records.append({
                "ticket_id": tid,
                "customer_id": cid,
                "order_id": oid,
                "status": guard_status,
                "category": "",
                "true_category": true_cat,
                "tokens_used": 0
            })
            continue
            
        # 2. PII Redaction
        redacted_text, _, _, _ = redact_pii(raw_text)
        assert verify_no_pii_remains(redacted_text), f"Unmasked PII detected in ticket {tid}!"
        
        # 3. Token Budget Check
        if token_budget is not None and running_tokens >= token_budget:
            records.append({
                "ticket_id": tid,
                "customer_id": cid,
                "order_id": oid,
                "status": "budget_exhausted",
                "category": "",
                "true_category": true_cat,
                "tokens_used": 0
            })
            continue
            
        # 4. Model Call with Retry
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": redacted_text}
        ]
        
        resp = call_mock_with_retry(messages, tid, retry_log=retry_log)
        call_tokens = resp.usage.total_tokens
        running_tokens += call_tokens
        
        # 5. Output Validation
        reply_content = resp.choices[0].message.content
        status, parsed_cat = validate_response(reply_content)
        
        records.append({
            "ticket_id": tid,
            "customer_id": cid,
            "order_id": oid,
            "status": status,
            "category": parsed_cat if parsed_cat else "",
            "true_category": true_cat,
            "tokens_used": call_tokens
        })
        
    res_df = pd.DataFrame(records)
    return res_df, running_tokens, retry_log

def main():
    OUTPUTS = "outputs"
    os.makedirs(OUTPUTS, exist_ok=True)
    
    # 1. Audit PII Redaction across all tickets
    tickets = pd.read_csv(os.path.join("data", "raw", "tickets.csv"), keep_default_na=False)
    tot_email, tot_phone, tot_order = 0, 0, 0
    all_redacted = []
    for txt in tickets["text"]:
        r_txt, ne, np_, no = redact_pii(txt)
        tot_email += ne
        tot_phone += np_
        tot_order += no
        all_redacted.append(r_txt)
        
    print("=== PII REDACTION AUDIT (164 TICKETS) ===")
    print(f"Emails replaced: {tot_email}")
    print(f"Phone numbers replaced: {tot_phone}")
    print(f"Order IDs replaced: {tot_order}")
    print(f"Total substitutions: {tot_email + tot_phone + tot_order}")
    assert tot_email == 55 and tot_phone == 91 and tot_order == 132 and (tot_email + tot_phone + tot_order) == 278
    
    leftover_pii = sum(not verify_no_pii_remains(t) for t in all_redacted)
    print(f"Leftover PII instances found: {leftover_pii}")
    assert leftover_pii == 0
    
    # 2. Run Zero-shot evaluation
    print("\n=== RUNNING ZERO-SHOT EVALUATION ===")
    df_zs, tokens_zs, retry_log_zs = run_triage_pipeline(prompt_type="zero-shot", token_budget=None, log_retries=True)
    print(f"Total TransientLLMError retries recorded: {len(retry_log_zs)}")
    assert len(retry_log_zs) == 19, f"Expected 19 retries, got {len(retry_log_zs)}"
    
    zs_counts = df_zs["status"].value_counts().to_dict()
    print("Zero-shot Status Breakdown:", zs_counts)
    assert zs_counts.get("classified_json", 0) == 135
    assert zs_counts.get("classified_keyword", 0) == 12
    assert zs_counts.get("flagged_invalid_output", 0) == 9
    assert zs_counts.get("skipped_empty", 0) == 4
    assert zs_counts.get("flagged_long", 0) == 4
    
    classified_zs = df_zs[df_zs["status"].isin(["classified_json", "classified_keyword"])].copy()
    acc_zs = (classified_zs["category"] == classified_zs["true_category"]).mean() * 100
    print(f"Zero-shot Classified Tickets: {len(classified_zs)}")
    print(f"Zero-shot Accuracy: {acc_zs:.1f}% ({(classified_zs['category'] == classified_zs['true_category']).sum()} / {len(classified_zs)})")
    print(f"Zero-shot Total Tokens: {tokens_zs}, Tokens/Classified: {tokens_zs / len(classified_zs):.1f}")
    assert abs(acc_zs - 85.7) <= 0.5
    
    print("\nZero-shot Confusion Matrix (classified tickets):")
    conf_zs = pd.crosstab(classified_zs["true_category"], classified_zs["category"], rownames=["True"], colnames=["Predicted"])
    print(conf_zs)
    
    # 3. Run Few-shot evaluation
    print("\n=== RUNNING FEW-SHOT EVALUATION ===")
    df_fs, tokens_fs, _ = run_triage_pipeline(prompt_type="few-shot", token_budget=None, log_retries=False)
    fs_counts = df_fs["status"].value_counts().to_dict()
    print("Few-shot Status Breakdown:", fs_counts)
    
    classified_fs = df_fs[df_fs["status"].isin(["classified_json", "classified_keyword"])].copy()
    acc_fs = (classified_fs["category"] == classified_fs["true_category"]).mean() * 100
    print(f"Few-shot Classified Tickets: {len(classified_fs)}")
    print(f"Few-shot Accuracy: {acc_fs:.1f}% ({(classified_fs['category'] == classified_fs['true_category']).sum()} / {len(classified_fs)})")
    print(f"Few-shot Total Tokens: {tokens_fs}, Tokens/Classified: {tokens_fs / len(classified_fs):.1f}")
    assert acc_fs == 100.0
    
    # 4. Token Budget Demonstration (Zero-shot with TOKEN_BUDGET = 6000)
    print("\n=== TOKEN BUDGET DEMO (Budget = 6000, Zero-shot) ===")
    df_budget, tokens_b, _ = run_triage_pipeline(prompt_type="zero-shot", token_budget=6000, log_retries=False)
    exhausted_count = (df_budget["status"] == "budget_exhausted").sum()
    print(f"Budget Demo Total Tokens: {tokens_b}")
    print(f"Tickets ending budget_exhausted: {exhausted_count}")
    assert exhausted_count >= 1, "Budget demo must have at least one budget_exhausted ticket!"
    
    # Save outputs/triage_budget_demo.csv
    demo_cols = ["ticket_id", "customer_id", "status", "category", "tokens_used"]
    df_budget[demo_cols].to_csv(os.path.join(OUTPUTS, "triage_budget_demo.csv"), index=False)
    print("Saved outputs/triage_budget_demo.csv")
    
    # 5. Final Routing Pipeline (Few-shot with TOKEN_BUDGET = 100000)
    print("\n=== GENERATING FINAL TRIAGE RESULTS (Few-shot, Routing) ===")
    df_final, tokens_final, _ = run_triage_pipeline(prompt_type="few-shot", token_budget=100000, log_retries=False)
    
    # Merge customer segments to get segment_name and is_premium
    segments = pd.read_csv(os.path.join(OUTPUTS, "customer_segments.csv"))
    df_final = df_final.merge(segments[["customer_id", "segment_name", "is_premium"]], on="customer_id", how="left")
    df_final["is_premium"] = df_final["is_premium"].fillna(False)
    df_final["segment_name"] = df_final["segment_name"].fillna("")
    
    # Apply Routing Rules
    routes = []
    priorities = []
    for _, r in df_final.iterrows():
        st = r["status"]
        cat = r["category"]
        is_prem = bool(r["is_premium"])
        
        if st in ["skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"]:
            routes.append("human_review")
            priorities.append("NORMAL")
        elif cat == "fraud":
            routes.append("security_team")
            priorities.append("HIGH")
        elif is_prem:
            routes.append("premium_desk")
            priorities.append("HIGH")
        else:
            routes.append("support_queue")
            priorities.append("NORMAL")
            
    df_final["route"] = routes
    df_final["priority"] = priorities
    
    # Leave category and segment_name blank when not available / unclassified
    mask_unclass = ~df_final["status"].isin(["classified_json", "classified_keyword"])
    df_final.loc[mask_unclass, "category"] = ""
    df_final.loc[mask_unclass, "segment_name"] = ""
    
    out_cols = ["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]
    final_output = df_final[out_cols]
    final_csv_path = os.path.join(OUTPUTS, "triage_results.csv")
    final_output.to_csv(final_csv_path, index=False)
    print(f"Saved {final_csv_path} with {len(final_output)} rows.")
    print("\nRoute distribution:\n", final_output["route"].value_counts())
    print("\nPriority distribution:\n", final_output["priority"].value_counts())

if __name__ == "__main__":
    main()
