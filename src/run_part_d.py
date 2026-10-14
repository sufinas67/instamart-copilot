"""src/run_part_d.py -- Executes Part D: RAG evaluation, ReAct refund agent scenarios, safe-stops, and resumability demos."""
import os
import sys
import glob
import json
import time
sys.path.insert(0, os.path.abspath("."))
import pandas as pd
import mock_llm
from src.rag import chunk_documents, PolicyRetriever, evaluate_retrieval
from src.agent import run_agent, resume_agent

OUTPUTS_DIR = "outputs"
STATE_DIR = "state"
LEDGER_PATH = os.path.join(OUTPUTS_DIR, "refund_ledger.jsonl")
RUNS_PATH = os.path.join(OUTPUTS_DIR, "agent_runs.jsonl")

def cleanup_previous_runs():
    if os.path.exists(LEDGER_PATH):
        os.remove(LEDGER_PATH)
    if os.path.exists(RUNS_PATH):
        os.remove(RUNS_PATH)
    for f in glob.glob(os.path.join(STATE_DIR, "*.json")):
        try:
            os.remove(f)
        except Exception:
            pass

def main():
    cleanup_previous_runs()
    print("=== PART D: POLICY RAG AND REFUND AGENT ===")
    
    # 1. RAG Chunks Check
    chunks_10, counts_10 = chunk_documents(overlap=10)
    print(f"Chunks (overlap=10): Total={len(chunks_10)}, per-file={counts_10}")
    
    # 2. RAG Evaluation
    eval_10_df, _ = evaluate_retrieval(overlap=10)
    ans_10 = eval_10_df[~eval_10_df["is_unanswerable"]]
    unans_10 = eval_10_df[eval_10_df["is_unanswerable"]]
    doc_hits_10 = int(ans_10["doc_match"].sum())
    fact_hits_10 = int(ans_10["fact_match"].sum())
    fb_hits_10 = int(unans_10["fact_match"].sum())
    
    eval_0_df, _ = evaluate_retrieval(overlap=0)
    ans_0 = eval_0_df[~eval_0_df["is_unanswerable"]]
    doc_hits_0 = int(ans_0["doc_match"].sum())
    fact_hits_0 = int(ans_0["fact_match"].sum())
    failing_0 = ans_0[~ans_0["fact_match"]]
    failing_q_name = failing_0.iloc[0]["question"] if len(failing_0) > 0 else "None"
    
    # 3. Run 7 Scenarios
    scenarios_df = pd.read_csv(os.path.join("data", "raw", "agent_scenarios.csv"), keep_default_na=False)
    scenario_results = []
    
    print("\n--- Running 7 Refund Agent Scenarios ---")
    for _, r in scenarios_df.iterrows():
        sid = r["scenario_id"]
        cid = r["customer_id"]
        msg = r["message"]
        exp_oc = r["expected_outcome"]
        exp_amt = float(r["expected_refund_inr"])
        
        mock_llm.reset()
        res = run_agent(msg, sid)
        
        ref_amt = 0.0
        if res["status"] == "REFUND":
            for step in res["steps"]:
                if step["tool"] == "issue_refund":
                    ref_amt = float(step["observation"].get("refund_inr", 0.0))
        elif res["status"] == "ESCALATE":
            for step in res["steps"]:
                if step["tool"] == "escalate_to_human":
                    ref_amt = float(step["observation"].get("refund_inr", 0.0))
                    
        scenario_results.append({
            "scenario_id": sid,
            "status": res["status"],
            "expected_outcome": exp_oc,
            "refund_amount": ref_amt,
            "expected_refund_inr": exp_amt,
            "model_calls": res["llm_calls"],
            "tokens_used": res["tokens_used"],
            "final_answer": res["final_answer"]
        })
        
        run_record = {
            "scenario_id": sid,
            "status": res["status"],
            "final_answer": res["final_answer"],
            "llm_calls": res["llm_calls"],
            "tokens_used": res["tokens_used"],
            "steps": res["steps"]
        }
        with open(RUNS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(run_record) + "\n")
            
    sc_df = pd.DataFrame(scenario_results)
    print("\n=== SEVEN-SCENARIO RESULTS TABLE ===")
    print(sc_df[["scenario_id", "status", "refund_amount", "model_calls", "tokens_used"]].to_string(index=False))
    
    # 4. Safe Stop Demonstrations
    print("\n--- Safe Stop Demonstrations ---")
    s1_row = scenarios_df.iloc[0]
    mock_llm.reset()
    res_token_demo = run_agent(s1_row["message"], "S1_token_demo", token_limit=200)
    print(f"Token limit demo status: {res_token_demo['status']}")
    assert res_token_demo["status"] == "FALLBACK_TOKEN_LIMIT"
    
    mock_llm.reset()
    res_time_demo = run_agent(s1_row["message"], "S1_time_demo", time_limit_seconds=0.0)
    print(f"Timeout demo status: {res_time_demo['status']}")
    assert res_time_demo["status"] == "FALLBACK_TIMEOUT"
    
    # 5. Crash and Resume Demonstration on S1
    print("\n--- Crash and Resume Demonstration on S1 ---")
    mock_llm.reset()
    crash_run_id = f"crash_demo_{int(time.time()*1000)}"
    crash_occurred = False
    try:
        run_agent(s1_row["message"], "S1_crash", run_id=crash_run_id, crash_after_tool_step=2)
    except RuntimeError as e:
        crash_occurred = True
        print(f"Crash simulated successfully: {e}")
    assert crash_occurred
    
    with open(os.path.join(STATE_DIR, f"{crash_run_id}.json"), "r", encoding="utf-8") as f:
        pre_crash_ckpt = json.load(f)
    pre_crash_tools = pre_crash_ckpt["tools_executed"]
    print(f"Tools executed before crash: {pre_crash_tools}")
    
    resumed_res = resume_agent(crash_run_id)
    post_resume_tools = resumed_res["tools_executed"]
    print(f"Tools executed across crash and resume: {post_resume_tools}")
    print(f"Final status after resume: {resumed_res['status']}, Final answer: {resumed_res['final_answer']}")
    
    assert post_resume_tools == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert resumed_res["status"] == "REFUND"
    
    # 6. Part D PASS/FAIL Table
    print("\n=== PART D VERIFICATION TABLE ===")
    checks = [
        ("Chunk count (overlap 10)", len(chunks_10), 15, len(chunks_10) == 15),
        ("Policy QA Doc Hits (10 overlap)", doc_hits_10, 12, doc_hits_10 == 12),
        ("Policy QA Fact Hits (10 overlap)", fact_hits_10, 12, fact_hits_10 == 12),
        ("Policy QA Fallbacks (10 overlap)", fb_hits_10, 2, fb_hits_10 == 2),
        ("Policy QA Zero-overlap Failure", fact_hits_0, 10, fact_hits_0 < 12),
        ("S1 outcome & refund", f"{sc_df.loc[0, 'status']} INR {sc_df.loc[0, 'refund_amount']:.2f}", "REFUND INR 128.25", sc_df.loc[0, 'status'] == "REFUND" and abs(sc_df.loc[0, 'refund_amount'] - 128.25) < 0.01),
        ("S1 model calls", sc_df.loc[0, 'model_calls'], 5, sc_df.loc[0, 'model_calls'] == 5),
        ("S2 outcome & refund", f"{sc_df.loc[1, 'status']} INR {sc_df.loc[1, 'refund_amount']:.2f}", "REFUND INR 135.00", sc_df.loc[1, 'status'] == "REFUND" and abs(sc_df.loc[1, 'refund_amount'] - 135.0) < 0.01),
        ("S3 outcome & refund", f"{sc_df.loc[2, 'status']} INR {sc_df.loc[2, 'refund_amount']:.2f}", "ESCALATE INR 1999.00", sc_df.loc[2, 'status'] == "ESCALATE" and abs(sc_df.loc[2, 'refund_amount'] - 1999.0) < 0.01),
        ("S4 outcome", sc_df.loc[3, 'status'], "DENIED", sc_df.loc[3, 'status'] == "DENIED"),
        ("S5 outcome", sc_df.loc[4, 'status'], "DENIED", sc_df.loc[4, 'status'] == "DENIED"),
        ("S6 outcome", sc_df.loc[5, 'status'], "NOT_FOUND", sc_df.loc[5, 'status'] == "NOT_FOUND"),
        ("S7 fallback max iterations", sc_df.loc[6, 'status'], "FALLBACK_MAX_ITERATIONS", sc_df.loc[6, 'status'] == "FALLBACK_MAX_ITERATIONS"),
        ("S7 model calls", sc_df.loc[6, 'model_calls'], 6, sc_df.loc[6, 'model_calls'] == 6),
        ("S1 token limit demo", res_token_demo["status"], "FALLBACK_TOKEN_LIMIT", res_token_demo["status"] == "FALLBACK_TOKEN_LIMIT"),
        ("S1 timeout demo", res_time_demo["status"], "FALLBACK_TIMEOUT", res_time_demo["status"] == "FALLBACK_TIMEOUT"),
        ("Crash & Resume 4 tools executed", str(post_resume_tools), "['get_order', 'search_policy', 'calculate_refund', 'issue_refund']", post_resume_tools == ["get_order", "search_policy", "calculate_refund", "issue_refund"])
    ]
    
    chk_df = pd.DataFrame(checks, columns=["check", "mine", "reference", "PASS/FAIL"])
    chk_df["PASS/FAIL"] = chk_df["PASS/FAIL"].apply(lambda p: "PASS" if p else "FAIL")
    print(chk_df.to_string(index=False))
    print(f"\nZero-overlap failing question: '{failing_q_name}'")
    assert (chk_df["PASS/FAIL"] == "PASS").all(), "Some Part D checks failed!"

if __name__ == "__main__":
    main()
