"""src/agent.py -- ReAct Refund Agent with safety limits, state persistence, and resume capability."""
import os
import sys
import json
import time
import sqlite3
sys.path.insert(0, os.path.abspath("."))
import pandas as pd
import mock_llm
from src.rag import PolicyRetriever

DB_PATH = os.path.join("data", "instamart.db")
LEDGER_PATH = os.path.join("outputs", "refund_ledger.jsonl")
STATE_DIR = "state"
os.makedirs("outputs", exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)

_retriever = None
def get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = PolicyRetriever(overlap=10)
    return _retriever

def get_order(order_id: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?", (order_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return {"error": "order not found"}
    return {
        "order_id": row[0],
        "status": row[1],
        "order_amount": float(row[2]),
        "delivery_minutes": int(row[3]) if row[3] is not None and not pd.isna(row[3]) else None
    }

def search_policy(query: str):
    r = get_retriever()
    results = r.retrieve(query, k=3)
    compact = [{"source": res["source"], "text": res["text"][:100]} for res in results]
    return {"results": compact}

def calculate_refund(order_amount: float, delivery_minutes: int):
    amt = float(order_amount)
    deliv = int(delivery_minutes) if delivery_minutes is not None else 0
    if deliv > 45:
        return {"refund_inr": amt}
    return {"refund_inr": 0.0}

def issue_refund(order_id: str, refund_inr: float):
    amt = float(refund_inr)
    entry = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "tool": "issue_refund", "order_id": order_id, "refund_inr": amt}
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"refund_inr": amt}

def escalate_to_human(order_id: str, refund_inr: float):
    amt = float(refund_inr)
    entry = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "tool": "escalate_to_human", "order_id": order_id, "refund_inr": amt}
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"refund_inr": amt}

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human
}

TOOLS_SPEC = [
    {"name": "get_order", "description": "Fetches order details from database.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}},
    {"name": "search_policy", "description": "Searches policy documents.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "calculate_refund", "description": "Calculates refund based on delivery time.", "parameters": {"type": "object", "properties": {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "integer"}}, "required": ["order_amount", "delivery_minutes"]}},
    {"name": "issue_refund", "description": "Issues automatic refund under INR 1000.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, "required": ["order_id", "refund_inr"]}},
    {"name": "escalate_to_human", "description": "Escalates refund over INR 1000 to human lead.", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}, "required": ["order_id", "refund_inr"]}}
]

def run_agent(user_message: str, scenario_id: str, max_iterations: int = 6, token_limit: int = 6000, time_limit_seconds: float = 10.0, run_id: str = None, crash_after_tool_step: int = None):
    if run_id is None:
        run_id = f"run_{scenario_id}_{int(time.time()*1000)}"
    start_time = time.monotonic()
    messages = [
        {"role": "system", "content": "AGENT. You are an autonomous refund assistant. Reason and call tools."},
        {"role": "user", "content": user_message}
    ]
    steps, tokens_used, llm_calls, tools_executed = [], 0, 0, []
    
    for it in range(max_iterations):
        if (time.monotonic() - start_time) >= time_limit_seconds:
            return {"scenario_id": scenario_id, "status": "FALLBACK_TIMEOUT", "final_answer": "Execution time limit exceeded. Handed over to human agent with context.", "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used, "tools_executed": tools_executed, "run_id": run_id}
            
        resp = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += resp.usage.total_tokens
        choice = resp.choices[0]
        
        if choice.message.tool_calls:
            if tokens_used > token_limit:
                return {"scenario_id": scenario_id, "status": "FALLBACK_TOKEN_LIMIT", "final_answer": "Session token budget exceeded. Handed over to human agent with context.", "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used, "tools_executed": tools_executed, "run_id": run_id}
            
            tc = choice.message.tool_calls[0]
            call_spec = {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
            messages.append({"role": "assistant", "content": None, "tool_calls": [call_spec]})
            
            tool_fn = TOOL_MAP[tc.name]
            obs = tool_fn(**tc.arguments)
            tools_executed.append(tc.name)
            messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": json.dumps(obs)})
            steps.append({"step": len(steps) + 1, "tool": tc.name, "arguments": tc.arguments, "observation": obs})
            
            ckpt = {"run_id": run_id, "scenario_id": scenario_id, "messages": messages, "tokens_used": tokens_used, "llm_calls": llm_calls, "step_number": len(steps), "tools_executed": tools_executed, "steps": steps}
            with open(os.path.join(STATE_DIR, f"{run_id}.json"), "w", encoding="utf-8") as f:
                json.dump(ckpt, f, indent=2)
                
            if crash_after_tool_step is not None and len(tools_executed) == crash_after_tool_step:
                raise RuntimeError(f"Simulated crash after tool step {crash_after_tool_step}")
        else:
            final_content = choice.message.content or ""
            raw_status = final_content.split(":")[0].strip() if ":" in final_content else "COMPLETED"
            return {"scenario_id": scenario_id, "status": raw_status, "final_answer": final_content, "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used, "tools_executed": tools_executed, "run_id": run_id}
            
    return {"scenario_id": scenario_id, "status": "FALLBACK_MAX_ITERATIONS", "final_answer": "Maximum iteration limit reached. Handed over to human agent with context.", "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used, "tools_executed": tools_executed, "run_id": run_id}

def resume_agent(run_id: str, max_iterations: int = 6, token_limit: int = 6000, time_limit_seconds: float = 10.0):
    ckpt_path = os.path.join(STATE_DIR, f"{run_id}.json")
    with open(ckpt_path, "r", encoding="utf-8") as f:
        ckpt = json.load(f)
    messages = ckpt["messages"]
    tokens_used = ckpt["tokens_used"]
    llm_calls = ckpt["llm_calls"]
    tools_executed = list(ckpt["tools_executed"])
    steps = list(ckpt["steps"])
    scenario_id = ckpt["scenario_id"]
    start_time = time.monotonic()
    
    for it in range(len(tools_executed), max_iterations):
        if (time.monotonic() - start_time) >= time_limit_seconds:
            return {"scenario_id": scenario_id, "status": "FALLBACK_TIMEOUT", "final_answer": "Execution time limit exceeded. Handed over to human agent with context.", "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used, "tools_executed": tools_executed, "run_id": run_id}
        resp = mock_llm.chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += resp.usage.total_tokens
        choice = resp.choices[0]
        if choice.message.tool_calls:
            if tokens_used > token_limit:
                return {"scenario_id": scenario_id, "status": "FALLBACK_TOKEN_LIMIT", "final_answer": "Session token budget exceeded. Handed over to human agent with context.", "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used, "tools_executed": tools_executed, "run_id": run_id}
            tc = choice.message.tool_calls[0]
            call_spec = {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
            messages.append({"role": "assistant", "content": None, "tool_calls": [call_spec]})
            tool_fn = TOOL_MAP[tc.name]
            obs = tool_fn(**tc.arguments)
            tools_executed.append(tc.name)
            messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": json.dumps(obs)})
            steps.append({"step": len(steps) + 1, "tool": tc.name, "arguments": tc.arguments, "observation": obs})
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump({"run_id": run_id, "scenario_id": scenario_id, "messages": messages, "tokens_used": tokens_used, "llm_calls": llm_calls, "step_number": len(steps), "tools_executed": tools_executed, "steps": steps}, f, indent=2)
        else:
            final_content = choice.message.content or ""
            raw_status = final_content.split(":")[0].strip() if ":" in final_content else "COMPLETED"
            return {"scenario_id": scenario_id, "status": raw_status, "final_answer": final_content, "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used, "tools_executed": tools_executed, "run_id": run_id}
    return {"scenario_id": scenario_id, "status": "FALLBACK_MAX_ITERATIONS", "final_answer": "Maximum iteration limit reached. Handed over to human agent with context.", "steps": steps, "llm_calls": llm_calls, "tokens_used": tokens_used, "tools_executed": tools_executed, "run_id": run_id}
