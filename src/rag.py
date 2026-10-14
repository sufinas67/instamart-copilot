"""src/part_d/rag.py -- Document chunking, TF-IDF vector retrieval, RAG QA assistant, and overlap evaluation."""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mock_llm

POLICY_DOCS = [
    "refund_policy.txt",
    "delivery_policy.txt",
    "escalation_policy.txt",
    "payment_policy.txt",
    "privacy_policy.txt"
]

def chunk_documents(overlap=10, chunk_size=40):
    """
    Splits policy documents into chunks of chunk_size words with specified overlap words.
    First chunk: 0..39, next starts (chunk_size - overlap) words later.
    Stops after first chunk whose end reaches or passes the last word of the document.
    """
    step = chunk_size - overlap
    chunks = []
    chunk_counts = {}
    
    for fn in POLICY_DOCS:
        p = os.path.join("data", "policies", fn)
        with open(p, "r", encoding="utf-8") as f:
            words = f.read().strip().split()
            
        count = 0
        i = 0
        while i < len(words):
            chunk_words = words[i : i + chunk_size]
            chunks.append({
                "source": fn,
                "text": " ".join(chunk_words),
                "chunk_id": f"{fn}_{count}"
            })
            count += 1
            if i + chunk_size >= len(words):
                break
            i += step
        chunk_counts[fn] = count
        
    return chunks, chunk_counts

class PolicyRetriever:
    def __init__(self, overlap=10):
        self.chunks, self.chunk_counts = chunk_documents(overlap=overlap)
        self.vectorizer = TfidfVectorizer()
        self.chunk_matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])
        
    def retrieve(self, query: str, k: int = 3):
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.chunk_matrix)[0]
        top_indices = sims.argsort()[::-1][:k]
        return [
            {
                "source": self.chunks[i]["source"],
                "score": float(sims[i]),
                "text": self.chunks[i]["text"]
            }
            for i in top_indices
        ]

def answer_policy_question(question: str, retriever: PolicyRetriever):
    """Answers question using retrieved policy context and mock_llm Policy Assistant."""
    top_chunks = retriever.retrieve(question, k=3)
    context_lines = []
    for c in top_chunks:
        context_lines.append(f"[{c['source']}]\n{c['text']}")
    context_block = "\n".join(context_lines)
    
    user_content = f"CONTEXT:\n{context_block}\nQUESTION:\n{question}"
    messages = [
        {
            "role": "system",
            "content": "POLICY ASSISTANT. Answer only from the supplied context. If the answer cannot be found in the context, say: I could not find that information in the policy documents."
        },
        {"role": "user", "content": user_content}
    ]
    resp = mock_llm.chat(messages)
    return resp.choices[0].message.content, [c["source"] for c in top_chunks]

def evaluate_retrieval(overlap=10):
    retriever = PolicyRetriever(overlap=overlap)
    eval_df = pd.read_csv(os.path.join("data", "raw", "policy_eval.csv"), keep_default_na=False)
    
    results = []
    for _, r in eval_df.iterrows():
        q = r["question"]
        exp_doc = r["expected_doc"]
        exp_fact = r["expected_fact"]
        ans, sources = answer_policy_question(q, retriever)
        
        if exp_doc == "NONE":
            is_fallback = ans.startswith("I could not find")
            results.append({
                "question": q,
                "expected_doc": exp_doc,
                "expected_fact": "(Fallback)",
                "sources": sources,
                "answer": ans,
                "doc_match": True,
                "fact_match": is_fallback,
                "is_unanswerable": True
            })
        else:
            doc_ok = exp_doc in sources
            fact_ok = exp_fact in ans
            results.append({
                "question": q,
                "expected_doc": exp_doc,
                "expected_fact": exp_fact,
                "sources": sources,
                "answer": ans,
                "doc_match": doc_ok,
                "fact_match": fact_ok,
                "is_unanswerable": False
            })
            
    res_df = pd.DataFrame(results)
    return res_df, retriever.chunk_counts

def main():
    print("=== TASK 1: CHUNK DOCUMENTS (OVERLAP = 10) ===")
    chunks_10, counts_10 = chunk_documents(overlap=10)
    print(f"Total chunks: {len(chunks_10)}")
    print(f"Chunks per file: {counts_10}")
    assert len(chunks_10) == 15
    assert counts_10 == {"refund_policy.txt": 4, "delivery_policy.txt": 3, "escalation_policy.txt": 3, "payment_policy.txt": 3, "privacy_policy.txt": 2}
    
    print("\n=== TASK 2: RETRIEVE TEST ===")
    retriever_10 = PolicyRetriever(overlap=10)
    test_q = "refund policy late delivery"
    top_test = retriever_10.retrieve(test_q, k=3)
    print(f"Top 3 sources for '{test_q}': {[t['source'] for t in top_test]}")
    assert all(t["source"] == "refund_policy.txt" for t in top_test)
    
    print("\n=== TASK 4: EVALUATION WITH OVERLAP = 10 ===")
    eval_10_df, _ = evaluate_retrieval(overlap=10)
    ans_10 = eval_10_df[~eval_10_df["is_unanswerable"]]
    unans_10 = eval_10_df[eval_10_df["is_unanswerable"]]
    doc_hits_10 = ans_10["doc_match"].sum()
    fact_hits_10 = ans_10["fact_match"].sum()
    fb_hits_10 = unans_10["fact_match"].sum()
    print(f"Answerable Doc Hits: {doc_hits_10} / 12")
    print(f"Answerable Fact Hits: {fact_hits_10} / 12")
    print(f"Unanswerable Fallbacks: {fb_hits_10} / 2")
    assert doc_hits_10 == 12 and fact_hits_10 == 12 and fb_hits_10 == 2
    
    print("\n=== TASK 4: EVALUATION WITH OVERLAP = 0 ===")
    eval_0_df, counts_0 = evaluate_retrieval(overlap=0)
    ans_0 = eval_0_df[~eval_0_df["is_unanswerable"]]
    doc_hits_0 = ans_0["doc_match"].sum()
    fact_hits_0 = ans_0["fact_match"].sum()
    print(f"Zero Overlap Doc Hits: {doc_hits_0} / 12")
    print(f"Zero Overlap Fact Hits: {fact_hits_0} / 12")
    failed_0 = ans_0[~ans_0["fact_match"]]
    print("\nQuestions failing with zero overlap:")
    for _, f_row in failed_0.iterrows():
        print(f"- Question: {f_row['question']}")
        print(f"  Expected Fact: '{f_row['expected_fact']}'")
        print(f"  Answer Received: '{f_row['answer']}'")
        print("  Explanation: The sentence containing the factual answer was split across a chunk boundary, causing retrieval or text extraction to miss the key phrase.")
        
    assert fact_hits_0 < 12, "At least one answerable question must fail with zero overlap!"
    print("\nALL POLICY RAG TASKS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
