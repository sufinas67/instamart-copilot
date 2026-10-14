# Instamart Customer Intelligence & Support Copilot — Capstone Project Brief

Swiggy Instamart delivers groceries, dairy, snacks, personal-care items and small electronics in six metro cities within minutes. As the order base grows, three problems keep surfacing in the weekly leadership review: nobody trusts the raw order exports (duplicate rows, mixed spellings, impossible values), the retention team cannot tell which customers are about to stop ordering, and the support desk is drowning in thousands of free-text tickets that mix refunds, fraud reports, app crashes and late deliveries. The Head of Customer Experience has asked a small data and AI team to build one internal tool that fixes all three: a Customer Intelligence & Support Copilot.

You are that team. Over 21 days you will build the Copilot as one repository, in four connected Parts. Part A turns the messy exports into a trusted analytical database. Part B uses that database to segment customers and to predict who is likely to lapse. Part C reads support tickets, protects customer privacy, and routes each ticket using the segments from Part B. Part D builds a policy question-answering assistant and a refund agent that queries the database from Part A and reuses the policy retrieval built in the same Part. Every Part consumes what the previous Parts produced, so earlier mistakes travel forward — and the numeric checkpoints in this brief let you catch them early.

All data in this project is synthetic and fictional, produced by a script given below. The brand name is used only to frame the scenario; nothing here is real Instamart data, policy or pricing. All monetary values are in Indian Rupees (INR).

Total marks: 100 · One deliverable: a single public GitHub repository containing the complete Copilot (data pipeline, SQL, models, support-triage pipeline, policy assistant, refund agent) and one README.md that documents it.

Part A: Data Foundation: Clean, Store, Query (22 marks)
Part B: Customer Segments and Lapse Prediction (28 marks)
Part C: Support Ticket Triage with Guardrails (22 marks)
Part D: Policy RAG Assistant and Refund Agent (28 marks)
