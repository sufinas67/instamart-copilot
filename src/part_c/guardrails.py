import sys, os
sys.path.insert(0, os.path.abspath('.'))
"""src/part_c/guardrails.py -- Input guardrails, token estimation, and regex-based PII redaction."""
import re
import mock_llm

MAX_INPUT_TOKENS = 400

# Strict regex patterns specified in brief
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_PATTERN = re.compile(r"\b[6-9]\d{9}\b")
ORDER_ID_PATTERN = re.compile(r"ORD-\d{6}")

def check_input_guardrails(text: str):
    """
    Evaluates input guardrails:
    (a) Empty or only whitespace -> skipped_empty
    (b) Estimated tokens > MAX_INPUT_TOKENS (400) -> flagged_long
    (c) Otherwise -> ok
    """
    if not text or text.strip() == "":
        return "skipped_empty"
    
    tokens = mock_llm.est_tokens(text)
    if tokens > MAX_INPUT_TOKENS:
        return "flagged_long"
        
    return "ok"

def redact_pii(text: str):
    """
    Applies exact regular expressions and placeholders in order:
    1. Email -> [EMAIL]
    2. Phone -> [PHONE]
    3. Order ID -> [ORDER_ID]
    Returns (redacted_text, count_email, count_phone, count_order)
    """
    if not text:
        return "", 0, 0, 0
        
    t, n_email = EMAIL_PATTERN.subn("[EMAIL]", text)
    t, n_phone = PHONE_PATTERN.subn("[PHONE]", t)
    t, n_order = ORDER_ID_PATTERN.subn("[ORDER_ID]", t)
    
    return t, n_email, n_phone, n_order

def verify_no_pii_remains(text: str) -> bool:
    """Verifies that no unmasked email, phone, or order ID remains in the text."""
    if not text:
        return True
    return (
        len(EMAIL_PATTERN.findall(text)) == 0 and
        len(PHONE_PATTERN.findall(text)) == 0 and
        len(ORDER_ID_PATTERN.findall(text)) == 0
    )
