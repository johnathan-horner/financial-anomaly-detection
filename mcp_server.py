"""FastMCP server for the anomaly detector.

Every tool calls the real detector and audit log in the `anomaly` package; nothing is simulated and no
language model is called. Scoring is deterministic: the same request returns the same score.
Run with:  python mcp_server.py        (stdio transport)
"""

from __future__ import annotations

import os
from functools import lru_cache

from fastmcp import FastMCP

from anomaly.audit import AuditLog
from anomaly.rules import RULES
from anomaly.service import FraudService

mcp = FastMCP("Financial Anomaly Detection")


@lru_cache(maxsize=1)
def service() -> FraudService:
    return FraudService(audit=AuditLog(os.environ.get("ANOMALY_AUDIT_DB", "audit_log.sqlite3")))


@mcp.tool
def detect_transaction_anomaly(customer_id: str, amount: float, merchant_category: str, country: str = "US",
                               lat: float | None = None, lon: float | None = None) -> dict:
    """Score a new card transaction for a known customer.

    Returns a score from 0 to 1, a lane (clear, review_queue or hold) and plain-language reasons.
    Location defaults to the customer's home. Scoring does not change the customer's history, so the
    same request always gets the same score. Each call is written to the audit log.
    """
    return service().score_transaction(customer_id, amount, merchant_category, country, lat, lon, remember=False)


@mcp.tool
def investigate_customer_pattern(customer_id: str, days_back: int = 30) -> dict:
    """Summarize a customer's recent spending: count, typical and largest amounts, categories, countries."""
    return service().customer_profile(customer_id, days_back)


@mcp.tool
def analyze_merchant_risk(merchant_category: str) -> dict:
    """Transaction count, median amount and labeled anomaly rate for a merchant category (synthetic data)."""
    return service().category_stats(merchant_category)


@mcp.tool
def get_investigation_queue() -> list[dict]:
    """Flags waiting for an analyst, highest score first."""
    return service().review_queue()


@mcp.tool
def record_analyst_decision(transaction_id: str, reviewer: str, action: str, note: str = "") -> dict:
    """Record an analyst's decision on a flagged transaction. action is confirmed_fraud or marked_legit."""
    return service().record_review(transaction_id, reviewer, action, note)


@mcp.tool
def get_system_metrics() -> dict:
    """Precision and recall measured on the held-out synthetic test period, plus the thresholds in use."""
    m = service().metrics()
    return {"model_version": m["model_version"], "thresholds": m["thresholds"], "test": m["test"]}


@mcp.resource("model://detector")
def detector_info() -> dict:
    """Model version, thresholds and how each rule was weighted."""
    s = service()
    return {"model_version": s.detector.version, **s.thresholds()}


@mcp.resource("data://risk_thresholds")
def risk_thresholds() -> dict:
    """The rules and the score thresholds that decide each lane."""
    return {"rules": RULES, "thresholds": service().thresholds()}


@mcp.prompt("fraud_investigation")
def fraud_investigation_prompt(transaction_id: str) -> str:
    """Steps for reviewing one flagged transaction."""
    return (
        f"Review flagged transaction {transaction_id}. Use get_investigation_queue to read its score and "
        "reasons, then investigate_customer_pattern for the customer's usual behavior. Decide whether the "
        "reasons are explained by normal behavior (travel, a planned large purchase) or not, and record the "
        "outcome with record_analyst_decision. The data is synthetic."
    )


if __name__ == "__main__":
    mcp.run()
