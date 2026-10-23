"""Score new transactions against each customer's history. Used by the app and the MCP server."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import datetime

import pandas as pd

from anomaly.audit import AuditLog
from anomaly.config import METRICS_FILE, ROOT
from anomaly.detector import Detector
from anomaly.features import build_features
from anomaly.rules import RULES

HISTORY_FILE = ROOT / "data" / "recent_history.csv.gz"
HISTORY_COLUMNS = ["transaction_id", "timestamp", "customer_id", "amount", "merchant_category", "country",
                   "lat", "lon", "home_lat", "home_lon"]


class FraudService:
    def __init__(self, detector: Detector | None = None, history: pd.DataFrame | None = None,
                 audit: AuditLog | None = None):
        self.detector = detector or Detector.load()
        if history is None:
            history = pd.read_csv(HISTORY_FILE, parse_dates=["timestamp"])
        self.history = history.copy()
        self.history["timestamp"] = pd.to_datetime(self.history["timestamp"])
        self.audit = audit or AuditLog()
        self.homes = self.history.groupby("customer_id")[["home_lat", "home_lon"]].first()

    def customers(self) -> list[str]:
        return sorted(self.homes.index)

    def score_transaction(self, customer_id: str, amount: float, merchant_category: str, country: str = "US",
                          lat: float | None = None, lon: float | None = None,
                          timestamp: datetime | str | None = None) -> dict:
        """Score one new transaction, add it to the customer's history and write it to the audit log."""
        if customer_id not in self.homes.index:
            raise KeyError(f"unknown customer {customer_id}")
        if amount <= 0:
            raise ValueError("amount must be positive")
        home = self.homes.loc[customer_id]
        past = self.history[self.history["customer_id"] == customer_id]
        if timestamp is not None:
            ts = pd.Timestamp(timestamp)
        else:
            # Default: the day after this customer's latest transaction, at their usual hour.
            usual = int(past["timestamp"].dt.hour.median()) if len(past) else 12
            ts = (past["timestamp"].max() + pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=usual)
        row = {
            "transaction_id": "N" + uuid.uuid4().hex[:10], "timestamp": ts, "customer_id": customer_id,
            "amount": float(amount), "merchant_category": merchant_category, "country": country,
            "lat": float(home.home_lat if lat is None else lat), "lon": float(home.home_lon if lon is None else lon),
            "home_lat": float(home.home_lat), "home_lon": float(home.home_lon),
        }
        feats = build_features(pd.concat([past[HISTORY_COLUMNS], pd.DataFrame([row])], ignore_index=True))
        new = feats[feats["transaction_id"] == row["transaction_id"]]
        decision = self.detector.decide(new)[0]
        self.history = pd.concat([self.history, pd.DataFrame([row])], ignore_index=True)
        self.audit.record_score(row["transaction_id"], customer_id, self.detector.version, row["amount"],
                                decision.score, decision.lane, decision.reasons)
        return {"transaction_id": row["transaction_id"], "customer_id": customer_id, "amount": row["amount"],
                "merchant_category": merchant_category, "country": country, "timestamp": ts.isoformat(),
                "model_version": self.detector.version, **asdict(decision)}

    def customer_profile(self, customer_id: str, days_back: int = 30) -> dict:
        h = self.history[self.history["customer_id"] == customer_id]
        if h.empty:
            raise KeyError(f"unknown customer {customer_id}")
        h = h[h["timestamp"] >= h["timestamp"].max() - pd.Timedelta(days=days_back)]
        return {
            "customer_id": customer_id, "days": days_back, "transactions": len(h),
            "median_amount": round(float(h["amount"].median()), 2),
            "max_amount": round(float(h["amount"].max()), 2),
            "usual_categories": h["merchant_category"].value_counts().head(5).to_dict(),
            "countries": sorted(h["country"].unique().tolist()),
            "typical_hour": int(h["timestamp"].dt.hour.median()),
        }

    def category_stats(self, merchant_category: str) -> dict:
        h = self.history[self.history["merchant_category"] == merchant_category]
        if h.empty:
            raise KeyError(f"unknown merchant category {merchant_category}")
        out = {"merchant_category": merchant_category, "transactions": len(h),
               "median_amount": round(float(h["amount"].median()), 2)}
        if "is_anomaly" in h and h["is_anomaly"].notna().any():
            out["labeled_anomaly_rate"] = round(float(h["is_anomaly"].mean()), 4)
        return out

    def review_queue(self) -> list[dict]:
        return self.audit.review_queue()

    def record_review(self, transaction_id: str, reviewer: str, action: str, note: str = "") -> dict:
        return self.audit.record_review(transaction_id, reviewer, action, note)

    @staticmethod
    def metrics() -> dict:
        return json.loads(METRICS_FILE.read_text())

    def thresholds(self) -> dict:
        return {"flag": round(self.detector.flag_threshold, 4), "hold": round(self.detector.hold_threshold, 4),
                "rule_precision": self.detector.rule_precision, "rules": RULES}
