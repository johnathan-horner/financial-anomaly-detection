"""Append-only audit log in SQLite. Every score and every analyst decision is a new row."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from anomaly.config import AUDIT_DB, LANE_REVIEW

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_events (
    event_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    transaction_id  TEXT NOT NULL,
    customer_id     TEXT NOT NULL,
    event_type      TEXT NOT NULL CHECK (event_type IN ('scored', 'review')),
    model_version   TEXT NOT NULL,
    amount          REAL NOT NULL,
    score           REAL NOT NULL,
    lane            TEXT NOT NULL,
    reasons         TEXT NOT NULL,      -- JSON list of plain-language reasons
    reviewer_action TEXT NOT NULL,      -- none, pending, confirmed_fraud, marked_legit
    reviewer        TEXT,
    note            TEXT,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_txn ON audit_events(transaction_id);
"""
COLUMNS = ("event_id", "transaction_id", "customer_id", "event_type", "model_version", "amount", "score",
           "lane", "reasons", "reviewer_action", "reviewer", "note", "created_at")
ACTIONS = ("confirmed_fraud", "marked_legit")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class AuditLog:
    def __init__(self, path: Path | str = AUDIT_DB):
        self.path = str(path)
        with sqlite3.connect(self.path) as conn:
            conn.executescript(SCHEMA)

    def _insert(self, row: dict) -> None:
        cols = [c for c in COLUMNS if c != "event_id"]
        with sqlite3.connect(self.path) as conn:
            conn.execute(f"INSERT INTO audit_events ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                         [row.get(c) for c in cols])

    def record_score(self, transaction_id: str, customer_id: str, model_version: str, amount: float,
                     score: float, lane: str, reasons: list[str]) -> None:
        self._insert({
            "transaction_id": transaction_id, "customer_id": customer_id, "event_type": "scored",
            "model_version": model_version, "amount": amount, "score": score, "lane": lane,
            "reasons": json.dumps(reasons), "reviewer_action": "pending" if lane == LANE_REVIEW else "none",
            "created_at": _now(),
        })

    def record_review(self, transaction_id: str, reviewer: str, action: str, note: str = "") -> dict:
        if action not in ACTIONS:
            raise ValueError(f"action must be one of {ACTIONS}")
        latest = self.latest(transaction_id)
        if latest is None:
            raise KeyError(f"unknown transaction {transaction_id}")
        row = {**latest, "event_type": "review", "reviewer_action": action, "reviewer": reviewer,
               "note": note, "created_at": _now()}
        self._insert(row)
        return self.latest(transaction_id)

    def events(self, transaction_id: str | None = None, limit: int | None = None) -> list[dict]:
        sql, args = f"SELECT {', '.join(COLUMNS)} FROM audit_events", ()
        if transaction_id:
            sql, args = sql + " WHERE transaction_id = ?", (transaction_id,)
        sql += " ORDER BY event_id" + (f" DESC LIMIT {int(limit)}" if limit else "")
        with sqlite3.connect(self.path) as conn:
            rows = [dict(zip(COLUMNS, r)) for r in conn.execute(sql, args)]
        for r in rows:
            r["reasons"] = json.loads(r["reasons"])
        return rows

    def latest(self, transaction_id: str) -> dict | None:
        rows = self.events(transaction_id)
        if not rows:
            return None
        row = rows[-1]
        row["reasons"] = json.dumps(row["reasons"])
        return row

    def review_queue(self) -> list[dict]:
        sql = f"""
            SELECT {', '.join('e.' + c for c in COLUMNS)} FROM audit_events e
            JOIN (SELECT transaction_id, MAX(event_id) AS last_id FROM audit_events GROUP BY transaction_id) l
              ON e.event_id = l.last_id
            WHERE e.reviewer_action = 'pending' ORDER BY e.score DESC
        """
        with sqlite3.connect(self.path) as conn:
            rows = [dict(zip(COLUMNS, r)) for r in conn.execute(sql)]
        for r in rows:
            r["reasons"] = json.loads(r["reasons"])
        return rows
