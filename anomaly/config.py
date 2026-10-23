"""Paths and constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "transactions.csv"
ARTIFACTS = ROOT / "artifacts"
MODEL_FILE = ARTIFACTS / "detector.joblib"
METRICS_FILE = ARTIFACTS / "metrics.json"
AUDIT_DB = ROOT / "audit_log.sqlite3"

MODEL_VERSION = "iforest-rules-v1"

LANE_CLEAR = "clear"
LANE_REVIEW = "review_queue"      # flagged, lower confidence: a person decides
LANE_HOLD = "hold"                # flagged, high confidence: hold the transaction
