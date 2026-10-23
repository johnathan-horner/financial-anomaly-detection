"""Isolation Forest plus rules, combined into one score with a lane and plain reasons.

The model's percentile and each rule are put on the same scale by calibrating them on a
validation period: the model percentile maps to an estimated probability of anomaly (isotonic
regression) and each rule maps to its observed precision. Evidence is combined as
1 - product(1 - p), so two independent signals score higher than either alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.isotonic import IsotonicRegression

from anomaly import rules
from anomaly.config import LANE_CLEAR, LANE_HOLD, LANE_REVIEW, MODEL_FILE, MODEL_VERSION
from anomaly.features import FEATURES

FEATURE_LABELS = {
    "log_amount_ratio": "Amount far from this customer's usual",
    "hour_distance": "Unusual time of day for this customer",
    "log_km_from_home": "Far from home",
    "log_speed_kmh": "Fast travel since the last transaction",
    "n_last_10min": "Several transactions in the last 10 minutes",
    "n_last_60min": "Several transactions in the last hour",
    "n_small_last_10min": "Several small charges in the last 10 minutes",
    "new_category": "First purchase in this merchant category",
    "foreign": "Purchase outside the US",
}


@dataclass
class Decision:
    score: float           # combined estimate that this is an anomaly, 0 to 1
    model_score: float     # Isolation Forest percentile, 0 to 1
    lane: str
    reasons: list[str] = field(default_factory=list)
    rules_hit: list[str] = field(default_factory=list)


@dataclass
class Detector:
    model: IsolationForest
    train_scores: np.ndarray          # sorted raw anomaly scores on training data
    medians: np.ndarray
    scales: np.ndarray
    calibrator: IsotonicRegression | None = None
    rule_precision: dict = field(default_factory=dict)
    flag_threshold: float = 0.5
    hold_threshold: float = 1.01
    version: str = MODEL_VERSION

    @classmethod
    def fit(cls, train: pd.DataFrame, seed: int = 0) -> Detector:
        x = train[FEATURES].to_numpy(dtype=float)
        model = IsolationForest(n_estimators=200, max_samples=4096, random_state=seed, n_jobs=-1).fit(x)
        raw = np.sort(-model.score_samples(x))
        med = np.median(x, axis=0)
        mad = np.median(np.abs(x - med), axis=0)
        scale = np.where(mad > 0, 1.4826 * mad, x.std(axis=0) + 1e-6)
        return cls(model, raw, med, scale)

    def model_scores(self, feats: pd.DataFrame) -> np.ndarray:
        raw = -self.model.score_samples(feats[FEATURES].to_numpy(dtype=float))
        return np.searchsorted(self.train_scores, raw, side="right") / len(self.train_scores)

    def rule_hits(self, feats: pd.DataFrame) -> list[list[rules.RuleHit]]:
        return [rules.evaluate(r) for r in feats.to_dict("records")]

    def calibrate(self, val: pd.DataFrame) -> None:
        """Fit the model calibration and per-rule precision on validation data."""
        y = val["is_anomaly"].to_numpy()
        self.calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
        self.calibrator.fit(self.model_scores(val), y)
        hits = self.rule_hits(val)
        for name in rules.RULES:
            fired = np.array([any(h.rule == name for h in hs) for hs in hits])
            # Laplace smoothing keeps a rule that never fired on validation from scoring 0 or 1.
            self.rule_precision[name] = round(float((y[fired].sum() + 1) / (fired.sum() + 2)), 4)

    def combined(self, feats: pd.DataFrame):
        """Return (combined score, model percentile, any-rule-fired, rule hits)."""
        m = self.model_scores(feats)
        p_model = self.calibrator.predict(m) if self.calibrator is not None else m
        hits = self.rule_hits(feats)
        keep = 1.0 - p_model
        for i, hs in enumerate(hits):
            for h in hs:
                keep[i] *= 1.0 - self.rule_precision.get(h.rule, h.severity)
        fired = np.array([len(hs) > 0 for hs in hits])
        return 1.0 - keep, m, fired, hits

    def lane_for(self, score: float) -> str:
        if score >= self.hold_threshold:
            return LANE_HOLD
        if score >= self.flag_threshold:
            return LANE_REVIEW
        return LANE_CLEAR

    def model_reasons(self, row: pd.Series, top: int = 2) -> list[str]:
        """Features furthest from normal in the risky direction (higher), plus amount in either direction."""
        x = row[FEATURES].to_numpy(dtype=float)
        z = (x - self.medians) / self.scales
        risk = np.where(np.array(FEATURES) == "log_amount_ratio", np.abs(z), z)
        out = []
        for i in np.argsort(-risk)[:top]:
            if risk[i] < 2:
                continue
            name = FEATURES[i]
            if name == "log_amount_ratio" and pd.notna(row["typical_amount"]):
                out.append(f"Amount ${row['amount']:,.2f} vs typical ${row['typical_amount']:,.2f}")
            elif name == "log_km_from_home":
                out.append(f"{row['km_from_home']:,.0f} km from home")
            else:
                out.append(FEATURE_LABELS[name])
        return out or ["Unusual combination of amount, time, place and category for this customer"]

    def decide(self, feats: pd.DataFrame) -> list[Decision]:
        score, m, _, hits = self.combined(feats)
        out = []
        for i, (_, row) in enumerate(feats.iterrows()):
            lane = self.lane_for(score[i])
            reasons = [h.reason for h in hits[i]]
            if lane != LANE_CLEAR and not reasons:
                reasons += [f"Model: {r}" for r in self.model_reasons(row)]
            out.append(Decision(float(score[i]), float(m[i]), lane, reasons, [h.rule for h in hits[i]]))
        return out

    def save(self, path=MODEL_FILE) -> None:
        path.parent.mkdir(exist_ok=True)
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path=MODEL_FILE) -> Detector:
        return joblib.load(path)
