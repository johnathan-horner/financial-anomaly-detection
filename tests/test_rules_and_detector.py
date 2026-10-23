import numpy as np
import pandas as pd

from anomaly import rules
from anomaly.config import LANE_CLEAR, LANE_HOLD, LANE_REVIEW


def row(**kw):
    base = {"amount": 30.0, "typical_amount": 30.0, "km_from_prev": 2.0, "speed_kmh": 1.0, "hours_from_prev": 5.0,
            "n_small_last_10min": 0, "n_last_60min": 0, "new_category": 0, "merchant_category": "grocery",
            "timestamp": pd.Timestamp("2026-03-01 14:00")}
    return {**base, **kw}


def test_ordinary_row_fires_no_rule():
    assert rules.evaluate(row()) == []


def test_each_rule_fires_with_a_plain_reason():
    cases = {
        "amount_spike": row(amount=300.0),
        "impossible_travel": row(km_from_prev=8000.0, speed_kmh=4000.0, hours_from_prev=2.0),
        "card_testing": row(amount=1.0, n_small_last_10min=2),
        "velocity": row(n_last_60min=3),
        "night_new_category": row(new_category=1, timestamp=pd.Timestamp("2026-03-01 03:00")),
    }
    for name, r in cases.items():
        hits = rules.evaluate(r)
        assert [h.rule for h in hits] == [name], name
        assert hits[0].reason and "_" not in hits[0].reason


def test_amount_rule_waits_for_history():
    assert rules.evaluate(row(amount=900.0, typical_amount=float("nan"))) == []


def test_impossible_travel_reason_uses_minutes_under_an_hour():
    (hit,) = rules.evaluate(row(km_from_prev=8000.0, speed_kmh=16000.0, hours_from_prev=0.5))
    assert "30 minutes earlier" in hit.reason


def test_scores_are_probabilities_and_rules_add_evidence(small_detector, small_feats):
    score, m, fired, _ = small_detector.combined(small_feats)
    assert np.all((score >= 0) & (score <= 1)) and np.all((m >= 0) & (m <= 1))
    # Where a rule fired, the combined score is at least that rule's calibrated precision.
    p_min = min(small_detector.rule_precision.values())
    assert np.all(score[fired] >= p_min - 1e-9)


def test_rule_precision_is_measured_not_hardcoded(small_detector):
    assert set(small_detector.rule_precision) == set(rules.RULES)
    assert all(0 < p < 1 for p in small_detector.rule_precision.values())


def test_lanes(small_detector):
    small_detector.flag_threshold, small_detector.hold_threshold = 0.4, 0.9
    assert small_detector.lane_for(0.95) == LANE_HOLD
    assert small_detector.lane_for(0.5) == LANE_REVIEW
    assert small_detector.lane_for(0.1) == LANE_CLEAR


def test_every_flag_has_a_reason(small_detector, small_feats):
    small_detector.flag_threshold, small_detector.hold_threshold = 0.4, 0.9
    flagged = [d for d in small_detector.decide(small_feats.head(3000)) if d.lane != LANE_CLEAR]
    assert flagged
    assert all(d.reasons for d in flagged)


def test_detector_beats_chance_on_held_out_days(small_detector, small_feats):
    score, *_ = small_detector.combined(small_feats)
    y = small_feats["is_anomaly"].to_numpy()
    top = score >= np.quantile(score, 0.98)
    assert y[top].mean() > 5 * y.mean()


def test_model_reasons_only_cite_risky_deviations(small_detector, small_feats):
    row = small_feats.iloc[0].copy()
    row["km_from_home"], row["log_km_from_home"] = 0.0, 0.0   # exactly at home: unusual but not risky
    reasons = small_detector.model_reasons(row)
    assert not any("km from home" in r for r in reasons)
