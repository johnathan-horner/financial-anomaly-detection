import pytest

from anomaly.config import LANE_CLEAR, LANE_REVIEW


def test_score_and_review_are_append_only(audit):
    audit.record_score("T1", "C1", "v1", 50.0, 0.6, LANE_REVIEW, ["reason a"])
    audit.record_score("T2", "C1", "v1", 20.0, 0.1, LANE_CLEAR, [])
    assert [q["transaction_id"] for q in audit.review_queue()] == ["T1"]

    latest = audit.record_review("T1", "ana", "marked_legit", "customer confirmed travel")
    assert latest["reviewer_action"] == "marked_legit"
    assert audit.review_queue() == []
    events = audit.events("T1")
    assert [e["event_type"] for e in events] == ["scored", "review"]
    assert events[0]["reviewer_action"] == "pending"
    assert events[1]["reasons"] == ["reason a"] and events[1]["reviewer"] == "ana"
    assert all(e["created_at"].endswith("+00:00") for e in events)


def test_rejects_unknown_action_and_transaction(audit):
    audit.record_score("T1", "C1", "v1", 50.0, 0.6, LANE_REVIEW, [])
    with pytest.raises(ValueError):
        audit.record_review("T1", "ana", "delete")
    with pytest.raises(KeyError):
        audit.record_review("nope", "ana", "marked_legit")
