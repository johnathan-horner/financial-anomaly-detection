"""Uses the committed model (artifacts/) and history (data/recent_history.csv.gz)."""
import asyncio
import json
from pathlib import Path

import pandas as pd
import pytest
from fastmcp import Client
from streamlit.testing.v1 import AppTest

from anomaly.config import LANE_CLEAR, LANE_HOLD, METRICS_FILE
from anomaly.detector import Detector
from anomaly.service import FraudService


@pytest.fixture(scope="module")
def svc(tmp_path_factory):
    from anomaly.audit import AuditLog
    return FraudService(audit=AuditLog(tmp_path_factory.mktemp("a") / "audit.sqlite3"))


def test_ordinary_purchases_clear(svc):
    lanes = []
    for c in svc.customers()[:15]:
        p = svc.customer_profile(c)
        lanes.append(svc.score_transaction(c, p["median_amount"], next(iter(p["usual_categories"])))["lane"])
    assert lanes.count(LANE_CLEAR) >= 14


def test_card_testing_burst_is_flagged(svc):
    c = svc.customers()[20]
    base = svc.history[svc.history.customer_id == c]["timestamp"].max() + pd.Timedelta(days=1)
    results = [svc.score_transaction(c, 1.25, "online", timestamp=base + pd.Timedelta(minutes=k)) for k in range(4)]
    assert results[-1]["lane"] != LANE_CLEAR
    assert any("under $5" in r for r in results[-1]["reasons"])


def test_impossible_travel_is_held(svc):
    c = svc.customers()[21]
    base = svc.history[svc.history.customer_id == c]["timestamp"].max() + pd.Timedelta(days=1)
    svc.score_transaction(c, 30, "grocery", timestamp=base)
    r = svc.score_transaction(c, 400, "online", "BR", -23.55, -46.63, timestamp=base + pd.Timedelta(hours=1))
    assert r["lane"] == LANE_HOLD
    assert any("km from the previous transaction" in x for x in r["reasons"])


def test_unknown_customer_and_bad_amount(svc):
    with pytest.raises(KeyError):
        svc.score_transaction("nobody", 10, "grocery")
    with pytest.raises(ValueError):
        svc.score_transaction(svc.customers()[0], -5, "grocery")


def test_committed_metrics_match_committed_model():
    m = json.loads(METRICS_FILE.read_text())
    det = Detector.load()
    assert m["thresholds"]["flag"] == round(det.flag_threshold, 4)
    assert m["thresholds"]["hold"] == round(det.hold_threshold, 4)
    assert m["thresholds"]["picked_on"].startswith("validation")
    assert m["test"]["combined"]["recall"] > 0


def test_mcp_tools(tmp_path, monkeypatch):
    monkeypatch.setenv("ANOMALY_AUDIT_DB", str(tmp_path / "mcp.sqlite3"))
    import mcp_server
    mcp_server.service.cache_clear()

    async def run():
        async with Client(mcp_server.mcp) as client:
            names = {t.name for t in await client.list_tools()}
            assert names == {"detect_transaction_anomaly", "investigate_customer_pattern", "analyze_merchant_risk",
                             "get_investigation_queue", "get_audit_trail", "get_system_metrics"}
            cust = mcp_server.service().customers()[3]
            prof = (await client.call_tool("investigate_customer_pattern", {"customer_id": cust})).data
            assert prof["transactions"] > 0
            risky = (await client.call_tool("detect_transaction_anomaly", {
                "customer_id": cust, "amount": prof["median_amount"] * 30, "merchant_category": "jewelry",
                "country": "RU", "lat": 55.76, "lon": 37.62})).data
            assert risky["lane"] != "clear" and risky["reasons"]
            again = (await client.call_tool("detect_transaction_anomaly", {
                "customer_id": cust, "amount": prof["median_amount"] * 30, "merchant_category": "jewelry",
                "country": "RU", "lat": 55.76, "lon": 37.62})).data
            assert again["score"] == risky["score"] and again["reasons"] == risky["reasons"]
            stats = (await client.call_tool("analyze_merchant_risk", {"merchant_category": "grocery"})).data
            assert stats["transactions"] > 0
            metrics = (await client.call_tool("get_system_metrics", {})).data
            assert "combined" in metrics["test"]
            # Read-only: scoring through MCP writes nothing to the audit log.
            assert mcp_server.service().audit.events() == []
            queue = (await client.call_tool("get_investigation_queue", {})).data
            assert queue == []
            svc = mcp_server.service()
            logged = svc.score_transaction(cust, prof["median_amount"] * 30, "jewelry", "RU", 55.76, 37.62)
            trail = (await client.call_tool("get_audit_trail", {"transaction_id": logged["transaction_id"]})).data
            assert [e["event_type"] for e in trail] == ["scored"]
    asyncio.run(run())
    mcp_server.service.cache_clear()


def test_app_scores_a_scenario(tmp_path, monkeypatch):
    monkeypatch.setenv("ANOMALY_AUDIT_DB", str(tmp_path / "app.sqlite3"))
    at = AppTest.from_file(str(Path(__file__).parent.parent / "app.py"), default_timeout=120).run()
    assert not at.exception
    next(b for b in at.button if b.label.startswith("Card testing")).click().run()
    assert not at.exception
    assert any("Review queue" in w.value or "Hold" in w.value for w in list(at.warning) + list(at.error))
