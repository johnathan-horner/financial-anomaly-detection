"""Streamlit demo: score card transactions, review flags, and read the audit log and test results."""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from anomaly.audit import AuditLog
from anomaly.config import LANE_CLEAR, LANE_HOLD
from anomaly.generate import CATEGORIES, FOREIGN
from anomaly.service import FraudService

st.set_page_config(page_title="Transaction Anomaly Detection", layout="wide")

def md(text: str) -> str:
    """Escape $ so Streamlit does not render amounts as math."""
    return text.replace("$", "\\$")


LANE_TEXT = {"hold": "Hold (high confidence)", "review_queue": "Review queue (lower confidence)", "clear": "Clear"}


@st.cache_resource
def get_service() -> FraudService:
    return FraudService(audit=AuditLog(os.environ.get("ANOMALY_AUDIT_DB", "audit_log.sqlite3")))


def show_result(r: dict) -> None:
    box = st.error if r["lane"] == LANE_HOLD else st.success if r["lane"] == LANE_CLEAR else st.warning
    box(md(f"{LANE_TEXT[r['lane']]}: score {r['score']:.2f}  (transaction {r['transaction_id']}, "
           f"${r['amount']:,.2f} {r['merchant_category']}, {r['country']})"))
    for reason in r["reasons"]:
        st.markdown(md(f"- {reason}"))


def score_tab(svc: FraudService) -> None:
    customers = svc.customers()
    customer = st.selectbox("Customer (synthetic)", customers, index=1)
    profile = svc.customer_profile(customer)
    st.caption(md(f"Last 30 days: {profile['transactions']} transactions, typical ${profile['median_amount']:,.2f}, "
               f"usual categories {', '.join(profile['usual_categories'])}, usual hour {profile['typical_hour']}:00"))

    st.markdown("**Try a scenario**")
    cols = st.columns(4)
    usual_cat = next(iter(profile["usual_categories"]))
    if cols[0].button("Ordinary purchase"):
        st.session_state["results"] = [svc.score_transaction(customer, profile["median_amount"], usual_cat)]
    if cols[1].button("Card testing (4 x $1.50)"):
        base = svc.history[svc.history.customer_id == customer]["timestamp"].max() + pd.Timedelta(days=1)
        st.session_state["results"] = [
            svc.score_transaction(customer, 1.5, "online", timestamp=base + pd.Timedelta(minutes=k))
            for k in range(4)]
    if cols[2].button("Used abroad 1 hour later"):
        base = svc.history[svc.history.customer_id == customer]["timestamp"].max() + pd.Timedelta(days=1)
        a = svc.score_transaction(customer, profile["median_amount"], usual_cat, timestamp=base)
        f = FOREIGN[1]
        b = svc.score_transaction(customer, profile["median_amount"] * 4, "online", f[0], f[1], f[2],
                                  timestamp=base + pd.Timedelta(hours=1))
        st.session_state["results"] = [a, b]
    if cols[3].button("Large jewelry purchase"):
        st.session_state["results"] = [svc.score_transaction(customer, profile["median_amount"] * 20, "jewelry")]

    with st.expander("Or enter a transaction"):
        c1, c2, c3 = st.columns(3)
        amount = c1.number_input("Amount ($)", min_value=0.5, value=float(profile["median_amount"]))
        category = c2.selectbox("Merchant category", list(CATEGORIES))
        country = c3.selectbox("Country", ["US"] + [f[0] for f in FOREIGN])
        if st.button("Score", type="primary"):
            lat = lon = None
            if country != "US":
                lat, lon = next((f[1], f[2]) for f in FOREIGN if f[0] == country)
            st.session_state["results"] = [svc.score_transaction(customer, amount, category, country, lat, lon)]

    for r in st.session_state.get("results", []):
        show_result(r)


def review_tab(svc: FraudService) -> None:
    queue = svc.review_queue()
    st.write(f"{len(queue)} flag(s) waiting for an analyst.")
    reviewer = st.text_input("Analyst name", value="demo-analyst")
    for item in queue:
        with st.container(border=True):
            st.markdown(md(f"**{item['transaction_id']}**  customer {item['customer_id']}, ${item['amount']:,.2f}, "
                           f"score {item['score']:.2f}"))
            for reason in item["reasons"]:
                st.markdown(md(f"- {reason}"))
            c1, c2 = st.columns(2)
            if c1.button("Confirm fraud", key=f"f-{item['transaction_id']}"):
                svc.record_review(item["transaction_id"], reviewer, "confirmed_fraud")
                st.rerun()
            if c2.button("Mark legitimate", key=f"l-{item['transaction_id']}"):
                svc.record_review(item["transaction_id"], reviewer, "marked_legit")
                st.rerun()


def audit_tab(svc: FraudService) -> None:
    events = svc.audit.events()
    if not events:
        st.info("Nothing scored yet.")
        return
    df = pd.DataFrame(events)
    df["reasons"] = df["reasons"].apply(lambda r: "; ".join(r))
    st.dataframe(df.iloc[::-1], hide_index=True, width="stretch")
    st.download_button("Download CSV", df.to_csv(index=False), "audit_log.csv", "text/csv")


def results_tab(svc: FraudService) -> None:
    m = svc.metrics()
    d, t, th = m["data"], m["test"], m["thresholds"]
    st.markdown(f"Synthetic data: {d['transactions']:,} transactions, {d['customers']:,} customers, "
                f"{d['anomaly_rate']:.1%} labeled anomalies. Split by time: train {d['split']['train']:,}, "
                f"validation {d['split']['validation']:,}, test {d['split']['test']:,} "
                f"({d['test_anomalies']:,} anomalies). Thresholds were picked on validation only.")
    rows = [{"approach": k.replace("_", " "), **{x: t[k][x] for x in
             ("precision", "recall", "f1", "false_positives_per_1000_normal")}}
            for k in ("combined", "rules_only", "model_only")]
    st.markdown("**Test period**")
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    c1, c2 = st.columns(2)
    c1.markdown("**Lanes on the test period**")
    c1.dataframe(pd.DataFrame(t["lanes"]).T, width="stretch")
    c2.markdown("**Recall by anomaly type**")
    c2.dataframe(pd.DataFrame(t["recall_by_anomaly_type"]).T, width="stretch")
    st.caption(f"Flag threshold {th['flag']} ({th['flag_rule']}). Hold threshold {th['hold']} ({th['hold_rule']}).")


def main() -> None:
    st.title("Transaction Anomaly Detection")
    st.info("Synthetic data only. Scores are decision support for an analyst, not a fraud determination.")
    svc = get_service()
    tabs = st.tabs(["Score a transaction", "Review queue", "Audit log", "Test results"])
    with tabs[0]:
        score_tab(svc)
    with tabs[1]:
        review_tab(svc)
    with tabs[2]:
        audit_tab(svc)
    with tabs[3]:
        results_tab(svc)


main()
