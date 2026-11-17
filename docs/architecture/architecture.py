"""Transaction Anomaly Detection: training (offline), the live app and the local MCP server. No AWS services."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagrams import Diagram
from diagrams.onprem.client import User

from style import EDGE, GRAPH, NODE, area, flow, icon, primary, soft

with Diagram("Transaction Anomaly Detection  |  Live architecture",
             filename=str(Path(__file__).with_name("architecture")), outformat="png", direction="TB",
             show=False, graph_attr=GRAPH, node_attr=NODE, edge_attr=EDGE):
    user = User("Fraud analyst\n(browser)")

    with area("Training  (offline, laptop CPU)", "plan"):
        gen = icon("database-green", "328k synthetic transactions\n6 fraud patterns +\nhard negatives")
        fit = icon("scikitlearn", "Fit Isolation Forest\ncalibrate + pick thresholds\non validation period")
        model = icon("layers-green", "Detector file in repo\nflag 0.457, hold 0.871")

    with area("Local only", "plan"):
        mcp = icon("network-ink", "FastMCP server\nread-only tools\nfor an AI assistant")

    with area("Streamlit Community Cloud", "host"):
        app = icon("streamlit", "Streamlit app\nscore, review queue,\naudit log, results")
        feats = icon("cpu-ink", "Features from this\ncustomer's past only")
        with area("Detector"):
            iforest = icon("scikitlearn", "Isolation Forest")
            rules = icon("list-checks-ink", "5 plain rules\ne.g. card testing,\nimpossible travel")
        score = icon("gauge-red", "Calibrated combined\nscore + plain reasons")
        with area("Lanes"):
            clear = icon("shield-check-red", "Clear")
            queue = icon("user-check-ink", "Review queue\nanalyst decides")
            hold = icon("hand-red", "Hold\n95.4% precise on test")
        audit = icon("sqlite", "Audit log (SQLite)\nappend-only")

    gen >> flow("train / val / test by time") >> fit
    fit >> flow("save") >> model
    model >> soft("loaded at startup") >> iforest

    user >> flow("transaction, decisions") >> app
    app >> primary("transaction") >> feats
    feats >> flow() >> iforest
    feats >> flow() >> rules
    iforest >> flow("model score") >> score
    rules >> flow("rule hits") >> score
    score >> flow("below flag") >> clear
    score >> flow("flag to hold") >> queue
    score >> flow("hold or above") >> hold
    queue >> flow("score + decision") >> audit
    hold >> flow("score") >> audit
    clear >> flow("score") >> audit
    mcp >> soft("read-only: scores (not logged),\nreads queue + audit trail") >> feats
