"""Transaction Anomaly Detection: what runs today (traced from origin/main).

Run from this directory:  python architecture.py
"""
from diagrams.aws.general import User

from archstyle import arch, box, e, layer, linkedin, outside

OUT = "architecture"
with arch("Transaction Anomaly Detection", "What runs today (no AWS at runtime)", OUT):
    user = User("Visitor / analyst")

    with outside("Streamlit Community Cloud"):
        with layer("App", margin=40):
            app = box("Streamlit app (app.py)\nScore | Review queue |\nAudit log | Test results")
        with layer("Scoring (anomaly/service.py)"):
            feats = box("Past-only features\n9 per transaction\n(customer's own history)")
            model = box("Isolation Forest\n(scikit-learn)")
            rules = box("5 rules\nplain-language reasons")
            combine = box("Calibrated score\n1 - prod(1 - p)\nclear / review / hold")
        with layer("Governance"):
            audit = box("Audit log (SQLite)\nappend-only rows\nscores + analyst decisions")
            queue = box("Review queue\nlatest row still 'pending'\n(score 0.457 to 0.871)")
        committed = layer("Committed artifacts", margin=36)
        committed.dot.graph_attr["labeljust"] = "r"  # the model edge enters from the left
        with committed:
            artifacts = box("detector.joblib (Isolation Forest,\ncalibrator, thresholds)\nmetrics.json (test results)\nrecent_history.csv.gz (30 days)")

    local = outside("Local only (not hosted)")
    local.dot.graph_attr["labeljust"] = "r"
    with local:
        client = box("MCP client\n(AI assistant)")
        mcp = box("FastMCP server (stdio)\n6 read-only tools\nown copy of anomaly/ + artifacts/")

    offline = outside("Offline build (not at runtime)")
    offline.dot.graph_attr["labeljust"] = "r"
    with offline:
        gen = box("anomaly/generate.py\nsynthetic data, seed 7")
        train = box("anomaly/train.py\ntime split, calibrate,\npick thresholds")

    user >> e("score a scenario; confirm fraud\nor mark legit in review tab") >> app
    app >> e("score_transaction") >> feats
    feats >> e("feature row") >> model
    feats >> e("feature row") >> rules
    model >> e("percentile") >> combine
    rules >> e("rule hits") >> combine
    combine >> e("writes 'scored' row") >> audit
    audit >> e("latest row per txn\nwhere pending") >> queue
    queue >> e("analyst decision:\n  writes 'review' row", constraint="false") >> audit
    feats >> e("reads customer\nhistory", headport="n") >> artifacts
    model >> e("loads detector", headport="w") >> artifacts

    client >> e("MCP tool calls\nover stdio") >> mcp

    gen >> e("328,448 transactions") >> train
    train >> e("writes model + metrics\n(committed to repo)") >> artifacts

linkedin(OUT)
