# Arrow verification: Transaction Anomaly Detection

Diagram: `docs/architecture/architecture.png` (What runs today, no AWS at runtime).
Code read from `origin/main` at d14d740. No arrow touches an AWS service, so no AWS doc applies.
No second diagram: there is no IaC (CDK, Terraform, SAM) on main.

| From | To | Action | Code evidence | AWS doc |
|---|---|---|---|---|
| Visitor / analyst | Streamlit app | Picks a synthetic customer, runs a scenario (ordinary, card testing 4 x $1.50, abroad 1 hour later, large jewelry) or a manual transaction; in the review tab clicks Confirm fraud / Mark legitimate | app.py:38-74 (score tab), app.py:80-96 (review tab) | N/A (not AWS) |
| Streamlit app | Past-only features | `FraudService.score_transaction(...)` builds features for the new row from that customer's earlier rows | app.py:49-74; anomaly/service.py:37-66; anomaly/features.py:8-18, 29-88 | N/A (not AWS) |
| Past-only features | Isolation Forest | Feature row scored with `score_samples`, mapped to a percentile of training scores | anomaly/detector.py:67-69, 87 | N/A (not AWS) |
| Past-only features | 5 rules | Feature row evaluated by `rules.evaluate`, each hit has a plain-language reason | anomaly/detector.py:71-72; anomaly/rules.py:22-53 | N/A (not AWS) |
| Isolation Forest | Calibrated score | Percentile -> isotonic calibration -> probability | anomaly/detector.py:85-88, 74-78 | N/A (not AWS) |
| 5 rules | Calibrated score | Each rule hit contributes its validation precision; combined as 1 - product(1 - p); lane by thresholds | anomaly/detector.py:89-102, 122-131 | N/A (not AWS) |
| Calibrated score | Audit log (SQLite) | Writes a `scored` row (`pending` if lane is review_queue) when `log=True` (app default) | anomaly/service.py:70-72; anomaly/audit.py:51-58; DB path app.py:27 (`ANOMALY_AUDIT_DB`, default `audit_log.sqlite3`) | N/A (not AWS) |
| Audit log | Review queue | Query: latest row per transaction where `reviewer_action = 'pending'`, highest score first | anomaly/audit.py:90-101; anomaly/service.py:107-108; app.py:81 | N/A (not AWS) |
| Review queue | Audit log | Analyst decision appends a `review` row (`confirmed_fraud` or `marked_legit`) | app.py:91-96; anomaly/service.py:110-111; anomaly/audit.py:60-69 | N/A (not AWS) |
| Past-only features | Committed artifacts | Reads customer history from `data/recent_history.csv.gz` (loaded once into memory) | anomaly/service.py:18, 27-29, 52 | N/A (not AWS) |
| Isolation Forest | Committed artifacts | Loads `artifacts/detector.joblib` (whole Detector: model, calibrator, rule precisions, thresholds); metrics.json read for Test results | anomaly/service.py:26; anomaly/detector.py:137-139; anomaly/config.py:8-9; anomaly/service.py:113-115; app.py:110-111 | N/A (not AWS) |
| MCP client (AI assistant) | FastMCP server (local) | MCP tool calls over stdio; 6 tools, 2 resources, 1 prompt; scoring passes `remember=False, log=False` | mcp_server.py:20, 28-93, 96-97 (`mcp.run()`, stdio per docstring :6); README.md:122-126 | N/A (not AWS) |
| anomaly/generate.py | anomaly/train.py | Synthetic data (seed 7): 328,448 transactions, read from `data/transactions.csv` or generated in-process | anomaly/generate.py:137-152; anomaly/train.py:80; artifacts/metrics.json `data.transactions` | N/A (not AWS) |
| anomaly/train.py | Committed artifacts | Writes `detector.joblib` and `metrics.json` (committed; README.md:108-109) | anomaly/train.py:126-127; anomaly/detector.py:133-135 | N/A (not AWS) |

UNVERIFIED arrows: none (no AWS arrows).

Not drawn as an arrow: no code in the repo writes `data/recent_history.csv.gz` (only anomaly/service.py:18 reads it).
The diagram's merged artifacts box says "30 days". The file holds 109,491 rows for 2,000 customers, 2026-03-02 to 2026-04-01, which matches the last 30 days of the generated 90-day set.
