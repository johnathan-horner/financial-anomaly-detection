# Contributing

1. Python 3.11: `pip install -r requirements-dev.txt`
2. `ruff check anomaly tests app.py mcp_server.py` and `pytest` must pass. CI runs both, plus a gitleaks scan.
3. If you change the generator, features, rules or detector, rerun `python -m anomaly.train` and commit
   `artifacts/` so the README numbers match `artifacts/metrics.json`.
4. Never commit credentials, `.env` files or `audit_log.sqlite3`.
