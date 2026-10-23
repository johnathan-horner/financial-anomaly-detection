import pandas as pd
import pytest

from anomaly.audit import AuditLog
from anomaly.detector import Detector
from anomaly.features import build_features
from anomaly.generate import START, generate


@pytest.fixture(scope="session")
def small_data() -> pd.DataFrame:
    return generate(customers=80, days=45, seed=11)


@pytest.fixture(scope="session")
def small_feats(small_data) -> pd.DataFrame:
    return build_features(small_data)


@pytest.fixture(scope="session")
def small_detector(small_feats) -> Detector:
    day = (small_feats["timestamp"] - pd.Timestamp(START)).dt.days
    det = Detector.fit(small_feats[day < 30])
    det.calibrate(small_feats[day >= 30])
    return det


@pytest.fixture
def audit(tmp_path) -> AuditLog:
    return AuditLog(tmp_path / "audit.sqlite3")
