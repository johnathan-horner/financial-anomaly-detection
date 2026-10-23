import numpy as np
import pandas as pd

from anomaly.features import FEATURES, build_features
from anomaly.generate import generate


def test_generator_is_deterministic_and_labeled(small_data):
    again = generate(customers=80, days=45, seed=11)
    pd.testing.assert_frame_equal(small_data, again)
    assert 0.005 < small_data["is_anomaly"].mean() < 0.06
    assert set(small_data.loc[small_data.is_anomaly == 0, "anomaly_type"]) == {"none"}
    assert small_data["transaction_id"].is_unique


def test_includes_hard_negatives(small_data):
    # Legitimate trips abroad exist, so "foreign" alone is not a fraud label.
    normal_abroad = small_data[(small_data.is_anomaly == 0) & (small_data.country != "US")]
    assert len(normal_abroad) > 0


def test_features_only_use_the_past(small_data):
    cust = small_data.customer_id.iloc[0]
    rows = small_data[small_data.customer_id == cust].reset_index(drop=True)
    cut = len(rows) // 2
    full = build_features(rows).set_index("transaction_id")[FEATURES]
    early = build_features(rows.iloc[:cut]).set_index("transaction_id")[FEATURES]
    pd.testing.assert_frame_equal(full.loc[early.index], early)


def test_feature_values_on_a_handmade_history():
    t0 = pd.Timestamp("2026-02-01 12:00")
    rows = [{"transaction_id": f"X{i}", "timestamp": t0 + pd.Timedelta(days=i), "customer_id": "C1",
             "amount": 20.0, "merchant_category": "grocery", "country": "US", "lat": 40.71, "lon": -74.01,
             "home_lat": 40.71, "home_lon": -74.01} for i in range(6)]
    rows.append({**rows[-1], "transaction_id": "X6", "timestamp": rows[-1]["timestamp"] + pd.Timedelta(hours=1),
                 "amount": 200.0, "merchant_category": "jewelry", "country": "GB", "lat": 51.51, "lon": -0.13})
    f = build_features(pd.DataFrame(rows)).set_index("transaction_id").loc["X6"]
    assert np.isclose(f["log_amount_ratio"], np.log(10))
    assert f["new_category"] == 1 and f["foreign"] == 1
    assert 5500 < f["km_from_prev"] < 5600
    assert f["speed_kmh"] > 5000
    assert f["n_last_60min"] == 1
