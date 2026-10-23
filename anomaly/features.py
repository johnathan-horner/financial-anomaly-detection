"""Per-transaction features computed only from that customer's earlier transactions."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "log_amount_ratio",   # log(amount / customer's typical amount so far)
    "hour_distance",      # hours away from the customer's usual time of day
    "log_km_from_home",
    "log_speed_kmh",      # implied travel speed since the previous transaction
    "n_last_10min",
    "n_last_60min",
    "n_small_last_10min",  # earlier charges under $5 in the last 10 minutes
    "new_category",
    "foreign",
]
SMALL_AMOUNT = 5.0
MIN_HISTORY = 5


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _per_customer(g: pd.DataFrame) -> pd.DataFrame:
    t = g["timestamp"].values.astype("datetime64[s]").astype(np.int64)
    amount = g["amount"].values
    n = len(g)

    prior_median = g["amount"].expanding().median().shift(1).values
    prior_count = np.arange(n)
    typical = np.where(prior_count >= MIN_HISTORY, prior_median, np.nan)

    hours = g["timestamp"].dt.hour.values + g["timestamp"].dt.minute.values / 60
    angle = hours / 24 * 2 * np.pi
    mean_sin = pd.Series(np.sin(angle)).expanding().mean().shift(1).values
    mean_cos = pd.Series(np.cos(angle)).expanding().mean().shift(1).values
    usual_hour = (np.arctan2(mean_sin, mean_cos) % (2 * np.pi)) / (2 * np.pi) * 24
    diff = np.abs(hours - usual_hour)
    hour_distance = np.where(prior_count >= MIN_HISTORY, np.minimum(diff, 24 - diff), 0.0)

    km_prev = np.zeros(n)
    dt_hours = np.full(n, np.inf)
    km_prev[1:] = haversine_km(g["lat"].values[:-1], g["lon"].values[:-1], g["lat"].values[1:], g["lon"].values[1:])
    dt_hours[1:] = np.maximum((t[1:] - t[:-1]) / 3600, 1 / 60)
    speed = np.where(np.isfinite(dt_hours), km_prev / dt_hours, 0.0)

    start10 = np.searchsorted(t, t - 600, side="left")
    start60 = np.searchsorted(t, t - 3600, side="left")
    small = (amount < SMALL_AMOUNT).astype(int)
    small_cum = np.concatenate([[0], np.cumsum(small)])
    idx = np.arange(n)

    seen, new_cat = set(), np.zeros(n, dtype=int)
    for i, cat in enumerate(g["merchant_category"].values):
        new_cat[i] = int(i >= MIN_HISTORY and cat not in seen)
        seen.add(cat)

    return pd.DataFrame({
        "typical_amount": typical,
        "log_amount_ratio": np.nan_to_num(np.log(amount / typical), nan=0.0),
        "hour_distance": hour_distance,
        "km_from_home": haversine_km(g["home_lat"].values, g["home_lon"].values, g["lat"].values, g["lon"].values),
        "km_from_prev": km_prev,
        "hours_from_prev": np.where(np.isfinite(dt_hours), dt_hours, np.nan),
        "speed_kmh": speed,
        "n_last_10min": idx - start10,
        "n_last_60min": idx - start60,
        "n_small_last_10min": small_cum[idx] - small_cum[start10],
        "new_category": new_cat,
        "foreign": (g["country"].values != "US").astype(int),
    }, index=g.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with feature columns added. Rows keep their original order."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    ordered = df.sort_values(["customer_id", "timestamp"], kind="stable")
    feats = pd.concat([_per_customer(g) for _, g in ordered.groupby("customer_id", sort=False)])
    out = df.join(feats)
    out["log_km_from_home"] = np.log1p(out["km_from_home"])
    out["log_speed_kmh"] = np.log1p(out["speed_kmh"])
    return out
