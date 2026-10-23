"""Seeded synthetic card transactions with labeled anomalies.

Everything is generated here; no real customer data is used. Each row has `is_anomaly` and
`anomaly_type`. Normal behavior includes hard cases on purpose (legitimate large purchases and
real trips abroad) so the detector cannot reach 100% precision by flagging anything unusual.

Usage:  python -m anomaly.generate --customers 2000 --days 90
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from anomaly.config import DATA_FILE

START = datetime(2026, 1, 1)

CITIES = [  # (name, lat, lon) for home locations and domestic trips
    ("New York", 40.71, -74.01), ("Chicago", 41.88, -87.63), ("Houston", 29.76, -95.37),
    ("Phoenix", 33.45, -112.07), ("Philadelphia", 39.95, -75.17), ("San Antonio", 29.42, -98.49),
    ("San Diego", 32.72, -117.16), ("Dallas", 32.78, -96.80), ("Austin", 30.27, -97.74),
    ("Jacksonville", 30.33, -81.66), ("Columbus", 39.96, -83.00), ("Charlotte", 35.23, -80.84),
    ("Indianapolis", 39.77, -86.16), ("Seattle", 47.61, -122.33), ("Denver", 39.74, -104.99),
    ("Boston", 42.36, -71.06), ("Nashville", 36.16, -86.78), ("Detroit", 42.33, -83.05),
    ("Portland", 45.52, -122.68), ("Atlanta", 33.75, -84.39), ("Miami", 25.76, -80.19),
    ("Minneapolis", 44.98, -93.27), ("Raleigh", 35.78, -78.64), ("Kansas City", 39.10, -94.58),
]
FOREIGN = [  # (country, lat, lon)
    ("GB", 51.51, -0.13), ("FR", 48.86, 2.35), ("BR", -23.55, -46.63), ("NG", 6.52, 3.38),
    ("RU", 55.76, 37.62), ("CN", 31.23, 121.47), ("IN", 19.08, 72.88), ("MX", 19.43, -99.13),
]
CATEGORIES = {  # category: typical amount multiplier relative to the customer's base spend
    "grocery": 1.0, "restaurant": 0.8, "gas": 0.9, "pharmacy": 0.6, "coffee": 0.15,
    "retail": 1.5, "online": 1.2, "utilities": 2.0, "entertainment": 1.0, "travel": 4.0,
    "electronics": 5.0, "jewelry": 8.0,
}
EVERYDAY = ["grocery", "restaurant", "gas", "pharmacy", "coffee", "retail", "online", "utilities", "entertainment"]


def _jitter(rng, lat, lon, km=15.0):
    return lat + rng.normal(0, km / 111), lon + rng.normal(0, km / 85)


def _customers(rng, n):
    rows = []
    for i in range(n):
        city = CITIES[rng.integers(len(CITIES))]
        rows.append({
            "customer_id": f"C{i:05d}",
            "home_city": city[0], "home_lat": city[1], "home_lon": city[2],
            "base_spend": float(np.exp(rng.normal(3.4, 0.5))),    # median about $30
            "active_hour": float(np.clip(rng.normal(14, 3), 8, 20)),
            "rate_per_day": float(rng.uniform(0.6, 3.0)),
            "categories": list(rng.choice(EVERYDAY, size=int(rng.integers(3, 7)), replace=False)),
        })
    return rows


def _normal_txns(rng, c, days):
    """Ordinary spending, with occasional domestic trips, legit big buys and trips abroad."""
    out = []
    trip_city, trip_left = None, 0
    abroad, abroad_left = None, 0
    for d in range(days):
        if trip_left == 0 and rng.random() < 0.01:
            trip_city, trip_left = CITIES[rng.integers(len(CITIES))], int(rng.integers(2, 6))
        if abroad_left == 0 and rng.random() < 0.0015:   # rare vacation abroad, starts on a new day
            abroad, abroad_left = FOREIGN[rng.integers(len(FOREIGN))], int(rng.integers(3, 8))
        for _ in range(rng.poisson(c["rate_per_day"])):
            hour = float(np.clip(rng.normal(c["active_hour"], 3), 6, 23.5))
            ts = START + timedelta(days=d, hours=hour, minutes=float(rng.uniform(0, 59)))
            cat = str(rng.choice(c["categories"]))
            amount = c["base_spend"] * CATEGORIES[cat] * float(np.exp(rng.normal(0, 0.45)))
            country, (lat, lon) = "US", _jitter(rng, c["home_lat"], c["home_lon"])
            if abroad_left:
                country, (lat, lon) = abroad[0], _jitter(rng, abroad[1], abroad[2])
                cat = str(rng.choice(["restaurant", "travel", "retail"]))
            elif trip_left:
                lat, lon = _jitter(rng, trip_city[1], trip_city[2])
            if rng.random() < 0.004:  # legitimate large purchase: hard negative
                cat = str(rng.choice(["electronics", "travel", "jewelry"]))
                amount = c["base_spend"] * float(rng.uniform(8, 25))
            out.append((ts, round(max(amount, 1.0), 2), cat, country, lat, lon, 0, "none"))
        trip_left = max(0, trip_left - 1)
        abroad_left = max(0, abroad_left - 1)
    return out


def _anomalies(rng, c, days, normal):
    """Inject labeled fraud-like events after a warm-up period."""
    out = []
    n_events = rng.poisson(0.9 * days / 90)
    for _ in range(n_events):
        d = int(rng.integers(14, days))
        ts = START + timedelta(days=d, hours=float(rng.uniform(0, 24)))
        kind = str(rng.choice(["amount_spike", "impossible_travel", "card_testing", "velocity_burst",
                               "odd_hour_new_category", "subtle_unusual_purchase"],
                              p=[0.2, 0.15, 0.2, 0.15, 0.15, 0.15]))
        home = _jitter(rng, c["home_lat"], c["home_lon"])
        if kind == "amount_spike":
            cat = str(rng.choice(["electronics", "jewelry", "online"]))
            out.append((ts, round(c["base_spend"] * float(rng.uniform(10, 40)), 2), cat, "US", *home, 1, kind))
        elif kind == "impossible_travel":
            # A home purchase, then the card is used abroad a couple of hours later.
            out.append((ts, round(c["base_spend"] * 0.8, 2), str(rng.choice(c["categories"])), "US", *home, 0, "none"))
            f = FOREIGN[rng.integers(len(FOREIGN))]
            later = ts + timedelta(hours=float(rng.uniform(0.5, 3)))
            out.append((later, round(c["base_spend"] * float(rng.uniform(2, 8)), 2), "online", f[0],
                        *_jitter(rng, f[1], f[2]), 1, kind))
        elif kind == "card_testing":
            for k in range(int(rng.integers(4, 10))):
                t = ts + timedelta(seconds=float(rng.uniform(20, 90)) * k)
                out.append((t, round(float(rng.uniform(0.5, 3.0)), 2), "online", "US", *home, 1, kind))
        elif kind == "velocity_burst":
            for k in range(int(rng.integers(6, 11))):
                t = ts + timedelta(minutes=float(rng.uniform(2, 5)) * k)
                cat = str(rng.choice(["retail", "electronics", "online"]))
                out.append((t, round(c["base_spend"] * float(rng.uniform(1, 3)), 2), cat, "US", *home, 1, kind))
        elif kind == "odd_hour_new_category":
            unused = [k for k in CATEGORIES if k not in c["categories"]]
            t = START + timedelta(days=d, hours=float(rng.uniform(1, 5)))
            out.append((t, round(c["base_spend"] * float(rng.uniform(3, 6)), 2), str(rng.choice(unused)), "US",
                        *home, 1, kind))
        else:  # subtle: no single rule fires; only the combination is unusual
            unused = [k for k in CATEGORIES if k not in c["categories"] and k not in ("electronics", "jewelry")]
            cat = str(rng.choice(unused)) if unused else "online"
            amount = c["base_spend"] * CATEGORIES[cat] * float(rng.uniform(3, 6))
            t = START + timedelta(days=d, hours=float(rng.uniform(22, 24.9)) % 24)
            out.append((t, round(amount, 2), cat, "US", *_jitter(rng, c["home_lat"], c["home_lon"], 120), 1, kind))
    return out


def generate(customers: int = 2000, days: int = 90, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for c in _customers(rng, customers):
        txns = _normal_txns(rng, c, days)
        txns += _anomalies(rng, c, days, txns)
        for ts, amount, cat, country, lat, lon, label, kind in txns:
            rows.append({
                "timestamp": ts, "customer_id": c["customer_id"], "amount": amount,
                "merchant_category": cat, "country": country, "lat": round(lat, 4), "lon": round(lon, 4),
                "home_lat": c["home_lat"], "home_lon": c["home_lon"],
                "is_anomaly": label, "anomaly_type": kind,
            })
    df = pd.DataFrame(rows).sort_values(["timestamp", "customer_id"]).reset_index(drop=True)
    df.insert(0, "transaction_id", [f"T{i:07d}" for i in range(len(df))])
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--customers", type=int, default=2000)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    df = generate(args.customers, args.days, args.seed)
    DATA_FILE.parent.mkdir(exist_ok=True)
    df.to_csv(DATA_FILE, index=False)
    print(f"{len(df)} transactions, {int(df.is_anomaly.sum())} anomalies ({df.is_anomaly.mean():.2%}) -> {DATA_FILE}")
    print(df[df.is_anomaly == 1].anomaly_type.value_counts().to_string())


if __name__ == "__main__":
    main()
