"""Plain rules. Each returns a severity (0 to 1) and a reason a person can read."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RuleHit:
    rule: str
    severity: float
    reason: str


AMOUNT_SPIKE_RATIO = 8.0
TRAVEL_SPEED_KMH = 900.0
TRAVEL_MIN_KM = 500.0
CARD_TEST_COUNT = 2
VELOCITY_COUNT = 3

RULES = {
    "amount_spike": f"Amount at least {AMOUNT_SPIKE_RATIO:.0f}x the customer's typical amount",
    "impossible_travel": f"Over {TRAVEL_MIN_KM:.0f} km from the last transaction at more than {TRAVEL_SPEED_KMH:.0f} km/h",
    "card_testing": f"A charge under $5 with at least {CARD_TEST_COUNT} other small charges in 10 minutes",
    "velocity": f"At least {VELOCITY_COUNT} earlier transactions in the past hour",
    "night_new_category": "Between 1am and 5am in a merchant category the customer has never used",
}


def evaluate(row) -> list[RuleHit]:
    """Rules that fire for one feature row (a dict or pandas row)."""
    hits = []
    typical = row["typical_amount"]
    if not math.isnan(typical) and row["amount"] >= AMOUNT_SPIKE_RATIO * typical:  # NaN until 5 prior txns
        hits.append(RuleHit("amount_spike", 0.9,
                            f"${row['amount']:,.2f} is {row['amount'] / typical:.1f}x this customer's typical "
                            f"${typical:,.2f}"))
    if row["km_from_prev"] >= TRAVEL_MIN_KM and row["speed_kmh"] >= TRAVEL_SPEED_KMH:
        gap = row["hours_from_prev"]
        when = f"{gap * 60:.0f} minutes" if gap < 1 else f"{gap:.1f} hours"
        hits.append(RuleHit("impossible_travel", 0.95,
                            f"Used {row['km_from_prev']:,.0f} km from the previous transaction {when} earlier"))
    if row["amount"] < 5 and row["n_small_last_10min"] >= CARD_TEST_COUNT:
        hits.append(RuleHit("card_testing", 0.9,
                            f"{int(row['n_small_last_10min']) + 1} charges under $5 within 10 minutes"))
    if row["n_last_60min"] >= VELOCITY_COUNT:
        hits.append(RuleHit("velocity", 0.75, f"{int(row['n_last_60min']) + 1} transactions within one hour"))
    hour = row["timestamp"].hour
    if 1 <= hour < 5 and row["new_category"]:
        hits.append(RuleHit("night_new_category", 0.6,
                            f"{row['timestamp']:%H:%M} purchase in a new category ({row['merchant_category']})"))
    return hits
