"""Definitions of feature groups used across models."""
from typing import List

BASE_FEATURES: List[str] = [
    "impression_time",
    "txn_amount_sum",
    "txn_count",
    "event_count",
    "last_transaction",
    "last_event",
]

OFFER_METADATA_FEATURES: List[str] = [
    "offer_category",
    "offer_value",
    "channel",
]

BEHAVIORAL_FEATURES: List[str] = [
    "ctr_7d",
    "ctr_30d",
    "txn_mean_30d",
    "event_rate_7d",
]

ALL_FEATURES = BASE_FEATURES + OFFER_METADATA_FEATURES + BEHAVIORAL_FEATURES
