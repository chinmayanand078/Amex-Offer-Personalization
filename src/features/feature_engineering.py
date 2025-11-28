"""Behavioral feature engineering for ranking models."""
import logging
from datetime import timedelta
from typing import List

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def _compute_rate(counts: pd.Series, window_days: int) -> pd.Series:
    return counts / window_days


def add_recency_features(df: pd.DataFrame) -> pd.DataFrame:
    now = df["impression_time"].max()
    df["days_since_last_txn"] = (now - df["last_transaction"]).dt.days.fillna(-1)
    df["days_since_last_event"] = (now - df["last_event"]).dt.days.fillna(-1)
    return df


def add_behavioral_features(df: pd.DataFrame, windows: List[int] = config.WINDOWS) -> pd.DataFrame:
    df = df.copy()
    for window in windows:
        window_delta = timedelta(days=window)
        recent = df[df["impression_time"] >= df["impression_time"].max() - window_delta]
        ctr = recent.groupby(config.ITEM_COL)[config.TARGET_COL].mean().rename(f"ctr_{window}d")
        txn_mean = (
            recent.groupby(config.GROUP_COL)["txn_amount_sum"].mean().rename(f"txn_mean_{window}d")
        )
        df = df.merge(ctr, on=config.ITEM_COL, how="left")
        df = df.merge(txn_mean, on=config.GROUP_COL, how="left")
    df = add_recency_features(df)
    df["event_rate_7d"] = _compute_rate(df["event_count"], 7)
    return df


def select_feature_matrix(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    return df[features].copy()
