"""Data loading utilities for impressions, transactions, events, and metadata."""
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from src import config
from src.utils import reduce_memory_usage


def _read_csv(path: Path, parse_dates: Optional[Iterable[str]] = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    df = pd.read_csv(path, parse_dates=parse_dates)
    return reduce_memory_usage(df)


def load_impressions(path: Path = config.IMPRESSIONS_FILE) -> pd.DataFrame:
    return _read_csv(path, parse_dates=["impression_time"])


def load_transactions(path: Path = config.TRANSACTIONS_FILE) -> pd.DataFrame:
    return _read_csv(path, parse_dates=["transaction_time"])


def load_events(path: Path = config.EVENTS_FILE) -> pd.DataFrame:
    return _read_csv(path, parse_dates=["event_time"])


def load_metadata(path: Path = config.METADATA_FILE) -> pd.DataFrame:
    return _read_csv(path)
