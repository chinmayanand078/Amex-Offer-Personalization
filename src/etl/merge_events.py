"""Aggregate user events for recency and frequency features."""
import pandas as pd

from src import config


def aggregate_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=[config.GROUP_COL, "event_count", "last_event"])

    grouped = events.groupby(config.GROUP_COL).agg(
        event_count=("event_type", "count"),
        last_event=("event_time", "max"),
    )
    return grouped.reset_index()


def merge_events(impressions: pd.DataFrame, event_agg: pd.DataFrame) -> pd.DataFrame:
    return impressions.merge(event_agg, how="left", on=config.GROUP_COL)
