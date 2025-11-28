"""Functions to enrich impression logs with metadata."""
import pandas as pd

from src import config


def merge_impressions_with_metadata(
    impressions: pd.DataFrame, metadata: pd.DataFrame
) -> pd.DataFrame:
    merged = impressions.merge(metadata, how="left", on=config.ITEM_COL)
    merged[config.TARGET_COL] = merged[config.TARGET_COL].fillna(0)
    return merged
