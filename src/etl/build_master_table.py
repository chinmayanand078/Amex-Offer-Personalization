"""ETL orchestration to build the unified training table."""
import logging
from pathlib import Path

import pandas as pd

from src import config
from src.etl.load_data import load_events, load_impressions, load_metadata, load_transactions
from src.etl.merge_events import aggregate_events, merge_events
from src.etl.merge_impressions import merge_impressions_with_metadata
from src.etl.merge_transactions import aggregate_transactions, merge_transactions
from src.utils import ensure_dir

logger = logging.getLogger(__name__)


def build_master_table(output_path: Path = config.MASTER_TABLE) -> pd.DataFrame:
    logger.info("Loading raw datasets")
    impressions = load_impressions()
    metadata = load_metadata()
    transactions = load_transactions()
    events = load_events()

    logger.info("Merging impressions with metadata")
    master = merge_impressions_with_metadata(impressions, metadata)

    logger.info("Aggregating transactions and events")
    txn_agg = aggregate_transactions(transactions)
    evt_agg = aggregate_events(events)

    logger.info("Joining aggregates")
    master = merge_transactions(master, txn_agg)
    master = merge_events(master, evt_agg)

    logger.info("Casting datetime columns and filling missing values")
    for col in config.DATETIME_COLS:
        if col in master.columns:
            master[col] = pd.to_datetime(master[col])

    master.fillna({
        "txn_amount_sum": 0.0,
        "txn_count": 0,
        "event_count": 0,
    }, inplace=True)

    logger.info("Saving master table to %s", output_path)
    ensure_dir(output_path.parent)
    master.to_parquet(output_path, index=False)
    return master
