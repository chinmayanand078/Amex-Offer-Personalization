"""Aggregate transaction history for each customer/offer pair."""
import pandas as pd

from src import config


def aggregate_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame(columns=[config.GROUP_COL, config.ITEM_COL, "txn_amount_sum", "txn_count"])

    grouped = transactions.groupby([config.GROUP_COL, config.ITEM_COL]).agg(
        txn_amount_sum=("amount", "sum"),
        txn_count=("amount", "count"),
        last_transaction=("transaction_time", "max"),
    )
    grouped = grouped.reset_index()
    return grouped


def merge_transactions(
    impressions: pd.DataFrame, txn_agg: pd.DataFrame
) -> pd.DataFrame:
    return impressions.merge(txn_agg, how="left", on=[config.GROUP_COL, config.ITEM_COL])
