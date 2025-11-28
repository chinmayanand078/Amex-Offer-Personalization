"""Parallel wrappers around feature engineering steps."""
import logging
from multiprocessing import Pool
from typing import Callable

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def _apply_fn(args):
    fn, chunk = args
    return fn(chunk)


def parallel_apply(
    df: pd.DataFrame, fn: Callable[[pd.DataFrame], pd.DataFrame], n_workers: int = config.N_WORKERS
) -> pd.DataFrame:
    if n_workers <= 1 or len(df) < n_workers * 10_000:
        logger.info("Processing features in a single process")
        return fn(df)

    logger.info("Processing features with %s workers", n_workers)
    splits = np.array_split(df, n_workers)
    with Pool(processes=n_workers) as pool:
        parts = pool.map(_apply_fn, [(fn, split) for split in splits])
    return pd.concat(parts, ignore_index=True)
