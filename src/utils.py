"""Common helpers for logging and reproducibility."""
import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)


def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numerical columns to save memory when loading large CSVs."""
    for col in df.select_dtypes(include=["int", "float"]).columns:
        col_min, col_max = df[col].min(), df[col].max()
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], downcast="float")
        else:
            if col_min >= 0:
                df[col] = pd.to_numeric(df[col], downcast="unsigned")
            else:
                df[col] = pd.to_numeric(df[col], downcast="integer")
    return df
