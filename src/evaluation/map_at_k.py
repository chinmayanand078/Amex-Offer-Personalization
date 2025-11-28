"""Mean Average Precision at K implementation."""
import numpy as np
import pandas as pd

from src import config


def map_at_k(y_true: pd.Series, y_score: pd.Series, group_ids: pd.Series, k: int = config.MAP_K) -> float:
    data = pd.DataFrame({"y_true": y_true, "y_score": y_score, "group": group_ids})
    aps = []
    for _, group in data.sort_values("y_score", ascending=False).groupby("group"):
        top_k = group.head(k)
        relevant = top_k["y_true"].values
        if relevant.sum() == 0:
            continue
        precision = np.cumsum(relevant) / (np.arange(len(relevant)) + 1)
        ap = (precision * relevant).sum() / min(k, relevant.sum())
        aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0
