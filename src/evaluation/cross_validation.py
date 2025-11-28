"""Cross-validation utilities for ranking models."""
import logging
from typing import Callable, Dict, Tuple

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GroupKFold

from src import config
from src.evaluation.map_at_k import map_at_k

logger = logging.getLogger(__name__)


def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    train_fn: Callable,
    predict_fn: Callable,
) -> Dict[str, float]:
    gkf = GroupKFold(n_splits=config.CV_FOLDS)
    scores = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        logger.info("Starting fold %s", fold + 1)
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        groups_val = groups.iloc[val_idx]

        smote = SMOTE(**config.SMOTE_PARAMS)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        group_sizes = (
            pd.Series(groups.iloc[train_idx])
            .value_counts()
            .reindex(pd.Series(groups.iloc[train_idx]).unique())
            .fillna(1)
            .astype(int)
            .values
        )

        model = train_fn(X_res, y_res, group_sizes)
        preds = predict_fn(model, X_val)
        score = map_at_k(y_val, preds, groups_val)
        scores.append(score)
        logger.info("Fold %s MAP@%s: %.4f", fold + 1, config.MAP_K, score)

    return {"map@k_mean": float(pd.Series(scores).mean()), "map@k_std": float(pd.Series(scores).std())}
