"""High-level training pipeline for Amex offer personalization."""
import logging
from pathlib import Path
from typing import Dict

import pandas as pd

from src import config
from src.evaluation.cross_validation import run_cv
from src.features.feature_engineering import add_behavioral_features, select_feature_matrix
from src.features.feature_sets import ALL_FEATURES
from src.features.parallel_transform import parallel_apply
from src.models.cat_ranker import predict_cat_ranker, train_cat_ranker
from src.models.xgb_ranker import predict_xgb_ranker, train_xgb_ranker
from src.utils import configure_logging, ensure_dir, save_json, seed_everything

logger = logging.getLogger(__name__)


def _prepare_features(master: pd.DataFrame) -> pd.DataFrame:
    engineered = parallel_apply(master, add_behavioral_features, n_workers=config.N_WORKERS)
    engineered = engineered.sort_values([config.GROUP_COL, config.ITEM_COL])
    return engineered


def _train_and_eval_model(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
):
    if model_name == "xgboost":
        train_fn, predict_fn = train_xgb_ranker, predict_xgb_ranker
    elif model_name == "catboost":
        train_fn, predict_fn = train_cat_ranker, predict_cat_ranker
    else:
        raise ValueError(f"Unknown model {model_name}")

    cv_results = run_cv(X, y, groups, train_fn, predict_fn)
    logger.info("%s CV results: %s", model_name, cv_results)
    return cv_results


def run_training(master_path: Path = config.MASTER_TABLE) -> Dict[str, Dict[str, float]]:
    configure_logging()
    seed_everything(config.RANDOM_STATE)

    logger.info("Loading master table from %s", master_path)
    master = pd.read_parquet(master_path)
    master = _prepare_features(master)

    feature_cols = [c for c in ALL_FEATURES if c in master.columns]
    X = select_feature_matrix(master, feature_cols)
    y = master[config.TARGET_COL]
    groups = master[config.GROUP_COL]

    results = {}
    for model_name in ["xgboost", "catboost"]:
        results[model_name] = _train_and_eval_model(model_name, X, y, groups)

    ensure_dir(config.RESULTS_DIR)
    save_json(results, config.LEADERBOARD_PATH)
    return results


if __name__ == "__main__":
    run_training()
