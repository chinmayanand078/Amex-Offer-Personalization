"""XGBoost ranking wrapper."""
from typing import Tuple

import pandas as pd
from xgboost import XGBRanker

from src import config


def train_xgb_ranker(X_train: pd.DataFrame, y_train: pd.Series, group_sizes) -> XGBRanker:
    model = XGBRanker(**config.XGB_PARAMS)
    model.fit(X_train, y_train, group=group_sizes)
    return model


def predict_xgb_ranker(model: XGBRanker, X: pd.DataFrame) -> pd.Series:
    return pd.Series(model.predict(X), index=X.index)
