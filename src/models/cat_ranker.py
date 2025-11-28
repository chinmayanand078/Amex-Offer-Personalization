"""CatBoost ranking wrapper."""
import pandas as pd
from catboost import CatBoostRanker, Pool

from src import config


def train_cat_ranker(X_train: pd.DataFrame, y_train: pd.Series, group_ids) -> CatBoostRanker:
    train_pool = Pool(X_train, label=y_train, group_id=group_ids)
    model = CatBoostRanker(**config.CAT_PARAMS)
    model.fit(train_pool)
    return model


def predict_cat_ranker(model: CatBoostRanker, X: pd.DataFrame) -> pd.Series:
    return pd.Series(model.predict(X), index=X.index)
