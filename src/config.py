"""Project-wide configuration and defaults."""
from pathlib import Path

# Paths
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
MASTER_TABLE = DATA_DIR / "master_table.parquet"
LEADERBOARD_PATH = RESULTS_DIR / "leaderboard.json"
METRICS_CSV = RESULTS_DIR / "model_metrics.csv"

# Input files
IMPRESSIONS_FILE = DATA_DIR / "impressions.csv"
TRANSACTIONS_FILE = DATA_DIR / "transactions.csv"
EVENTS_FILE = DATA_DIR / "events.csv"
METADATA_FILE = DATA_DIR / "offer_metadata.csv"

# Data loading
READ_CHUNKSIZE = 500_000
DATETIME_COLS = [
    "impression_time",
    "transaction_time",
    "event_time",
]

# Feature engineering
N_WORKERS = 4
WINDOWS = [7, 14, 30]
TARGET_COL = "clicked"
GROUP_COL = "customer_id"
ITEM_COL = "offer_id"

# Modeling
RANDOM_STATE = 42
CV_FOLDS = 5
MAP_K = 7
TEST_SIZE = 0.2

XGB_PARAMS = {
    "objective": "rank:pairwise",
    "learning_rate": 0.08,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 200,
    "min_child_weight": 1,
    "random_state": RANDOM_STATE,
    "tree_method": "hist",
}

CAT_PARAMS = {
    "loss_function": "YetiRank",
    "iterations": 300,
    "depth": 8,
    "learning_rate": 0.08,
    "random_seed": RANDOM_STATE,
    "l2_leaf_reg": 3.0,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.8,
    "verbose": False,
}

SMOTE_PARAMS = {
    "sampling_strategy": 0.5,
    "k_neighbors": 5,
    "random_state": RANDOM_STATE,
}
