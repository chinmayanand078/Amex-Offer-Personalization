# Personalization of Amex Digital Offers

Production-ready reference project for ranking American Express digital offers.
The repository demonstrates scalable ETL across impressions, transactions, and
user events, rich behavioral feature engineering, and end-to-end model training
with XGBoost and CatBoost rankers. The pipeline is optimized for large datasets
(30M+ rows) using multiprocessing and includes reproducible evaluation via
MAP@7 and group-based cross-validation.

## Project layout
```
Amex-Offer-Personalization/
 ├── src/                   # Source code for the pipeline
 │   ├── config.py          # Paths, constants, and hyperparameters
 │   ├── utils.py           # Logging utilities and helpers
 │   ├── etl/               # ETL modules for each data source
 │   ├── features/          # Feature definitions and parallel transforms
 │   ├── models/            # XGBoost and CatBoost ranker wrappers
 │   ├── evaluation/        # Metrics and cross-validation helpers
 │   └── training_pipeline.py
 ├── notebooks/             # EDA and experiment notebooks
 ├── scripts/               # Convenience shell scripts to run the project
 ├── results/               # Leaderboard and metrics artifacts
 └── data/                  # Local data directory (ignored in git)
```

## Quickstart
1. Install dependencies (ideally in a virtual environment):
   ```bash
   pip install -r requirements.txt
   ```
2. Place the raw CSVs in `data/` as configured in `src/config.py`.
3. Run the ETL to build the master table:
   ```bash
   bash scripts/run_etl.sh
   ```
4. Train and evaluate models:
   ```bash
   bash scripts/run_train.sh
   bash scripts/run_eval.sh
   ```

## Data expectations
The ETL modules expect CSV files for impressions, transactions, events, and
static metadata. You can adjust file names, dtypes, and chunk sizes in
`src/config.py`. The pipeline uses multiprocessing for feature generation and
balancing (SMOTE) to manage label imbalance.

## Reproducibility
* Deterministic seeds are set in `src/config.py`.
* Cross-validation uses `GroupKFold` to prevent leakage across customers.
* Metrics and leaderboard entries are persisted to `results/`.

## Notebook experiments
Two starter notebooks are included for exploratory analysis and training
experiments. They read from the processed master table and reuse the functions
from `src/` so results stay consistent with the scripted pipeline.

## License
This project is provided for educational and portfolio purposes.
