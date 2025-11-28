#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
from src.training_pipeline import run_training

results = run_training()
print("Training complete. Leaderboard:")
print(results)
PY
