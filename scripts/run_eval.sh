#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
import json
from pathlib import Path

from src import config

if not config.LEADERBOARD_PATH.exists():
    raise SystemExit("Leaderboard not found. Run scripts/run_train.sh first.")

with config.LEADERBOARD_PATH.open() as f:
    leaderboard = json.load(f)

print("Leaderboard results (MAP@K):")
for model, metrics in leaderboard.items():
    print(f"- {model}: mean={metrics['map@k_mean']:.4f}, std={metrics['map@k_std']:.4f}")
PY
