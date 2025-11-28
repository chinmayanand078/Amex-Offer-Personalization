#!/usr/bin/env bash
set -euo pipefail
python - <<'PY'
from src.etl.build_master_table import build_master_table

df = build_master_table()
print(f"Built master table with {len(df):,} rows")
PY
