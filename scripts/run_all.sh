#!/usr/bin/env bash
# Full national pipeline — run overnight.
# Skips steps where output already exists.
# Usage: bash scripts/run_all.sh 2>&1 | tee logs/run_all.log

set -euo pipefail
BASE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BASE"
mkdir -p logs

ts() { date "+%H:%M:%S"; }

echo "[$(ts)] ── Step 1: Download + filter OSM PBFs (all 50 states)"
python3 scripts/01_fetch_data.py

echo "[$(ts)] ── Step 2: Fetch Census block group era data (all 50 states)"
Rscript scripts/01b_fetch_building_era.R

echo "[$(ts)] ── Step 3: Extract street attributes from OSM (all states)"
python3 scripts/02_extract_streets.py

echo "[$(ts)] ── Step 4: Spatial join — attach era to each street"
python3 scripts/02b_join_era.py

echo "[$(ts)] ── Step 5: Divergence analysis"
python3 scripts/03_divergence_analysis.py

echo "[$(ts)] ── Step 6: Era analysis + figures"
python3 scripts/04_era_analysis.py

echo "[$(ts)] ── Step 7: Render report"
cd reports && quarto render street_type_divergence.qmd
cd "$BASE"

echo "[$(ts)] ── All done."
