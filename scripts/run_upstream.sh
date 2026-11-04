#!/usr/bin/env bash
# Reproduce the JTH paper's baselines with the official code (external/JTH).
#
#   bash scripts/run_upstream.sh            # fast baselines, both directions
#   WITH_SLOW=1 bash scripts/run_upstream.sh  # + Jaccard j2c and LogReg (hours)
#
# Outputs: data/upstream_split/{train,test}.csv, results/upstream/*.pkl.gz,
# results/upstream/results.csv (one row per prediction file).
# Existing prediction files are skipped, so the script can be re-run.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UP="$ROOT/external/JTH/code"
DATA="$ROOT/data/raw"
SPLIT="$ROOT/data/upstream_split"
RES="$ROOT/results/upstream"
PY=(uv run --project "$ROOT" python)

mkdir -p "$RES"
cd "$RES"  # feature_classifier_baseline.py writes features.csv to the cwd

if [[ ! -f "$SPLIT/test.csv" ]]; then
  "${PY[@]}" "$UP/utils/binary_splitter.py" "$DATA/history.csv" "$SPLIT"
fi

run() {  # run <out-file> <script> <args...>
  local out="$RES/$1"; shift
  if [[ -f "$out" ]]; then echo "skip $(basename "$out")"; return; fi
  echo "== $(basename "$out")"
  local start=$SECONDS
  "${PY[@]}" "$@" "$out"
  echo "   done in $((SECONDS - start))s"
}

for m in c2j j2c; do
  run "random_$m.pkl.gz"      "$UP/baselines/random_baseline.py" "$DATA/candidates.csv" "$DATA/jobs.csv" "$SPLIT/test.csv" --mode "$m"
  run "popularity_$m.pkl.gz"  "$UP/baselines/online_popularity_baseline.py" "$SPLIT/train.csv" "$SPLIT/test.csv" --mode "$m"
  run "poprecency_$m.pkl.gz"  "$UP/baselines/popularity_recency_baseline.py" "$SPLIT/train.csv" "$SPLIT/test.csv" "$DATA/candidates.csv" "$DATA/jobs.csv" --mode "$m" --lambda 0.015
  run "pasttemporal_$m.pkl.gz" "$UP/baselines/past_temporal_baseline.py" "$DATA/candidates.csv" "$DATA/jobs.csv" "$SPLIT/test.csv" --mode "$m"
  run "cf_$m.pkl.gz"          "$UP/baselines/collab_cf_temporal.py" "$SPLIT/train.csv" "$SPLIT/test.csv" --mode "$m"
  run "cfrecency_$m.pkl.gz"   "$UP/baselines/hybrid_recency_cf_baseline.py" "$SPLIT/train.csv" "$SPLIT/test.csv" "$DATA/candidates.csv" "$DATA/jobs.csv" --mode "$m" --max_age 90
done
run "jaccard_c2j.pkl.gz" "$UP/baselines/jaccard_baseline.py" "$DATA/candidates.csv" "$DATA/jobs.csv" "$SPLIT/test.csv" --mode c2j --use_human_features --use_llm_features

if [[ "${WITH_SLOW:-0}" == 1 ]]; then
  run "jaccard_j2c.pkl.gz" "$UP/baselines/jaccard_baseline.py" "$DATA/candidates.csv" "$DATA/jobs.csv" "$SPLIT/test.csv" --mode j2c --use_human_features --use_llm_features
  run "logreg_c2j.pkl.gz" "$UP/baselines/feature_classifier_baseline.py" "$DATA/candidates.csv" "$DATA/jobs.csv" "$SPLIT/train.csv" "$SPLIT/test.csv" --mode c2j --neg_ratio 1.0 --model logreg --use_temporal_features --use_llm_features
fi

# The positional prediction file comes after the gold file in evaluator.py.
# evaluator.py keys results.csv rows by the prediction path and SKIPS paths it
# already has, so paths are passed relative to the repo root (stable across
# machines). To re-score a regenerated file, delete its row from results.csv.
cd "$ROOT"
for f in results/upstream/*.pkl.gz; do
  m=$(basename "$f" .pkl.gz); m=${m##*_}
  "${PY[@]}" "$UP/utils/evaluator.py" --mode "$m" --k 1 5 10 20 \
    --train "$SPLIT/train.csv" --candidates "$DATA/candidates.csv" --jobs "$DATA/jobs.csv" \
    --out "$RES/results.csv" "$SPLIT/test.csv" "$f" > /dev/null
done
echo "results -> $RES/results.csv"
