#!/bin/bash
# Ministral 3 3B. The default checkpoint ships FP8 weights that need a kernel
# package we don't have, so use Mistral's BF16 release of the same model.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python3}
while pgrep -f run_newmodels.sh > /dev/null; do sleep 60; done
export HF_HUB_CACHE=/tmp/hf_am_newmodels
M=mistralai/Ministral-3-3B-Instruct-2512-BF16; T=ministral3_3b
echo "=== NEW $T (BF16) START $(date)"
if $PY bbq_firstparty.py --model $M --dtype bfloat16 --items-csv multilingual_items.csv \
     --smoke 2 --out smoke_$T.json > smoke_$T.log 2>&1; then
  $PY bbq_firstparty.py --model $M --dtype bfloat16 --per-category 200 --out bbq11_$T.json >> bbq11_$T.log 2>&1
  echo "=== NEW $T bbq11 rc=$? $(date)"
  $PY bbq_firstparty.py --model $M --dtype bfloat16 --items-csv multilingual_items.csv --out ml_$T.json >> ml_$T.log 2>&1
  echo "=== NEW $T ml rc=$? $(date)"
else
  echo "=== NEW $T SMOKE FAILED $(date)"; tail -5 smoke_$T.log
fi
rm -rf /tmp/hf_am_newmodels
echo "=== MINISTRAL DONE $(date)"
