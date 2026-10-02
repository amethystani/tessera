#!/bin/bash
# First-party BBQ sweep on all 11 categories, 200 items each (2,200 per checkpoint),
# for all six checkpoints. Balanced allocation, since usable items per category range
# from 432 to 5,280. Run validate_bbq_matching.py first to check matcher coverage.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python3}

echo "=== matcher validation ==="
$PY validate_bbq_matching.py || { echo "MATCHER VALIDATION FAILED - aborting"; exit 1; }

run () {
  echo "=== START $2 $(date) ==="
  $PY bbq_firstparty.py --model "$1" --per-category 200 \
      --out "bbq11_$2.json" >> "bbq11_$2.log" 2>&1
  echo "=== DONE $2 rc=$? $(date) ==="
}

run Qwen/Qwen2.5-7B-Instruct            qwen
run allenai/OLMo-2-1124-7B-SFT          olmo_sft
run allenai/OLMo-2-1124-7B-DPO          olmo_dpo
run allenai/OLMo-2-1124-7B-Instruct     olmo_inst
run mistralai/Mistral-7B-Instruct-v0.3  mistral
run microsoft/Phi-3.5-mini-instruct     phi
echo "ALL BBQ11 RUNS COMPLETE $(date)"
