#!/bin/bash
# Catalan (CaBBQ) on all 12 checkpoints with the batched bbq_firstparty.py.
# The six newer checkpoints are downloaded to a scratch cache and deleted after
# each run to save disk.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python3}
NEWCACHE=/tmp/hf_am_newmodels

new () {
  echo "=== CA $2 START $(date)"
  export HF_HUB_CACHE=$NEWCACHE
  $PY bbq_firstparty.py --model "$1" --dtype bfloat16 --items-csv cabbq_items.csv \
      --out "ca_$2.json" >> "ca_$2.log" 2>&1
  echo "=== CA $2 rc=$? $(date)"
  rm -rf "$NEWCACHE"; unset HF_HUB_CACHE
}
old () {
  echo "=== CA $2 START $(date)"
  $PY bbq_firstparty.py --model "$1" --items-csv cabbq_items.csv \
      --out "ca_$2.json" >> "ca_$2.log" 2>&1
  echo "=== CA $2 rc=$? $(date)"
}

new Qwen/Qwen3-4B-Instruct-2507                  qwen3_4b
new google/gemma-4-E2B-it                        gemma4_e2b
new microsoft/Phi-4-mini-instruct                phi4_mini
new ibm-granite/granite-4.0-micro                granite4_micro
new allenai/Olmo-3-7B-Instruct                   olmo3_7b
new mistralai/Ministral-3-3B-Instruct-2512-BF16  ministral3_3b
old Qwen/Qwen2.5-7B-Instruct                qwen
old allenai/OLMo-2-1124-7B-SFT              olmo_sft
old allenai/OLMo-2-1124-7B-DPO              olmo_dpo
old allenai/OLMo-2-1124-7B-Instruct         olmo_inst
old mistralai/Mistral-7B-Instruct-v0.3      mistral
old microsoft/Phi-3.5-mini-instruct         phi
echo "=== CABBQ ALL DONE $(date)"
