#!/bin/bash
# Newer model families on English BBQ (same 2,200 items as the original six) and
# on the multilingual set (MBBQ en/es/nl/tr + KoBBQ ko), then the original six on
# the multilingual set. Each new model is downloaded, run, and deleted.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python3}
NEWCACHE=/tmp/hf_am_newmodels
while pgrep -f "natural_model_screen.py" > /dev/null || pgrep -f "bbq_firstparty.py" > /dev/null; do sleep 60; done

new () {   # $1 model id, $2 tag
  echo "=== NEW $2 START $(date)"
  export HF_HUB_CACHE=$NEWCACHE
  if ! $PY bbq_firstparty.py --model "$1" --dtype bfloat16 --items-csv multilingual_items.csv \
        --smoke 2 --out "smoke_$2.json" > "smoke_$2.log" 2>&1; then
    echo "=== NEW $2 SMOKE FAILED $(date)"; tail -5 "smoke_$2.log"
  else
    $PY bbq_firstparty.py --model "$1" --dtype bfloat16 --per-category 200 \
        --out "bbq11_$2.json" >> "bbq11_$2.log" 2>&1; echo "=== NEW $2 bbq11 rc=$? $(date)"
    $PY bbq_firstparty.py --model "$1" --dtype bfloat16 --items-csv multilingual_items.csv \
        --out "ml_$2.json" >> "ml_$2.log" 2>&1; echo "=== NEW $2 ml rc=$? $(date)"
  fi
  rm -rf "$NEWCACHE"; unset HF_HUB_CACHE
}
old () {   # original checkpoints, already cached under ~/.cache, original fp16
  echo "=== OLD $2 START $(date)"
  $PY bbq_firstparty.py --model "$1" --items-csv multilingual_items.csv \
      --out "ml_$2.json" >> "ml_$2.log" 2>&1; echo "=== OLD $2 ml rc=$? $(date)"
}
new Qwen/Qwen3-4B-Instruct-2507             qwen3_4b
new google/gemma-4-E2B-it                   gemma4_e2b
new mistralai/Ministral-3-3B-Instruct-2512  ministral3_3b
new microsoft/Phi-4-mini-instruct           phi4_mini
new ibm-granite/granite-4.0-micro           granite4_micro
new allenai/Olmo-3-7B-Instruct              olmo3_7b
old Qwen/Qwen2.5-7B-Instruct                qwen
old allenai/OLMo-2-1124-7B-SFT              olmo_sft
old allenai/OLMo-2-1124-7B-DPO              olmo_dpo
old allenai/OLMo-2-1124-7B-Instruct         olmo_inst
old mistralai/Mistral-7B-Instruct-v0.3      mistral
old microsoft/Phi-3.5-mini-instruct         phi
echo "=== ALL DONE $(date)"
