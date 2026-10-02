#!/bin/bash
# Resume of run_newmodels.sh after qwen3_4b, gemma4_e2b, phi4_mini and
# granite4_micro had finished. Runs olmo3_7b and ministral3_3b (BF16 release), then
# the original six checkpoints on the multilingual set.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python3}
NEWCACHE=/tmp/hf_am_newmodels

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
old () {   # original checkpoints, already cached, original fp16, multilingual only
  echo "=== OLD $2 START $(date)"
  $PY bbq_firstparty.py --model "$1" --items-csv multilingual_items.csv \
      --out "ml_$2.json" >> "ml_$2.log" 2>&1; echo "=== OLD $2 ml rc=$? $(date)"
}

new allenai/Olmo-3-7B-Instruct                   olmo3_7b
new mistralai/Ministral-3-3B-Instruct-2512-BF16  ministral3_3b
old Qwen/Qwen2.5-7B-Instruct                qwen
old allenai/OLMo-2-1124-7B-SFT              olmo_sft
old allenai/OLMo-2-1124-7B-DPO              olmo_dpo
old allenai/OLMo-2-1124-7B-Instruct         olmo_inst
old mistralai/Mistral-7B-Instruct-v0.3      mistral
old microsoft/Phi-3.5-mini-instruct         phi
echo "=== ALL DONE $(date)"
