#!/bin/bash
# Template robustness: same 2,200 BBQ items per checkpoint with reworded prompts.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python3}
while pgrep -f "natural_model_screen.py" > /dev/null || pgrep -f "bbq_firstparty.py" > /dev/null; do sleep 60; done
run () {
  echo "=== START $2 $(date) ==="
  $PY bbq_firstparty.py --model "$1" --per-category 200 --variant para \
      --out "bbq11para_$2.json" >> "bbq11para_$2.log" 2>&1
  echo "=== DONE $2 rc=$? $(date) ==="
}
run Qwen/Qwen2.5-7B-Instruct            qwen
run mistralai/Mistral-7B-Instruct-v0.3  mistral
run allenai/OLMo-2-1124-7B-SFT          olmo_sft
run allenai/OLMo-2-1124-7B-DPO          olmo_dpo
run allenai/OLMo-2-1124-7B-Instruct     olmo_inst
run microsoft/Phi-3.5-mini-instruct     phi
