#!/bin/bash
# WinoGender certificate screen on all 480 non-neutral items instead of the
# 160-item sample, so the result doesn't depend on which 160 were drawn.
# 480 items x 16 draws x 3 conditions = 23,040 generations per checkpoint.
# Waits for any running BBQ sweep first, since two 7B models won't fit on the GPU.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python3}

while pgrep -f "bbq_firstparty.py" > /dev/null; do
  echo "waiting for BBQ sweep to finish... $(date)"
  sleep 120
done
echo "=== GPU free, starting full-480 screens $(date) ==="

run () {
  echo "=== START $2 $(date) ==="
  $PY natural_model_screen.py --model "$1" --limit 480 --draws 16 \
      --out "screen_${2}_n480_d16.json" >> "screen_${2}_n480_d16.log" 2>&1
  echo "=== DONE $2 rc=$? $(date) ==="
}

run Qwen/Qwen2.5-7B-Instruct            qwen
run allenai/OLMo-2-1124-7B-SFT          olmo_sft
run allenai/OLMo-2-1124-7B-DPO          olmo_dpo
run allenai/OLMo-2-1124-7B-Instruct     olmo_inst
run mistralai/Mistral-7B-Instruct-v0.3  mistral
run microsoft/Phi-3.5-mini-instruct     phi
echo "ALL N480 RUNS COMPLETE $(date)"
