#!/bin/bash
# Behaviour-derived vs authored refusal direction on the full 480-item pool.
# At n=160 the authored direction had no instrument failures on Qwen, so there was
# nothing to remove. On the full pool it has 3 on Qwen and 22 on Phi. Qwen runs
# first, then the clean checkpoints, then Phi, which is expected to fail since the
# behavioural direction needs at least 8 abstentions on each side.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python3}
while pgrep -f "natural_model_screen.py" > /dev/null || pgrep -f "bbq_firstparty.py" > /dev/null; do
  sleep 60
done
run () {
  echo "=== START $2 $(date) ==="
  $PY natural_model_screen.py --model "$1" --limit 480 --draws 16 \
      --direction behavioral --out "screen_${2}_n480_dir_behavioral.json" \
      >> "screen_${2}_n480_dir_behavioral.log" 2>&1
  echo "=== DONE $2 rc=$? $(date) ==="
}
run Qwen/Qwen2.5-7B-Instruct           qwen
run allenai/OLMo-2-1124-7B-SFT         olmo_sft
run allenai/OLMo-2-1124-7B-DPO         olmo_dpo
run allenai/OLMo-2-1124-7B-Instruct    olmo_inst
run mistralai/Mistral-7B-Instruct-v0.3 mistral
run microsoft/Phi-3.5-mini-instruct    phi
echo "ALL DIRECTION-FULL RUNS COMPLETE $(date)"
