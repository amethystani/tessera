#!/bin/bash
# Authored-probe vs behaviour-derived refusal direction on the n=160 screen,
# Phi-3.5-mini and Qwen2.5-7B only. Items, seed, draws, conditions and parser are
# held fixed, so any change in instrument failures comes from the direction.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python3}

while pgrep -f "natural_model_screen.py" > /dev/null || \
      pgrep -f "bbq_firstparty.py" > /dev/null; do
  echo "waiting for GPU... $(date)"
  sleep 120
done
echo "=== GPU free, starting direction comparison $(date) ==="

run () {
  echo "=== START $2 ($3) $(date) ==="
  $PY natural_model_screen.py --model "$1" --limit 160 --draws 16 \
      --direction "$3" --out "screen_${2}_n160_dir_${3}.json" \
      >> "screen_${2}_n160_dir_${3}.log" 2>&1
  echo "=== DONE $2 ($3) rc=$? $(date) ==="
}

run microsoft/Phi-3.5-mini-instruct  phi   behavioral
run Qwen/Qwen2.5-7B-Instruct         qwen  behavioral
echo "ALL DIRECTION RUNS COMPLETE $(date)"
