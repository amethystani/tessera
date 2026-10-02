#!/bin/bash
# Catalan run for OLMo-2-DPO, retried with a smaller batch after an OOM on a shared GPU.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python3}
while pgrep -f "[r]etry_cabbq_sft.sh" > /dev/null; do sleep 30; done
echo "=== RETRY olmo_dpo START $(date)"
$PY bbq_firstparty.py --model allenai/OLMo-2-1124-7B-DPO --batch-size 8 \
    --items-csv cabbq_items.csv --out ca_olmo_dpo.json >> ca_olmo_dpo_retry.log 2>&1
echo "=== RETRY olmo_dpo rc=$? $(date)"
