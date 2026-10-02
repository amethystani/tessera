#!/bin/bash
# Catalan run for OLMo-2-SFT. The first attempt hit an OOM in the forced-logprob
# batch (24 x full-vocab logits) while the GPU was shared, so retry with a small batch.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-python3}
while pgrep -f "[r]un_cabbq.sh" > /dev/null; do sleep 30; done
echo "=== RETRY olmo_sft START $(date)"
$PY bbq_firstparty.py --model allenai/OLMo-2-1124-7B-SFT --batch-size 8 \
    --items-csv cabbq_items.csv --out ca_olmo_sft.json >> ca_olmo_sft_retry.log 2>&1
echo "=== RETRY olmo_sft rc=$? $(date)"
