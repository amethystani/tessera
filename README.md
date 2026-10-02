# Tessera

Code, data and results for the paper **Abstention Masquerading as Debiasing: Partial
Identification and Calibrated Certificates for LLM Fairness Evaluation**.

BBQ's ambiguous-context bias score is `s_AMB = (1 - accuracy) * b`, where accuracy is the rate at
which the model answers "unknown". A model that abstains more gets a lower score without being any
less biased in the answers it does give. The paper does three things with that identity:

1. Splits 32 published method/model results into a coverage part (more abstention) and a
   disposition part (less bias among committed answers). The median coverage share is 80.4%, committed-answer
   bias got worse in 7 of the 32, and the identified set for the underlying preference widened in 31.
2. Repeats the analysis on our own runs: 12 checkpoints (2024 to 2026 releases), 11 BBQ categories, and
   five further languages (Spanish, Dutch, Turkish, Korean, Catalan). The score mostly ranks checkpoints by how
   often they abstain. Catalan is the one language where this breaks down, and the recovered bias `b`
   shows why.
3. Builds a compatibility certificate for checking whether several elicitation conditions (direct,
   forced choice, refusal-ablated) can share one answer law. The plug-in version rejects about 21% of
   compatible items regardless of draw count. An exact-binomial correction brings that to 0.05% or less. It is applied to
   138,240 WinoGender generations.

## Layout

```
paper/        LaTeX source, bibliography, generated tables/ and figures/
src/          decomposition of published scores, bias metrics, refusal-ablation experiment
experiments/  model runs, analyses, and their outputs
  lab_results/   per-item model outputs and summaries (CSV/JSON), one file set per checkpoint
  logs/          stdout from the runs
  run/           shell scripts that launched the sweeps
data/         BBQ, WinoGender, and the transcribed published numbers (see data/README.md)
results/      output of the published-table decomposition
scripts/      verify_paper_claims.py
```

## Reproducing the paper's numbers

Everything below runs on a laptop from the committed outputs. No model is called.

```bash
pip install -r requirements.txt

# every number quoted in the paper against the stored results
python scripts/verify_paper_claims.py

# regenerate the paper's tables and figures from the stored results
cd experiments && python make_paper_assets.py

# build the PDF
cd ../paper && latexmk -pdf main.tex      # or: tectonic main.tex
```

`verify_paper_claims.py` recomputes each claim from its source file and compares it with the text of
`paper/main.tex`, so a table or sentence that drifts from the data fails the check.

## Re-running the models

The per-item outputs in `experiments/lab_results/` came from a single GPU machine. To regenerate them you
need `torch`, `transformers` and enough memory for a 7B model in fp16.

```bash
cd experiments

# BBQ on one checkpoint (baseline, instructed, forced choice), 200 items per category
python bbq_firstparty.py --model Qwen/Qwen2.5-7B-Instruct --per-category 200 --out bbq11_qwen.json

# WinoGender certificate screen, all 480 items, 16 draws per condition
python natural_model_screen.py --model Qwen/Qwen2.5-7B-Instruct --limit 480 --draws 16

# the full sweeps as they were run
bash run/run_bbq_all11.sh
bash run/run_screen_full480.sh
bash run/run_newmodels.sh
```

Checkpoints: Qwen2.5-7B-Instruct, OLMo-2-1124-7B (SFT, DPO, Instruct), Mistral-7B-Instruct-v0.3,
Phi-3.5-mini-instruct, Qwen3-4B-Instruct-2507, gemma-4-E2B-it, Phi-4-mini-instruct, granite-4.0-micro,
Olmo-3-7B-Instruct and Ministral-3-3B-Instruct-2512.

## Analyses

| Script | What it produces |
|---|---|
| `src/decap_decomposition.py` | coverage/disposition split of the 32 published pairs |
| `src/decomposition_robustness.py` | symmetric split and rounding sensitivity |
| `experiments/bbq_firstparty.py` | BBQ runs with saved raw responses |
| `experiments/bbq_by_category.py` | 66 checkpoint x category cells, variance shares, Manski coverage |
| `experiments/abstainer_disposition.py` | what the abstained items would have been |
| `experiments/template_robustness.py` | same analysis under reworded prompts |
| `experiments/multilingual_analysis.py` | six-language replication |
| `experiments/check_theory.py` | the compatibility certificate |
| `experiments/draw_count_calibration.py` | false-positive rate against draw count |
| `experiments/power_by_magnitude.py` | detection power by size of violation |
| `experiments/natural_model_screen.py` | certificate on real WinoGender generations |
| `experiments/violation_taxonomy.py` | which condition is the odd one out in each violation |
| `experiments/mechanism_inference.py`, `mechanism_abstention.py` | "someone" items and the recovery gap |
| `experiments/parser_audit.py`, `winogender_parser_audit.py` | checks on the response parsers |

## Notes

- Response parsers were audited against saved raw text. The BBQ parser agreed with an independent
  classifier on 161 of 161 sampled draws, and the WinoGender parser on 3,838 of 3,840.
- The refusal-ablated condition is a comparison condition, not ground truth. The paper reports where the
  certificate finds it failing.
- Published-table results depend on one paper's harness (DeCAP). The reasons other papers couldn't be
  used are in `data/decap_source_notes.md`.
