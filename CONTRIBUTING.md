# Contributing

Thanks for looking. Most useful things here are small: another checkpoint, another BBQ-style dataset,
a number that doesn't reproduce, or a check you think the stored outputs can answer.

## Getting set up

```bash
pip install -r requirements.txt matplotlib
make verify      # every number in the paper against the stored results
```

Re-running models needs a GPU and the extra packages listed in `requirements.txt`.

## Before opening a pull request

- `make verify` should pass.
- If you changed an analysis script or its output, rerun `make assets plots summaries` and commit what
  they produce. CI checks that the generated tables and figures match.
- Numbers in `paper/` come from generated files. Change the script, not the table.
- New model runs should save the raw response text for every draw, as `bbq_firstparty.py` does.
  Several checks (parser audits, re-scoring) depend on it.

## Adding a checkpoint

`experiments/run/run_newmodels.sh` has the pattern. Run the BBQ sweep and the multilingual sweep, put
the outputs in `experiments/lab_results/` with the usual file prefixes, and open a pull request. The
analysis scripts pick up new files by name.

## Questions

Open an issue. There are templates for the common cases.
