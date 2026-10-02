# Changelog

## 0.1.1

- Results page with a headline table, plots and a copy of every summary file under `results/`
- 80 tests: the certificate, both response parsers, the data item counts, and a GPU-free check that
  each stored BBQ summary can be re-derived from its per-item CSV
- CI now lints with ruff, runs the tests, and installs the pinned versions in `constraints.txt`
- Weekly run against the newest dependency releases, and a release workflow that builds the paper
- `docs/paper-map.md` maps paper sections, tables and figures to the scripts and files behind them
- Dataset documentation with source, license and item counts
- Docstrings on the core functions and the table and figure generators
- Paper: code link as a footnote on the first page (switch in the preamble), accented author name fixed
  in the references, method figure redrawn in TikZ
- Cleanup of leftover paths and unused code from the old layout

## 0.1.0

First release of the code, data and results behind the paper.

- Decomposition of 32 published BBQ method/model results into coverage and disposition
- First-party BBQ runs on 12 checkpoints, 11 categories, and 5 further languages
  (Spanish, Dutch, Turkish, Korean, Catalan)
- Compatibility certificate with an exact-binomial correction, calibration and power studies
- WinoGender certificate screen on 138,240 generations
- Parser audits for the BBQ and WinoGender response parsers
- `scripts/verify_paper_claims.py` checks every number in the paper against the stored results
