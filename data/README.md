# Data

| Path | What it is | Source |
|---|---|---|
| `bbq/` | BBQ item files, 11 categories (jsonl) | Parrish et al. (2022), https://github.com/nyu-mll/BBQ, CC-BY-4.0 |
| `winogender/` | WinoGender sentences and occupation statistics | Rudinger et al. (2018), https://github.com/rudinger/winogender-schemas |
| `decap_bbq_ambiguous.csv` | Ambiguous-split accuracy and bias score for 5 methods x 8 models | Table 11(a) of Bae, Choi & Lee (2025), arXiv:2503.19426, transcribed by hand |
| `decap_source_notes.md` | Other papers checked as sources for the decomposition and why they were not usable | |

The multilingual item sets (MBBQ, KoBBQ, CaBBQ) are not copied here. They are built from the
public releases by `experiments/multilingual_prepare.py` and `experiments/cabbq_prepare.py`.
The resulting `multilingual_items.csv` (MBBQ and KoBBQ) and `cabbq_items.csv` are checked in
under `experiments/` with their match reports.
