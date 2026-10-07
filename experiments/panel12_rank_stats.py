"""Twelve checkpoints on the 11-category BBQ panel: does the score track abstention?

Section 3.3 reports the correlation for the six original checkpoints. All twelve checkpoints were also run on
the same 2,200-item, 11-category panel (bbq11_*), so this gives the twelve-checkpoint version on that panel.
The multilingual table uses a different panel for English (MBBQ, six categories).

With twelve checkpoints there are too many permutations to enumerate, so the p-value is a Monte Carlo
permutation test (200,000 shuffles, fixed seed).

Output: results_bbq11_12ckpt_rank.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from bbq_by_category import cell_stats

HERE = Path(__file__).parent
LAB = HERE / "lab_results"
TAGS = ["qwen", "olmo_sft", "olmo_dpo", "olmo_inst", "mistral", "phi", "qwen3_4b", "gemma4_e2b",
        "phi4_mini", "granite4_micro", "olmo3_7b", "ministral3_3b"]
SHUFFLES = 200_000


def main() -> None:
    abstention, score, inside, cells = [], [], 0, 0
    for tag in TAGS:
        summary = json.loads((LAB / f"bbq11_{tag}.json").read_text())["baseline"]
        abstention.append(summary["abstention_rate"])
        score.append(summary["s_AMB"])
        df = pd.read_csv(LAB / f"bbq11_{tag}_per_item.csv", keep_default_na=False, na_values=[""])
        for _, g in df.groupby("category"):
            inside += int(cell_stats(g)["inside"])
            cells += 1
    a, s = np.array(abstention), np.array(score)
    rho = float(spearmanr(a, s)[0])
    rng = np.random.default_rng(11)
    hits = sum(abs(spearmanr(a, rng.permutation(s))[0]) >= abs(rho) - 1e-12 for _ in range(SHUFFLES))
    out = {
        "n_checkpoints": len(TAGS),
        "n_items_per_checkpoint": 2200,
        "n_categories": 11,
        "spearman_abstention_vs_score": rho,
        "p_monte_carlo_permutation": (hits + 1) / (SHUFFLES + 1),
        "n_shuffles": SHUFFLES,
        "manski_cells_inside": inside,
        "manski_cells_total": cells,
    }
    (HERE / "results_bbq11_12ckpt_rank.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
