"""Summarise the six-checkpoint first-party BBQ runs written by bbq_firstparty.py.

Checks whether the coverage/disposition split replicates on our own prompts,
model calls and parsing instead of a published table.

Usage: python3 bbq_firstparty_summary.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MODELS = {
    "qwen": "Qwen2.5-7B",
    "olmo_sft": "OLMo-2-SFT",
    "olmo_dpo": "OLMo-2-DPO",
    "olmo_inst": "OLMo-2-Instruct",
    "mistral": "Mistral-7B",
    "phi": "Phi-3.5-mini",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="bbq",
                    help="result-file prefix: 'bbq' (3 categories) or "
                         "'bbq11' (all 11 categories)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--rank-stats", action="store_true",
                    help="also emit rank correlations with bootstrap CIs")
    a = ap.parse_args()
    out_path = Path(a.out) if a.out else Path(
        f"results_{a.prefix}_firstparty_summary.json")

    rows = []
    for key, name in MODELS.items():
        d = json.loads(Path(f"lab_results/{a.prefix}_{key}.json").read_text())
        rows.append({
            "model": name,
            "n_items": d.get("n_items"),
            "baseline_abstention": d["baseline"]["abstention_rate"],
            "baseline_s_AMB": d["baseline"]["s_AMB"],
            "instructed_abstention": d["instructed"]["abstention_rate"],
            "instructed_s_AMB": d["instructed"]["s_AMB"],
            "mechanical_fraction": d["mechanical_fraction"],
            "forced_logprob_biased_rate": d["forced_biased_rate"],
            "manski_lo": d["manski_lo"],
            "manski_hi": d["manski_hi"],
            "forced_rate_inside_interval": d["forced_rate_inside_interval"],
        })

    s_amb = [r["baseline_s_AMB"] for r in rows]
    biased = [r["forced_logprob_biased_rate"] for r in rows]
    abst = [r["baseline_abstention"] for r in rows]
    n_inside = sum(r["forced_rate_inside_interval"] for r in rows)

    print(f"{'Model':16s} {'base abst':>10s} {'base s_AMB':>11s} "
          f"{'instr abst':>11s} {'mech %':>8s} {'forced biased':>14s} {'inside?':>8s}")
    for r in rows:
        print(f"{r['model']:16s} {r['baseline_abstention']*100:9.1f}% "
              f"{r['baseline_s_AMB']:11.4f} {r['instructed_abstention']*100:10.1f}% "
              f"{r['mechanical_fraction']*100:7.1f}% "
              f"{r['forced_logprob_biased_rate']*100:13.1f}% "
              f"{'yes' if r['forced_rate_inside_interval'] else 'NO':>8s}")

    summary = {
        "n_models": len(rows),
        "baseline_s_AMB_min": min(s_amb), "baseline_s_AMB_max": max(s_amb),
        "baseline_s_AMB_ratio": max(s_amb) / min(s_amb),
        "baseline_abstention_min": min(abst), "baseline_abstention_max": max(abst),
        "forced_biased_rate_min": min(biased), "forced_biased_rate_max": max(biased),
        "forced_biased_rate_range_pp": (max(biased) - min(biased)) * 100,
        "n_models_manski_interval_contains_truth": n_inside,
        "rows": rows,
    }
    summary["source_prefix"] = a.prefix
    summary["n_items_per_model"] = rows[0].get("n_items")
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nbaseline s_AMB: {min(s_amb):.4f}-{max(s_amb):.4f} "
          f"({max(s_amb)/min(s_amb):.1f}x range)")
    print(f"forced-logprob biased rate: {min(biased)*100:.1f}%-{max(biased)*100:.1f}% "
          f"({(max(biased)-min(biased))*100:.1f} pp range)")
    print(f"Manski interval contains forced truth: {n_inside}/{len(rows)}")
    print(f"\nWrote {out_path}")
    if a.rank_stats:
        rs = rank_stats(str(out_path))
        print(f"\nSpearman(abstention, score)      = "
              f"{rs['spearman_abstention_vs_score']:+.3f} "
              f"(p={rs['p_abstention_vs_score']:.3f})")
        print(f"Spearman(score, disposition)     = "
              f"{rs['spearman_score_vs_disposition']:+.3f} "
              f"(p={rs['p_score_vs_disposition']:.3f}), bootstrap 95% CI "
              f"[{rs['score_vs_disposition_boot_ci'][0]:+.2f}, "
              f"{rs['score_vs_disposition_boot_ci'][1]:+.2f}]")
        print("Wrote results_bbq11_rank_stats.json")


def rank_stats(summary_path: str = "results_bbq11_firstparty_summary.json") -> dict:
    """Rank correlations between reported score, abstention, and measured truth.

    Reported with their uncertainty because n=6 checkpoints is far too few to
    establish a rank relationship: the score-vs-disposition correlation has a
    bootstrap CI spanning almost the entire possible range. The abstention-vs-
    score relationship is the one that survives, and it is the mechanism this
    paper is about.
    """
    import numpy as np
    from scipy.stats import spearmanr

    d = json.loads(Path(summary_path).read_text())
    rows = d["rows"]
    s = np.array([r["baseline_s_AMB"] for r in rows])
    t = np.array([r["forced_logprob_biased_rate"] for r in rows])
    a = np.array([r["baseline_abstention"] for r in rows])

    rng = np.random.default_rng(7)
    boots = []
    for _ in range(20000):
        idx = rng.integers(0, len(s), len(s))
        if len(set(s[idx])) < 3:
            continue
        r, _ = spearmanr(s[idx], t[idx])
        if not np.isnan(r):
            boots.append(r)
    boots = np.array(boots)

    rho_at, p_at = spearmanr(a, s)
    rho_st, p_st = spearmanr(s, t)
    out = {
        "n_models": len(rows),
        "spearman_abstention_vs_score": float(rho_at),
        "p_abstention_vs_score": float(p_at),
        "spearman_score_vs_disposition": float(rho_st),
        "p_score_vs_disposition": float(p_st),
        "score_vs_disposition_boot_ci": [float(np.percentile(boots, 2.5)),
                                         float(np.percentile(boots, 97.5))],
        "note": "score-vs-disposition CI spans nearly the full range at n=6; "
                "not an established relationship in either direction.",
    }
    Path("results_bbq11_rank_stats.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


if __name__ == "__main__":
    main()
