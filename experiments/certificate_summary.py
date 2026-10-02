"""Collect the numbers used in the WinoGender section into one JSON file.

Reads the per-checkpoint screens and the analyses run on them, all under
lab_results/:
  screen_{tag}_n480_d16.json            certificate summary per checkpoint
  screen_{tag}_n480_d16_per_item.csv    per-item counts
  model_null_calibration_n480.json      per-model parametric bootstrap null
  violation_taxonomy_n480.json          violation classes
  mechanism_inference_n480.json         recovery-gap stratum tests
  mechanism_abstention_n480.json        abstention-lean stratum tests (Holm)

make_paper_assets.py builds the paper tables from the output and
scripts/verify_paper_claims.py checks the prose against it.

Output: results_n480_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

LAB = Path(__file__).parent / "lab_results"
OUT = Path(__file__).parent / "results_n480_summary.json"

MODELS = {
    "qwen": "Qwen2.5-7B",
    "olmo_sft": "OLMo-2-SFT",
    "olmo_dpo": "OLMo-2-DPO",
    "olmo_inst": "OLMo-2-Instruct",
    "mistral": "Mistral-7B",
    "phi": "Phi-3.5-mini",
}


def by_tag(records, key="file"):
    out = {}
    for r in records:
        tag = r[key].split("screen_")[1].split("_n480")[0]
        out[tag] = r
    return out


def main() -> None:
    null = json.loads((LAB / "model_null_calibration_n480.json").read_text())
    null_rows = by_tag(null["results"])
    tax_rows = by_tag(json.loads((LAB / "violation_taxonomy_n480.json").read_text()))
    mech = json.loads((LAB / "mechanism_inference_n480.json").read_text())
    mech_rows = by_tag([r for r in mech["results"] if r["metric"] == "direct_err"])
    abst = json.loads((LAB / "mechanism_abstention_n480.json").read_text())
    abst_rows = by_tag(abst["results"])
    reps = null["reps"]

    rows = []
    tot_obs = tot_null = 0.0
    for tag, name in MODELS.items():
        s = json.loads((LAB / f"screen_{tag}_n480_d16.json").read_text())
        assert s["n_items"] == 480 and s["n_draws"] == 16, tag
        df = pd.read_csv(LAB / f"screen_{tag}_n480_d16_per_item.csv")
        n = null_rows[tag]
        obs = n["observed_naive_incompatible"]
        assert obs == s["n_items"] - s["n_naive_compatible"], f"{tag}: null vs screen disagree"
        tot_obs += obs
        tot_null += n["null_naive_mean"]
        tc = tax_rows[tag].get("counts", {})
        m, a = mech_rows[tag], abst_rows[tag]
        rows.append({
            "tag": tag, "model": name, "n_items": s["n_items"],
            "naive_incompatible": obs,
            "corrected_incompatible": s["n_items"] - s["n_corrected_compatible"],
            "null_mean": n["null_naive_mean"], "null_p95": n["null_naive_p95"],
            "excess": obs - n["null_naive_mean"],
            "noise_share": n["null_naive_mean"] / obs,
            # A p of 0.0 from `reps` replications means "below 1/reps", not zero.
            "p_naive_floor": 1.0 / reps if n["p_value_naive"] == 0 else n["p_value_naive"],
            "p_naive_is_floor": n["p_value_naive"] == 0,
            "n_instrument": tc.get("instrument_failure", 0),
            "n_policy": tc.get("policy_effect", 0),
            "n_other": (tc.get("forced_outlier", 0) + tc.get("no_majority", 0)
                        + tc.get("all_agree", 0)),
            "direct_disclosure": float(((df.n_direct_occ + df.n_direct_part) / 16).mean()),
            "forced_disclosure": float(((df.n_forced_occ + df.n_forced_part) / 16).mean()),
            "n_someone": m["n_someone"], "n_named": m["n_named"],
            "recovery_gap": m["difference"], "recovery_gap_lo": m["ci_lo"],
            "recovery_gap_hi": m["ci_hi"], "recovery_gap_p": m["p_permutation"],
            "abstention": a["overall_abstention_rate"],
            "abstention_someone": a["abstention_someone"],
            "abstention_named": a["abstention_named"],
            "abstention_lean": a["difference"],
            "abstention_lean_lo": a["ci_lo"], "abstention_lean_hi": a["ci_hi"],
            "abstention_lean_p": a["p_permutation"],
            "abstention_lean_p_holm": a["p_holm"],
        })

    summary = {
        "n_items": 480, "n_draws": 16, "n_conditions": 3,
        "generations_per_checkpoint": 480 * 16 * 3,
        "generations_total": 6 * 480 * 16 * 3,
        "null_reps": reps,
        "pooled_naive_observed": tot_obs, "pooled_null_mean": tot_null,
        "pooled_noise_share": tot_null / tot_obs,
        "total_violations": sum(r["corrected_incompatible"] for r in rows),
        "total_instrument": sum(r["n_instrument"] for r in rows),
        "total_policy": sum(r["n_policy"] for r in rows),
        "total_other": sum(r["n_other"] for r in rows),
        "n_checkpoints_zero_violations": sum(r["corrected_incompatible"] == 0 for r in rows),
        "n_abstention_lean_significant_holm": abst["n_significant_holm"],
        "n_abstaining_checkpoints": abst["n_testable"],
        "rows": rows,
    }
    assert summary["total_violations"] == (summary["total_instrument"]
        + summary["total_policy"] + summary["total_other"]), "taxonomy does not partition"
    OUT.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"{'model':16s} {'naive':>6s} {'null':>7s} {'noise%':>7s} {'corr':>5s} "
          f"{'inst':>5s} {'pol':>4s} {'oth':>4s} {'abst%':>6s} {'lean pp':>8s} {'gap D':>7s}")
    for r in rows:
        print(f"{r['model']:16s} {r['naive_incompatible']:6d} {r['null_mean']:7.1f} "
              f"{r['noise_share']*100:6.1f}% {r['corrected_incompatible']:5d} "
              f"{r['n_instrument']:5d} {r['n_policy']:4d} {r['n_other']:4d} "
              f"{r['abstention']*100:5.1f}% {r['abstention_lean']*100:+8.2f} "
              f"{r['recovery_gap']:+7.3f}")
    print(f"\npooled: obs={tot_obs:.0f} null={tot_null:.1f} "
          f"noise share={summary['pooled_noise_share']*100:.1f}%")
    print(f"violations: {summary['total_violations']} = "
          f"{summary['total_instrument']} instrument + {summary['total_policy']} policy "
          f"+ {summary['total_other']} other")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
