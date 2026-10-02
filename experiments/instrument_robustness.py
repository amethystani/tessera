"""Does the Qwen2.5 recovery-gap result depend on the refusal-ablation instrument?

The corrected certificate flags instrument failures on two checkpoints, so we
re-estimate the gap D two ways:

  (a) swap in the behaviour-derived direction, which has no instrument failures
      on Qwen2.5 (violation_taxonomy_n480_dir_behavioral)
  (b) keep the generic direction but drop every item the corrected certificate
      rejects for that checkpoint

Output: results_instrument_robustness.json
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from mechanism_inference import prepare, bootstrap_diff, permutation_p

LAB = Path(__file__).parent / "lab_results"
OUT = Path(__file__).parent / "results_instrument_robustness.json"
REPS = 10000


def gap(df, rng):
    s = df[df.someone].direct_err.to_numpy(float)
    n = df[~df.someone].direct_err.to_numpy(float)
    boots = bootstrap_diff(rng, s, n, REPS)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"D": float(s.mean() - n.mean()), "ci": [float(lo), float(hi)],
            "p_permutation": float(permutation_p(rng, s, n, REPS)),
            "n_someone": int(len(s)), "n_named": int(len(n))}


def flagged(tax_file, tag):
    for t in json.loads((LAB / tax_file).read_text()):
        if f"screen_{tag}_" in t["file"]:
            return {it["sentid"] for it in t.get("items", [])}
    return set()


def main():
    rng = np.random.default_rng(20260925)
    out = {}
    for tag in ("qwen", "mistral"):
        generic = prepare(str(LAB / f"screen_{tag}_n480_d16_per_item.csv"))
        behav = prepare(str(LAB / f"screen_{tag}_n480_dir_behavioral_per_item.csv"))
        drop = flagged("violation_taxonomy_n480.json", tag)
        out[tag] = {
            "generic": gap(generic, rng),
            "behavioral": gap(behav, rng),
            "behavioral_instrument_failures": sum(
                1 for s in flagged("violation_taxonomy_n480_dir_behavioral.json", tag)),
            "generic_minus_flagged": gap(generic[~generic.sentid.isin(drop)], rng),
            "n_flagged_dropped": len(drop),
        }
        for k in ("generic", "behavioral", "generic_minus_flagged"):
            r = out[tag][k]
            print(f"{tag:8s} {k:22s} D={r['D']:+.3f} [{r['ci'][0]:+.3f},{r['ci'][1]:+.3f}] p={r['p_permutation']:.4f}")
    OUT.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
