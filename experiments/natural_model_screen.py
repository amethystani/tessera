"""Run the compatibility certificate on real model output.

For each item i and each condition z (refusal-ablated forced choice, direct,
forced) we draw N stochastic completions at temperature > 0. Greedy decoding
gives one fixed answer per item, which can't estimate the response distribution
that compatible() and necessary_dependence() are defined over. Draws are scored
with validation_experiment.parse_response, the same parser used for single
greedy decodes. That gives a per-item, per-condition distribution over
{occupation, participant}, where unobserved mass is the share of draws that
abstained or didn't parse.

For each item it runs check_theory.compatible() and necessary_dependence() on the
resulting 3 x K matrix and reports:
  - the fraction of items whose three conditions share one answer law
  - the distribution and mean of the necessary-dependence certificate gamma*
  - a cross-check against the single-greedy-decode recovery error on the same items

The indirect/crosswise condition is left out. In the greedy screen that came
before this one it failed on task competence and wasn't measuring anything.

Usage:
  python3 natural_model_screen.py --model Qwen/Qwen2.5-7B-Instruct --limit 30 --draws 8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from validation_experiment import (  # noqa: E402
    FORCED, DIRECT, query, parse_response, find_refusal_direction,
    find_behavioral_refusal_direction, ablate_direction,
)
from metrics import stereotype_answer  # noqa: E402
from check_theory import compatible, necessary_dependence, simultaneous_lower_bounds  # noqa: E402

CATEGORIES = ("occupation", "participant")  # column order for the certificate matrices


def item_counts(model, tokenizer, template: str, sentence: str, occupation: str,
                participant: str, n_draws: int, seed_base: int) -> np.ndarray:
    """
    Draw n_draws stochastic completions for one item under one condition and
    return raw counts [n(occupation), n(participant)], not proportions.

    simultaneous_lower_bounds needs raw counts. Plugging proportions straight
    into compatible() or necessary_dependence() gives false rejections, which
    the null simulation in calibration_ledger.py shows (naive check rejected in
    all 1,000 null repetitions, the simultaneous-bound check in none). With few
    draws per item this is the usual outcome.
    """
    counts = {"occupation": 0, "participant": 0}
    for d in range(n_draws):
        resp = query(model, tokenizer, template.format(sentence=sentence, a=occupation, b=participant),
                    temperature=0.8, seed=seed_base + d)
        _, pred = parse_response(resp, occupation, participant)
        if pred == occupation:
            counts["occupation"] += 1
        elif pred == participant:
            counts["participant"] += 1
    return np.array([counts["occupation"], counts["participant"]], dtype=int)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--items", default="../data/winogender/all_sentences.tsv")
    ap.add_argument("--stats", default="../data/winogender/occupations-stats.tsv")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--out", default="natural_screen_results.json")
    ap.add_argument("--direction", choices=["generic", "behavioral"],
                    default="generic",
                    help="refusal direction: authored probe prompts (generic) "
                         "or the model's own abstention behaviour on these "
                         "items (behavioral)")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    sent = pd.read_csv(a.items, sep="\t")
    stats = pd.read_csv(a.stats, sep="\t")
    parts = sent.sentid.str.replace(".txt", "", regex=False).str.split(".")
    sent["occupation"], sent["participant"] = parts.str[0], parts.str[1]
    sent["gender"] = parts.str[3]
    sent = sent[sent.gender != "neutral"].merge(stats, on="occupation")
    sent = sent.sample(n=min(a.limit, len(sent)), random_state=42).reset_index(drop=True)
    sent["stereotype_answer"] = [
        stereotype_answer(o, p, b, g) for o, p, b, g
        in zip(sent.occupation, sent.participant, sent.bergsma_pct_female, sent.gender)]
    print(f"{len(sent)} items x {a.draws} draws x 3 conditions = "
          f"{len(sent) * a.draws * 3} generate() calls")

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None)
    model.eval()

    # Ground truth: ablate the refusal direction, sample FORCED under it.
    print("\n[1/3] Ground-truth condition (ablated, forced) ...")
    direction_meta = {"source": a.direction}
    if a.direction == "behavioral":
        d, n_abs, n_ans = find_behavioral_refusal_direction(
            model, tok, sent, DIRECT)
        direction_meta.update(n_abstained=n_abs, n_answered=n_ans)
        print(f"  behavioural direction from {n_abs} abstained / "
              f"{n_ans} answered items")
        if d is None:
            raise SystemExit(
                f"cannot build behavioural direction: {n_abs} abstained, "
                f"{n_ans} answered (need >=8 each). This model does not "
                f"supply the contrast on these items.")
    else:
        d = find_refusal_direction(model, tok)
    handles = ablate_direction(model, d)
    truth_counts = []
    for i, r in enumerate(sent.itertuples()):
        truth_counts.append(item_counts(model, tok, FORCED, r.sentence, r.occupation,
                                        r.participant, a.draws, seed_base=1000 + i * 100))
    for h in handles:
        h.remove()

    print("[2/3] Direct condition (intact model) ...")
    direct_counts = []
    for i, r in enumerate(sent.itertuples()):
        direct_counts.append(item_counts(model, tok, DIRECT, r.sentence, r.occupation,
                                         r.participant, a.draws, seed_base=2000 + i * 100))

    print("[3/3] Forced condition (intact model) ...")
    forced_counts = []
    for i, r in enumerate(sent.itertuples()):
        forced_counts.append(item_counts(model, tok, FORCED, r.sentence, r.occupation,
                                         r.participant, a.draws, seed_base=3000 + i * 100))

    # --- Per-item certificates -------------------------------------------------
    # Bonferroni across items: each item's own panel already self-corrects across
    # its (conditions x categories) cells inside simultaneous_lower_bounds; alpha
    # is additionally divided by n_items here to hold a family-wise rate of 0.05
    # across the whole screen, per that function's own docstring instruction.
    print("\nComputing per-item compatibility certificates "
          "(naive plug-in AND Bonferroni-corrected) ...")
    n_items = len(sent)
    alpha_per_item = 0.05 / n_items
    per_item = []
    for i, r in enumerate(sent.itertuples()):
        counts = np.stack([truth_counts[i], direct_counts[i], forced_counts[i]])  # 3 x 2, ints
        totals = np.array([a.draws, a.draws, a.draws])
        panel_naive = counts / a.draws  # plug-in proportions - upward-biased, do not trust alone

        naive_compatible = compatible(panel_naive)
        naive_gamma, _, _ = necessary_dependence(panel_naive)

        lower = simultaneous_lower_bounds(counts, totals, alpha=alpha_per_item)
        corrected_compatible = compatible(lower)
        corrected_gamma, _, _ = necessary_dependence(lower)

        truth_dist, direct_dist, forced_dist = panel_naive[0], panel_naive[1], panel_naive[2]
        truth_aligned = truth_dist[0] if r.occupation == r.stereotype_answer else truth_dist[1]
        direct_aligned = direct_dist[0] if r.occupation == r.stereotype_answer else direct_dist[1]
        forced_aligned = forced_dist[0] if r.occupation == r.stereotype_answer else forced_dist[1]
        per_item.append({
            "sentid": r.sentid, "sentence": r.sentence,
            "n_truth_occ": int(truth_counts[i][0]), "n_truth_part": int(truth_counts[i][1]),
            "n_direct_occ": int(direct_counts[i][0]), "n_direct_part": int(direct_counts[i][1]),
            "n_forced_occ": int(forced_counts[i][0]), "n_forced_part": int(forced_counts[i][1]),
            "naive_compatible": naive_compatible, "naive_gamma_star": naive_gamma,
            "corrected_compatible": corrected_compatible, "corrected_gamma_star": corrected_gamma,
            "truth_aligned": truth_aligned, "direct_aligned": direct_aligned,
            "forced_aligned": forced_aligned,
        })

    df = pd.DataFrame(per_item)
    df.to_csv(a.out.replace(".json", "_per_item.csv"), index=False)

    n_naive = int(df.naive_compatible.sum())
    n_corrected = int(df.corrected_compatible.sum())
    print(f"\n{'=' * 66}")
    print(f"Compatible with one common law - naive plug-in:      "
          f"{n_naive}/{len(df)} ({n_naive/len(df):.1%})")
    print(f"Compatible with one common law - Bonferroni-corrected: "
          f"{n_corrected}/{len(df)} ({n_corrected/len(df):.1%})  "
          f"(alpha={0.05/len(df):.5f} per item)")
    if n_corrected > n_naive:
        print(f"  -> correction flipped {n_corrected - n_naive} item(s) from "
              f"'incompatible' to 'compatible': exactly the false-positive "
              f"pattern the null simulation predicted at this draw count.")
    print(f"\nMean itemwise gamma* - naive: {df.naive_gamma_star.mean():.4f}  |  "
          f"corrected: {df.corrected_gamma_star.mean():.4f}")
    print(f"Items with corrected gamma*=0 exactly: "
          f"{(df.corrected_gamma_star < 1e-9).sum()}/{len(df)}")

    direct_err = float((df.direct_aligned - df.truth_aligned).abs().mean())
    forced_err = float((df.forced_aligned - df.truth_aligned).abs().mean())
    print(f"\nMean |direct_aligned - truth_aligned| across items: {direct_err:.4f}")
    print(f"Mean |forced_aligned - truth_aligned| across items: {forced_err:.4f}")

    summary = {
        "model": a.model, "n_items": len(df), "n_draws": a.draws,
        "n_naive_compatible": n_naive, "frac_naive_compatible": n_naive / len(df),
        "n_corrected_compatible": n_corrected, "frac_corrected_compatible": n_corrected / len(df),
        "mean_naive_gamma_star": float(df.naive_gamma_star.mean()),
        "mean_corrected_gamma_star": float(df.corrected_gamma_star.mean()),
        "median_corrected_gamma_star": float(df.corrected_gamma_star.median()),
        "n_corrected_gamma_star_zero": int((df.corrected_gamma_star < 1e-9).sum()),
        "mean_direct_recovery_error": direct_err,
        "mean_forced_recovery_error": forced_err,
        "direction": direction_meta,
    }
    Path(a.out).write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {a.out} and {a.out.replace('.json', '_per_item.csv')}")


if __name__ == "__main__":
    main()
