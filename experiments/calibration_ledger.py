"""Reference regimes with known ground truth for checking the certificate.

No model is involved. A ledger is a fixed array of true categorical answers, and
an instrument is a disclosure or rewriting rule applied to it. Three regimes,
each simulated at a chosen sample size:

  PURE_SUPPRESSION      instruments reveal or hide the true answer, and
                        disclosure may depend on the category. A common law
                        exists, so the audit should not reject.
  INSTRUMENT_REWRITING  instrument z changes some categories under a known
                        transition rule and rewrite rate. The audit should
                        reject once the rewriting is large enough to show up in
                        the marginals. We report a detection curve over rewrite
                        rate.
  SHARED_REWRITING      every instrument applies the same wrong transform.
                        The audit cannot reject (gamma* = 0) even though all
                        instruments are far from the ledger. Only a comparison
                        against the held-out ledger would show it.

Usage: python3 calibration_ledger.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from check_theory import compatible, necessary_dependence, simultaneous_lower_bounds

RNG = np.random.default_rng(0)
CATEGORIES = ("A", "B")  # binary ledger for a clean first pass; extend later
N_LEDGER = 4000
N_TRIALS_PER_INSTRUMENT = 500  # completions drawn per instrument, per replicate
N_REPLICATES = 200
ALPHA = 0.05


@dataclass
class RegimeResult:
    regime: str
    param: float                 # the varied knob (rewrite rate, etc.)
    false_rejection_rate: float  # P(audit rejects compatibility) under this regime
    mean_gamma_star: float
    detected_rewriting_rate: float | None  # audit's gamma* as an estimate of true rewrite mass
    true_tv_from_ledger: float   # instruments' true distance from the ledger (not seen by audit)


def make_ledger(p_a: float = 0.5, n: int = N_LEDGER) -> np.ndarray:
    """True answers, frozen before any disclosure. 0 = A, 1 = B."""
    return (RNG.random(n) >= p_a).astype(int)


def sample_instrument(ledger: np.ndarray, disclose_prob: dict, n_draw: int) -> np.ndarray:
    """
    Draw n_draw ledger entries (with replacement, i.i.d.), apply a
    category-dependent disclosure probability, return observed category codes
    with -1 for "not disclosed" (conservatively unobserved).
    disclose_prob: {0: P(reveal | true=A), 1: P(reveal | true=B)}
    """
    idx = RNG.integers(0, len(ledger), size=n_draw)
    truth = ledger[idx]
    revealed = RNG.random(n_draw) < np.array([disclose_prob[t] for t in truth])
    out = np.where(revealed, truth, -1)
    return out


def counts_from_observations(obs_list: list[np.ndarray], n_cat: int = 2):
    counts = np.zeros((len(obs_list), n_cat), dtype=int)
    totals = np.zeros(len(obs_list), dtype=int)
    for z, obs in enumerate(obs_list):
        totals[z] = len(obs)
        for k in range(n_cat):
            counts[z, k] = (obs == k).sum()
    return counts, totals


def run_replicate_pure_suppression(ledger: np.ndarray, n_instruments: int = 3) -> tuple[bool, float]:
    """Each instrument has its own (fixed, arbitrary, category-dependent)
    disclosure rates but the SAME underlying ledger. A common law genuinely
    exists, so the audit should not reject except at the chosen false-rejection
    rate."""
    obs = []
    for _ in range(n_instruments):
        # Arbitrary, instrument-specific, category-dependent disclosure --
        # allowed under the pure-suppression null, still one common law.
        dz = {0: RNG.uniform(0.2, 0.9), 1: RNG.uniform(0.2, 0.9)}
        obs.append(sample_instrument(ledger, dz, N_TRIALS_PER_INSTRUMENT))
    counts, totals = counts_from_observations(obs)
    lower = simultaneous_lower_bounds(counts, totals, alpha=ALPHA)
    rejected = not compatible(lower)
    gamma_hat = necessary_dependence(counts / totals[:, None])[0]
    return rejected, gamma_hat


def run_replicate_instrument_rewriting(ledger: np.ndarray, bias_spread: float,
                                       n_instruments: int = 3) -> tuple[bool, float, float]:
    """
    Each instrument has its own asymmetric transition matrix - a
    category-directed bias, not a symmetric flip. A symmetric flip at rate r
    applied to a balanced (50/50) ledger leaves the marginal at exactly 0.5
    regardless of r (it is the channel's fixed point), so heterogeneous FLIP
    RATES alone cannot separate instruments' marginals from each other in
    that case. Real broken instruments are not symmetric noise sources; they
    are directionally biased (e.g. a stereotype-consistent default), so this
    is also the more realistic model.

    Instrument z has P(shown=A | true=A) = 1 - e_A(z) and
    P(shown=A | true=B) = e_B(z), where both error rates are shifted toward A
    as `bias_spread` grows, spread evenly across instruments so heterogeneity
    - not the ledger's own base rate - is what the audit must detect.
    """
    biases = np.linspace(-bias_spread, bias_spread, n_instruments)
    obs = []
    true_tvs = []
    p_ledger = np.array([(ledger == 0).mean(), (ledger == 1).mean()])
    for bias in biases:
        idx = RNG.integers(0, len(ledger), size=N_TRIALS_PER_INSTRUMENT)
        truth = ledger[idx]
        # bias > 0 => this instrument over-reports A; bias < 0 => over-reports B.
        p_shown_A_given_A = np.clip(0.9 + bias, 0.05, 0.95)
        p_shown_A_given_B = np.clip(0.1 + bias, 0.05, 0.95)
        u = RNG.random(N_TRIALS_PER_INSTRUMENT)
        shown_truth = np.where(truth == 0, (u < p_shown_A_given_A).astype(int) ^ 1,
                                            (u < p_shown_A_given_B).astype(int) ^ 1)
        # shown_truth: 0=A, 1=B. Above: if true=A, show A w.p. p_shown_A_given_A.
        reveal = RNG.random(N_TRIALS_PER_INSTRUMENT) < 0.85
        obs.append(np.where(reveal, shown_truth, -1))
        disc = shown_truth[reveal]
        p_instr = np.array([(disc == 0).mean(), (disc == 1).mean()]) if len(disc) else np.array([0.5, 0.5])
        true_tvs.append(0.5 * np.abs(p_ledger - p_instr).sum())
    counts, totals = counts_from_observations(obs)
    lower = simultaneous_lower_bounds(counts, totals, alpha=ALPHA)
    rejected = not compatible(lower)
    gamma_hat = necessary_dependence(counts / totals[:, None])[0]
    return rejected, gamma_hat, float(np.mean(true_tvs))


def run_replicate_shared_rewriting(ledger: np.ndarray, n_instruments: int = 3) -> tuple[bool, float, float]:
    """
    Every instrument applies the SAME wrong transform: always report "A"
    regardless of the true ledger answer. Perfect cross-instrument agreement,
    gamma* == 0 - but every instrument is far from the ledger in TV. This is
    the blind spot: an audit using only cross-instrument comparison cannot
    catch it.
    """
    p_a = (ledger == 0).mean()
    obs = [np.zeros(N_TRIALS_PER_INSTRUMENT, dtype=int) for _ in range(n_instruments)]  # always "A"
    counts, totals = counts_from_observations(obs)
    lower = simultaneous_lower_bounds(counts, totals, alpha=ALPHA)
    rejected = not compatible(lower)
    gamma_hat = necessary_dependence(counts / totals[:, None])[0]
    true_tv = 0.5 * (abs(1 - p_a) + abs(0 - (1 - p_a)))  # TV(always-A, true law)
    return rejected, gamma_hat, float(true_tv)


def main() -> None:
    results: list[RegimeResult] = []

    # --- Pure suppression: false-rejection rate at the chosen alpha ---------
    ledger = make_ledger(0.5)
    outs = [run_replicate_pure_suppression(ledger) for _ in range(N_REPLICATES)]
    rej = [o[0] for o in outs]; gam = [o[1] for o in outs]
    results.append(RegimeResult(
        regime="pure_suppression", param=float("nan"),
        false_rejection_rate=float(np.mean(rej)), mean_gamma_star=float(np.mean(gam)),
        detected_rewriting_rate=None, true_tv_from_ledger=0.0))

    # --- Instrument-dependent rewriting: detection curve over rewrite rate --
    for spread in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
        outs = [run_replicate_instrument_rewriting(ledger, spread) for _ in range(N_REPLICATES)]
        rej = [o[0] for o in outs]; gam = [o[1] for o in outs]; tv = [o[2] for o in outs]
        results.append(RegimeResult(
            regime="instrument_rewriting", param=spread,
            false_rejection_rate=float(np.mean(rej)), mean_gamma_star=float(np.mean(gam)),
            detected_rewriting_rate=float(np.mean(gam)), true_tv_from_ledger=float(np.mean(tv))))

    # --- Shared rewriting: the blind spot ------------------------------------
    outs = [run_replicate_shared_rewriting(ledger) for _ in range(N_REPLICATES)]
    rej = [o[0] for o in outs]; gam = [o[1] for o in outs]; tv = [o[2] for o in outs]
    results.append(RegimeResult(
        regime="shared_rewriting", param=float("nan"),
        false_rejection_rate=float(np.mean(rej)), mean_gamma_star=float(np.mean(gam)),
        detected_rewriting_rate=float(np.mean(gam)), true_tv_from_ledger=float(np.mean(tv))))

    print(f"{'regime':<22}{'param':>8}{'false_rej':>12}{'mean_gamma*':>14}{'true_TV':>10}")
    for r in results:
        p = f"{r.param:.2f}" if r.param == r.param else "-"
        print(f"{r.regime:<22}{p:>8}{r.false_rejection_rate:12.3f}{r.mean_gamma_star:14.4f}"
              f"{r.true_tv_from_ledger:10.3f}")

    out = Path(__file__).with_name("calibration_results.json")
    out.write_text(json.dumps([asdict(r) for r in results], indent=2))
    print(f"\nWrote {out}")

    print("\n--- Read this before trusting the audit on any real model ---")
    pure = results[0]
    print(f"Pure suppression false-rejection rate: {pure.false_rejection_rate:.3f} "
          f"(target ~{ALPHA:.2f} or below; Bonferroni is conservative, so below is expected)")
    shared = results[-1]
    print(f"Shared rewriting: mean gamma* = {shared.mean_gamma_star:.4f} "
          f"(near 0, as predicted) while true TV from ledger = {shared.true_tv_from_ledger:.3f}.")
    print("The audit cannot detect this regime by construction. That is a property of")
    print("the method and not a bug. Cross-instrument compatibility can refute")
    print("recovery but never certify it.")


if __name__ == "__main__":
    main()
