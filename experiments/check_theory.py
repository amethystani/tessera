"""Compatibility certificate and its checks on synthetic data.

Rows are interventions and columns are substantive answer categories. Entries
are unconditional response probabilities, and any omitted row mass counts as
unobserved. This file does not decide which responses count as substantive.
No language model is used.

Run: python3 check_theory.py   (needs numpy and scipy)
Writes synthetic_checks.json next to this file.
"""
from __future__ import annotations

from itertools import combinations, product
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.stats import beta


def validate(p):
    """Return the panel as a float array, raising ValueError unless it is a nonempty non-negative matrix with rows summing to at most one."""
    p = np.asarray(p, dtype=float)
    if p.ndim != 2 or min(p.shape) < 1:
        raise ValueError("Expected a nonempty intervention-by-category matrix")
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Probabilities must be finite and nonnegative")
    if (p.sum(axis=1) > 1 + 1e-10).any():
        raise ValueError("Each row must sum to at most one")
    return p


def compatible(p):
    """Existence of one answer law and arbitrary answer-dependent disclosure."""
    p = validate(p)
    return bool(p.max(axis=0).sum() <= 1 + 1e-10)


def necessary_dependence(p):
    """Sharp minimum P(not all Y_z equal), over all completions and couplings.

    Maximize common mass sum(t), with max(t, p_z) summing to at most 1
    for every z. Auxiliary v variables linearize (p_z - t)_+.
    This is a lower bound on necessary dependence, not an estimate of actual
    individual answer changes or an upper bound on measurement error.
    """
    p = validate(p)
    z_count, k_count = p.shape
    n_vars = k_count + z_count * k_count
    c = np.zeros(n_vars)
    c[:k_count] = -1
    rows, rhs = [], []
    for z in range(z_count):
        row = np.zeros(n_vars)
        row[:k_count] = 1
        row[k_count + z * k_count:k_count + (z + 1) * k_count] = 1
        rows.append(row)
        rhs.append(1.0)
        for k in range(k_count):
            row = np.zeros(n_vars)
            row[k] = -1
            row[k_count + z * k_count + k] = -1
            rows.append(row)
            rhs.append(-p[z, k])
    result = linprog(c, A_ub=rows, b_ub=rhs, bounds=(0, None), method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    t = result.x[:k_count]
    # Construct feasible full answer laws, for independent inspection.
    q = np.maximum(p, t)
    q[:, 0] += 1 - q.sum(axis=1)
    assert np.all(q >= p - 1e-8)
    assert np.allclose(q.sum(axis=1), 1)
    value = float(np.clip(1 + result.fun, 0, 1))
    assert np.isclose(value, 1 - q.min(axis=0).sum(), atol=1e-8)
    return value, t, q


def explicit_joint_reference(p):
    """Independent exponential-size formulation, for small verification only."""
    p = validate(p)
    z_count, k_count = p.shape
    atoms = np.array(list(product(range(k_count), repeat=z_count)))
    cost = (~np.all(atoms == atoms[:, :1], axis=1)).astype(float)
    rows, rhs = [], []
    for z in range(z_count):
        for k in range(k_count):
            rows.append(-(atoms[:, z] == k).astype(float))
            rhs.append(-p[z, k])
    result = linprog(cost, A_ub=rows, b_ub=rhs,
                     A_eq=np.ones((1, len(atoms))), b_eq=[1],
                     bounds=(0, None), method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    return float(result.fun)


def simultaneous_lower_bounds(counts, totals, alpha=0.05):
    """Bonferroni one-sided exact binomial bounds for one fixed panel.

    totals includes all outputs, including nonanswers. IID trials within each
    condition are required. For multiple panels, supply a divided alpha or
    build a single family; this function alone does not correct across panels.
    """
    counts = np.asarray(counts, dtype=int)
    totals = np.asarray(totals, dtype=int)
    if counts.ndim != 2 or totals.shape != (counts.shape[0],):
        raise ValueError("Invalid count or total shape")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    if (totals <= 0).any() or (counts < 0).any():
        raise ValueError("Counts must be nonnegative and totals positive")
    if (counts.sum(axis=1) > totals).any():
        raise ValueError("Answered counts cannot exceed all-query totals")
    lower = np.zeros(counts.shape)
    tail = alpha / counts.size
    for z, k in np.ndindex(counts.shape):
        if counts[z, k] > 0:
            lower[z, k] = beta.ppf(tail, counts[z, k],
                                   totals[z] - counts[z, k] + 1)
    return lower


def main():
    rng = np.random.default_rng(20260908)
    panels = {
        "binary_selection_only_despite_opposite_answered_majorities":
            [[0.45, 0.05], [0.05, 0.45]],
        "binary_irreducible_difference": [[0.70, 0.10], [0.10, 0.65]],
        "three_way_failure_all_pairs_pass":
            [[0.40, 0, 0], [0, 0.40, 0], [0, 0, 0.40]],
        "fully_observed_identical": [[0.2, 0.3, 0.5]] * 3,
        "fully_observed_disjoint": np.eye(3).tolist(),
        "all_unobserved": np.zeros((3, 3)).tolist(),
        "agreed_rewriting_invisible_without_anchor": [[1, 0], [1, 0]],
    }
    expected = [0, 0.35, 0.10, 0, 1, 0, 0]
    examples = {}
    for (name, p), target in zip(panels.items(), expected):
        value, t, q = necessary_dependence(p)
        assert np.isclose(value, target)
        assert np.isclose(value, explicit_joint_reference(p))
        assert compatible(p) == (value < 1e-8)
        examples[name] = {
            "unconditional_probabilities": p,
            "pure_suppression_compatible": compatible(p),
            "necessary_dependence": round(value, 10),
            "common_mass_by_category": t.tolist(),
            "witness_completed_laws": q.tolist(),
        }
    three_way = np.asarray(panels["three_way_failure_all_pairs_pass"])
    assert all(compatible(three_way[list(pair)])
               for pair in combinations(range(3), 2))

    # Coarsening over items conceals opposite changes.
    per_item = np.asarray([[[.7, .1], [.1, .7]],
                           [[.1, .7], [.7, .1]]])
    item_values = [necessary_dependence(p)[0] for p in per_item]
    pooled_value = necessary_dependence(per_item.mean(axis=0))[0]
    assert np.allclose(item_values, [.4, .4])
    assert np.isclose(pooled_value, 0)

    comparisons = 0
    for z_count, k_count in [(2, 2), (2, 4), (3, 2), (3, 3), (4, 3)]:
        for _ in range(30):
            p = rng.dirichlet(np.ones(k_count + 1), size=z_count)[:, :k_count]
            value, _, _ = necessary_dependence(p)
            assert np.isclose(value, explicit_joint_reference(p), atol=1e-8)
            assert compatible(p) == (value < 1e-8)
            if z_count == 2:
                expected_pair = max(0, p.max(axis=0).sum() - 1)
                assert np.isclose(value, expected_pair)
            if k_count == 2:
                expected_binary = max(0, p[:, 0].max() + p[:, 1].max() - 1)
                assert np.isclose(value, expected_binary)
            # Lower category probabilities can only weaken a certificate.
            lower = p * rng.uniform(.4, 1, size=p.shape)
            assert necessary_dependence(lower)[0] <= value + 1e-8
            comparisons += 1

    # Fixed synthetic counts, not sampled LLM responses.
    synthetic_counts = np.array([[800, 0, 0], [0, 800, 0], [0, 0, 800]])
    lower = simultaneous_lower_bounds(synthetic_counts, np.array([2000] * 3))
    lower_gamma = necessary_dependence(lower)[0]
    assert 0 < lower_gamma < .1
    zero_lower = simultaneous_lower_bounds(np.zeros((2, 2), int), np.array([5, 5]))
    assert np.all(zero_lower == 0)

    # 1,000 draws at a null boundary. Illustrative finite-sample check only;
    # theoretical error control comes from exact marginal bounds + union bound.
    trials, n = 1000, 500
    naive_positives, certified_positives = 0, 0
    for _ in range(trials):
        counts = rng.multinomial(n, [.3, .4, .3], size=3)
        naive_positives += not compatible(counts / n)
        lo = simultaneous_lower_bounds(counts, np.array([n] * 3))
        certified_positives += not compatible(lo)
    assert certified_positives / trials < .05

    output = {
        "status": "SYNTHETIC ONLY; no LLM inference or empirical bias result",
        "seed": 20260908,
        "examples": examples,
        "aggregation_counterexample": {
            "per_item_dependence": item_values, "pooled_dependence": pooled_value},
        "randomized_independent_LP_comparisons_passed": comparisons,
        "synthetic_counts_three_way_95pct_lower_bound": lower_gamma,
        "null_boundary_simulation": {
            "trials": trials, "samples_per_condition": n,
            "naive_positive_count": naive_positives,
            "certified_positive_count": certified_positives},
        "common_mode_blind_spot": {
            "independent_reference_law": [.5, .5],
            "each_agreed_intervention_law": [1, 0],
            "actual_TV_from_reference": .5,
            "unanchored_necessary_dependence": 0},
    }
    path = Path(__file__).with_name("synthetic_checks.json")
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(path),
                      "randomized_comparisons_passed": comparisons,
                      "three_way_necessary_dependence": .1,
                      "three_way_confidence_lower_bound": lower_gamma,
                      "null_boundary_simulation": output["null_boundary_simulation"],
                      "status": output["status"]}, indent=2))


if __name__ == "__main__":
    main()
