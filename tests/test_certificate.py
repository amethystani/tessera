import numpy as np
import pytest

from check_theory import (compatible, explicit_joint_reference,
                          necessary_dependence, simultaneous_lower_bounds)


def test_identical_rows_are_compatible():
    p = np.array([[0.6, 0.3], [0.6, 0.3], [0.6, 0.3]])
    assert compatible(p)
    assert necessary_dependence(p)[0] == pytest.approx(0, abs=1e-9)


def test_opposed_full_disclosure_is_incompatible():
    p = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert not compatible(p)
    assert necessary_dependence(p)[0] == pytest.approx(1.0)


def test_disclosure_can_mask_a_difference():
    # one condition shows only 0.4 of the mass on A, the other only 0.4 on B.
    # Hidden mass can always be assigned so that both share one law.
    p = np.array([[0.4, 0.0], [0.0, 0.4]])
    assert compatible(p)


def test_two_category_closed_form():
    # for K=2, gamma* = max(0, sum_k max_z p[z,k] - 1)
    rng = np.random.default_rng(1)
    for _ in range(50):
        a = rng.uniform(0, 1, 3)
        b = rng.uniform(0, 1, 3) * (1 - a)
        p = np.column_stack([a, b])
        expected = max(0.0, p.max(axis=0).sum() - 1)
        assert necessary_dependence(p)[0] == pytest.approx(expected, abs=1e-7)


def test_lp_matches_explicit_joint_formulation():
    rng = np.random.default_rng(7)
    for _ in range(25):
        z, k = rng.integers(2, 4), rng.integers(2, 4)
        raw = rng.dirichlet(np.ones(k + 1), size=z)[:, :k]
        assert necessary_dependence(raw)[0] == pytest.approx(
            explicit_joint_reference(raw), abs=1e-6)


def test_compatible_iff_zero_dependence():
    rng = np.random.default_rng(3)
    for _ in range(50):
        p = rng.dirichlet(np.ones(4), size=3)[:, :3] * rng.uniform(0.3, 1, (3, 1))
        zero = necessary_dependence(p)[0] < 1e-8
        assert compatible(p) == zero


def test_completed_laws_dominate_the_observed_masses():
    p = np.array([[0.5, 0.2], [0.1, 0.7], [0.3, 0.3]])
    _, _, q = necessary_dependence(p)
    assert np.all(q >= p - 1e-8)
    assert np.allclose(q.sum(axis=1), 1)


@pytest.mark.parametrize("bad", [
    np.array([[0.7, 0.5]]),        # row sums above one
    np.array([[-0.1, 0.5]]),       # negative
    np.array([0.2, 0.3]),          # not two-dimensional
    np.array([[np.nan, 0.1]]),     # not finite
])
def test_invalid_panels_are_rejected(bad):
    with pytest.raises(ValueError):
        compatible(bad)


def test_lower_bounds_sit_below_the_empirical_share():
    counts = np.array([[12, 3], [8, 7]])
    totals = np.array([16, 16])
    lo = simultaneous_lower_bounds(counts, totals)
    assert np.all(lo <= counts / totals)
    assert np.all(lo >= 0)


def test_lower_bound_is_zero_when_nothing_was_seen():
    lo = simultaneous_lower_bounds(np.array([[0, 5]]), np.array([16]))
    assert lo[0, 0] == 0
    assert lo[0, 1] > 0


def test_more_draws_tighten_the_bound():
    small = simultaneous_lower_bounds(np.array([[8, 8]]), np.array([16]))
    large = simultaneous_lower_bounds(np.array([[512, 512]]), np.array([1024]))
    assert large[0, 0] > small[0, 0]
    assert large[0, 0] < 0.5


def test_smaller_alpha_gives_a_looser_bound():
    c, t = np.array([[10, 6]]), np.array([16])
    assert (simultaneous_lower_bounds(c, t, alpha=0.01)
            <= simultaneous_lower_bounds(c, t, alpha=0.2)).all()


def test_bound_inputs_are_checked():
    with pytest.raises(ValueError):
        simultaneous_lower_bounds(np.array([[10, 10]]), np.array([16]))   # 20 > 16
    with pytest.raises(ValueError):
        simultaneous_lower_bounds(np.array([[1, 1]]), np.array([16]), alpha=1.5)
