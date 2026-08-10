"""Unit tests for lib/utils/doe_space_sampling.py.

Only `get_lhs_sampling` runs during a normal fit, so the other samplers were
entirely unexercised. The seeding behaviour matters most: scipy.stats.qmc uses
its own Generator rather than numpy's global state, so a regression that drops
the `seed=` argument would silently destroy the reproducibility guarantee that
RANDOM_SEED is supposed to provide, without failing anything else.
"""

import numpy as np
import pytest

from lib.utils.doe_space_sampling import (
    get_lhs_sampling,
    get_sobol_sampling,
    get_spacefilled_DoE,
    create_doe_sampling_with_corners,
    l2_dist,
    pairwise_dist,
)

RANGES = np.array([[0.0, 1.0], [-2.0, 2.0], [10.0, 20.0]])


def within_ranges(samples: np.ndarray, ranges: np.ndarray) -> bool:
    return bool(
        np.all(samples >= ranges[:, 0] - 1e-12) and np.all(samples <= ranges[:, 1] + 1e-12)
    )


# ---------------------------------------------------------------------------
# LHS — the sampler the framework actually uses
# ---------------------------------------------------------------------------

def test_lhs_shape_and_bounds():
    s = get_lhs_sampling(16, RANGES, seed=0)
    assert s.shape == (16, 3)
    assert within_ranges(s, RANGES)


def test_lhs_is_reproducible_with_a_seed():
    """This is the guarantee RANDOM_SEED depends on."""
    a = get_lhs_sampling(20, RANGES, seed=123)
    b = get_lhs_sampling(20, RANGES, seed=123)
    np.testing.assert_array_equal(a, b)


def test_lhs_differs_across_seeds():
    a = get_lhs_sampling(20, RANGES, seed=1)
    b = get_lhs_sampling(20, RANGES, seed=2)
    assert not np.allclose(a, b)


def test_lhs_ignores_numpy_global_seed():
    """scipy.stats.qmc has its own Generator: np.random.seed must NOT control it.

    If this test ever starts failing, the sampler has been changed to draw from
    numpy's global state, and the explicit `seed=` plumbing can be simplified.
    Until then, seeding numpy alone is not enough for a reproducible fit.
    """
    np.random.seed(42)
    a = get_lhs_sampling(12, RANGES)
    np.random.seed(42)
    b = get_lhs_sampling(12, RANGES)
    assert not np.allclose(a, b)


def test_lhs_stratification_covers_every_stratum():
    """One sample per stratum per axis is the defining property of an LHS."""
    n = 25
    s = get_lhs_sampling(n, RANGES, seed=7)
    for axis in range(RANGES.shape[0]):
        lo, hi = RANGES[axis]
        strata = np.floor((s[:, axis] - lo) / (hi - lo) * n).astype(int)
        strata = np.clip(strata, 0, n - 1)
        assert len(set(strata.tolist())) == n, f"axis {axis} does not cover all strata"


@pytest.mark.parametrize("optimization", ["random-cd", None])
def test_lhs_accepts_optimization_modes(optimization):
    s = get_lhs_sampling(8, RANGES, optimization=optimization, seed=3)
    assert s.shape == (8, 3)
    assert within_ranges(s, RANGES)


# ---------------------------------------------------------------------------
# Sobol — available but not currently wired into the optimizers
# ---------------------------------------------------------------------------

def test_sobol_shape_bounds_and_seeding():
    a = get_sobol_sampling(16, RANGES, seed=5)
    b = get_sobol_sampling(16, RANGES, seed=5)
    assert a.shape == (16, 3)
    assert within_ranges(a, RANGES)
    np.testing.assert_array_equal(a, b)


def test_sobol_is_more_uniform_than_random():
    """Low-discrepancy sequences should beat uniform random on min-pair spacing."""
    unit = np.array([[0.0, 1.0]] * 3)
    sobol = get_sobol_sampling(64, unit, seed=0)
    rng = np.random.default_rng(0)
    random = rng.uniform(size=(64, 3))

    def min_spacing(points):
        d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        return d.min()

    assert min_spacing(sobol) > min_spacing(random)


# ---------------------------------------------------------------------------
# space-filling DoE + distance helpers
# ---------------------------------------------------------------------------

def test_spacefilled_doe_returns_points_in_bounds_with_finite_criterion():
    """NOTE: despite its docstring, get_spacefilled_DoE does NOT include corners.

    The corner-based sampler is commented out at doe_space_sampling.py:61 in
    favour of create_doe_sampling_random_points, so this asserts the real
    behaviour: n_samples uniform points plus a finite Morris-Mitchell phi_p.
    """
    ranges = np.array([[0.0, 1.0], [0.0, 1.0]])
    sampling, phi_p = get_spacefilled_DoE(8, ranges, q=2)

    sampling = np.asarray(sampling)
    assert sampling.shape == (8, 2)
    assert within_ranges(sampling, ranges)
    assert np.isfinite(phi_p) and phi_p > 0


def test_corner_sampler_includes_every_corner_and_the_centre():
    """create_doe_sampling_with_corners is the variant that guarantees corners."""
    ranges = np.array([[0.0, 1.0], [0.0, 1.0]])
    sampling = np.asarray(create_doe_sampling_with_corners(8, ranges))

    assert sampling.shape == (8, 2)
    present = {tuple(np.round(p, 12)) for p in sampling}
    corners = {(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)}
    assert corners <= present
    assert (0.5, 0.5) in present, "centre point must be included"


def test_corner_sampler_requires_enough_points_for_the_corners():
    """With n axes it must place 2**n corners + 1 centre, so fewer samples is an error."""
    ranges = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        create_doe_sampling_with_corners(4, ranges)   # needs at least 2**3 + 1 = 9


def test_l2_dist_matches_numpy():
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 2.0, 2.0])
    assert float(l2_dist(a, b)) == pytest.approx(3.0)


def test_pairwise_dist_returns_all_unique_pairs():
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    ranges = np.array([[0.0, 1.0], [0.0, 1.0]])
    d = np.asarray(pairwise_dist(pts, ranges))
    assert d.shape == (6,)          # C(4,2)
    assert np.all(d > 0)
