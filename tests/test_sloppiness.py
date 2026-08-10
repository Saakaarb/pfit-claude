"""Unit tests for the post-fit sloppiness / identifiability diagnostic.

The diagnostic runs on every fit, so its contract matters twice over:

1. It must NEVER break a fit. It is wrapped in a bare try/except for exactly
   that reason, and these tests hold that guarantee against a loss function that
   raises, returns NaN, or has the wrong shape.
2. Its verdict must be meaningful. A diagnostic that reported "well determined"
   for everything would be worse than useless, so a genuinely sloppy problem
   (one flat direction) is checked to be reported as such.
"""

import numpy as np
import pytest

import jax.numpy as jnp

from lib.utils.sloppiness import MAX_NPAR, run_sloppiness_analysis


def make_constants(**overrides):
    """The subset of `constants` the diagnostic touches."""
    c = {
        "min_limits": jnp.array([-1.0, -1.0]),
        "max_limits": jnp.array([1.0, 1.0]),
        "is_logscale": jnp.array([0, 0]),
    }
    c.update(overrides)
    return c


def quadratic_loss(curvature):
    """A loss with prescribed curvature along each axis, minimised at the origin."""
    k = jnp.array(curvature)

    def loss(constants, theta):
        return jnp.sum(k * theta**2)

    return loss


# ---------------------------------------------------------------------------
# it must never break a fit
# ---------------------------------------------------------------------------

def test_returns_none_and_does_not_raise_when_the_loss_raises(tmp_path):
    def exploding_loss(constants, theta):
        raise RuntimeError("solver blew up")

    result = run_sloppiness_analysis(exploding_loss, [make_constants()],
                                     np.zeros(2), ["a", "b"], tmp_path)
    assert result is None


def test_does_not_raise_when_the_loss_returns_nan(tmp_path):
    def nan_loss(constants, theta):
        return jnp.array(np.nan)

    run_sloppiness_analysis(nan_loss, [make_constants()],
                            np.zeros(2), ["a", "b"], tmp_path)


def test_does_not_raise_on_a_bad_output_directory():
    run_sloppiness_analysis(quadratic_loss([1.0, 1.0]), [make_constants()],
                            np.zeros(2), ["a", "b"], "/nonexistent/path/xyz")


def test_skips_when_there_are_too_many_parameters(tmp_path):
    """Above MAX_NPAR the finite-difference Hessian is too expensive to attempt."""
    n = MAX_NPAR + 1
    constants = make_constants(
        min_limits=jnp.full(n, -1.0), max_limits=jnp.full(n, 1.0),
        is_logscale=jnp.zeros(n, dtype=int),
    )
    result = run_sloppiness_analysis(
        quadratic_loss(np.ones(n)), [constants], np.zeros(n),
        [f"p{i}" for i in range(n)], tmp_path,
    )
    assert result is None
    assert not (tmp_path / "sloppiness_report.txt").exists()


# ---------------------------------------------------------------------------
# its verdict must be meaningful
# ---------------------------------------------------------------------------

def test_writes_report_and_plot_for_a_well_conditioned_problem(tmp_path):
    run_sloppiness_analysis(quadratic_loss([1.0, 1.0]), [make_constants()],
                            np.zeros(2), ["alpha", "beta"], tmp_path)

    report = tmp_path / "sloppiness_report.txt"
    assert report.exists()
    assert (tmp_path / "sloppiness_spectrum.png").exists()

    text = report.read_text()
    assert "alpha" in text and "beta" in text


def test_detects_a_flat_direction_as_non_identifiable(tmp_path):
    """A loss that is completely flat in one parameter must be flagged.

    This is the case the diagnostic exists to catch: the data constrains one
    combination of parameters and says nothing about the other.
    """
    result = run_sloppiness_analysis(
        quadratic_loss([1.0, 0.0]), [make_constants()],
        np.zeros(2), ["constrained", "flat"], tmp_path,
    )

    assert result is not None
    text = (tmp_path / "sloppiness_report.txt").read_text().lower()
    assert "identifiab" in text


def test_spectrum_ordering_reflects_the_curvature(tmp_path):
    """Eigenvalues must span the imposed curvature ratio, stiffest first."""
    result = run_sloppiness_analysis(
        quadratic_loss([100.0, 1.0]), [make_constants()],
        np.zeros(2), ["stiff", "sloppy"], tmp_path,
    )

    assert result is not None
    eigenvalues = np.asarray(result.get("eigenvalues"))
    assert eigenvalues.shape == (2,)
    assert np.all(np.diff(np.abs(eigenvalues)) <= 0), "eigenvalues must be sorted stiffest-first"
    ratio = np.abs(eigenvalues[0]) / max(np.abs(eigenvalues[-1]), 1e-30)
    assert ratio > 10, f"expected a wide spectrum, got ratio {ratio}"


def test_averages_over_multiple_experiments(tmp_path):
    """With two experiments the Hessian must come from the averaged loss."""
    result = run_sloppiness_analysis(
        quadratic_loss([1.0, 1.0]), [make_constants(), make_constants()],
        np.zeros(2), ["a", "b"], tmp_path,
    )
    assert result is not None
    assert (tmp_path / "sloppiness_report.txt").exists()
