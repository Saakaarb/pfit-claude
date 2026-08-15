"""Unit tests for the problem object and parameter scaling.

These cover the numerical plumbing every fit depends on, without paying for a
fit: the [-1, 1] scaling round-trip (a sign error here silently fits the wrong
parameter), and `CreatedClass`'s per-experiment constants and loss averaging —
the multi-experiment execution model that CLAUDE.md calls out as critical and
that no test previously touched.
"""

import numpy as np
import pytest

import jax.numpy as jnp

from lib.algorithms.NODE.helper_functions import scale_value, unscale_value
from lib.utils.helper_functions import CreatedClass, get_input_reader
from tests.conftest import FIXTURES


# ---------------------------------------------------------------------------
# scaling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,lo,hi", [
    (0.5, 0.0, 1.0), (-3.0, -10.0, 10.0), (1e-4, 1e-6, 1e-2), (7.0, 7.0, 42.0),
])
def test_scale_unscale_round_trip(value, lo, hi):
    assert unscale_value(scale_value(value, lo, hi), lo, hi) == pytest.approx(value)


def test_scale_maps_bounds_to_minus_one_and_one():
    assert scale_value(2.0, 2.0, 8.0) == pytest.approx(-1.0)
    assert scale_value(8.0, 2.0, 8.0) == pytest.approx(1.0)
    assert scale_value(5.0, 2.0, 8.0) == pytest.approx(0.0)


def test_generated_script_scaling_matches_the_framework():
    """The JAX copies in generated_script.py must agree with the NODE helpers.

    They are separate implementations (one jitted, one plain numpy); a drift
    between them would move the optimizer and the reported parameters apart.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "decay_gs", FIXTURES / "decay_session" / "generated" / "generated_script.py")
    gs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gs)

    lo, hi = -2.0, 1.0          # log10 bounds, as the framework stores them
    for scaled in (-1.0, -0.25, 0.0, 0.5, 1.0):
        linear = unscale_value(scaled, lo, hi)
        # is_logscale=0 must match the plain helper exactly
        assert float(gs.unscale_value(scaled, lo, hi, 0)) == pytest.approx(linear)
        # is_logscale=1 must be the same value, exponentiated
        assert float(gs.unscale_value(scaled, lo, hi, 1)) == pytest.approx(10.0**linear)
        # and scale_value must inverse it
        assert float(gs.scale_value(10.0**linear, lo, hi, 1)) == pytest.approx(scaled)


# ---------------------------------------------------------------------------
# CreatedClass: per-experiment constants and loss aggregation
# ---------------------------------------------------------------------------

def build_problem(fixture_name: str, loss_fn=None, write_fn=None) -> CreatedClass:
    """Assemble a CreatedClass from a committed fixture, as fit_generic_system does."""
    session = FIXTURES / fixture_name
    reader = get_input_reader(session / "inputs" / "user_input.yaml")

    experiments = []
    for i, exp in enumerate(reader.experiments):
        data = np.genfromtxt(session / "inputs" / exp["filename"], delimiter=",")
        experiments.append({
            "t_eval": data[:, 0],
            "dataset": data[:, 1:],
            "y0": jnp.array(reader.get_y0(i)),
        })

    return CreatedClass(
        experiments=experiments, input_reader=reader,
        compute_loss_problem=loss_fn or (lambda c, x: jnp.array(0.0)),
        write_problem_result=write_fn or (lambda c, x: jnp.zeros((2, 2))),
    )


def test_constants_are_built_per_experiment():
    problem = build_problem("decay_multiexp")
    assert len(problem.constants_list) == 2

    # each experiment carries its OWN initial condition, from its config overrides
    np.testing.assert_allclose(np.asarray(problem.constants_list[0]["init_cond"]), [1.0, 0.0])
    np.testing.assert_allclose(np.asarray(problem.constants_list[1]["init_cond"]), [2.0, 0.5])

    # and its own data / time grid
    for c in problem.constants_list:
        assert c["dataset"].shape == (21, 2)
        assert c["num_steps"] == 21
        assert float(c["init_time"]) == pytest.approx(0.0)
        assert float(c["final_time"]) == pytest.approx(10.0)


def test_datasets_differ_between_experiments():
    """Guards against every experiment accidentally getting experiment 0's data."""
    problem = build_problem("decay_multiexp")
    a = np.asarray(problem.constants_list[0]["dataset"])
    b = np.asarray(problem.constants_list[1]["dataset"])
    assert not np.allclose(a, b)


def test_loss_is_averaged_over_experiments():
    """The framework averages per-experiment losses; the user function returns one."""
    calls = []

    def fake_loss(constants, x):
        # Derive the "loss" from this experiment's own initial condition, so the
        # result can only be right if each experiment's constants were passed in.
        # (No float() here — this runs under trace, where values are abstract.)
        calls.append(1)
        return jnp.sum(constants["init_cond"])

    problem = build_problem("decay_multiexp", loss_fn=fake_loss)
    total = float(problem._compute_loss(jnp.array([0.0, 0.0])))

    # experiment 1 ICs sum to 1.0 + 0.0; experiment 2 to 2.0 + 0.5
    assert total == pytest.approx((1.0 + 2.5) / 2)
    assert len(calls) == 2                  # called once per experiment, not once overall


def test_single_experiment_loss_is_not_averaged_away():
    problem = build_problem("decay_session", loss_fn=lambda c, x: jnp.array(0.75))
    assert float(problem._compute_loss(jnp.array([0.0, 0.0]))) == pytest.approx(0.75)


def test_limit_setters_apply_to_every_experiment():
    problem = build_problem("decay_multiexp")
    problem.set_min_limit([-2.0, -2.0])
    problem.set_max_limit([1.0, 1.0])
    problem.set_is_logscale([1, 1])
    for c in problem.constants_list:
        np.testing.assert_allclose(np.asarray(c["min_limits"]), [-2.0, -2.0])
        np.testing.assert_allclose(np.asarray(c["max_limits"]), [1.0, 1.0])
        np.testing.assert_allclose(np.asarray(c["is_logscale"]), [1, 1])


def test_write_problem_result_emits_one_file_per_experiment(tmp_path):
    problem = build_problem("decay_multiexp",
                            write_fn=lambda c, x: jnp.column_stack(
                                [c["t_eval"], c["dataset"][:, 0]]))
    reader = problem.input_reader
    reader.output_dir = tmp_path

    problem.write_problem_result(np.array([0.0, 0.0]), reader, label="result")

    assert (tmp_path / "result_solution_exp1.csv").exists()
    assert (tmp_path / "result_solution_exp2.csv").exists()
    # the two experiments must not write identical content
    a = np.genfromtxt(tmp_path / "result_solution_exp1.csv", delimiter=",")
    b = np.genfromtxt(tmp_path / "result_solution_exp2.csv", delimiter=",")
    assert not np.allclose(a, b)


def test_compute_all_losses_sanitises_non_finite_values():
    """NaN/inf from a broken solve must become 1e10, never propagate to the swarm."""
    def nan_loss(constants, x):
        return jnp.where(x[0] > 0, jnp.nan, 0.5)

    problem = build_problem("decay_session", loss_fn=nan_loss)
    population = np.array([[-1.0, 0.0], [1.0, 0.0]])
    losses = problem._compute_all_losses(population)

    assert losses[0] == pytest.approx(0.5)
    assert losses[1] == pytest.approx(1e10)
    assert np.all(np.isfinite(losses))


def test_compute_all_losses_handles_population_not_divisible_by_devices():
    """Sharding pads the population to a multiple of the device count and must
    strip the padding again — an off-by-one here silently drops particles."""
    problem = build_problem("decay_session",
                            loss_fn=lambda c, x: jnp.abs(x[0]))
    for n_particles in (1, 3, 7, 20):
        population = np.linspace(-1, 1, n_particles).reshape(-1, 1)
        population = np.column_stack([population, np.zeros(n_particles)])
        losses = problem._compute_all_losses(population)
        assert losses.shape == (n_particles,)
        np.testing.assert_allclose(losses, np.abs(population[:, 0]), atol=1e-12)
