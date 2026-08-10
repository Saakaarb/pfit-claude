"""End-to-end fits on the two original session fixtures.

The data-driven cases in test_cases.yml carry the parameters that generated the
data, so a converged fit is checked against ground truth rather than merely
being checked for "did not raise" — which is all this file asserted before,
because every case had a blank `expected` and `run_driver` returned None.
"""

import importlib.util
import shutil

import numpy as np
import pytest
import yaml

from tests import helper_functions
from tests.conftest import REPO_ROOT

pytestmark = pytest.mark.slow


def load_tests():
    with open("tests/test_cases.yml", "r") as f:
        return yaml.safe_load(f)["tests"]


def resolve_function(name):
    return getattr(helper_functions, name)


@pytest.mark.parametrize("case", load_tests(), ids=lambda x: x["name"])
def test_functions(case):
    func = resolve_function(case["function"])
    args = case["input"]
    expected = case.get("expected", None)

    # Run the function with the provided arguments
    if args is not None:
        result = func(*args)
    else:
        result = func()

    # If no expected value is provided, just check that no exception was raised
    if expected is None or expected == []:
        print(f"Test '{case['name']}' ran successfully (no expected value to check).")
        assert True
        return

    np_result = np.array(result, dtype="float")
    np_expected = np.array(expected, dtype="float")
    assert np_result.shape == np_expected.shape, (
        f"fit returned {np_result.shape} parameters, expected {np_expected.shape}"
    )

    frac_err = np.divide(np_result - np_expected, np_expected)
    standard = np.zeros_like(frac_err)
    assert np.allclose(
        frac_err, standard, rtol=1e-2, atol=1e-1
    ), f"Expected {standard}, got {frac_err}"


def load_generated_script(session):
    """Import a session's generated_script.py under a unique module name."""
    path = session / "generated" / "generated_script.py"
    spec = importlib.util.spec_from_file_location(f"gs_{session.name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vanderpol_fit_is_locally_optimal(tmp_path):
    """Ground-truth-free check: the fitted point must beat random alternatives.

    The van der Pol fixture's data has no identifiable true `mu` (integrating to
    t=100 makes pointwise error phase-dominated, so the residual has no
    minimum), which is why this asserts optimality of the loss instead of
    accuracy of the parameter. A fit that returned its starting guess, or a
    NaN, or a bound, would fail here.
    """
    import jax.numpy as jnp

    from lib.utils.helper_functions import get_input_reader
    from fit_parameters import run_driver

    session = tmp_path / "vanderpol_session"
    shutil.copytree(REPO_ROOT / "tests" / "vanderpol_session", session)

    reader = get_input_reader(session / "inputs" / "user_input.xml")
    result = np.atleast_1d(np.asarray(run_driver(session, reader), dtype=float))

    assert result.shape == (1,)
    assert np.all(np.isfinite(result))
    assert reader.min_axis_values[0] < result[0] < reader.max_axis_values[0], (
        f"fitted mu={result[0]} sits on or outside the search bounds"
    )

    # rebuild the loss exactly as the framework does, then compare the fitted
    # point against random points drawn from the same (log) search space
    gs = load_generated_script(session)
    data = np.genfromtxt(session / "inputs" / reader.filename_data, delimiter=",")
    lo = np.log10(reader.min_axis_values[0])
    hi = np.log10(reader.max_axis_values[0])
    constants = {
        "dataset": jnp.array(data[:, 1:]),
        "t_eval": jnp.array(data[:, 0]),
        "init_cond": jnp.array(reader.get_y0(0)),
        "init_time": float(data[0, 0]),
        "stepsize_rtol": jnp.array(reader.stepsize_rtol),
        "stepsize_atol": jnp.array(reader.stepsize_atol),
        "init_timestep": reader.init_timestep,
        "fixed_parameters": {},
        "error_loss": reader.error_loss,
        "min_limits": jnp.array([lo]),
        "max_limits": jnp.array([hi]),
        "is_logscale": jnp.array(reader.axis_logscale),
    }

    def loss_at(mu):
        scaled = 2.0 * (np.log10(mu) - lo) / (hi - lo) - 1.0
        return float(gs._compute_loss_problem(constants, jnp.array([scaled])))

    fitted_loss = loss_at(result[0])
    assert np.isfinite(fitted_loss)

    rng = np.random.default_rng(0)
    alternatives = 10 ** rng.uniform(lo, hi, size=12)
    beaten = sum(fitted_loss <= loss_at(mu) for mu in alternatives)
    assert beaten >= 11, (
        f"fitted loss {fitted_loss:.4e} was not better than "
        f"{len(alternatives) - beaten} of {len(alternatives)} random points"
    )
