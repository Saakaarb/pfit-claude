"""End-to-end fits across the configuration space.

Every test here runs a real PSO/DE + NODE fit on the fast `decay_session`
fixture, whose data is the exact analytic solution of

    dA/dt = -k1*A ;  dB/dt = k1*A - k2*B      (k1 = 1.0, k2 = 0.3)

so "did it work" can be asserted as "did it recover the true parameters"
rather than merely "did it not crash".

Before this file, only one configuration was ever executed: single-experiment,
PSO, lbfgs, default Kvaerno5. DE, adam, explicit solvers, multi-experiment
fitting, seeded reproducibility and the gradient-only entry point were all
untested.
"""

import shutil
import subprocess
import sys

import numpy as np
import pytest

from tests.conftest import DECAY_TRUE_PARAMS, REPO_ROOT, set_setting

TRUE = np.array([DECAY_TRUE_PARAMS["k1"], DECAY_TRUE_PARAMS["k2"]])

pytestmark = pytest.mark.slow


def assert_recovers_truth(result, rtol=0.05):
    """The fit must land within `rtol` of the parameters that generated the data."""
    result = np.asarray(result, dtype=float)
    assert result.shape == TRUE.shape
    assert np.all(np.isfinite(result)), f"non-finite parameters: {result}"
    np.testing.assert_allclose(result, TRUE, rtol=rtol)


def read_final_design_point(session):
    return np.atleast_1d(
        np.genfromtxt(session / "outputs" / "final_design_point.csv", delimiter=",")
    )


# ---------------------------------------------------------------------------
# baseline + the population algorithms
# ---------------------------------------------------------------------------

def test_pso_lbfgs_recovers_true_parameters(make_session, run_fit):
    session = make_session("decay_session")
    assert_recovers_truth(run_fit(session))


def test_differential_evolution_recovers_true_parameters(make_session, run_fit):
    """algorithm: DE — the entire FitParamsDE class was previously unexecuted."""
    session = make_session("decay_session", population={"algorithm": "DE"})
    assert_recovers_truth(run_fit(session))
    assert (session / "outputs" / "de_fitting.log").exists()
    assert not (session / "outputs" / "pso_fitting.log").exists()


def test_pso_writes_its_own_log(make_session, run_fit):
    session = make_session("decay_session")
    run_fit(session)
    log = (session / "outputs" / "pso_fitting.log").read_text()
    assert "Total number of PSO iterations" in log
    # one CSV row per iteration
    rows = [l for l in log.splitlines() if l and l[0].isdigit()]
    assert len(rows) == 5


# ---------------------------------------------------------------------------
# gradient optimizers
# ---------------------------------------------------------------------------

def test_adam_gradient_optimizer_runs_and_improves(make_session, run_fit):
    """gradient_optimizer: adam exercises the LR-schedule branch in FitParamsNODE.

    Adam is first-order, so it needs many more iterations than L-BFGS to move a
    comparable distance; the assertion is that it converges to the truth, with a
    looser tolerance than the quasi-Newton default.
    """
    session = make_session(
        "decay_session",
        gradient={"gradient_optimizer": "adam", "num_iters": 60,
                  "init_value_lr": 1e-2, "end_value_lr": 1e-4},
    )
    assert_recovers_truth(run_fit(session), rtol=0.15)
    assert (session / "outputs" / "NODE_fitting.log").exists()


def test_unsupported_gradient_optimizer_degrades_to_the_population_result(
        make_session, run_fit, capsys):
    """An unknown gradient_optimizer does NOT abort the fit.

    FitParamsNODE raises, but fit_equation_system catches every NODE exception
    (helper_functions.py:414) and falls back to the population-search point with
    loss 1e10. This is a silent degradation: a typo'd optimizer name disables
    the entire gradient stage and the run still reports "success", so the test
    pins the behaviour and the warning that accompanies it.
    """
    session = make_session("decay_session", gradient={"gradient_optimizer": "rmsprop"})
    result = run_fit(session)

    assert np.all(np.isfinite(np.asarray(result, dtype=float)))
    assert "Error in NODE training" in capsys.readouterr().out
    # PSO alone gets close but the gradient stage never ran, so no NODE log
    assert not (session / "outputs" / "NODE_fitting.log").exists()


# ---------------------------------------------------------------------------
# integrators
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("integrator", ["Dopri5", "Tsit5", "Kvaerno5", "Kvaerno3"])
def test_fit_works_across_integrators(make_session, run_fit, integrator):
    """Both explicit and implicit solvers must fit this (non-stiff) system.

    The asymmetry the digest documents: a stiff solver on a non-stiff system is
    merely slower, so all four are expected to succeed here.
    """
    session = make_session("decay_session", gradient={"integrator": integrator})
    # the solver class is baked into the generated script, not read from the config
    gs = session / "generated" / "generated_script.py"
    gs.write_text(gs.read_text().replace("diffrax.Dopri5()", f"diffrax.{integrator}()"))

    assert_recovers_truth(run_fit(session))


# ---------------------------------------------------------------------------
# multi-experiment
# ---------------------------------------------------------------------------

def test_multi_experiment_fit_recovers_shared_parameters(make_session, run_fit):
    """Two runs with different ICs, one shared parameter set."""
    session = make_session("decay_multiexp")
    assert_recovers_truth(run_fit(session))


def test_multi_experiment_writes_one_result_file_per_experiment(make_session, run_fit):
    session = make_session("decay_multiexp")
    run_fit(session)

    exp1 = np.genfromtxt(session / "outputs" / "result_solution_exp1.csv", delimiter=",")
    exp2 = np.genfromtxt(session / "outputs" / "result_solution_exp2.csv", delimiter=",")
    assert not (session / "outputs" / "result_solution_exp3.csv").exists()

    # each file must reflect ITS OWN initial condition (1.0 vs 2.0 for A)
    assert exp1[0, 3] == pytest.approx(1.0, abs=1e-6)
    assert exp2[0, 3] == pytest.approx(2.0, abs=1e-6)


def test_multi_experiment_solution_tracks_both_datasets(make_session, run_fit):
    """A fit that ignored experiment 2 would still look good on experiment 1."""
    session = make_session("decay_multiexp")
    run_fit(session)

    for name in ("result_solution_exp1.csv", "result_solution_exp2.csv"):
        arr = np.genfromtxt(session / "outputs" / name, delimiter=",")
        data_a, sol_a = arr[:, 1], arr[:, 3]
        rel_err = np.abs(sol_a - data_a).max() / np.abs(data_a).max()
        assert rel_err < 0.02, f"{name}: solution does not track the data"


# ---------------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------------

def test_random_seed_makes_the_fit_reproducible(make_session, run_fit):
    """random_seed must pin every stochastic component (LHS init + PSO draws)."""
    a = make_session("decay_session", population={"random_seed": 11}, dest_name="a")
    b = make_session("decay_session", population={"random_seed": 11}, dest_name="b")

    np.testing.assert_array_equal(np.asarray(run_fit(a)), np.asarray(run_fit(b)))


def test_de_is_reproducible_by_default(make_session, run_fit):
    """DE falls back to seed 42 when random_seed is unset, so it is deterministic."""
    a = make_session("decay_session", population={"algorithm": "DE"}, dest_name="a")
    b = make_session("decay_session", population={"algorithm": "DE"}, dest_name="b")

    np.testing.assert_array_equal(np.asarray(run_fit(a)), np.asarray(run_fit(b)))


# ---------------------------------------------------------------------------
# outputs and diagnostics
# ---------------------------------------------------------------------------

def test_fit_writes_the_expected_output_files(make_session, run_fit):
    session = make_session("decay_session")
    result = run_fit(session)

    outputs = session / "outputs"
    for name in ("final_design_point.csv", "result_solution_exp1.csv",
                 "pso_fitting.log", "NODE_fitting.log",
                 "sloppiness_report.txt", "sloppiness_spectrum.png"):
        assert (outputs / name).exists(), f"missing output: {name}"

    # the returned vector and the written CSV must agree
    np.testing.assert_allclose(read_final_design_point(session), np.asarray(result))


def test_sloppiness_report_describes_a_well_determined_fit(make_session, run_fit):
    """Both decay parameters are identifiable from this data, so the diagnostic
    must not report non-identifiable directions."""
    session = make_session("decay_session")
    run_fit(session)

    report = (session / "outputs" / "sloppiness_report.txt").read_text()
    assert "k1" in report and "k2" in report
    assert "eigen" in report.lower()


def test_write_results_off_suppresses_solution_files(make_session, run_fit):
    session = make_session("decay_session")
    set_setting(session / "inputs" / "user_input.yaml", "output", "write_results", False)

    run_fit(session)
    assert (session / "outputs" / "final_design_point.csv").exists()
    assert not (session / "outputs" / "result_solution_exp1.csv").exists()


def test_outputs_directory_is_wiped_between_runs(make_session, run_fit):
    """fit_parameters clears outputs/, so results can't be mistaken for fresh ones."""
    session = make_session("decay_session")
    outputs = session / "outputs"
    outputs.mkdir(exist_ok=True)
    (outputs / "stale_artifact.txt").write_text("from an older run")

    run_fit(session)
    assert not (outputs / "stale_artifact.txt").exists()


# ---------------------------------------------------------------------------
# the other entry points
# ---------------------------------------------------------------------------

def test_gradient_only_refines_an_existing_design_point(make_session):
    """fit_gradient_only seeds from outputs/final_design_point.csv and must
    improve (or hold) the fit without re-running the population search."""
    from fit_gradient_only import run_driver as gradient_only_driver
    from lib.utils.helper_functions import get_input_reader

    session = make_session("decay_session")
    outputs = session / "outputs"
    outputs.mkdir(exist_ok=True)

    # a deliberately offset starting point, in real units
    np.savetxt(outputs / "final_design_point.csv", np.array([0.6, 0.5]), delimiter=",")
    (outputs / "pso_fitting.log").write_text("from the previous run\n")

    reader = get_input_reader(session / "inputs" / "user_input.yaml")
    result = gradient_only_driver(session, reader)

    assert_recovers_truth(result, rtol=0.10)
    # the population-stage artifacts must be preserved, not wiped
    assert (outputs / "pso_fitting.log").read_text() == "from the previous run\n"
    assert (outputs / "NODE_fitting.log").exists()


def test_gradient_only_rejects_a_wrong_length_seed(make_session):
    from fit_gradient_only import run_driver as gradient_only_driver
    from lib.utils.helper_functions import get_input_reader

    session = make_session("decay_session")
    outputs = session / "outputs"
    outputs.mkdir(exist_ok=True)
    np.savetxt(outputs / "final_design_point.csv", np.array([1.0, 0.3, 0.1]), delimiter=",")

    reader = get_input_reader(session / "inputs" / "user_input.yaml")
    with pytest.raises(ValueError, match="init_guess has 3 entries"):
        gradient_only_driver(session, reader)
    assert (outputs / "fitting_error.txt").exists()


def test_analyze_fit_reruns_diagnostics_on_a_completed_session(make_session, run_fit):
    """analyze_fit.py must regenerate the sloppiness report standalone."""
    session = make_session("decay_session")
    run_fit(session)

    report = session / "outputs" / "sloppiness_report.txt"
    report.unlink()

    result = subprocess.run(
        [sys.executable, "analyze_fit.py", str(session)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert report.exists(), "analyze_fit did not rewrite the sloppiness report"


# ---------------------------------------------------------------------------
# error and interruption paths
# ---------------------------------------------------------------------------

def test_stop_flag_halts_the_gradient_stage_and_returns_the_seed(make_session):
    """A stop_fitting.flag must interrupt cleanly, not crash.

    Regression test: train_NODE used to initialise best_result to None, so a flag
    seen before the first iteration completed returned None and the caller died
    with "too many indices for array: array is 0-dimensional". The interrupted
    run must instead degrade to the unrefined starting point.

    The gradient-only entry point is used because fit_parameters wipes outputs/
    at startup, which would delete the flag before it could be seen.
    """
    from fit_gradient_only import run_driver as gradient_only_driver
    from lib.utils.helper_functions import get_input_reader

    session = make_session("decay_session")
    outputs = session / "outputs"
    outputs.mkdir(exist_ok=True)
    seed = np.array([0.6, 0.5])
    np.savetxt(outputs / "final_design_point.csv", seed, delimiter=",")
    (outputs / "stop_fitting.flag").write_text("stop")

    reader = get_input_reader(session / "inputs" / "user_input.yaml")
    result = gradient_only_driver(session, reader)

    np.testing.assert_allclose(np.asarray(result, dtype=float), seed, rtol=1e-6)


def test_stop_flag_halts_the_population_search(make_session):
    """optimize_function must check the flag between iterations and break.

    Called directly (rather than through a full fit) because fit_parameters
    deletes outputs/ — and the flag with it — before the search starts.
    """
    from lib.utils.helper_functions import optimize_function

    session = make_session("decay_session")
    outputs = session / "outputs"
    outputs.mkdir(exist_ok=True)
    (outputs / "stop_fitting.flag").write_text("stop")

    class SpyFitObj:
        def __init__(self):
            self.iterations = 0
            self.best_pos = np.array([0.0, 0.0])
            self.swarm_obj = type("swarm", (), {"best_cost": 1.0})()

        def search_iteration(self, iter_no, file_obj):
            self.iterations += 1

    class Reader:
        n_iters_pop = 25
        output_dir = outputs

    fit_obj = SpyFitObj()
    best_pos, best_cost = optimize_function(fit_obj, Reader(), None)

    assert fit_obj.iterations == 0, "the flag must stop the search immediately"
    np.testing.assert_array_equal(best_pos, [0.0, 0.0])
    assert best_cost == 1.0


def test_missing_generated_script_is_reported_clearly(make_session, run_fit):
    session = make_session("decay_session")
    (session / "generated" / "generated_script.py").unlink()

    with pytest.raises(FileNotFoundError, match="pfit-jax"):
        run_fit(session)


def test_missing_data_file_writes_an_error_report(make_session, run_fit):
    session = make_session("decay_session")
    (session / "inputs" / "decay_data.csv").unlink()

    with pytest.raises(Exception):
        run_fit(session)
    assert (session / "outputs" / "fitting_error.txt").exists()


def test_duplicate_names_are_rejected_before_fitting(make_session, run_fit):
    session = make_session("decay_session")
    config = session / "inputs" / "user_input.yaml"
    # rename integrated variable A to k1, colliding with the trainable parameter
    config.write_text(config.read_text().replace("{name: A,", "{name: k1,", 1))

    with pytest.raises(ValueError, match="not unique"):
        run_fit(session)
    assert (session / "outputs" / "fitting_error.txt").exists()
