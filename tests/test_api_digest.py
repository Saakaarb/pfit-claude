"""Guard the generated API digests in lib/LLM/api/.

The digests are the LLM skills' only source of truth for the diffrax / jax /
optax / scipy APIs. Two ways they can silently become harmful:

1. They drift out of date after a dependency bump, so the skills read API
   details that no longer match what the code will run against.
2. A claim in a digest stops being true (a solver disappears, a signature
   changes), which is exactly the failure the digests exist to prevent.

These tests catch both.
"""

import importlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
API_DIR = REPO_ROOT / "lib" / "LLM" / "api"
GENERATOR = REPO_ROOT / "tools" / "gen_api_context.py"

DIGESTS = ["diffrax.md", "jax.md", "optax.md", "population_optimizers.md", "MANIFEST.md"]


@pytest.mark.parametrize("name", DIGESTS)
def test_digest_exists(name):
    assert (API_DIR / name).exists(), (
        f"{name} is missing — run: ./venv/bin/python3 tools/gen_api_context.py"
    )


@pytest.mark.parametrize("name", [d for d in DIGESTS if d != "MANIFEST.md"])
def test_digest_version_stamp_matches_installed(name):
    """Every digest header pins the versions it was generated from."""
    text = (API_DIR / name).read_text()
    match = re.search(r"\*\*Pinned to: (.+?)\*\*", text)
    assert match, f"{name} has no version stamp header"

    for entry in match.group(1).split(", "):
        package, _, recorded = entry.partition("==")
        installed = getattr(importlib.import_module(package), "__version__", None)
        assert installed == recorded, (
            f"{name} is STALE: records {package}=={recorded} but {installed} is "
            f"installed. Regenerate: ./venv/bin/python3 tools/gen_api_context.py"
        )


def test_generator_is_idempotent(tmp_path):
    """Re-running the generator on an unchanged env must not rewrite anything.

    This both proves the digests are current and keeps the generator safe to run
    from a skill without dirtying the git tree.
    """
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--log-file", str(tmp_path / "gen.log")],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "done: 4 digest(s) produced, 0 file(s) updated" in result.stdout, (
        "Generator rewrote a digest, so the committed digests were out of date. "
        "Commit the regenerated files.\n" + result.stdout
    )


def documented_solvers() -> list[str]:
    """Solver names listed as usable in the digest's INTEGRATOR table."""
    text = (API_DIR / "diffrax.md").read_text()
    table = text.split("| Solver | Kind |")[1].split("**Excluded")[0]
    return re.findall(r"^\| `(\w+)` \|", table, re.MULTILINE)


def test_documented_solvers_are_usable_integrators():
    """Every listed solver must satisfy all three constraints the framework imposes.

    Adaptive (PIDController needs an error estimate), deterministic (the term is
    an ODETerm), and constructible as a bare `diffrax.<Name>()` — which is
    literally how the generated script instantiates it.
    """
    import diffrax

    names = documented_solvers()
    assert len(names) >= 10, "solver table looks truncated"

    for name in names:
        solver = getattr(diffrax, name, None)
        assert solver is not None, f"digest lists diffrax.{name}, which does not exist"
        assert issubclass(solver, diffrax.AbstractAdaptiveSolver), (
            f"{name} is listed but is not adaptive; it cannot be used with "
            f"PIDController"
        )
        solver()  # must construct with no arguments, or this raises


def test_excluded_solvers_are_really_unusable():
    """The exclusion list must not hide a solver that would in fact work."""
    import diffrax

    text = (API_DIR / "diffrax.md").read_text()
    excluded_block = text.split("**Excluded as unusable")[1].split("##")[0]
    # Only the "- _reason_: `A`, `B`" bullets name solvers; the surrounding
    # prose also contains backticked words such as `INTEGRATOR`.
    excluded = [
        name
        for line in excluded_block.splitlines()
        if line.startswith("- _")
        for name in re.findall(r"`(\w+)`", line)
    ]
    assert excluded, "exclusion list is empty"

    for name in excluded:
        solver = getattr(diffrax, name)
        adaptive = issubclass(solver, diffrax.AbstractAdaptiveSolver)
        try:
            solver()
            constructible = True
        except Exception:
            constructible = False
        bases = {b.__name__ for b in solver.__mro__}
        sde_only = bool(
            {"AbstractSRK", "AbstractStratonovichSolver", "AbstractItoSolver"} & bases
        ) and not ({"AbstractERK", "AbstractRungeKutta"} & bases)
        assert not (adaptive and constructible and not sde_only), (
            f"{name} is excluded but appears usable — the filter is too aggressive"
        )


def test_stiff_solvers_are_available():
    """The stiff/non-stiff choice must remain open to the orchestrator."""
    import diffrax

    stiff = [
        n for n in documented_solvers()
        if issubclass(getattr(diffrax, n), diffrax.AbstractImplicitSolver)
    ]
    assert len(stiff) >= 3, f"too few stiff solvers available: {stiff}"
    assert "Kvaerno5" in stiff, "Kvaerno5 (the default INTEGRATOR) must be listed"


def test_documented_results_codes_exist():
    """The RESULTS table drives the loss failure mask — every code must be real."""
    from diffrax import RESULTS

    text = (API_DIR / "diffrax.md").read_text()
    codes = re.findall(r"`RESULTS\.(\w+)`", text)
    assert "successful" in codes and "max_steps_reached" in codes

    for code in set(codes):
        assert hasattr(RESULTS, code), f"digest lists RESULTS.{code}, which does not exist"


def test_integrators_used_by_sessions_are_valid():
    """Any integrator named in a committed session config must be a real adaptive solver."""
    import diffrax
    import yaml

    sessions = REPO_ROOT / "sessions"
    if not sessions.is_dir():
        pytest.skip("no sessions directory")

    checked = 0
    for config_path in sessions.glob("*/inputs/user_input.yaml"):
        config = yaml.safe_load(config_path.read_text()) or {}
        name = (config.get("gradient_opt") or {}).get("integrator")
        if not name:
            continue
        checked += 1
        solver = getattr(diffrax, name, None)
        assert solver is not None, f"{config_path}: integrator {name} does not exist"
        assert issubclass(solver, diffrax.AbstractAdaptiveSolver), (
            f"{config_path}: integrator {name} is not adaptive and cannot be used "
            f"with the PIDController in _integrate_system"
        )
    if checked == 0:
        pytest.skip("no session sets integrator explicitly")


def test_failure_mask_is_exhaustive_everywhere():
    """The loss must mask on `!= successful`, not on an enumerated code subset.

    diffrax defines 14 result codes; only `successful` yields a usable
    trajectory. Enumerating a subset (the old
    `max_steps_reached or singular` form) lets inf/NaN trajectories from
    dt_min_reached / nonlinear_divergence / nonfinite be scored as a genuine
    fit, which silently produces NaN gradients in the NODE stage.
    """
    targets = [REPO_ROOT / "lib" / "utils" / "output_sample.py"]
    targets += sorted(REPO_ROOT.glob("sessions/*/generated/generated_script.py"))
    targets += sorted(REPO_ROOT.glob("tests/*/generated/generated_script.py"))

    checked = 0
    for path in targets:
        text = path.read_text()
        if "failed" not in text:
            continue
        checked += 1
        rel = path.relative_to(REPO_ROOT)
        assert "jnp.invert(result == RESULTS.successful)" in text, (
            f"{rel}: failure mask is not the exhaustive form"
        )
        assert "RESULTS.max_steps_reached" not in text, (
            f"{rel}: still enumerates individual failure codes"
        )
    assert checked >= 5, "expected to check the template plus the example scripts"


def test_saveat_preserves_data_row_alignment():
    """Every generated script must save at exactly the data times.

    The saved solution is differenced against `dataset` row for row, so the save
    grid has to equal the data grid. `SaveAt(ts=t_eval)` guarantees that for any
    `init_time`; `SaveAt(t0=True, ts=t_eval[1:])` only matches while
    `INITIAL_TIME == t_eval[0]`, and silently misaligns every residual otherwise
    (row 0 compares the model at init_time against the data at t_eval[0], and
    t_eval[0] is never evaluated). That regression shipped in
    sessions/sliding_basepoint, which sets INITIAL_TIME=0.0 against data
    starting at 9.94e-4, and nothing caught it.
    """
    targets = [REPO_ROOT / "lib" / "utils" / "output_sample.py"]
    targets += sorted(REPO_ROOT.glob("sessions/*/generated/generated_script.py"))
    targets += sorted(REPO_ROOT.glob("tests/*/generated/generated_script.py"))
    targets += sorted(REPO_ROOT.glob("tests/fixtures/*/generated/generated_script.py"))

    checked = 0
    for path in targets:
        text = path.read_text()
        if "SaveAt" not in text:
            continue
        checked += 1
        rel = path.relative_to(REPO_ROOT)
        assert "diffrax.SaveAt(ts=t_eval)" in text, f"{rel}: SaveAt is not ts=t_eval"
        # ignore comments — the template deliberately documents the wrong form
        code = "\n".join(l.split("#")[0] for l in text.splitlines())
        assert "t0=True" not in code, (
            f"{rel}: uses SaveAt(t0=True, ...), which misaligns the residuals "
            f"whenever INITIAL_TIME differs from t_eval[0]"
        )
    assert checked >= 8, "expected the template plus every committed generated script"


def test_saveat_alignment_holds_when_initial_time_precedes_the_data():
    """Prove the invariant against diffrax rather than trusting the source text."""
    import diffrax
    import jax.numpy as jnp
    import numpy as np

    t_eval = jnp.array([0.5, 1.0, 1.5, 2.0])
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t, y, args: -0.5 * y), diffrax.Kvaerno5(),
        t0=0.0, t1=2.0, dt0=1e-4, y0=jnp.array([1.0]), max_steps=10000, throw=False,
        saveat=diffrax.SaveAt(ts=t_eval),
        stepsize_controller=diffrax.PIDController(rtol=1e-8, atol=1e-10),
    )
    # init_time (0.0) is deliberately earlier than t_eval[0] (0.5)
    assert np.allclose(np.asarray(sol.ts), np.asarray(t_eval)), (
        "SaveAt(ts=t_eval) must save exactly at the data times"
    )
