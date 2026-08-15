"""Shared fixtures for the pfit-claude test suite.

The integration tests all follow the same shape: copy a committed session
fixture into a tmp directory, optionally patch a handful of settings in
user_input.yaml, and run a fit. Copying matters — a fit deletes and rewrites
`outputs/`, so running against the committed fixture in place would dirty the
repo and make tests order-dependent.
"""

import shutil
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"

# Ground truth used to generate every decay fixture's data (see the config header).
DECAY_TRUE_PARAMS = {"k1": 1.0, "k2": 0.3}
DECAY_PARAM_ORDER = ["k1", "k2"]

# Robertson's classic rate constants; tests/robertson_session's data is a
# noiseless solution of the system at these values.
ROBERTSON_TRUE_PARAMS = [0.04, 3.0e7, 1.0e4]

sys.path.insert(0, str(REPO_ROOT))


def _split_header(text: str) -> tuple[str, str]:
    """Separate the leading `#` comment block from the YAML body.

    The fixtures document their ground-truth parameters in that header, and a
    PyYAML round-trip drops every comment — so it is preserved by hand.
    """
    lines = text.splitlines(keepends=True)
    cut = 0
    for i, line in enumerate(lines):
        if line.strip() and not line.lstrip().startswith("#"):
            cut = i
            break
    else:
        cut = len(lines)
    return "".join(lines[:cut]), "".join(lines[cut:])


def _rewrite_config(config_path: Path, mutate) -> None:
    header, body = _split_header(config_path.read_text())
    config = yaml.safe_load(body)
    mutate(config)
    config_path.write_text(
        header + yaml.safe_dump(config, sort_keys=False, default_flow_style=False)
    )


def set_setting(config_path: Path, section: str, key: str, value) -> None:
    """Set (or insert) `key: value` inside a named section of user_input.yaml.

    `section` is a top-level key such as population_opt or gradient_opt.
    num_iters exists in both optimizer sections, which is exactly why the
    section must be given explicitly.
    """

    def mutate(config):
        if section not in config:
            raise ValueError(f"{config_path} has no '{section}' section")
        config[section][key] = value

    _rewrite_config(config_path, mutate)


def remove_setting(config_path: Path, section: str, key: str) -> None:
    """Delete a key, so the reader falls back to its default."""

    def mutate(config):
        config[section].pop(key, None)

    _rewrite_config(config_path, mutate)


@pytest.fixture(autouse=True)
def clear_jax_caches_between_tests():
    """Release compiled XLA executables after every test.

    `CreatedClass._compute_loss` is jitted with `static_argnums=(0,)`, so every
    problem object becomes a distinct compilation key and its executables stay
    alive for the life of the process. A suite that runs a dozen independent
    fits therefore accumulates them all; clearing between tests keeps each fit
    as isolated as it would be in a fresh `python fit_parameters.py` run.
    """
    yield
    try:
        import gc

        import jax

        jax.clear_caches()
        gc.collect()
    except Exception:  # never let cleanup fail a test
        pass


@pytest.fixture
def make_session(tmp_path):
    """Factory: copy a committed fixture session into tmp_path and patch its config.

    Usage:
        session = make_session("decay_session",
                               population={"num_iters": 2},
                               gradient={"gradient_optimizer": "adam"})
    """

    def _make(name: str, population: dict | None = None,
              gradient: dict | None = None, dest_name: str | None = None) -> Path:
        source = FIXTURES / name
        if not source.is_dir():
            raise ValueError(f"no such fixture: {source}")
        dest = tmp_path / (dest_name or name)
        shutil.copytree(source, dest)

        config_path = dest / "inputs" / "user_input.yaml"
        for key, value in (population or {}).items():
            set_setting(config_path, "population_opt", key, value)
        for key, value in (gradient or {}).items():
            set_setting(config_path, "gradient_opt", key, value)
        return dest

    return _make


@pytest.fixture
def run_fit():
    """Run the full two-stage fit on a session directory and return the result."""

    def _run(session_dir: Path):
        from fit_parameters import run_driver
        from lib.utils.helper_functions import get_input_reader

        reader = get_input_reader(session_dir / "inputs" / "user_input.yaml")
        return run_driver(session_dir, reader)

    return _run
