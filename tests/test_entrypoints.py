"""Unit tests for the CLI entry-point helpers.

`resolve_session_dir` and `resolve_device_count` run before JAX is imported —
`resolve_device_count` in particular decides the XLA host device count, and it
must never raise, because a malformed config there would kill the process before
the real (well-reported) validation in the fitting path ever runs.

These helpers were previously untested: the suite called `run_driver` directly
and bypassed both.
"""

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

import fit_parameters
from fit_gradient_only import load_init_guess
from tests.conftest import FIXTURES


@pytest.fixture
def sessions_root(tmp_path, monkeypatch):
    """A tmp cwd containing sessions/<name>, mirroring the real layout."""
    root = tmp_path / "sessions"
    root.mkdir()
    shutil.copytree(FIXTURES / "decay_session", root / "decay_session")
    monkeypatch.chdir(tmp_path)
    return root


# ---------------------------------------------------------------------------
# resolve_session_dir
# ---------------------------------------------------------------------------

def test_resolve_session_dir_accepts_a_bare_name(sessions_root, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fit_parameters.py", "decay_session"])
    assert fit_parameters.resolve_session_dir() == Path("sessions/decay_session")


def test_resolve_session_dir_accepts_a_full_path(sessions_root, monkeypatch):
    target = sessions_root / "decay_session"
    monkeypatch.setattr(sys, "argv", ["fit_parameters.py", str(target)])
    assert fit_parameters.resolve_session_dir() == target


def test_resolve_session_dir_rejects_a_missing_session(sessions_root, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fit_parameters.py", "nope"])
    with pytest.raises(ValueError, match="does not exist"):
        fit_parameters.resolve_session_dir()


def test_resolve_session_dir_falls_back_to_most_recent(sessions_root, monkeypatch):
    """With no argument it must pick the newest session, not an arbitrary one."""
    import time

    newer = sessions_root / "newer_session"
    shutil.copytree(FIXTURES / "decay_session", newer)
    time.sleep(0.01)
    os.utime(newer, None)
    # ctime ordering is what the implementation sorts on; make it unambiguous
    newest = max(sessions_root.iterdir(), key=lambda d: d.stat().st_ctime)

    monkeypatch.setattr(sys, "argv", ["fit_parameters.py"])
    resolved = fit_parameters.resolve_session_dir()
    # the helper returns a path relative to cwd, so compare by name
    assert resolved.name == newest.name
    assert resolved.is_dir()


def test_resolve_session_dir_exits_when_sessions_is_empty(tmp_path, monkeypatch):
    (tmp_path / "sessions").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["fit_parameters.py"])
    with pytest.raises(SystemExit):
        fit_parameters.resolve_session_dir()


# ---------------------------------------------------------------------------
# resolve_device_count
# ---------------------------------------------------------------------------

def test_resolve_device_count_reads_processors_from_config(sessions_root):
    # the decay fixture sets processors: 2
    assert fit_parameters.resolve_device_count(sessions_root / "decay_session") == 2


def test_resolve_device_count_falls_back_when_config_is_missing(tmp_path):
    """Must not raise — it runs before the real validation path."""
    empty = tmp_path / "no_config_here"
    (empty / "inputs").mkdir(parents=True)
    assert fit_parameters.resolve_device_count(empty) == (os.cpu_count() or 1)


def test_resolve_device_count_falls_back_on_malformed_config(tmp_path):
    session = tmp_path / "broken"
    (session / "inputs").mkdir(parents=True)
    # unbalanced bracket: not parseable as YAML at all
    (session / "inputs" / "user_input.yaml").write_text("population_opt: [unclosed\n")
    assert fit_parameters.resolve_device_count(session) == (os.cpu_count() or 1)


def test_resolve_device_count_is_at_least_one(sessions_root):
    from tests.conftest import set_setting

    config = sessions_root / "decay_session" / "inputs" / "user_input.yaml"
    set_setting(config, "population_opt", "processors", 0)
    assert fit_parameters.resolve_device_count(sessions_root / "decay_session") >= 1


# ---------------------------------------------------------------------------
# gradient-only seed loading
# ---------------------------------------------------------------------------

def test_load_init_guess_reads_the_previous_design_point(tmp_path):
    session = tmp_path / "s"
    (session / "outputs").mkdir(parents=True)
    np.savetxt(session / "outputs" / "final_design_point.csv",
               np.array([1.5, 0.25]), delimiter=",")
    np.testing.assert_allclose(load_init_guess(session), [1.5, 0.25])


def test_load_init_guess_handles_a_single_parameter(tmp_path):
    """A 1-parameter CSV loads as a 0-d array; it must still come back as 1-d."""
    session = tmp_path / "s"
    (session / "outputs").mkdir(parents=True)
    (session / "outputs" / "final_design_point.csv").write_text("2.5\n")
    guess = load_init_guess(session)
    assert guess.shape == (1,)
    assert guess[0] == pytest.approx(2.5)


def test_load_init_guess_errors_when_no_previous_fit_exists(tmp_path):
    session = tmp_path / "s"
    (session / "outputs").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="No initial guess"):
        load_init_guess(session)
