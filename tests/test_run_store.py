"""Run isolation, snapshot execution, seed lineage, and explicit plot selection."""
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from lib.utils.run_store import new_run, resolve_run, run_config
from lib.utils.yamlread import read_input_file


def test_snapshots_survive_edits_and_failed_runs(make_session):
    session = make_session("decay_session")
    reader = read_input_file(session / "inputs/user_input.yaml")
    original = (session / "inputs/user_input.yaml").read_bytes()
    with new_run(session, reader, "full") as (first, snapshot):
        (first / "final_design_point.csv").write_text("1\n0.3\n")
        assert (snapshot / "inputs/user_input.yaml").read_bytes() == original
        execution = read_input_file(run_config(first, session))
        assert (snapshot / "inputs" / execution.experiments[0]["filename"]).is_file()
    (session / "generated/generated_script.py").write_text("raise RuntimeError('edited')")
    with pytest.raises(RuntimeError, match="interrupted solve"):
        with new_run(session, reader, "full") as (second, _):
            (second / "final_design_point.csv").write_text("99\n99\n")
            raise RuntimeError("interrupted solve")
    assert first != second
    assert resolve_run(session) == second
    assert resolve_run(session, require_seed=True) == first
    assert json.loads((second / "run_manifest.json").read_text())["status"] == "failed"
    manifest = json.loads((first / "run_manifest.json").read_text())
    for relative, digest in manifest["sha256"].items():
        assert hashlib.sha256((first / relative).read_bytes()).hexdigest() == digest


def test_driver_uses_snapshot_and_preserves_previous_run(make_session, monkeypatch):
    import fit_parameters
    import fit_gradient_only
    import lib.utils.helper_functions as helpers
    session = make_session("decay_session")
    reader = read_input_file(session / "inputs/user_input.yaml")

    def full(config, output, generated, sources):
        assert config == output / "snapshot/inputs/run_config.yaml"
        assert generated == output / "snapshot/generated"
        assert sources == output / "snapshot"
        (output / "final_design_point.csv").write_text("1\n0.3\n")
        (output / "pso_fitting.log").write_text("first run\n")
        print("snapshot fit executed")
        return np.array([1, .3])

    def gradient(config, output, generated, sources, seed):
        np.testing.assert_allclose(seed, [1, .3])
        assert (output / "seed_design_point.csv").is_file()
        assert not (output / "pso_fitting.log").exists()
        (output / "final_design_point.csv").write_text("1.1\n0.31\n")
        return np.array([1.1, .31])

    monkeypatch.setattr(helpers, "fit_generic_system", full)
    monkeypatch.setattr(helpers, "fit_gradient_only_system", gradient)
    fit_parameters.run_driver(session, reader)
    first = resolve_run(session)
    before = {p.relative_to(first): p.read_bytes() for p in first.rglob("*")
              if p.is_file()}
    fit_gradient_only.run_driver(session, reader, seed_run=first.name)
    second = resolve_run(session)
    assert first != second
    for relative, content in before.items():
        assert (first / relative).read_bytes() == content
    manifest = json.loads((second / "run_manifest.json").read_text())
    assert manifest["seed_source"] == str(first / "final_design_point.csv")
    fit_parameters.run_driver(session, reader)
    assert len(list((session / "outputs").glob("*/run_manifest.json"))) == 3


def test_plot_explicit_old_run_writes_inside_that_run(make_session):
    from tools.plot_fits import plot_session
    session = make_session("decay_session")
    reader = read_input_file(session / "inputs/user_input.yaml")
    with new_run(session, reader, "full") as (first, _):
        np.savetxt(first / "result_solution_exp1.csv", [[0, 1, 1], [1, .5, .4]], delimiter=",")
    with new_run(session, reader, "full") as (second, _):
        pass
    style = dict(surface="white", color_data="black", color_fit="blue",
                 color_ink="black", color_ink_muted="gray")
    output = plot_session(dict(name="decay", root=str(session), run=first.name,
                               panels=[dict(label="A", data_col=1, sim_col=2)]),
                          style, "unused", 50)
    assert Path(output) == first / "decay_fit.png"
    assert Path(output).is_file()
    assert not (second / "decay_fit.png").exists()


def test_custom_output_directory_and_interruption(make_session):
    from tests.conftest import _rewrite_config
    session = make_session("decay_session")
    _rewrite_config(session / "inputs/user_input.yaml",
                    lambda c: c.update(paths={"output_dir": "history"}))
    reader = read_input_file(session / "inputs/user_input.yaml")
    with pytest.raises(KeyboardInterrupt):
        with new_run(session, reader, "full") as (run, _):
            raise KeyboardInterrupt()
    assert run.parent == session / "history"
    assert resolve_run(session) == run
    assert json.loads((run / "run_manifest.json").read_text())["status"] == "interrupted"


def test_analysis_uses_archived_inputs_after_working_files_change(make_session, monkeypatch):
    import analyze_fit
    import sys
    session = make_session("decay_session")
    reader = read_input_file(session / "inputs/user_input.yaml")
    with new_run(session, reader, "full") as (run, snapshot):
        (run / "final_design_point.csv").write_text("1\n0.3\n")
    (session / "generated/generated_script.py").write_text("raise RuntimeError('wrong model')")
    (session / "inputs/decay_data.csv").write_text("0,999,999\n1,999,999\n")
    observed = []

    def inspect_analysis(loss, constants, point, names, output):
        assert Path(inspect.unwrap(loss).__code__.co_filename) == snapshot / "generated/generated_script.py"
        assert float(constants[0]["dataset"][0, 0]) != 999
        assert names == ["k1", "k2"]
        assert output == run
        observed.append(True)

    monkeypatch.setattr(analyze_fit, "run_sloppiness_analysis", inspect_analysis)
    monkeypatch.setattr(sys, "argv", ["analyze_fit.py", str(session), "--run", run.name])
    analyze_fit.main()
    assert observed
