"""Fast telemetry checks with quadratic objectives, never ODE fits."""
import json
import os
import signal
import time
from types import SimpleNamespace
import threading
from urllib.error import HTTPError
from urllib.request import urlopen, Request

import numpy as np
import pytest

from lib.utils.live_dashboard import ProgressState, make_server, launch_dashboard
from lib.utils.live_progress import emit_progress
from lib.utils.yamlread import read_input_file


def event(stage, iteration, loss, kind="iteration"):
    return dict(stage=stage, iteration=iteration, loss=loss, kind=kind,
                time="2026-09-12T12:00:00+00:00", parameters={"k": 1.0})


def test_handoff_is_continuous_and_does_not_change_losses(tmp_path):
    path = tmp_path / "live_progress.jsonl"
    records = [event("global", 1, 10), event("global", 2, 1),
               event("gradient", 0, None, "stage_start"),
               event("gradient", 0, 1.1), event("gradient", 1, .1)]
    path.write_text("".join(json.dumps(e) + "\n" for e in records))
    state = ProgressState(tmp_path)
    data = state.read()
    assert [p["x"] for p in data["points"]] == [1, 2, 2, 3]
    assert [p["loss"] for p in data["points"]] == [10, 1, 1.1, .1]
    assert data["handoff"] == 2
    assert state.read()["points"] == data["points"]  # no duplicated refresh data


def test_incomplete_line_is_not_consumed_and_zero_is_preserved(tmp_path):
    path = tmp_path / "live_progress.jsonl"
    line = json.dumps(event("gradient", 0, 0))
    path.write_text(line[:20])
    state = ProgressState(tmp_path)
    assert state.read()["points"] == []
    with path.open("a") as handle:
        handle.write(line[20:] + "\n")
    assert state.read()["points"][0] == dict(stage="gradient", iteration=0, x=0, loss=0)


def test_progress_in_physical_units_and_io_failure_is_nonfatal(tmp_path):
    optimizer = SimpleNamespace(input_reader=SimpleNamespace(
        output_dir=tmp_path, trainable_parameter_names=["a", "b"]),
        unscale_design_point=lambda p: 10.0 ** p)
    emit_progress(optimizer, "global", 1, .5, [1, -2])
    record = json.loads((tmp_path / "live_progress.jsonl").read_text())
    assert record["parameters"] == {"a": 10, "b": .01}
    optimizer.input_reader.output_dir = tmp_path / "missing"
    emit_progress(optimizer, "global", 2, .4, [1, -2])
    assert optimizer._progress_warning


@pytest.mark.parametrize("algorithm", ["PSO", "DE", "adam", "lbfgs"])
def test_real_optimizer_publishes_its_evaluated_best(make_session, algorithm):
    import jax.numpy as jnp
    from lib.algorithms.PSO.classes import FitParamsPSO
    from lib.algorithms.DE.classes import FitParamsDE
    from lib.algorithms.NODE.classes import FitParamsNODE
    session = make_session("decay_session")
    reader = read_input_file(session / "inputs/user_input.yaml")
    reader.output_dir = session / "telemetry_test"
    reader.output_dir.mkdir()
    reader.n_iters_pop = reader.n_iters_grad = 3
    reader.population_size = 8
    reader.random_seed = 1
    problem = SimpleNamespace(compute_all_losses=lambda p: np.sum(p*p, axis=1),
                              _compute_all_losses=lambda p: np.sum(p*p, axis=1),
                              _compute_loss=lambda p: jnp.sum(p*p))
    if algorithm == "PSO":
        optimizer = FitParamsPSO(reader, problem)
        for i in range(3):
            optimizer.search_iteration(i, None)
        expected = optimizer.unscale_design_point(optimizer.swarm_obj.best_pos)
    elif algorithm == "DE":
        optimizer = FitParamsDE(reader, problem)
        position, _ = optimizer.run(reader.output_dir / "de.log")
        expected = optimizer.unscale_design_point(position)
    else:
        reader.gradient_optimizer = algorithm
        optimizer = FitParamsNODE(reader, problem, init_guess=np.array([1., .3]))
        position, _ = optimizer.train_NODE()
        expected = optimizer.unscale_design_point(np.asarray(position))
    data = ProgressState(reader.output_dir).read()
    np.testing.assert_allclose(list(data["latest"]["parameters"].values()), expected)
    assert len(data["points"]) == 3
    if algorithm in ("adam", "lbfgs"):
        assert data["points"][0]["iteration"] == 0  # seed evaluation is not lost


def test_server_is_read_only_and_does_not_serve_run_files(tmp_path):
    (tmp_path / "secret.txt").write_text("not exposed")
    server = make_server(tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        assert server.server_address[0] == "127.0.0.1"
        with urlopen(url + "/") as response:
            assert b"Loss evolution" in response.read()
        with urlopen(url + "/api/progress") as response:
            assert json.load(response)["run_id"] == tmp_path.name
        for path in ("/secret.txt", "/../secret.txt"):
            with pytest.raises(HTTPError) as error:
                urlopen(url + path)
            assert error.value.code == 404
        with pytest.raises(HTTPError) as error:
            urlopen(Request(url + "/api/progress", data=b"stop", method="POST"))
        assert error.value.code == 501
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_automatic_viewer_launch_and_stop(tmp_path):
    info = launch_dashboard(tmp_path)
    try:
        with urlopen(info["url"] + "api/progress", timeout=5) as response:
            assert json.load(response)["run_id"] == tmp_path.name
    finally:
        os.kill(info["pid"], signal.SIGTERM)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if json.loads((tmp_path / "live_server.json").read_text())["status"] == "stopped":
                break
            time.sleep(.05)
        os.waitpid(info["pid"], 0)
    assert json.loads((tmp_path / "live_server.json").read_text())["status"] == "stopped"
