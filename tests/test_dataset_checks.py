"""
Regression tests for tools/check_dataset.py.

Each dataset check exists because the framework accepts the violation without
complaint. The tests below therefore assert on the CHECK, not on the framework:
a check that stops firing is a silent-wrong-answer mode reopening.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOL = REPO_ROOT / "tools" / "check_dataset.py"
VENV_PYTHON = REPO_ROOT / "venv" / "bin" / "python3"
PYTHON = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

GOOD_CSV = "0,1.0,0.0\n1,0.5,0.5\n2,0.2,0.8\n"

BASE_CONFIG = """\
experiments:
{experiments}
model:
  trainable_parameters:
    - {{name: k1, min_val: 0.1, max_val: 10.0, logscale: false}}
  integrated_variables:
    - {{name: y1, init_val: 1}}
    - {{name: y2, init_val: 0}}
{extra}
"""

COLUMNWISE_MODEL = """\
def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
        a = dataset[:, 0]
        b = dataset[:, 1]
        return np.sqrt(np.mean(np.square(solution[:, 0] - a) + np.square(solution[:, 1] - b)))
"""

WHOLE_ARRAY_MODEL = """\
def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
        return np.sqrt(np.mean(np.square(solution - dataset)))
"""


def build_session(tmp_path, csvs, model=COLUMNWISE_MODEL, extra="", variables=None):
    """Create a minimal session directory and return its path."""
    session = tmp_path / "session"
    (session / "inputs").mkdir(parents=True)
    (session / "generated").mkdir(parents=True)

    for name, content in csvs.items():
        (session / "inputs" / name).write_text(content)

    experiments = "".join(f"  - data_file: {name}\n" for name in csvs)
    config = BASE_CONFIG.format(experiments=experiments, extra=extra)
    if variables is not None:
        block = "".join(f"    - {{name: {v}, init_val: 0}}\n" for v in variables)
        head, _, tail = config.partition("  integrated_variables:\n")
        # replace only the variable list, keeping any trailing sections
        rest = tail.split("\n", len(variables))[-1] if variables else tail
        config = head + "  integrated_variables:\n" + block + rest
    (session / "inputs" / "user_input.yaml").write_text(config)
    (session / "generated" / "user_model.py").write_text(model)
    return session


def run_tool(session, tmp_path):
    result = subprocess.run(
        [PYTHON, str(TOOL), str(session), "--log-file", str(tmp_path / "check.log")],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    return result.returncode, result.stdout


def failed_ids(output):
    """Check ids that reported FAIL. Log lines are '<ts> [LEVEL] <id> FAIL ...'."""
    ids = set()
    for line in output.splitlines():
        tokens = line.split()
        if "FAIL" in tokens:
            ids.add(tokens[tokens.index("FAIL") - 1])
    return ids


def test_a_clean_session_passes(tmp_path):
    session = build_session(tmp_path, {"good.csv": GOOD_CSV})
    code, out = run_tool(session, tmp_path)
    assert code == 0, out
    assert not failed_ids(out), out


@pytest.mark.parametrize("check,csv", [
    # D1 - a single row collapses to 1-D and crashes the loader at all_data[:, 0]
    ("D1", "0,1.0,0.0\n"),
    # D2 - a trailing delimiter appends a phantom all-NaN column
    ("D2", "0,1.0,\n1,0.5,\n2,0.2,\n"),
    # D4 - non-monotonic time raises inside JIT, far from the CSV
    ("D4", "0,1.0,0.0\n2,0.5,0.5\n1,0.2,0.8\n"),
    # D5 - a duplicate timestamp is silently double-weighted in the loss
    ("D5", "0,1.0,0.0\n1,0.5,0.5\n1,0.2,0.8\n"),
    # D7 - a degenerate span returns y0 as the entire trajectory, without error
    ("D7", "5,1.0,0.0\n5,0.5,0.5\n"),
])
def test_structural_violations_are_caught(tmp_path, check, csv):
    session = build_session(tmp_path, {"bad.csv": csv})
    code, out = run_tool(session, tmp_path)
    assert check in failed_ids(out), f"{check} did not fire:\n{out}"
    assert code == 1


def test_initial_time_after_the_first_data_point_is_caught(tmp_path):
    """D6 - a save point before t0 raises inside JIT."""
    session = build_session(tmp_path, {"good.csv": GOOD_CSV},
                            extra="gradient_opt:\n  initial_time: 1.5\n")
    code, out = run_tool(session, tmp_path)
    assert "D6" in failed_ids(out), out
    assert code == 1


def test_nan_without_a_nan_safe_loss_is_caught(tmp_path):
    """D3x - the failure mode that flatlines the search with no error at all."""
    session = build_session(tmp_path, {"gappy.csv": "0,1.0,0.0\n1,,0.5\n2,0.2,0.8\n"})
    code, out = run_tool(session, tmp_path)
    assert "D3x" in failed_ids(out), out
    assert code == 1


def test_nan_with_a_nan_safe_loss_is_accepted(tmp_path):
    """The staggered-data workflow is legitimate and must not be flagged."""
    nan_safe = COLUMNWISE_MODEL.replace("np.mean", "np.nanmean")
    session = build_session(tmp_path, {"gappy.csv": "0,1.0,0.0\n1,,0.5\n2,0.2,0.8\n"},
                            model=nan_safe)
    code, out = run_tool(session, tmp_path)
    assert "D3x" not in failed_ids(out), out
    assert code == 0


def test_a_column_the_model_reads_but_the_csv_lacks_is_caught(tmp_path):
    """D8 - the model reads dataset[:, 1] from a single-observable CSV."""
    session = build_session(tmp_path, {"narrow.csv": "0,1.0\n1,0.5\n2,0.2\n"})
    code, out = run_tool(session, tmp_path)
    assert "D8" in failed_ids(out), out
    assert code == 1


def test_whole_array_difference_requires_one_column_per_state(tmp_path):
    """D8b - `solution - dataset` binds the observable count to the state count."""
    session = build_session(tmp_path, {"narrow.csv": "0,1.0\n1,0.5\n2,0.2\n"},
                            model=WHOLE_ARRAY_MODEL)
    code, out = run_tool(session, tmp_path)
    assert "D8b" in failed_ids(out), out
    assert code == 1


def test_whole_array_difference_is_not_flagged_when_widths_agree(tmp_path):
    session = build_session(tmp_path, {"good.csv": GOOD_CSV}, model=WHOLE_ARRAY_MODEL)
    code, out = run_tool(session, tmp_path)
    assert "D8b" not in failed_ids(out), out
    assert code == 0


def test_a_derived_observable_array_is_not_mistaken_for_the_whole_state(tmp_path):
    """
    boehm_stat5 differences a CONSTRUCTED observable array against the dataset.
    That form is unconstrained by the state dimension, so D8b must stay silent.
    """
    model = (
        "def _compute_loss_problem(solution_time, solution, dataset, "
        "trainable_parameters, fixed_parameters):\n"
        "        sim = np.stack([solution[:, 0], solution[:, 1]], axis=1)\n"
        "        return np.sqrt(np.mean(np.square(sim - dataset)))\n"
    )
    session = build_session(tmp_path, {"good.csv": GOOD_CSV}, model=model)
    code, out = run_tool(session, tmp_path)
    assert "D8b" not in failed_ids(out), out
    assert code == 0


def test_mismatched_column_counts_across_experiments_are_caught(tmp_path):
    """D9 - the mapping is positional, so a mismatch fits a different observable."""
    session = build_session(tmp_path, {
        "one.csv": GOOD_CSV,
        "two.csv": "0,1.0,0.0,9.0\n1,0.5,0.5,9.0\n2,0.2,0.8,9.0\n",
    })
    code, out = run_tool(session, tmp_path)
    assert "D9" in failed_ids(out), out
    assert code == 1


def test_every_shipped_session_passes(tmp_path):
    """
    The dataset checks describe latent risks, not current breakage. If a session
    in the repo starts failing, either the data changed or a check regressed.
    """
    sessions = sorted(
        d for d in (REPO_ROOT / "sessions").iterdir()
        if (d / "inputs" / "user_input.yaml").is_file()
    )
    assert sessions, "no sessions found"
    broken = []
    for session in sessions:
        code, out = run_tool(session, tmp_path)
        if code != 0:
            broken.append(f"{session.name}: {sorted(failed_ids(out))}")
    assert not broken, "sessions failing the dataset checks:\n" + "\n".join(broken)
