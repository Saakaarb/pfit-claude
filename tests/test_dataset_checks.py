"""
Regression tests for tools/check_dataset.py.

Each dataset check exists because the framework accepts the violation without
complaint. The tests below therefore assert on the CHECK, not on the framework:
a check that stops firing is a silent-wrong-answer mode reopening.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
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


# ---------------------------------------------------------------------------
# column declarations: the CSV is a bare numeric matrix, so `columns` is the
# only record of what each column means
# ---------------------------------------------------------------------------

COLUMNS_2 = """    columns:
      - {name: time, units: s}
      - {name: A, observes: y1}
      - {name: B, observes: y2}
"""


def build_declared(tmp_path, csv, columns=COLUMNS_2, model=COLUMNWISE_MODEL, extra=""):
    session = tmp_path / "declared"
    (session / "inputs").mkdir(parents=True)
    (session / "generated").mkdir(parents=True)
    (session / "inputs" / "d.csv").write_text(csv)
    (session / "inputs" / "user_input.yaml").write_text(
        "experiments:\n  - data_file: d.csv\n" + columns +
        "model:\n"
        "  trainable_parameters:\n"
        "    - {name: k1, min_val: 0.1, max_val: 10.0, logscale: false}\n"
        "  integrated_variables:\n"
        "    - {name: y1, init_val: 1.0}\n"
        "    - {name: y2, init_val: 0.0}\n" + extra)
    (session / "generated" / "user_model.py").write_text(model)
    return session


def test_declared_column_count_must_match_the_file(tmp_path):
    """D12 - a column added to the CSV shifts every index the loss uses."""
    session = build_declared(tmp_path, "0,1.0,0.0,9.0\n1,0.5,0.5,9.0\n")
    code, out = run_tool(session, tmp_path)
    assert "D12" in failed_ids(out), out
    assert code == 1


def test_a_header_is_skipped_and_checked_against_the_declaration(tmp_path):
    """D13 - a header is a second statement of the column map; it must agree."""
    session = build_declared(tmp_path, "time,A,B\n0,1.0,0.0\n1,0.5,0.5\n")
    code, out = run_tool(session, tmp_path)
    assert not failed_ids(out), out
    assert code == 0


def test_a_header_that_disagrees_with_the_declaration_is_caught(tmp_path):
    session = build_declared(tmp_path, "time,B,A\n0,1.0,0.0\n1,0.5,0.5\n")
    code, out = run_tool(session, tmp_path)
    assert "D13" in failed_ids(out), out


def test_a_headered_file_loads_to_the_same_array_as_a_bare_one(tmp_path):
    """The header must not reach the solver as data."""
    from lib.utils.dataset_io import load_dataset
    bare, headed = tmp_path / "a.csv", tmp_path / "b.csv"
    bare.write_text("0,1.0,0.0\n1,0.5,0.5\n")
    headed.write_text("time,A,B\n0,1.0,0.0\n1,0.5,0.5\n")
    assert np.array_equal(load_dataset(bare), load_dataset(headed))


def test_an_observed_state_disagreeing_with_row_0_is_reported(tmp_path):
    """D10 - automatable only because `observes` declares the column map."""
    session = build_declared(tmp_path, "0,0.5,0.0\n1,0.2,0.5\n")  # y1 init_val is 1.0
    code, out = run_tool(session, tmp_path)
    assert "D10 WARN" in out, out
    assert code == 0, "D10 is a warning, not a failure"


def test_an_observed_state_agreeing_with_row_0_is_not_reported(tmp_path):
    session = build_declared(tmp_path, "0,1.0,0.0\n1,0.5,0.5\n")
    code, out = run_tool(session, tmp_path)
    assert "D10 WARN" not in out, out


def test_a_declared_observable_the_model_does_not_define_is_caught(tmp_path):
    """D14 - the column names a model quantity that does not exist."""
    session = build_declared(
        tmp_path, GOOD_CSV,
        columns="    columns:\n      - {name: time}\n"
                "      - {name: A, observes: y1}\n      - {name: q, observes: Po}\n",
        extra="")
    cfg = session / "inputs" / "user_input.yaml"
    cfg.write_text(cfg.read_text().replace(
        "  integrated_variables:", "  observables:\n    - {name: Po}\n  integrated_variables:"))
    code, out = run_tool(session, tmp_path)
    assert "D14" in failed_ids(out), out
    assert code == 1


def test_declared_and_returned_observables_must_match(tmp_path):
    session = build_declared(
        tmp_path, GOOD_CSV,
        columns="    columns:\n      - {name: time}\n"
                "      - {name: A, observes: y1}\n      - {name: q, observes: Po}\n",
        model=COLUMNWISE_MODEL + '\n\ndef _observables(solution, trainable_parameters, '
              'fixed_parameters):\n        return {"WrongName": solution[:, 0]}\n')
    cfg = session / "inputs" / "user_input.yaml"
    cfg.write_text(cfg.read_text().replace(
        "  integrated_variables:", "  observables:\n    - {name: Po}\n  integrated_variables:"))
    code, out = run_tool(session, tmp_path)
    assert "D14" in failed_ids(out), out


def test_matching_observables_pass(tmp_path):
    session = build_declared(
        tmp_path, GOOD_CSV,
        columns="    columns:\n      - {name: time}\n"
                "      - {name: A, observes: y1}\n      - {name: q, observes: Po}\n",
        model=COLUMNWISE_MODEL + '\n\ndef _observables(solution, trainable_parameters, '
              'fixed_parameters):\n        return {"Po": solution[:, 0]}\n')
    cfg = session / "inputs" / "user_input.yaml"
    cfg.write_text(cfg.read_text().replace(
        "  integrated_variables:", "  observables:\n    - {name: Po}\n  integrated_variables:"))
    code, out = run_tool(session, tmp_path)
    assert not failed_ids(out), out
    assert code == 0


# ---------------------------------------------------------------------------
# source stamping: the fit imports generated_script.py and never reads
# user_model.py, so nothing else can tell whether the two still agree
# ---------------------------------------------------------------------------

from lib.utils.source_stamp import (  # noqa: E402
    normalized_hash, read_stamp, verify_stamp, write_stamp)


def build_stamped_session(tmp_path, model_body="        return 1.0\n"):
    session = tmp_path / "stamped"
    (session / "inputs").mkdir(parents=True)
    (session / "generated").mkdir(parents=True)
    (session / "inputs" / "user_input.yaml").write_text("experiments:\n  - data_file: d.csv\n")
    (session / "generated" / "user_model.py").write_text(
        "def _compute_loss_problem(a, b, dataset, c, d):\n" + model_body)
    (session / "generated" / "generated_script.py").write_text(
        "import jax\n\ndef _compute_loss_problem(constants, params):\n    return 1.0\n")
    write_stamp(session)
    return session


def test_a_fresh_stamp_verifies(tmp_path):
    ok, detail = verify_stamp(build_stamped_session(tmp_path))
    assert ok is True, detail


def test_editing_the_model_breaks_the_stamp(tmp_path):
    session = build_stamped_session(tmp_path)
    model = session / "generated" / "user_model.py"
    model.write_text(model.read_text().replace("return 1.0", "return 2.0"))
    ok, detail = verify_stamp(session)
    assert ok is False and "user_model.py" in detail


def test_editing_the_config_breaks_the_stamp(tmp_path):
    session = build_stamped_session(tmp_path)
    config = session / "inputs" / "user_input.yaml"
    config.write_text(config.read_text() + "gradient_opt:\n  max_steps: 99\n")
    ok, detail = verify_stamp(session)
    assert ok is False and "user_input.yaml" in detail


def test_a_comment_only_edit_does_not_break_the_stamp(tmp_path):
    """Timestamps flagged these; content should not. Otherwise the check gets
    ignored for crying wolf."""
    session = build_stamped_session(tmp_path)
    model = session / "generated" / "user_model.py"
    model.write_text(model.read_text() + "\n# explanatory note\n\n")
    ok, detail = verify_stamp(session)
    assert ok is True, detail


def test_touching_the_script_does_not_make_a_stale_one_verify(tmp_path):
    """The failure mode mtime comparison cannot see."""
    session = build_stamped_session(tmp_path)
    model = session / "generated" / "user_model.py"
    model.write_text(model.read_text().replace("return 1.0", "return 5.0"))
    (session / "generated" / "generated_script.py").touch()
    ok, _ = verify_stamp(session)
    assert ok is False


def test_an_unstamped_script_is_undecided_rather_than_passing(tmp_path):
    session = build_stamped_session(tmp_path)
    script = session / "generated" / "generated_script.py"
    script.write_text("\n".join(
        l for l in script.read_text().splitlines() if not l.startswith("# pfit-sources:")))
    ok, detail = verify_stamp(session)
    assert ok is None and "no source stamp" in detail


def test_stamping_twice_does_not_accumulate_lines(tmp_path):
    session = build_stamped_session(tmp_path)
    write_stamp(session)
    write_stamp(session)
    script = (session / "generated" / "generated_script.py").read_text()
    assert script.count("# pfit-sources:") == 1
    assert read_stamp(session / "generated" / "generated_script.py") is not None


def test_the_hash_ignores_comments_and_blank_lines(tmp_path):
    a = tmp_path / "a.py"; b = tmp_path / "b.py"
    a.write_text("x = 1\ny = 2\n")
    b.write_text("# lead\nx = 1   # trailing\n\n\ny = 2\n")
    assert normalized_hash(a) == normalized_hash(b)


# ---------------------------------------------------------------------------
# readiness: gradient-only requires a stored design point to seed from
# ---------------------------------------------------------------------------

READY_TOOL = REPO_ROOT / "tools" / "check_ready.py"


def build_runnable_session(tmp_path, with_seed: bool):
    session = tmp_path / "runnable"
    (session / "inputs").mkdir(parents=True)
    (session / "generated").mkdir(parents=True)
    (session / "outputs").mkdir(parents=True)
    (session / "inputs" / "user_input.yaml").write_text("experiments:\n  - data_file: d.csv\n")
    (session / "generated" / "user_model.py").write_text("x = 1\n")
    (session / "generated" / "generated_script.py").write_text("y = 2\n")
    (session / "outputs" / "de_fitting.log").write_text("1, 1.0, 0.1\n")
    if with_seed:
        (session / "outputs" / "final_design_point.csv").write_text("1.0,2.0\n")
    return session


def run_ready(session, tmp_path, mode="full"):
    result = subprocess.run(
        [PYTHON, str(READY_TOOL), str(session), "--mode", mode,
         "--log-file", str(tmp_path / "ready.log")],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    return result.returncode, result.stdout


def test_gradient_only_without_a_seed_is_blocked(tmp_path):
    """fit_gradient_only.py raises FileNotFoundError without it; better to say
    so in the pre-flight than in a traceback."""
    session = build_runnable_session(tmp_path, with_seed=False)
    code, out = run_ready(session, tmp_path, mode="gradient-only")
    assert "R6" in failed_ids(out), out
    assert code == 1


def test_gradient_only_with_a_seed_is_allowed(tmp_path):
    session = build_runnable_session(tmp_path, with_seed=True)
    code, out = run_ready(session, tmp_path, mode="gradient-only")
    assert "R6" not in failed_ids(out), out


def test_a_full_fit_does_not_require_a_seed(tmp_path):
    """The same session that blocks gradient-only must still allow a full fit."""
    session = build_runnable_session(tmp_path, with_seed=False)
    code, out = run_ready(session, tmp_path, mode="full")
    assert "R6" not in failed_ids(out), out
