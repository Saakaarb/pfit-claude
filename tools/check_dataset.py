#!/usr/bin/env python3
"""
Measure every dataset CSV of a session against the structural requirements the
framework imposes but does not enforce.

This tool REPORTS FACTS ONLY. It emits one line per check id (D1..D9) with a
PASS/FAIL verdict and the numbers behind it. The severity of each id, and what
to do about it, are owned by `lib/LLM/reference/validation_rules.md` — do not
duplicate that mapping here.

The loader mirrors `lib/utils/helper_functions.py` exactly: the same
`encoding='utf-8-sig'` open and the same
`np.genfromtxt(dtype=float, delimiter=',')` call, so what this tool sees is what
the fit will see.

Usage:
    ./venv/bin/python3 tools/check_dataset.py <session>
    ./venv/bin/python3 tools/check_dataset.py sessions/robertson_session
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

logger = logging.getLogger("check_dataset")

# jnp/np reductions that tolerate NaN. If the dataset carries NaN, the loss must
# use one of these (see staggered_data.md); a plain mean/sum propagates the NaN
# into the loss, which is then sanitised to error_loss for EVERY candidate.
NAN_SAFE = ("nanmean", "nansum", "nanmax", "nanmin", "nan_to_num", "isnan")


def setup_logging(log_file: str) -> None:
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)):
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(fmt)
        logger.addHandler(handler)


def resolve_session_dir(arg: str) -> Path:
    """Accept either a session name under sessions/ or a path to one."""
    candidate = Path(arg)
    session_dir = candidate if candidate.is_dir() else REPO_ROOT / "sessions" / arg
    if not session_dir.is_dir():
        raise SystemExit(f"session directory not found: {session_dir}")
    return session_dir


def load_like_the_framework(path: Path) -> np.ndarray:
    with open(path, "r", encoding="utf-8-sig") as handle:
        return np.genfromtxt(handle, dtype=float, delimiter=",")


def max_dataset_index(model_src: str) -> int | None:
    """
    Highest literal column index the model reads out of `dataset`.

    Matches `dataset[:, k]` and `dataset[i, k]`. Returns None when the model
    indexes `dataset` only with non-literals, in which case D8 cannot be decided
    statically and is reported as SKIP.
    """
    literals = [int(m) for m in re.findall(r"dataset\s*\[\s*[^,\]]+,\s*(\d+)\s*\]", model_src)]
    return max(literals) if literals else None


def differences_solution_against_dataset(model_src: str) -> bool:
    """
    True when the model subtracts the WHOLE `dataset` from the WHOLE `solution`,
    as in Robertson's `solution - dataset`.

    That form carries a stricter requirement than D8: every observable column
    lines up positionally with a STATE, so the observable count must equal the
    state dimension exactly. A mismatch raises on broadcast, or silently
    broadcasts against every state when one side has width 1.

    Deliberately narrow. A model that builds an observable array first and
    differences THAT (`sim - dataset`, as in boehm_stat5) is unconstrained by the
    state dimension, and matching it here would be a false positive.
    """
    body = "\n".join(
        line.split("#")[0] for line in model_src.splitlines()
        if not line.lstrip().startswith("def ")
    )
    return bool(
        re.search(r"solution\s*(?!\s*[\[.])\s*-\s*dataset\s*(?!\s*\[)", body)
        or re.search(r"dataset\s*(?!\s*\[)\s*-\s*solution\s*(?!\s*[\[.])", body)
    )


def report(check: str, ok: bool | None, message: str) -> bool:
    """
    Log one check line and return True when it failed.

    `ok=None` means the check could not be decided. Note that `ok` is often a
    numpy bool, for which `ok is False` is never true — hence `bool(ok)` rather
    than an identity test.
    """
    if ok is None:
        logger.info("%s SKIP  %s", check, message)
        return False
    if bool(ok):
        logger.info("%s PASS  %s", check, message)
        return False
    logger.error("%s FAIL  %s", check, message)
    return True


def check_experiment(name: str, data: np.ndarray, init_time: float | None,
                     max_idx: int | None, whole: bool = False,
                     n_states: int | None = None) -> tuple[int, int | None]:
    """Run the per-experiment checks. Returns (failures, column count or None)."""
    failures = 0

    # D1 - the array must load 2-D. A single row OR a single column collapses to
    # 1-D, and the framework's `all_data[:, 0]` then raises IndexError.
    if data.ndim != 2:
        report("D1", False, f"{name}: loaded 1-D with shape {data.shape} - a CSV with "
                            "one row or one column crashes the loader at all_data[:, 0]")
        return 1, None
    n_rows, n_cols = data.shape
    failures += report("D1", n_rows >= 2 and n_cols >= 2,
                       f"{name}: rows={n_rows} cols={n_cols} "
                       f"(need >= 2 rows and >= 2 columns: column 0 is time, so "
                       f"{n_cols - 1} observable column(s))")
    if n_cols < 2:
        return failures, n_cols

    # D2 - a trailing comma on every row appends a phantom all-NaN column, which
    # both inflates the observable count and injects NaN (feeding D3).
    trailing = bool(np.all(np.isnan(data[:, -1])))
    failures += report("D2", not trailing,
                       f"{name}: last column all-NaN = {trailing}"
                       + (" - looks like a trailing delimiter on every row" if trailing else ""))

    # D3 - NaN anywhere. Deliberate under staggered_data.md, fatal otherwise; the
    # cross-check against the loss body happens in check_session.
    n_nan = int(np.isnan(data).sum())
    # not counted as a failure here: NaN is legitimate under staggered_data.md.
    # The D3x cross-check against the loss body is what decides it.
    report("D3", n_nan == 0, f"{name}: {n_nan} NaN cell(s)"
           + (" - the loss MUST use nan-safe reductions (see D3 cross-check)" if n_nan else ""))

    t = data[:, 0]
    finite_t = np.isfinite(t).all()
    if not finite_t:
        failures += report("D4", False, f"{name}: time column contains NaN/inf")
        return failures, n_cols

    # D4 - SaveAt(ts=t_eval) requires strictly increasing ts. Non-monotonic time
    # raises inside JIT, where the traceback points at diffrax, not at the CSV.
    diffs = np.diff(t)
    failures += report("D4", bool(np.all(diffs > 0)),
                       f"{name}: time strictly increasing = {bool(np.all(diffs > 0))} "
                       f"(t=[{t[0]:g}, {t[-1]:g}], min step {diffs.min():g})")

    # D5 - duplicate times do NOT raise. The instant is saved twice and silently
    # double-weighted in the mean-square loss.
    n_dup = len(t) - len(np.unique(t))
    failures += report("D5", n_dup == 0,
                       f"{name}: {n_dup} duplicate time value(s)"
                       + (" - each is silently double-weighted in the loss" if n_dup else ""))

    # D6 - a save point before t0 raises inside JIT.
    if init_time is None:
        report("D6", None, f"{name}: no initial_time set, so t0 = t_eval[0] = {t[0]:g}")
    else:
        failures += report("D6", t[0] >= init_time,
                           f"{name}: t_eval[0]={t[0]:g} vs initial_time={init_time:g} "
                           "(a save point before t0 raises inside JIT)")

    # D7 - a degenerate span does NOT raise: the solve returns y0 as the whole
    # trajectory, so the run completes having integrated nothing.
    t0 = t[0] if init_time is None else init_time
    failures += report("D7", t[-1] > t0,
                       f"{name}: t_eval[-1]={t[-1]:g} vs t0={t0:g} "
                       "(a non-positive span returns y0 as the entire trajectory)")

    # D8 - every literal column the loss reads must exist.
    if max_idx is None:
        report("D8", None, f"{name}: model indexes dataset with no literal column index")
    else:
        failures += report("D8", max_idx < n_cols - 1,
                           f"{name}: model reads up to dataset[:, {max_idx}]; "
                           f"dataset has {n_cols - 1} column(s) (indices 0..{n_cols - 2})")

    # D8b - the whole-array form is stricter: one observable per state, in state
    # order. Robertson's `solution - dataset` is the canonical case.
    if whole and n_states is not None:
        failures += report("D8b", n_cols - 1 == n_states,
                           f"{name}: model differences `solution` against the whole `dataset`, so its "
                           f"{n_cols - 1} observable column(s) must equal the state "
                           f"dimension {n_states}, in the same order")
    elif whole:
        report("D8b", None, f"{name}: model differences `solution` against the whole `dataset`, but the state "
                            "dimension could not be read from the config")

    # D11 - which initial-condition regime this experiment is in. Not a failure
    # either way, but the framework never states it and the two behave very
    # differently, so it is reported for every experiment rather than inferred.
    if init_time is None or init_time == t[0]:
        report("D11", True,
               f"{name}: regime 1 - the solve starts at the first save point "
               f"(t0 = t_eval[0] = {t[0]:g}), so solution[0] IS y0 exactly for every "
               "candidate. Any disagreement between an OBSERVED state's init_val and "
               "row 0 below is a constant penalty no parameter can remove")
    else:
        report("D11", True,
               f"{name}: regime 2 - the state evolves over [{init_time:g}, {t[0]:g}] "
               "before the first comparison, so solution[0] is the evolved state, not "
               "y0. init_val and row 0 may legitimately differ")

    logger.info("D10 INFO  %s: data row 0 = %s", name,
                np.array2string(data[0], precision=6))
    logger.debug("%s: median dt=%g, t_span=%g", name,
                 float(np.median(diffs)), float(t[-1] - t[0]))
    return failures, n_cols


def check_session(session_dir: Path) -> int:
    config_path = session_dir / "inputs" / "user_input.yaml"
    if not config_path.is_file():
        raise SystemExit(f"no config at {config_path}")
    with open(config_path) as handle:
        config = yaml.safe_load(handle) or {}

    model_path = session_dir / "generated" / "user_model.py"
    model_src = model_path.read_text() if model_path.is_file() else ""
    if not model_src:
        logger.warning("no user_model.py at %s - D3 cross-check and D8 are skipped",
                       model_path)
    max_idx = max_dataset_index(model_src) if model_src else None
    whole = differences_solution_against_dataset(model_src) if model_src else False
    variables = (config.get("model") or {}).get("integrated_variables", [])
    n_states = len(variables) or None

    # initial_time may sit in either optimizer section; gradient_opt wins because
    # that is the stage whose alignment matters for the reported fit.
    init_time = None
    for section in ("gradient_opt", "population_opt"):
        value = (config.get(section) or {}).get("initial_time")
        if value is not None:
            init_time = float(value)
            break

    experiments = config.get("experiments") or []
    logger.info("session %s: %d experiment(s), initial_time=%s",
                session_dir.name, len(experiments), init_time)

    failures, col_counts, any_nan = 0, [], False
    for index, experiment in enumerate(experiments):
        filename = experiment.get("data_file")
        if not filename:
            logger.error("experiment %d has no data_file", index + 1)
            failures += 1
            continue
        path = session_dir / "inputs" / filename
        if not path.is_file():
            logger.error("experiment %d: missing %s", index + 1, path)
            failures += 1
            continue
        try:
            data = load_like_the_framework(path)
        except ValueError as exc:
            # genfromtxt raises on a ragged file; that is the one structural
            # requirement the loader already enforces, and its message is clear.
            logger.error("D0 FAIL  %s: not rectangular - %s", filename, exc)
            failures += 1
            continue

        exp_failures, n_cols = check_experiment(filename, data, init_time, max_idx,
                                                whole, n_states)
        failures += exp_failures
        if n_cols is not None:
            col_counts.append((filename, n_cols))
        if data.ndim == 2 and np.isnan(data).any():
            any_nan = True

    # D3 cross-check - NaN in the data is only supported when the loss is
    # nan-safe. Neither half is checked anywhere else.
    if any_nan:
        nan_safe = any(token in model_src for token in NAN_SAFE)
        failures += report("D3x", nan_safe,
                           f"dataset carries NaN and the loss uses a nan-safe "
                           f"reduction = {nan_safe} (without one the loss is NaN for "
                           "every candidate and the search flatlines at error_loss)")

    # D9 - all experiments must expose the same columns, in the same order. The
    # column mapping is positional and nothing in the config remaps it.
    if col_counts:
        distinct = sorted({count for _, count in col_counts})
        failures += report("D9", len(distinct) == 1,
                           f"column counts across experiments: {distinct}"
                           + ("" if len(distinct) == 1 else
                              f" - {col_counts}; the mapping is positional, so a "
                              "mismatch fits a different observable per file"))

    # D10 - reported, never judged. The config's initial conditions sit beside
    # each dataset's row 0 (printed per experiment above) so they can be compared
    # directly. Which column maps to which variable is known only to the loss
    # body, so the agent, not this tool, decides whether they agree - and only
    # for states the loss actually compares against a column.
    if variables:
        logger.info("D10 INFO  init_val per integrated variable: %s",
                    {v["name"]: v.get("init_val") for v in variables})
        overrides = [(index + 1, experiment["initial_conditions"])
                     for index, experiment in enumerate(experiments)
                     if experiment.get("initial_conditions")]
        for index, override in overrides:
            logger.info("D10 INFO  experiment %d overrides: %s", index, override)

    logger.info("%s: %d check(s) failed", session_dir.name, failures)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", help="session name under sessions/, or a path to one")
    parser.add_argument("--log-file", default=str(SCRIPT_DIR / "check_dataset.log"),
                        help="path to the log file")
    args = parser.parse_args()

    setup_logging(args.log_file)
    session_dir = resolve_session_dir(args.session)
    return 1 if check_session(session_dir) else 0


if __name__ == "__main__":
    sys.exit(main())
