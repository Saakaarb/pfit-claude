#!/usr/bin/env python3
"""
Measure every dataset CSV of a session against the structural requirements the
framework imposes but does not enforce.

This tool REPORTS FACTS ONLY. It emits one PASS/FAIL/SKIP line per check id
(D1..D15) with the numbers behind it, then an L1..L4 review of how the loss is
constructed. The severity of each id, and what to do about it, are owned by
`lib/LLM/reference/validation_rules.md` — do not duplicate that mapping here.

The loader is `lib/utils/dataset_io.load_dataset`, the same function the fit
uses, so what this tool sees is exactly what the fit will see.

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
sys.path.insert(0, str(REPO_ROOT))

from lib.utils.dataset_io import (  # noqa: E402
    has_trailing_delimiter, load_dataset, read_header)

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


def observable_keys(model_src: str) -> set | None:
    """
    The keys `_observables` returns, or None when the model does not define it.

    Read from the dict literal rather than by executing anything: the model is
    pseudocode and is never run as-is.
    """
    if "def _observables" not in model_src:
        return None
    body = model_src.split("def _observables", 1)[1].split("\ndef ", 1)[0]
    return set(re.findall(r"""["'](\w+)["']\s*:""", body))


def check_observables(declared: list, model_src: str, referenced: set) -> int:
    """
    D14/D15 - the three-way agreement between config, model and columns.

    This is the whole point of naming observables: each link is an exact string
    match, so none of it depends on reading the loss body's arithmetic.
    """
    failures = 0
    keys = observable_keys(model_src)

    if not declared and keys is None:
        report("D14", None, "no observables declared and none defined "
                            "(every measured column observes a state directly)")
    elif keys is None:
        failures += report("D14", False,
                           f"config declares observables {sorted(declared)} but "
                           f"user_model.py defines no _observables")
    else:
        failures += report("D14", set(declared) == keys,
                           f"declared {sorted(declared)} vs returned by "
                           f"_observables {sorted(keys)}")

    # D15 - a declared observable no column measures is dead weight, and usually
    # means a column's `observes` was left off.
    unused = sorted(set(declared) - referenced)
    failures += report("D15", not unused,
                       f"declared observable(s) not referenced by any column: {unused}"
                       if unused else
                       "every declared observable is referenced by a column")
    return failures


def check_declaration(name: str, data: np.ndarray, columns: list, path: Path) -> int:
    """
    D12/D13 - the config's `columns` block against the file it describes.

    The CSV is a bare numeric matrix, so nothing in it says what column 2 is.
    `columns` is the user's statement of that, and these two checks are the only
    thing keeping the statement honest as the data changes underneath it.
    """
    failures = 0

    # D12 - a column added to or removed from the CSV shifts every index the
    # loss uses. Positional mapping makes that silent; this makes it loud.
    declared, actual = len(columns), (data.shape[1] if data.ndim == 2 else 0)
    failures += report("D12", declared == actual,
                       f"{name}: config declares {declared} column(s), file has {actual}"
                       + ("" if declared == actual else
                          f" - declared {[c['name'] for c in columns]}"))

    # D13 - a header is optional, but when present it is a second statement of
    # the same thing and the two must not drift.
    header = read_header(path)
    if header is None:
        report("D13", None, f"{name}: no header row; `columns` is the only column map")
    else:
        expected = [c["name"] for c in columns]
        matches = len(header) == len(expected) and all(
            h.strip().lower() == e.lower() for h, e in zip(header, expected))
        failures += report("D13", matches,
                           f"{name}: header {header} vs declared {expected}")
    return failures


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
                     n_states: int | None = None,
                     path: Path | None = None,
                     n_declared: int | None = None) -> tuple[int, int | None]:
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
    # both inflates the observable count and injects NaN (feeding D3). Decided
    # from the raw text: an all-NaN last column is NOT evidence of this, because
    # an observable absent from one record of a multi-record set is expressed
    # exactly that way.
    # A row ending in a delimiter is textually IDENTICAL whether it is a typo or
    # a genuinely unmeasured final observable, so the raw text cannot separate
    # them. What separates them is arity: a typo yields one more column than the
    # config declares, an unmeasured observable yields exactly as many.
    trailing = has_trailing_delimiter(path) if path is not None else False
    phantom = trailing and n_declared is not None and n_cols > n_declared
    if n_declared is None:
        report("D2", None if not trailing else False,
               f"{name}: rows end with a delimiter = {trailing}; with no column "
               "declaration to compare against, this may be a phantom column or "
               "a legitimately unmeasured last observable")
        failures += 1 if trailing else 0
    else:
        failures += report("D2", not phantom,
                           f"{name}: rows end with a delimiter = {trailing}, file has "
                           f"{n_cols} column(s) against {n_declared} declared"
                           + (" - the extra one is a phantom from the trailing delimiter"
                              if phantom else ""))

    empty = [i for i in range(n_cols) if bool(np.all(np.isnan(data[:, i])))]
    if empty:
        logger.info("D2 INFO  %s: column(s) %s hold no measurements at all - "
                    "valid when that observable was not recorded in this record, "
                    "and skipped by a nan-safe loss", name, empty)

    # D3 - NaN anywhere. Deliberate under staggered_data.md, fatal otherwise; the
    # cross-check against the loss body happens in check_session.
    # Reported without a verdict: NaN is legitimate under staggered_data.md, and
    # D3x -- the cross-check against the loss body -- is what actually decides
    # whether it is a defect. Printing FAIL here would contradict a D3x PASS.
    n_nan = int(np.isnan(data).sum())
    logger.info("D3 INFO  %s: %d NaN cell(s)%s", name, n_nan,
                " - legitimate only if the loss is nan-safe, see D3x" if n_nan else "")

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

    logger.debug("%s: median dt=%g, t_span=%g", name,
                 float(np.median(diffs)), float(t[-1] - t[0]))
    return failures, n_cols


def check_initial_conditions(name: str, data: np.ndarray, columns: list,
                             y0: dict, init_time: float | None) -> None:
    """
    D10 - each directly-observed state's initial condition against data row 0.

    Only possible because `columns` declares which column observes which state;
    before that, the map lived as arbitrary Python inside the loss and this
    comparison could not be made by tooling at all. Derived columns are still
    beyond reach and are skipped rather than guessed at.

    Never a failure. In regime 1 a mismatch is a real defect -- the solve starts
    at the first save point, so solution[0] IS y0 and the gap is a constant
    penalty no parameter can remove -- but the severity call belongs to
    validation_rules.md, and in regime 2 the same gap is expected.
    """
    regime_1 = init_time is None or init_time == data[0, 0]
    for index, column in enumerate(columns):
        state = column.get("observes")
        if state is None or index >= data.shape[1]:
            continue
        measured, configured = float(data[0, index]), y0.get(state)
        if configured is None:
            continue
        gap = abs(measured - configured)
        # Scored against the column's own observed range, not against the values
        # themselves: the range is what the loss normalises by, so this is the
        # fraction of the signal that is permanently mis-fit. Scoring relative to
        # the value would read 100% whenever init_val is exactly zero, however
        # negligible the absolute gap.
        finite = data[:, index][np.isfinite(data[:, index])]
        span = float(finite.max() - finite.min()) if finite.size else 0.0
        fraction = gap / span if span > 0 else (0.0 if gap == 0 else float("inf"))
        verdict = "WARN" if (regime_1 and fraction > 1e-3) else "ok"
        logger.info(
            "D10 %s  %s: state '%s' (column %d '%s') init_val=%g vs row 0=%g "
            "- gap %.3g = %.2f%% of the column's range%s",
            verdict, name, state, index, column["name"], configured, measured,
            gap, 100.0 * fraction,
            " - in regime 1 this is a constant penalty no parameter can remove"
            if verdict == "WARN" else "")

    derived = [c["name"] for i, c in enumerate(columns[1:], 1)
               if not c.get("observes") and not c.get("uncertainty_of")]
    if derived:
        logger.info("D10 INFO  %s: derived column(s) %s have no state to compare "
                    "against; their init_val is unchecked", name, derived)


def review_loss(session_dir: Path, config: dict, experiments: list,
                model_src: str) -> None:
    """
    L1..L4 - feedback on how the loss is CONSTRUCTED, measured from the data
    alone. No solve, no fitted parameters, so it is available before the first
    fit and stays inside the cold-start invariant.

    Reported as facts, never as verdicts. In particular this does NOT try to
    detect whether the loss normalises: that would mean pattern-matching
    arbitrary Python, which is exactly the fragility the `columns` declaration
    exists to remove. The magnitudes below are the actionable part -- what the
    term weights WOULD be without normalisation -- and whether a normaliser is
    present is answered by reading the loss.
    """
    loss_body = (model_src.split("def _compute_loss_problem", 1)[1].split("\ndef ", 1)[0]
                 if "def _compute_loss_problem" in model_src else "")

    scales_per_exp = []
    for index, experiment in enumerate(experiments):
        columns = experiment.get("columns") or []
        path = session_dir / "inputs" / experiment["data_file"]
        if not path.is_file() or not columns:
            continue
        try:
            data = load_dataset(path)
        except ValueError:
            continue
        if data.ndim != 2:
            continue

        measurements = [(i, c) for i, c in enumerate(columns)
                        if i > 0 and not c.get("uncertainty_of") and i < data.shape[1]]
        scales = {}
        for i, column in enumerate([c for _, c in measurements]):
            values = data[:, measurements[i][0]]
            values = values[np.isfinite(values)]
            scales[column["name"]] = float(np.abs(values).max()) if values.size else 0.0
        scales_per_exp.append(max(scales.values(), default=0.0))

        if index == 0 and scales:
            logger.info("L1 INFO  column scales (max|data|): %s",
                        ", ".join(f"{k}={v:.3g}" for k, v in scales.items()))
            positive = [v for v in scales.values() if v > 0]
            if len(positive) > 1:
                ratio = max(positive) / min(positive)
                total = sum(v ** 2 for v in positive)
                share = ", ".join(f"{k} {100 * v ** 2 / total:.3f}%"
                                  for k, v in scales.items())
                logger.info("L1 INFO  largest/smallest scale ratio %.3g:1", ratio)
                logger.info("L1 INFO  IF residuals are squared and NOT normalised "
                            "per column, the implied weight share is: %s", share)

            # L3 - where the samples sit inside each observable's own range. A
            # heavily one-sided distribution means the loss is dominated by
            # whichever regime holds most of the samples, not by the interesting
            # one.
            for column_index, column in measurements:
                values = data[:, column_index]
                values = values[np.isfinite(values)]
                if values.size < 4:
                    continue
                low, high = values.min(), values.max()
                if high <= low:
                    continue
                fraction = (values - low) / (high - low)
                logger.info("L3 INFO  %s: %.0f%% of samples in the lowest decile of "
                            "its range, %.0f%% in the highest", column["name"],
                            100 * np.mean(fraction < 0.1), 100 * np.mean(fraction > 0.9))

    # L2 - declared uncertainties the loss may not be using. Reported, not judged:
    # the loss reads sigma columns by index, so this is a hint, not proof.
    sigma_columns = [c["name"] for e in experiments
                     for c in (e.get("columns") or []) if c.get("uncertainty_of")]
    if sigma_columns:
        mentioned = "sigma" in loss_body or any(s in loss_body for s in sigma_columns)
        logger.info("L2 INFO  uncertainty column(s) declared %s; the loss body "
                    "appears to reference them = %s (a declared uncertainty the "
                    "loss ignores discards information)",
                    sorted(set(sigma_columns)), mentioned)

    # L4 - losses are averaged across experiments UNWEIGHTED, so a scale spread
    # is a weighting the user did not choose.
    positive = [s for s in scales_per_exp if s > 0]
    if len(positive) > 1:
        logger.info("L4 INFO  %d experiments, data scales %.3g..%.3g (ratio %.3g:1); "
                    "experiment losses are averaged UNWEIGHTED",
                    len(scales_per_exp), min(positive), max(positive),
                    max(positive) / min(positive))


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

    global_y0 = {v["name"]: v.get("init_val") for v in variables}

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
            data = load_dataset(path)
        except ValueError as exc:
            # genfromtxt raises on a ragged file; that is the one structural
            # requirement the loader already enforces, and its message is clear.
            logger.error("D0 FAIL  %s: not rectangular - %s", filename, exc)
            failures += 1
            continue

        exp_failures, n_cols = check_experiment(filename, data, init_time, max_idx,
                                                whole, n_states, path,
                                                len(experiment.get("columns") or []) or None)
        failures += exp_failures

        columns = experiment.get("columns") or []
        if columns:
            failures += check_declaration(filename, data, columns, path)
            if data.ndim == 2 and data.shape[0]:
                y0 = dict(global_y0)
                y0.update(experiment.get("initial_conditions") or {})
                check_initial_conditions(filename, data, columns, y0, init_time)
        else:
            report("D12", None, f"{filename}: no `columns` block declared")
        if n_cols is not None:
            col_counts.append((filename, n_cols))
        if data.ndim == 2 and np.isnan(data).any():
            any_nan = True

    declared = [o["name"] for o in (config.get("model") or {}).get("observables", [])]
    referenced = {c.get("observes") for e in experiments
                  for c in (e.get("columns") or []) if c.get("observes")}
    failures += check_observables(declared, model_src, referenced)

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

    if variables:
        logger.info("D10 INFO  init_val per integrated variable: %s", global_y0)

    if experiments:
        review_loss(session_dir, config, experiments, model_src)

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
