#!/usr/bin/env python3
"""
Measure whether a session is ready to fit.

This tool REPORTS FACTS ONLY, one PASS/FAIL/SKIP line per check id (R1..R6).
The severity of each id, and what to do about it, are owned by
`lib/LLM/reference/validation_rules.md` — do not duplicate that mapping here.

The check that justifies the tool is R3. `fit_parameters.py` imports
`generated/generated_script.py`; it never reads `user_model.py`. So a model
edited after the last translation is simply not the model that gets fitted, and
nothing at run time notices — the run completes, the log looks normal, and the
parameters belong to the previous version of the equations. Comparing
modification times is the only way to catch it.

Usage:
    ./venv/bin/python3 tools/check_ready.py <session>
"""

import argparse
import logging
import sys
from pathlib import Path

SCRIPT_DIR_BOOTSTRAP = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR_BOOTSTRAP.parent))

from lib.utils.source_stamp import verify_stamp  # noqa: E402
from lib.utils.run_store import resolve_run, output_root

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

logger = logging.getLogger("check_ready")


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
    candidate = Path(arg)
    session_dir = candidate if candidate.is_dir() else REPO_ROOT / "sessions" / arg
    if not session_dir.is_dir():
        raise SystemExit(f"session directory not found: {session_dir}")
    return session_dir


def report(check: str, ok: bool | None, message: str, warn: bool = False) -> bool:
    """
    Log one check line and return True when it failed.

    `warn=True` marks an advisory check: it is reported when it does not hold,
    but does not make the session unfit to run. Printing those as FAIL would
    train the reader to ignore the ones that matter.
    """
    if ok is None:
        logger.info("%s SKIP  %s", check, message)
        return False
    if bool(ok):
        logger.info("%s PASS  %s", check, message)
        return False
    if warn:
        logger.warning("%s WARN  %s", check, message)
        return False
    logger.error("%s FAIL  %s", check, message)
    return True


def mtime(path: Path) -> float:
    return path.stat().st_mtime if path.is_file() else 0.0


def check_session(session_dir: Path, mode: str = "full", seed_run=None) -> int:
    config = session_dir / "inputs" / "user_input.yaml"
    model = session_dir / "generated" / "user_model.py"
    script = session_dir / "generated" / "generated_script.py"
    report_txt = session_dir / "generated" / "user_input_check.txt"
    outputs = output_root(session_dir)

    failures = 0

    # R1 - the inputs the whole pipeline is built from
    for check, path, produced_by in (
        ("R1", config, "/pfit-new"),
        ("R2", model, "/pfit-new"),
    ):
        failures += report(check, path.is_file(),
                           f"{path.relative_to(session_dir)} present"
                           + ("" if path.is_file() else f" - run {produced_by} first"))

    # R3 - the script that will actually run must be newer than everything it
    # was generated from. This is the silent one: the fit imports the script and
    # never looks at the model, so a stale script fits the previous equations.
    if not script.is_file():
        failures += report("R3", False,
                           "generated/generated_script.py missing - run /pfit-jax")
    else:
        # Content first. Timestamps are only consulted when there is no stamp to
        # compare, and the report says so rather than implying a content check.
        stamped, detail = verify_stamp(session_dir)
        if stamped is not None:
            failures += report("R3", stamped, f"{detail} (by content)")
        else:
            sources = {"user_model.py": model, "user_input.yaml": config}
            stale = {name: q for name, q in sources.items() if mtime(q) > mtime(script)}
            failures += report("R3", not stale,
                               f"{detail}; falling back to modification times, which "
                               + ("show it newer than its sources - but a clone or a "
                                  "copy can make that meaningless. Re-run /pfit-jax to "
                                  "stamp it" if not stale else
                                  f"show it OLDER than {sorted(stale)}. The fit imports "
                                  "the script and never reads the model, so it would "
                                  "fit the previous version. Re-run /pfit-jax"))

    # R4 - validation should have run, and should have run against the current
    # config rather than an earlier one
    if not report_txt.is_file():
        report("R4", None, "no user_input_check.txt - /pfit-check has not been run")
    else:
        current = mtime(report_txt) >= max(mtime(config), mtime(model))
        report("R4", current,
               "the validation report is current"
               if current else
               "the validation report predates the config or the model - re-run "
               "/pfit-check", warn=True)
        text = report_txt.read_text()
        for line in text.splitlines():
            if line.lower().startswith("number of critical errors"):
                count = line.split(":")[-1].strip()
                failures += report("R5", count in ("0", "none", "None"),
                                   f"last validation reported {line.split(':')[-1].strip()} "
                                   f"critical error(s)")
                break
        else:
            report("R5", None, "no critical-error count found in the report")

    # R6 - which entry points are actually available. fit_gradient_only.py seeds
    # from final_design_point.csv and refuses to start without it, so the seed
    # file is the thing to check -- not the presence of logs, which can survive a
    # run that died before writing a design point.
    try:
        seed = resolve_run(session_dir, seed_run, require_seed=True) / "final_design_point.csv"
    except FileNotFoundError as exc:
        failures += report("R6", False if mode == "gradient-only" else None,
                           f"{exc}; gradient-only requires a completed run with a design point")
    else:
        report("R6", True if mode == "gradient-only" else None,
               f"Seed available: {seed}; this fit creates a new run directory and preserves it")

    logger.info("%s: %d check(s) failed", session_dir.name, failures)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", help="session name under sessions/, or a path to one")
    parser.add_argument("--mode", choices=("full", "gradient-only"), default="full",
                        help="which entry point is intended; gradient-only makes the "
                             "stored design point a requirement rather than a note")
    parser.add_argument("--seed-run", help="run ID or directory used for gradient-only seeding")
    parser.add_argument("--log-file", default=str(SCRIPT_DIR / "check_ready.log"),
                        help="path to the log file")
    args = parser.parse_args()
    setup_logging(args.log_file)
    logger.info("checking %s for a %s run", args.session, args.mode)
    return 1 if check_session(resolve_session_dir(args.session), args.mode, args.seed_run) else 0


if __name__ == "__main__":
    sys.exit(main())
