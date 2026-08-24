#!/usr/bin/env python3
"""
Stamp a session's generated script with the sources it was generated from, or
verify an existing stamp.

`/pfit-jax` runs this with --write as its last step, so the script records what
it was translated from. `tools/check_ready.py` (and the fit entry points) then
compare content rather than timestamps. See `lib/utils/source_stamp.py` for why.

Usage:
    ./venv/bin/python3 tools/stamp_script.py <session> --write
    ./venv/bin/python3 tools/stamp_script.py <session>            # verify only
"""

import argparse
import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from lib.utils.source_stamp import verify_stamp, write_stamp  # noqa: E402

logger = logging.getLogger("stamp_script")


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", help="session name under sessions/, or a path to one")
    parser.add_argument("--write", action="store_true",
                        help="write or replace the stamp (default: verify only)")
    parser.add_argument("--log-file", default=str(SCRIPT_DIR / "stamp_script.log"))
    args = parser.parse_args()

    setup_logging(args.log_file)
    session_dir = resolve_session_dir(args.session)

    if args.write:
        stamp = write_stamp(session_dir)
        logger.info("%s: wrote %s", session_dir.name, stamp)
        return 0

    ok, detail = verify_stamp(session_dir)
    if ok is None:
        logger.warning("%s: %s", session_dir.name, detail)
        return 0
    if ok:
        logger.info("%s: %s", session_dir.name, detail)
        return 0
    logger.error("%s: %s", session_dir.name, detail)
    return 1


if __name__ == "__main__":
    sys.exit(main())
