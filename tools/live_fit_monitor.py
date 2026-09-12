#!/usr/bin/env python3
"""
Watch a fit that is running somewhere else.

A fit started with fit_parameters.py or fit_gradient_only.py raises this view by
itself, so this script is for the cases where that is not enough: a fit started
in another terminal, under nohup, on a machine you have since SSH'd back into,
or one whose live view you dismissed.

It only reads the per-iteration logs the optimizers already write and flush, so
it can attach to a fit already in progress and can be killed at any time without
touching it.

Usage:
    ./venv/bin/python3 tools/live_fit_monitor.py
    ./venv/bin/python3 tools/live_fit_monitor.py --config path/to/other.yaml

Every setting lives in live_fit_monitor.yaml beside this script; `session: auto`
attaches to whichever session's logs were written most recently. The rendering
itself lives in lib/utils/live_view.py, shared with the in-fit view.
"""

import argparse
import logging
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from lib.utils import live_view  # noqa: E402

DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "live_fit_monitor.yaml")

logger = logging.getLogger("live_view")


def setup_logging(log_file: str, to_stdout: bool) -> None:
    """Configure the shared live-view logger.

    The terminal renderer owns stdout and redraws in place, so the stdout
    handler is attached only when it is NOT running (a non-TTY, or a piped run,
    where the view degrades to one line per new iteration). The file handler is
    always attached, at DEBUG.
    """
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    handlers = [logging.FileHandler(log_file)]
    if to_stdout:
        handlers.append(logging.StreamHandler(sys.stdout))
    for handler in handlers:
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(fmt)
        logger.addHandler(handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="path to the YAML config (default: beside this script)")
    parser.add_argument("--run", help="run ID or directory; pair with an explicit session")
    parser.add_argument("--session", help="session name or path")
    parser.add_argument("--log-file", default=os.path.join(SCRIPT_DIR, "live_fit_monitor.log"),
                        help="path to the log file")
    args = parser.parse_args()

    config = live_view.load_config(args.config)
    if args.run:
        config["run"] = args.run
    if args.session:
        config["session"] = args.session

    interactive = sys.stdout.isatty() and config.get("renderer", "terminal") == "terminal"
    setup_logging(args.log_file, to_stdout=not interactive)
    logger.info("reading config from %s", args.config)

    def announce(message: str) -> None:
        if interactive:
            sys.stdout.write(message + "\n")
            sys.stdout.flush()
        else:
            logger.info(message)

    session_dir = live_view.wait_for_session(config, announce=announce)
    if not session_dir:
        logger.error(
            "no session found. Set `session:` in %s, or check `search_roots`.", args.config
        )
        return 1

    logger.info("following %s", os.path.relpath(session_dir, REPO_ROOT))
    return live_view.follow(session_dir, config, interactive=interactive)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.stdout.write(live_view.SHOW_CURSOR + "\n")
        sys.exit(0)
