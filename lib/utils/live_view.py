"""
Live view of a running fit's loss convergence.

Reads only the per-iteration logs the optimizers already write and flush --
outputs/pso_fitting.log or outputs/de_fitting.log for the population stage and
outputs/NODE_fitting.log for the gradient stage -- and draws a braille plot of
best-so-far loss against iteration. Nothing is written into the session and the
fitting process is never touched, so this is safe to attach to a fit already
running and safe to kill at any time.

Both logged curves are BEST-SO-FAR, so they are monotone by construction:
flatness means no improvement, not divergence.

Two ways in:

    attach(session_dir, output_dir)   context manager used by the fit entry
                                      points, so a fit shows the view by itself
    follow(session_dir, config)       the loop, used by tools/live_fit_monitor.py
                                      to watch a fit started elsewhere

This module must stay import-safe without jax: the entry points import it
before the XLA device count has been configured.
"""

import contextlib
import logging
import math
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "tools" / "live_fit_monitor.yaml"

logger = logging.getLogger("live_view")
logger.addHandler(logging.NullHandler())

# The failed-solve penalty, from lib/utils/yamlread.py. A best cost sitting here
# means every solve in the population failed (S1 in diagnosis_rules.md).
ERROR_LOSS = 5000.0

STAGE1_LOGS = ("de_fitting.log", "pso_fitting.log")
STAGE2_LOG = "NODE_fitting.log"

# "18, 3.4674E-02, 4.1023" -- the line every optimizer appends per iteration.
LOG_LINE = re.compile(r"^\s*(\d+)\s*,\s*([-+0-9.eEnaN]+)\s*,\s*([-+0-9.eEnaN]+)\s*$")

ESC = "\x1b"
HIDE_CURSOR = f"{ESC}[?25l"
SHOW_CURSOR = f"{ESC}[?25h"
CLEAR_EOL = f"{ESC}[K"
CLEAR_BELOW = f"{ESC}[J"

DEFAULTS = {
    "session": "auto",
    "search_roots": ["sessions"],
    "refresh_seconds": 1.0,
    "wait_for_session_seconds": 60,
    "y_scale": "log10",
    "max_width": 100,
    "min_height": 24,
    "renderer": "terminal",
    "exit_when_complete": True,
    "exit_on_error": True,
    "complete_quiet_seconds": 5.0,
    "startup_grace_seconds": 20.0,
    "auto_attach": True,
    "console_tail_lines": 12,
}


def load_config(path=None) -> dict:
    """Read the YAML config, falling back to built-in defaults for anything absent."""
    config = dict(DEFAULTS)
    path = Path(path) if path else DEFAULT_CONFIG
    try:
        import yaml

        with open(path) as handle:
            config.update(yaml.safe_load(handle) or {})
    except FileNotFoundError:
        logger.debug("no live-view config at %s; using defaults", path)
    except Exception as exc:
        logger.warning("could not read live-view config %s: %s", path, exc)
    return config


# --------------------------------------------------------------------------
# braille canvas
# --------------------------------------------------------------------------

# Bit for each dot in a 2-wide x 4-tall braille cell, indexed [col][row].
BRAILLE_DOTS = ((0x01, 0x02, 0x04, 0x40), (0x08, 0x10, 0x20, 0x80))


class BrailleCanvas:
    """A character grid addressed at 2x4 braille-dot resolution."""

    def __init__(self, width_chars: int, height_chars: int):
        self.w = max(1, width_chars)
        self.h = max(1, height_chars)
        self.dw = self.w * 2
        self.dh = self.h * 4
        self.cells = [[0] * self.w for _ in range(self.h)]

    def set(self, x: int, y: int) -> None:
        """Light the dot at (x, y) in dot coordinates, y = 0 at the top."""
        if not (0 <= x < self.dw and 0 <= y < self.dh):
            return
        self.cells[y // 4][x // 2] |= BRAILLE_DOTS[x % 2][y % 4]

    def line(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """Bresenham segment, so consecutive samples read as a curve."""
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            self.set(x0, y0)
            if x0 == x1 and y0 == y1:
                return
            err2 = 2 * err
            if err2 >= dy:
                err += dy
                x0 += sx
            if err2 <= dx:
                err += dx
                y0 += sy

    def rows(self) -> list[str]:
        # Empty cells render as a plain space rather than U+2800, so the panel
        # keeps a uniform advance width on terminals that treat the braille
        # blank as zero-width.
        return [
            "".join(chr(0x2800 + cell) if cell else " " for cell in row)
            for row in self.cells
        ]


# --------------------------------------------------------------------------
# reading the session
# --------------------------------------------------------------------------


def _mtime(path) -> float:
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def parse_log(path) -> list[tuple[int, float, float]]:
    """Return [(iteration, best_loss, seconds), ...] from an optimizer log.

    Header lines and any half-written final line are skipped: the fit appends to
    these files while we read them, so a torn read must not be fatal.
    """
    if not path or not os.path.isfile(path):
        return []
    try:
        with open(path, "r", errors="replace") as handle:
            raw = handle.read()
    except OSError as exc:
        logger.debug("could not read %s: %s", path, exc)
        return []

    points = []
    for line in raw.splitlines():
        match = LOG_LINE.match(line)
        if not match:
            continue
        try:
            points.append(
                (int(match.group(1)), float(match.group(2)), float(match.group(3)))
            )
        except ValueError:
            continue
    return points


def stage1_log_path(outputs_dir) -> tuple[str | None, str]:
    """Path and algorithm label for whichever population log exists."""
    for name in STAGE1_LOGS:
        path = os.path.join(outputs_dir, name)
        if os.path.isfile(path):
            return path, "DE" if name.startswith("de") else "PSO"
    return None, ""


def newest_log_mtime(outputs_dir) -> float:
    """Most recent mtime across the three iteration logs, or 0."""
    stamps = [
        _mtime(os.path.join(outputs_dir, name)) for name in STAGE1_LOGS + (STAGE2_LOG,)
    ]
    return max(stamps) if stamps else 0.0


def read_error_text(outputs_dir) -> str:
    path = os.path.join(outputs_dir, "fitting_error.txt")
    if not os.path.isfile(path):
        return ""
    try:
        with open(path, errors="replace") as handle:
            return handle.read().strip().replace("\n", " ")[:200]
    except OSError:
        return ""


def discover_session(search_roots: list[str]) -> str | None:
    """Session directory whose iteration logs were written most recently."""
    best_path, best_stamp = None, 0.0
    for root in search_roots:
        root_abs = os.path.join(REPO_ROOT, root)
        if not os.path.isdir(root_abs):
            continue
        for entry in sorted(os.listdir(root_abs)):
            session_dir = os.path.join(root_abs, entry)
            outputs_dir = os.path.join(session_dir, "outputs")
            if not os.path.isdir(outputs_dir):
                continue
            stamp = newest_log_mtime(outputs_dir)
            if stamp > best_stamp:
                best_path, best_stamp = session_dir, stamp
    return best_path


def resolve_session(config: dict) -> str | None:
    """Session directory to follow, from `session:` or by auto-detection."""
    search_roots = config.get("search_roots", DEFAULTS["search_roots"])
    name = config.get("session", "auto")
    if name and name != "auto":
        if os.path.isdir(os.path.join(REPO_ROOT, name)):
            return os.path.join(REPO_ROOT, name)
        for root in search_roots:
            candidate = os.path.join(REPO_ROOT, root, name)
            if os.path.isdir(candidate):
                return candidate
        return None
    return discover_session(search_roots)


def read_budgets(session_dir) -> dict:
    """Iteration budgets and optimizer names from the session's user_input.yaml.

    Only used for the progress bars and the ETA, so every failure degrades to
    "unknown total" rather than stopping the view.
    """
    budgets = {
        "n_iters_pop": None,
        "n_iters_grad": None,
        "algorithm": None,
        "gradient_optimizer": None,
    }
    config_path = os.path.join(session_dir, "inputs", "user_input.yaml")
    if not os.path.isfile(config_path):
        return budgets
    try:
        from lib.utils.yamlread import read_input_file

        reader = read_input_file(config_path)
        budgets["n_iters_pop"] = reader.n_iters_pop
        budgets["n_iters_grad"] = reader.n_iters_grad
        budgets["algorithm"] = (reader.algorithm or "PSO").upper()
        budgets["gradient_optimizer"] = reader.gradient_optimizer
    except Exception as exc:
        logger.warning("could not read budgets from %s: %s", config_path, exc)
    return budgets


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------


def fmt_clock(seconds) -> str:
    if seconds is None or seconds < 0 or seconds != seconds:
        return "--:--"
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def fmt_loss(value: float) -> str:
    if value != value:
        return "     NaN"
    return f"{value:.3E}"


def progress_bar(done: int, total, width: int = 14) -> str:
    if not total or total <= 0:
        return ""
    filled = max(0, min(width, round(width * done / total)))
    return "▕" + "█" * filled + "░" * (width - filled) + "▏"


def rate_per_iter(points, window: int = 5):
    """Mean seconds per iteration over the last `window` logged iterations."""
    if not points:
        return None
    recent = [p[2] for p in points[-window:] if p[2] > 0]
    if not recent:
        return None
    return sum(recent) / len(recent)


def render_panel(points, width: int, height: int, y_log: bool, x_total) -> list[str]:
    """Braille plot of best-so-far loss against iteration, with axes."""
    label_w = 10  # fmt_loss is 9 wide, so this leaves one space before the axis
    gutter = label_w + 1
    plot_w = max(4, width - gutter)
    canvas = BrailleCanvas(plot_w, height)

    xs = [p[0] for p in points]
    raw_ys = [p[1] for p in points]
    finite = [(x, y) for x, y in zip(xs, raw_ys) if y == y and abs(y) != float("inf")]
    if not finite:
        return [" " * width for _ in range(height + 2)]

    use_log = y_log and all(y > 0 for _, y in finite)
    ys = [(x, math.log10(y) if use_log else y) for x, y in finite]

    x_lo = min(1, xs[0])
    x_hi = max(x_total or 0, xs[-1], x_lo + 1)
    y_vals = [y for _, y in ys]
    y_lo, y_hi = min(y_vals), max(y_vals)
    if y_hi - y_lo < 1e-12:
        pad = abs(y_hi) * 0.05 + (0.5 if use_log else 1e-9)
        y_lo, y_hi = y_lo - pad, y_hi + pad
    else:
        pad = (y_hi - y_lo) * 0.06
        y_lo, y_hi = y_lo - pad, y_hi + pad

    def to_dot(x: float, y: float) -> tuple[int, int]:
        dx = round((x - x_lo) / (x_hi - x_lo) * (canvas.dw - 1))
        dy = round((y_hi - y) / (y_hi - y_lo) * (canvas.dh - 1))
        return dx, dy

    previous = None
    for x, y in ys:
        dot = to_dot(x, y)
        if previous is not None:
            canvas.line(previous[0], previous[1], dot[0], dot[1])
        else:
            canvas.set(dot[0], dot[1])
        previous = dot

    # y tick labels on the char rows nearest four evenly spaced values
    tick_rows = sorted({0, (height - 1) // 3, 2 * (height - 1) // 3, height - 1})
    labels = {}
    for row in tick_rows:
        frac = (row * 4) / max(1, canvas.dh - 1)
        value = y_hi - frac * (y_hi - y_lo)
        labels[row] = fmt_loss(10.0**value if use_log else value)

    lines = []
    for row, body in enumerate(canvas.rows()):
        label = labels.get(row, "")
        edge = "┤" if row in labels else "│"
        lines.append(f"{label:>{label_w}}{edge}{body}")

    # x axis with four ticks
    n_ticks = 4
    tick_cols = sorted({round(i * (plot_w - 1) / (n_ticks - 1)) for i in range(n_ticks)})
    axis = ["─"] * plot_w
    for col in tick_cols:
        axis[col] = "┬"
    lines.append(" " * label_w + "└" + "".join(axis))

    tick_line = [" "] * plot_w
    for col in tick_cols:
        value = round(x_lo + (col / max(1, plot_w - 1)) * (x_hi - x_lo))
        text = str(value)
        start = min(col, plot_w - len(text))
        if start < 0:
            continue
        if all(tick_line[start + i] == " " for i in range(len(text))):
            for i, char in enumerate(text):
                tick_line[start + i] = char
    lines.append(" " * gutter + "".join(tick_line))
    return lines


def build_frame(state: dict, width: int, height: int) -> list[str]:
    """Compose the whole view: header, two stage panels, footer notes."""
    lines: list[str] = []
    session = state["session_name"]
    elapsed = fmt_clock(time.time() - state["started"])

    head = f"pfit live · {session}"
    verb = "running" if state["attached"] else ("done" if state["in_process"] else "watching")
    tail = f"{verb} {elapsed}"
    lines.append(f"{head}{' ' * max(1, width - len(head) - len(tail))}{tail}")
    lines.append("─" * width)

    stage1 = state["stage1"]
    stage2 = state["stage2"]
    budgets = state["budgets"]

    # how much vertical room each panel gets
    n_panels = max(1, sum((bool(stage1), bool(stage2))))
    overhead = 6 + 2 * n_panels + 3
    plot_h = max(4, (height - overhead) // n_panels)

    def stage_block(label: str, points, total, note: str) -> None:
        if not points:
            lines.append(f"{label}  {note}")
            return
        done = points[-1][0]
        best = points[-1][1]
        rate = rate_per_iter(points)
        eta = rate * (total - done) if rate and total and total > done else None
        total_text = str(total) if total else "?"
        status = (
            f"{label}  {done:>4}/{total_text:<4} {progress_bar(done, total)}  "
            f"best {fmt_loss(best)}"
        )
        if rate:
            status += f"  {rate:.2f} s/it"
        if eta is not None:
            status += f"  eta {fmt_clock(eta)}"
        lines.append(status[:width])
        lines.extend(
            line[:width]
            for line in render_panel(points, width, plot_h, state["y_log"], total)
        )

    stage_block(
        f"Stage 1  {state['algorithm'] or budgets.get('algorithm') or 'population'}",
        stage1,
        budgets.get("n_iters_pop"),
        "waiting for the first iteration…",
    )
    lines.append("")
    optimizer = budgets.get("gradient_optimizer") or "gradient"
    stage_block(
        f"Stage 2  NODE ({optimizer})",
        stage2,
        budgets.get("n_iters_grad"),
        "not started" if stage1 else "waiting…",
    )

    lines.append("")

    # The handoff between stages. In a full fit both numbers describe nearly the
    # same parameter vector, since stage 2 is seeded from stage 1's best point,
    # so a large gap is the tolerance mismatch of S3. That premise fails under
    # fit_gradient_only.py, where the stage-1 log belongs to an earlier run --
    # hence the flag is raised only when this view watched stage 1 itself
    # advance. A missed flag is cheaper than a false one; S3 is /pfit-diagnose's
    # call, and this line only surfaces the numbers behind it.
    if stage1 and stage2:
        seed = stage1[-1][1]
        first = stage2[0][1]
        if seed > 0 and first > 0:
            ratio = first / seed
            flag = ""
            if state["saw_stage1"] and (ratio > 10 or ratio < 0.1):
                flag = "  ← stages disagree (S3)"
            lines.append(
                f"handoff   stage 1 {fmt_loss(seed)} → NODE first {fmt_loss(first)}"
                f"  (×{ratio:.2f}){flag}"[:width]
            )

    latest = stage2 or stage1
    if latest and abs(latest[-1][1] - ERROR_LOSS) < 1e-6:
        lines.append(
            f"⚠ best cost is the failed-solve penalty ({ERROR_LOSS:.1f}): every solve "
            f"is failing — S1"[:width]
        )

    if state["error_text"]:
        age = " (from a previous run)" if state["error_is_stale"] else ""
        lines.append(f"⚠ fitting_error.txt{age}: {state['error_text']}"[:width])

    if state["complete"]:
        if state["attached"]:
            lines.append("final_design_point.csv written — post-fit diagnostics running…")
        elif state["saw_progress"]:
            lines.append("fit complete — final_design_point.csv written")
        else:
            grace = state["grace_left"]
            lines.append(
                "showing the last completed fit — no new iteration yet"
                + (f", watching {grace:.0f}s more for one" if grace > 0 else "")
            )
    else:
        idle = time.time() - state["last_change"]
        if idle > 15:
            lines.append(f"no new iteration for {fmt_clock(idle)} (long solve, or stalled)")

    lines.append("both curves are best-so-far, so flat means no improvement — not divergence.")
    return [line[:width] for line in lines]


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------


class Screen:
    """In-place redraw that leaves the final frame in the scrollback."""

    def __init__(self, enabled: bool, stream=None):
        self.enabled = enabled
        self.stream = stream or sys.stdout
        self.last_lines = 0

    def __enter__(self):
        self._write(HIDE_CURSOR)
        return self

    def __exit__(self, *exc):
        self._write(SHOW_CURSOR)
        return False

    def _write(self, text: str) -> None:
        if not self.enabled:
            return
        try:
            self.stream.write(text)
            self.stream.flush()
        except (OSError, ValueError):  # terminal went away
            self.enabled = False

    def draw(self, lines: list[str]) -> None:
        if not self.enabled:
            return
        out = []
        if self.last_lines:
            out.append(f"{ESC}[{self.last_lines}A")
        out.append("\r")
        for line in lines:
            out.append(line + CLEAR_EOL + "\n")
        out.append(CLEAR_BELOW)
        self._write("".join(out))
        self.last_lines = len(lines)


def follow(session_dir, config: dict, stop_event=None, stream=None, interactive=None) -> int:
    """Draw the live view until the fit finishes or `stop_event` is set.

    Returns 0 normally, 1 if the fit reported an error. `stop_event` is what the
    in-process attachment uses: with it set, completion is decided by the caller
    (which knows when the fit function returned) rather than inferred from
    outputs/, so the view stays up through the post-fit diagnostics.
    """
    outputs_dir = os.path.join(session_dir, "outputs")
    session_name = os.path.basename(os.path.normpath(str(session_dir)))
    attached = stop_event is not None

    if interactive is None:
        interactive = sys.stdout.isatty() and config.get("renderer", "terminal") == "terminal"

    refresh = float(config.get("refresh_seconds", 1.0))
    max_width = int(config.get("max_width", 100))
    min_height = int(config.get("min_height", 24))
    exit_when_complete = bool(config.get("exit_when_complete", True)) and not attached

    state = {
        "session_name": session_name,
        "started": time.time(),
        "last_change": time.time(),
        "y_log": config.get("y_scale", "log10") == "log10",
        "budgets": read_budgets(session_dir),
        "stage1": [],
        "stage2": [],
        "algorithm": "",
        "error_text": "",
        "error_is_stale": False,
        "complete": False,
        "saw_progress": False,
        "saw_stage1": False,
        "grace_left": 0.0,
        "attached": attached,
        "in_process": attached,
    }

    # fit_gradient_only.py deliberately does NOT wipe outputs/, so a completed
    # previous run leaves final_design_point.csv and possibly fitting_error.txt
    # in place while a new fit is starting. Neither may be trusted until it is
    # newer than the iteration logs and has stayed that way while no new
    # iteration arrived. A fit that starts while this view is already attached
    # leaves no trace in outputs/ until its stage log is truncated, which is on
    # the far side of the JAX import -- hence a session that already looked
    # finished gets the longer startup grace, while one that finished under our
    # own eyes exits promptly.
    quiet = float(config.get("complete_quiet_seconds", 5.0))
    startup_grace = float(config.get("startup_grace_seconds", 20.0))
    monitor_started = time.time()
    complete_since = None
    status = 0

    fingerprint = None
    with Screen(interactive, stream) as screen:
        while True:
            path1, algorithm = stage1_log_path(outputs_dir)
            state["algorithm"] = algorithm
            state["stage1"] = parse_log(path1)
            state["stage2"] = parse_log(os.path.join(outputs_dir, STAGE2_LOG))
            log_stamp = newest_log_mtime(outputs_dir)

            error_path = os.path.join(outputs_dir, "fitting_error.txt")
            state["error_text"] = read_error_text(outputs_dir)
            state["error_is_stale"] = bool(state["error_text"]) and _mtime(error_path) < log_stamp

            design_point = os.path.join(outputs_dir, "final_design_point.csv")
            complete_now = (
                os.path.isfile(design_point) and _mtime(design_point) >= log_stamp - 1.0
            )
            complete_since = complete_since if complete_now else None
            if complete_now and complete_since is None:
                complete_since = time.time()
            state["complete"] = complete_now

            current = (len(state["stage1"]), len(state["stage2"]))
            if current != fingerprint:
                if fingerprint is not None:
                    state["saw_progress"] = True
                    if current[0] > fingerprint[0]:
                        state["saw_stage1"] = True
                fingerprint = current
                state["last_change"] = time.time()
                if not interactive:
                    stage = state["stage2"] or state["stage1"]
                    if stage:
                        logger.info(
                            "%s stage%d iter %d best %s",
                            session_name,
                            2 if state["stage2"] else 1,
                            stage[-1][0],
                            fmt_loss(stage[-1][1]),
                        )

            wait = quiet if state["saw_progress"] else startup_grace
            state["grace_left"] = (
                max(0.0, wait - (time.time() - complete_since)) if complete_since else 0.0
            )

            stopping = stop_event is not None and stop_event.is_set()
            if stopping:
                # last frame: the fit function has returned, so this is the
                # finished state rather than a fit still in its diagnostics
                state["attached"] = False
                state["saw_progress"] = True

            size = shutil.get_terminal_size(fallback=(max_width, min_height))
            width = max(48, min(max_width, size.columns - 1))
            height = max(min_height, size.lines - 1)
            screen.draw(build_frame(state, width, height))

            if stopping:
                return status
            if (
                exit_when_complete
                and complete_since is not None
                and time.time() - complete_since >= wait
            ):
                logger.info("fit complete; exiting")
                return status
            if (
                state["error_text"]
                and not state["error_is_stale"]
                and _mtime(error_path) >= monitor_started - 1.0
            ):
                logger.error("fit failed: %s", state["error_text"])
                status = 1
                if config.get("exit_on_error", True) and not attached:
                    return status

            if stop_event is not None:
                stop_event.wait(refresh)
            else:
                time.sleep(refresh)


def wait_for_session(config: dict, announce=None) -> str | None:
    """Resolve the session, waiting for one to start if none has logs yet."""
    deadline = time.time() + float(config.get("wait_for_session_seconds", 60))
    announced = False
    while True:
        session_dir = resolve_session(config)
        if session_dir and newest_log_mtime(os.path.join(session_dir, "outputs")):
            return session_dir
        if time.time() > deadline:
            return session_dir
        if not announced and announce:
            announce("no fit logs yet — waiting for one to start (Ctrl-C to quit)")
            announced = True
        time.sleep(1.0)


# --------------------------------------------------------------------------
# in-process attachment, used by the fit entry points
# --------------------------------------------------------------------------


def _enabled(config: dict) -> bool:
    """Whether a fit should raise the live view for itself.

    Requires a real terminal, so a piped run, a cron job and the pytest suite
    (which captures stdout) all keep their plain output untouched.
    """
    if os.environ.get("PFIT_LIVE", "1").strip().lower() in ("0", "false", "no", "off"):
        return False
    if not config.get("auto_attach", True):
        return False
    if config.get("renderer", "terminal") != "terminal":
        return False
    try:
        return sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def _follow_quietly(session_dir, config, stop_event, stream) -> None:
    """Render, but never let a drawing bug take the fit down with it."""
    try:
        follow(session_dir, config, stop_event=stop_event, stream=stream, interactive=True)
    except Exception:
        logger.exception("live view stopped; the fit is unaffected")
        try:
            stream.write(SHOW_CURSOR + "\n[live view stopped; the fit continues]\n")
            stream.flush()
        except Exception:
            pass


def _print_tail(log_path, n_lines: int) -> None:
    """Echo the end of the captured console output to the restored terminal."""
    if n_lines <= 0:
        return
    try:
        with open(log_path, errors="replace") as handle:
            tail = handle.read().splitlines()[-n_lines:]
    except OSError:
        return
    if not tail:
        return
    print(f"\n─── tail of {log_path} " + "─" * 12)
    for line in tail:
        print(line)


@contextlib.contextmanager
def attach(session_dir, output_dir, config=None):
    """Show the live view for the duration of a fit running in this process.

    The fit's own console output would fight the in-place redraw, so it is
    captured to outputs/run_stdout.log at the file-descriptor level -- which
    catches writes from C extensions too -- and the last lines are echoed back
    once the view comes down. Yields None (and changes nothing) when the live
    view is disabled or stdout is not a terminal.
    """
    config = config if isinstance(config, dict) else load_config(config)
    if not _enabled(config):
        yield None
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run_stdout.log"

    sys.stdout.flush()
    sys.stderr.flush()
    saved_out, saved_err = os.dup(1), os.dup(2)
    console = os.fdopen(os.dup(1), "w", buffering=1)
    log_file = open(log_path, "w")
    os.dup2(log_file.fileno(), 1)
    os.dup2(log_file.fileno(), 2)
    for stream in (sys.stdout, sys.stderr):
        try:  # keep the captured log tailable while the fit runs
            stream.reconfigure(line_buffering=True)
        except Exception:
            pass

    stop_event = threading.Event()
    thread = threading.Thread(
        target=_follow_quietly,
        args=(session_dir, config, stop_event, console),
        name="pfit-live-view",
        daemon=True,
    )
    thread.start()
    try:
        yield thread
    finally:
        stop_event.set()
        thread.join(timeout=float(config.get("refresh_seconds", 1.0)) + 5.0)
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        log_file.close()
        try:
            console.write(SHOW_CURSOR)
            console.flush()
        finally:
            console.close()
        _print_tail(log_path, int(config.get("console_tail_lines", 12)))
