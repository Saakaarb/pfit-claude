"""Loopback-only, read-only dashboard for one immutable run selection."""
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
import time

import yaml


ASSET = Path(__file__).with_name("live_dashboard.html")


class ProgressState:
    def __init__(self, run):
        self.run = Path(run).resolve()
        self.offset = 0
        self.events = []
        self.lock = threading.Lock()

    def read(self):
        with self.lock:
            path = self.run / "live_progress.jsonl"
            if path.is_file():
                if path.stat().st_size < self.offset:
                    self.offset, self.events = 0, []
                with path.open() as handle:
                    handle.seek(self.offset)
                    while line := handle.readline():
                        if not line.endswith("\n"):
                            break  # a writer has not finished this record yet
                        self.offset = handle.tell()
                        try:
                            self.events.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            manifest_path = self.run / "run_manifest.json"
            manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
            config_path = self.run / "snapshot/inputs/run_config.yaml"
            config = yaml.safe_load(config_path.read_text()) if config_path.is_file() else {}
            config = config or {}
            global_end = max((e["iteration"] for e in self.events if e["stage"] == "global"), default=0)
            points = {}
            for event in self.events:
                value = event.get("loss")
                if event["kind"] not in ("iteration", "stage_end") or value is None or not math.isfinite(value):
                    continue
                stage, iteration = event["stage"], event["iteration"]
                points[(stage, iteration)] = {"stage": stage, "iteration": iteration,
                                             "x": iteration + (global_end if stage == "gradient" else 0),
                                             "loss": value}
            latest = self.events[-1] if self.events else None
            failure = self.run / "fitting_error.txt"
            warning = failure.read_text(errors="replace")[-2000:] if failure.is_file() else None
            if latest and latest["kind"] == "solver_failure":
                warning = warning or "The gradient evaluation failed; the displayed parameters are the previous best or seed."
            return {
                "run_id": self.run.name,
                "session": Path(manifest.get("session", "session")).name,
                "status": manifest.get("status", "unknown"),
                "started_at": manifest.get("started_at"),
                "finished_at": manifest.get("finished_at"),
                "server_time": datetime.now(timezone.utc).isoformat(),
                "mode": manifest.get("mode"), "seed_source": manifest.get("seed_source"),
                "global_algorithm": config.get("population_opt", {}).get("algorithm", "PSO"),
                "gradient_optimizer": config.get("gradient_opt", {}).get("gradient_optimizer", "lbfgs"),
                "parameter_definitions": config.get("model", {}).get("trainable_parameters", []),
                "latest": latest, "points": list(points.values()),
                "handoff": global_end if any(e["stage"] == "gradient" for e in self.events) and global_end else None,
                "warning": warning,
            }


def make_server(run, port=0):
    state = ProgressState(run)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            host = self.headers.get("Host", "").split(":")[0].lower()
            if host not in ("127.0.0.1", "localhost"):
                self.send_error(403)
                return
            route = self.path.split("?", 1)[0]
            if route == "/":
                body, content_type = ASSET.read_bytes(), "text/html; charset=utf-8"
            elif route == "/api/progress":
                try:
                    body = json.dumps(state.read(), allow_nan=False).encode()
                except (OSError, ValueError) as exc:
                    self.send_error(503, "Run data temporarily unavailable")
                    return
                content_type = "application/json"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def launch_dashboard(run, port=0):
    """Launch the optional viewer separately, so it remains available after fitting."""
    run = Path(run).resolve()
    command = [sys.executable, str(Path(__file__).resolve().parents[2] / "tools/live_fit_server.py"),
               "--run-dir", str(run), "--port", str(port)]
    with (run / "live_server.log").open("a") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
    info_path = run / "live_server.json"
    for _ in range(100):
        if info_path.is_file():
            info = json.loads(info_path.read_text())
            if info.get("pid") == process.pid and info.get("status") == "serving":
                print(f"Live fit view: {info['url']} (viewer PID {process.pid})", flush=True)
                return info
        if process.poll() is not None:
            raise RuntimeError(f"Live view could not start; see {run / 'live_server.log'}")
        time.sleep(.05)
    process.terminate()
    process.wait(timeout=5)
    raise RuntimeError(f"Live view startup timed out; see {run / 'live_server.log'}")
