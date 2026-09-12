"""Per-run artifacts and self-contained input snapshots; never replace a run."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import threading
import uuid

import yaml


def output_root(session):
    session = Path(session)
    config = session / "inputs" / "user_input.yaml"
    if config.is_file():
        data = yaml.safe_load(config.read_text()) or {}
        return session / (data.get("paths") or {}).get("output_dir", "outputs")
    return session / "outputs"


def resolve_run(session, run=None, require_seed=False):
    """Select an explicit run, latest run, or latest successful seed; read legacy output."""
    # An explicit directory remains readable even if the working config was
    # subsequently removed or is currently being edited into invalid YAML.
    root = Path(run).parent if run and Path(run).is_dir() else output_root(session)
    if run:
        candidate = Path(run)
        candidate = candidate if candidate.is_dir() else root / run
        if not candidate.is_dir():
            raise FileNotFoundError(f"Run not found: {candidate}")
        candidates = [candidate]
    else:
        candidates = sorted((p for p in root.glob("*")
                             if p.is_dir() and (p / "run_manifest.json").is_file()),
                            key=lambda p: p.name, reverse=True)
        candidates.append(root)  # legacy flat outputs are never moved or removed
    for candidate in candidates:
        if require_seed:
            if not (candidate / "final_design_point.csv").is_file():
                continue
            manifest = candidate / "run_manifest.json"
            if manifest.is_file() and json.loads(manifest.read_text())["status"] != "completed":
                continue
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No {'initial guess' if require_seed else 'run'} found in {root}")


def run_sources(run, session):
    snapshot = Path(run) / "snapshot"
    return snapshot if snapshot.is_dir() else Path(session)


def run_config(run, session):
    sources = run_sources(run, session)
    runtime = sources / "inputs" / "run_config.yaml"
    return runtime if runtime.is_file() else sources / "inputs" / "user_input.yaml"


@contextmanager
def new_run(session, reader, mode, seed=None):
    session = Path(session).resolve()
    root = session / reader.output_dirname
    run = root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
                  + "_" + uuid.uuid4().hex[:8])
    run.mkdir(parents=True, exist_ok=False)
    manifest = {"run_id": run.name, "session": str(session), "mode": mode,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "status": "preparing", "pid": os.getpid(),
                "seed_source": str(Path(seed).resolve()) if seed else None,
                "python": platform.python_version(), "platform": platform.platform()}

    def save():
        temporary = run / "run_manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(run / "run_manifest.json")

    save()
    print(f"Run artifacts: {run}", flush=True)
    previous_handler = None
    if threading.current_thread() is threading.main_thread():
        previous_handler = signal.getsignal(signal.SIGTERM)
        def terminate(signum, frame):
            raise SystemExit(128 + signum)
        signal.signal(signal.SIGTERM, terminate)
    try:
        snapshot = run / "snapshot"
        inputs = snapshot / "inputs"
        inputs.mkdir(parents=True)
        original = session / reader.user_input_dirname / "user_input.yaml"
        shutil.copy2(original, inputs / "user_input.yaml")
        config = yaml.safe_load(original.read_text())
        for i, exp in enumerate(reader.experiments):
            source = session / reader.user_input_dirname / exp['filename']
            name = f"dataset_{i + 1}{source.suffix}"
            shutil.copy2(source, inputs / name)
            config["experiments"][i]["data_file"] = name
        # Preserve the exact original config, plus the path-adjusted config executed.
        config["paths"] = {"user_input_dir": "inputs",
                           "generated_dir": "generated", "output_dir": "outputs"}
        (inputs / "run_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        shutil.copytree(session / reader.generated_dirname, snapshot / "generated",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        if seed:
            shutil.copy2(seed, run / "seed_design_point.csv")
        repo = Path(__file__).resolve().parents[2]
        shutil.copytree(repo / "lib", snapshot / "framework" / "lib",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        shutil.copytree(repo / "tools", snapshot / "framework" / "tools",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.log"))
        for name in ("fit_parameters.py", "fit_gradient_only.py", "analyze_fit.py", "requirements.txt"):
            shutil.copy2(repo / name, snapshot / "framework" / name)
        manifest["packages"] = sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions())
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo,
                                  capture_output=True, text=True, check=False)
        manifest["git_commit"] = revision.stdout.strip() or None
        manifest["sha256"] = {str(p.relative_to(run)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in snapshot.rglob("*") if p.is_file()}
        if seed:
            manifest["sha256"]["seed_design_point.csv"] = hashlib.sha256(
                (run / "seed_design_point.csv").read_bytes()).hexdigest()
        manifest["status"] = "running"
        save()
        yield run, snapshot
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit))
                        else "failed", error=f"{type(exc).__name__}: {exc}")
        (run / "fitting_error.txt").write_text(manifest["error"] + "\n")
        raise
    else:
        manifest["status"] = "completed"
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()
        if previous_handler is not None:
            signal.signal(signal.SIGTERM, previous_handler)
