#!/usr/bin/env python3
"""Serve one run on localhost. Ctrl-C stops only the viewer, never the fit."""
import argparse
import json
import os
from pathlib import Path
import signal
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.utils.live_dashboard import make_server
from lib.utils.run_store import resolve_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path)
    source.add_argument("--session", help="session name or path; resolves once at startup")
    parser.add_argument("--run", help="run ID or directory when using --session")
    parser.add_argument("--port", type=int, default=0, help="0 chooses an available local port")
    args = parser.parse_args()
    if args.run_dir:
        run = args.run_dir.resolve()
    else:
        session = Path(args.session)
        if not session.is_dir():
            session = Path(__file__).resolve().parents[1] / "sessions" / args.session
        run = resolve_run(session, args.run).resolve()
    if not run.is_dir():
        parser.error(f"Run directory does not exist: {run}")
    server = make_server(run, args.port)
    info = {"url": f"http://127.0.0.1:{server.server_port}/", "pid": os.getpid(),
            "run_id": run.name, "status": "serving"}
    info_path = run / "live_server.json"
    def save():
        temporary = run / f"live_server.{os.getpid()}.tmp"
        temporary.write_text(json.dumps(info, indent=2) + "\n")
        temporary.replace(info_path)
    save()
    print(f"Live fit view: {info['url']} (PID {os.getpid()})", flush=True)
    def stop(signum, frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, stop)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        info["status"] = "stopped"
        if info_path.is_file() and json.loads(info_path.read_text()).get("pid") == os.getpid():
            save()


if __name__ == "__main__":
    main()
