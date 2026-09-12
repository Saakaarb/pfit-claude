import os
import sys
from pathlib import Path

import numpy as np

# Reuse the session/device resolution logic from the full two-stage entry point.
# jax is imported lazily (inside run_driver) so XLA device count can be set first.
from fit_parameters import (resolve_session_dir, resolve_device_count,
                            warn_if_script_is_stale)
from lib.utils.live_view import attach as attach_live_view
from lib.utils.yamlread import YAMLReader
from lib.utils.run_store import new_run, resolve_run, run_config


def load_init_guess(session_dir: Path, seed_run=None) -> np.ndarray:
    """Load the starting design point for the gradient-only run.

    Reads the selected completed run's ``final_design_point.csv`` — the best
    point from a previous run (population search + any prior gradient refinement).
    This is what lets a user re-run *just* the gradient stage (e.g. with more
    iterations or a different optimizer) starting from where they left off.

    Args:
        session_dir (Path): Path to the session directory.

    Returns:
        numpy.ndarray: Initial guess in real parameter units, in YAML trainable order.
    """
    guess_path = resolve_run(session_dir, seed_run, require_seed=True) / "final_design_point.csv"
    if not guess_path.exists():
        raise FileNotFoundError(
            f"No initial guess found at {guess_path}\n"
            "Run the full fit (python fit_parameters.py <session>) at least once first, "
            "or place a final_design_point.csv in the session's outputs/ directory."
        )
    guess = np.atleast_1d(np.genfromtxt(guess_path, delimiter=","))
    print(f"Loaded initial guess from {guess_path}: {guess}")
    return guess


def run_driver(session_dir: Path, input_reader: YAMLReader, seed_run=None, live_web=False, web_port=0):
    """
    Execute ONLY the gradient (NODE) refinement stage for the user's ODE system.

    Skips the population-based global search and seeds the gradient optimizer from
    the session's existing final_design_point.csv. Assumes generated_script.py
    already exists (created by the /pfit-jax skill).

    Both entry points create a new timestamped run directory. The seed is
    copied and its origin recorded; no prior run artifacts are overwritten.

    Args:
        session_dir (Path): Path to the session directory.
        input_reader (YAMLReader): Parsed YAML configuration.

    Returns:
        numpy.ndarray: The refined parameter set in real (unscaled) units.
    """
    import jax  # imported after XLA_FLAGS has been configured
    from lib.utils.helper_functions import fit_gradient_only_system

    print("Launching gradient-only driver script")
    print("Available devices: ", jax.devices("cpu"))

    session_path = Path(session_dir)
    generated_dir = session_path / input_reader.generated_dirname
    generated_script = generated_dir / "generated_script.py"

    if not generated_script.exists():
        raise FileNotFoundError(
            f"generated_script.py not found at {generated_script}\n"
            "Run the /pfit-jax Claude Code skill first to generate it."
        )

    seed_path = resolve_run(session_path, seed_run, require_seed=True) / "final_design_point.csv"
    if (seed_path.parent / "snapshot").is_dir():
        from lib.utils.yamlread import read_input_file
        seed_reader = read_input_file(run_config(seed_path.parent, session_path))
        if seed_reader.trainable_parameter_names != input_reader.trainable_parameter_names:
            raise ValueError("Seed run's trainable parameter names/order differ from this session")
    with new_run(session_path, input_reader, "gradient-only", seed=seed_path) as (run, snapshot):
        if live_web:
            from lib.utils.live_dashboard import launch_dashboard
            try:
                launch_dashboard(run, web_port)
            except Exception as exc:
                print(f"[live view] {exc}; fitting continues. Start tools/live_fit_server.py "
                      f"separately with --run-dir {run}", flush=True)
        init_guess = np.atleast_1d(np.genfromtxt(run / "seed_design_point.csv", delimiter=","))
        if init_guess.size != len(input_reader.trainable_parameter_names):
            raise ValueError(f"init_guess has {init_guess.size} entries but user_input.yaml defines "
                             f"{len(input_reader.trainable_parameter_names)} trainable parameters")
        if not np.all(np.isfinite(init_guess)):
            raise ValueError("Seed must contain finite values")
        with attach_live_view(session_path, run):
            return fit_gradient_only_system(snapshot / "inputs" / "run_config.yaml", run,
                                            snapshot / "generated", snapshot, init_guess)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("session", nargs="?")
    parser.add_argument("--seed-run", help="run ID or directory; defaults to latest successful run")
    parser.add_argument("--live-web", action="store_true", help="host this run on localhost")
    parser.add_argument("--web-port", type=int, default=0, help="dashboard port; 0 selects an available port")
    args = parser.parse_args()

    if not os.path.isdir("sessions"):
        raise ValueError(
            "No sessions directory found. Please create a sessions directory "
            "and add a session subdirectory as described in the README."
        )

    session_dir = resolve_session_dir([sys.argv[0], args.session] if args.session else [sys.argv[0]])
    warn_if_script_is_stale(session_dir)

    # Expose the requested number of CPU devices to JAX before the backend
    # initializes. (NODE itself is single-point, but we keep this consistent
    # with fit_parameters.py so the environment matches.)
    n_devices = resolve_device_count(session_dir)
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={n_devices}"
    print(f"Configuring JAX with {n_devices} CPU device(s) (processors from user_input.yaml)")

    from lib.utils.helper_functions import get_input_reader

    input_file_path = Path(session_dir) / "inputs" / "user_input.yaml"
    input_reader = get_input_reader(input_file_path)

    run_driver(session_dir, input_reader, seed_run=args.seed_run,
               live_web=args.live_web, web_port=args.web_port)
