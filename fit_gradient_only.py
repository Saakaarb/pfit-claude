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


def load_init_guess(session_dir: Path) -> np.ndarray:
    """Load the starting design point for the gradient-only run.

    Reads the session's existing ``outputs/final_design_point.csv`` — the best
    point from a previous run (population search + any prior gradient refinement).
    This is what lets a user re-run *just* the gradient stage (e.g. with more
    iterations or a different optimizer) starting from where they left off.

    Args:
        session_dir (Path): Path to the session directory.

    Returns:
        numpy.ndarray: Initial guess in real parameter units, in YAML trainable order.
    """
    guess_path = Path(session_dir) / "outputs" / "final_design_point.csv"
    if not guess_path.exists():
        raise FileNotFoundError(
            f"No initial guess found at {guess_path}\n"
            "Run the full fit (python fit_parameters.py <session>) at least once first, "
            "or place a final_design_point.csv in the session's outputs/ directory."
        )
    guess = np.atleast_1d(np.genfromtxt(guess_path, delimiter=","))
    print(f"Loaded initial guess from {guess_path}: {guess}")
    return guess


def run_driver(session_dir: Path, input_reader: YAMLReader):
    """
    Execute ONLY the gradient (NODE) refinement stage for the user's ODE system.

    Skips the population-based global search and seeds the gradient optimizer from
    the session's existing final_design_point.csv. Assumes generated_script.py
    already exists (created by the /pfit-jax skill).

    Unlike fit_parameters.py, this does NOT delete the output directory — the
    prior population-search logs and the seed design point are preserved.

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
    path_to_input = session_path / input_reader.user_input_dirname / "user_input.yaml"
    path_to_output_dir = session_path / input_reader.output_dirname
    generated_dir = session_path / input_reader.generated_dirname
    generated_script = generated_dir / "generated_script.py"

    if not generated_script.exists():
        raise FileNotFoundError(
            f"generated_script.py not found at {generated_script}\n"
            "Run the /pfit-jax Claude Code skill first to generate it."
        )

    # Load the seed BEFORE touching outputs (it lives there and must be preserved).
    init_guess = load_init_guess(session_path)

    # Outputs dir must exist; it is intentionally NOT wiped so population-search
    # logs and the seed point survive.
    path_to_output_dir.mkdir(parents=True, exist_ok=True)

    print("Launching gradient-only fitting process...")

    # See the note in fit_parameters.run_driver: a no-op unless stdout is a
    # terminal and the live view is enabled.
    with attach_live_view(session_path, path_to_output_dir):
        return fit_gradient_only_system(path_to_input, path_to_output_dir, generated_dir,
                                        session_path, init_guess)


if __name__ == "__main__":

    if not os.path.isdir("sessions"):
        raise ValueError(
            "No sessions directory found. Please create a sessions directory "
            "and add a session subdirectory as described in the README."
        )

    session_dir = resolve_session_dir()
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

    run_driver(session_dir, input_reader)
