import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# NOTE: jax is imported lazily (inside run_driver / __main__) so that the XLA
# device count can be configured from the session's PROCESSORS setting *before*
# the JAX backend initializes. XMLReader is import-safe (no jax dependency).
from lib.utils.xmlread import XMLReader


def resolve_session_dir() -> Path:
    """
    Determine which session directory to run.

    Uses the command-line argument if provided (either a full path or a session
    name under sessions/), otherwise falls back to the most recently created
    subdirectory of sessions/.

    Returns:
        Path: Path to the session directory to run.
    """
    sessions_root = Path("sessions")

    if len(sys.argv) == 2:
        arg = Path(sys.argv[1])
        session_dir = arg if arg.is_dir() else sessions_root / arg
        if not os.path.isdir(session_dir):
            raise ValueError(f"Session directory {session_dir} does not exist")
        return session_dir

    if not sessions_root.exists() or not any(sessions_root.iterdir()):
        print("No session_dir provided and sessions/ is empty.")
        print("Usage: python fit_parameters.py <session_dir>")
        sys.exit(1)

    session_dirs = [d for d in sessions_root.iterdir() if d.is_dir()]
    session_dirs.sort(key=lambda d: d.stat().st_ctime, reverse=True)
    session_dir = session_dirs[0]
    print(f"No session_dir provided. Using most recently created session: {session_dir}")
    return session_dir


def resolve_device_count(session_dir: Path) -> int:
    """
    Number of CPU devices to expose to JAX for population-parallel loss evaluation.

    Fully configurable from the session's user_input.xml via PROCESSORS, with no
    upper limit. Falls back to the number of logical CPUs when PROCESSORS is unset
    or the XML cannot be read.

    Args:
        session_dir (Path): Path to the session directory.

    Returns:
        int: Number of CPU devices to request from XLA (always >= 1).
    """
    fallback = os.cpu_count() or 1
    try:
        xml_path = Path(session_dir) / "inputs" / "user_input.xml"
        reader = XMLReader()
        reader.read_XML(ET.parse(xml_path).getroot())
        if reader.processors:
            return max(1, int(reader.processors))
    except Exception as exc:
        print(f"Could not read PROCESSORS from XML ({exc}); falling back to {fallback} CPU device(s)")
    return fallback


def run_driver(session_dir: Path, input_reader: XMLReader):
    """
    Execute the parameter fitting workflow for the user's ODE system.

    Assumes that generated/generated_script.py already exists — created by
    running the /pfit-jax Claude Code skill beforehand.

    Args:
        session_dir (Path): Path to the session directory.
        input_reader (XMLReader): Parsed XML configuration.
    """
    import jax  # imported after XLA_FLAGS has been configured
    from lib.utils.helper_functions import fit_generic_system

    print("Launching driver script")
    print("Available devices: ", jax.devices("cpu"))

    session_path = Path(session_dir)
    path_to_input = session_path / input_reader.user_input_dirname / "user_input.xml"
    path_to_output_dir = session_path / input_reader.output_dirname
    generated_dir = session_path / input_reader.generated_dirname
    generated_script = generated_dir / "generated_script.py"

    if not generated_script.exists():
        raise FileNotFoundError(
            f"generated_script.py not found at {generated_script}\n"
            "Run the /pfit-jax Claude Code skill first to generate it."
        )

    if path_to_output_dir.exists():
        shutil.rmtree(path_to_output_dir)
    path_to_output_dir.mkdir()

    print("Launching fitting process...")
    fit_generic_system(path_to_input, path_to_output_dir, generated_dir, session_path)


if __name__ == "__main__":

    if not os.path.isdir("sessions"):
        raise ValueError(
            "No sessions directory found. Please create a sessions directory "
            "and add a session subdirectory as described in the README."
        )

    session_dir = resolve_session_dir()

    # Expose exactly the requested number of CPU devices to JAX. This MUST happen
    # before JAX initializes its backend, so it is set here — from the jax-free
    # XML reader — prior to importing any jax-dependent module.
    n_devices = resolve_device_count(session_dir)
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={n_devices}"
    print(f"Configuring JAX with {n_devices} CPU device(s) (PROCESSORS from XML)")

    from lib.utils.helper_functions import get_input_reader

    input_file_path = Path(session_dir) / "inputs" / "user_input.xml"
    input_reader = get_input_reader(input_file_path)

    run_driver(session_dir, input_reader)
