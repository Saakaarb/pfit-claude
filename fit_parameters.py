import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
import jax

import shutil
import sys
from pathlib import Path
from lib.utils.helper_functions import fit_generic_system
from lib.utils.helper_functions import get_input_reader
from lib.utils.xmlread import XMLReader


def run_driver(session_dir: Path, input_reader: XMLReader):
    """
    Execute the parameter fitting workflow for the user's ODE system.

    Assumes that generated/generated_script.py already exists — created by
    running the /pfit-jax Claude Code skill beforehand.

    Args:
        session_dir (Path): Path to the session directory.
        input_reader (XMLReader): Parsed XML configuration.
    """
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

    sessions_root = Path("sessions")

    if len(sys.argv) == 2:
        session_dir = sessions_root / Path(sys.argv[1])
        if not os.path.isdir(session_dir):
            raise ValueError(f"Session directory {session_dir} does not exist")
    else:
        if not sessions_root.exists() or not any(sessions_root.iterdir()):
            print("No session_dir provided and sessions/ is empty.")
            print("Usage: python fit_parameters.py <session_dir>")
            sys.exit(1)
        session_dirs = [d for d in sessions_root.iterdir() if d.is_dir()]
        session_dirs.sort(key=lambda d: d.stat().st_ctime, reverse=True)
        session_dir = str(session_dirs[0])
        print(f"No session_dir provided. Using most recently created session: {session_dir}")

    input_file_path = Path(session_dir) / "inputs" / "user_input.xml"
    input_reader = get_input_reader(input_file_path)

    run_driver(session_dir, input_reader)
