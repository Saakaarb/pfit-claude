"""Shared fixtures for the pfit-claude test suite.

The integration tests all follow the same shape: copy a committed session
fixture into a tmp directory, optionally patch a handful of XML settings, and
run a fit. Copying matters — a fit deletes and rewrites `outputs/`, so running
against the committed fixture in place would dirty the repo and make tests
order-dependent.
"""

import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"

# Ground truth used to generate every decay fixture's data (see the XML header).
DECAY_TRUE_PARAMS = {"k1": 1.0, "k2": 0.3}
DECAY_PARAM_ORDER = ["k1", "k2"]

# Robertson's classic rate constants; tests/robertson_session's data is a
# noiseless solution of the system at these values.
ROBERTSON_TRUE_PARAMS = [0.04, 3.0e7, 1.0e4]

sys.path.insert(0, str(REPO_ROOT))


def set_xml_setting(xml_path: Path, section: str, key: str, value) -> None:
    """Set (or insert) `<P> key = value </P>` inside a named XML section.

    `section` is a top-level tag such as POPULATION_OPT or GRADIENT_OPT. The key
    is matched on the text left of the `=`, mirroring how `XMLReader` parses it.
    NUM_ITERS exists in both optimizer sections, which is exactly why the
    section must be given explicitly.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    block = root.find(section)
    if block is None:
        raise ValueError(f"{xml_path} has no <{section}> section")
    settings = block.find("SETTINGS")
    if settings is None:
        raise ValueError(f"<{section}> in {xml_path} has no <SETTINGS>")

    for p in settings.findall("P"):
        if p.text and p.text.split("=")[0].strip() == key:
            p.text = f" {key} = {value} "
            break
    else:
        new = ET.SubElement(settings, "P")
        new.text = f" {key} = {value} "

    tree.write(xml_path, encoding="utf-8", xml_declaration=True)


def remove_xml_setting(xml_path: Path, section: str, key: str) -> None:
    """Delete a `<P> key = ... </P>` entry, so the reader falls back to its default."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    settings = root.find(section).find("SETTINGS")
    for p in list(settings.findall("P")):
        if p.text and p.text.split("=")[0].strip() == key:
            settings.remove(p)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)


@pytest.fixture(autouse=True)
def clear_jax_caches_between_tests():
    """Release compiled XLA executables after every test.

    `CreatedClass._compute_loss` is jitted with `static_argnums=(0,)`, so every
    problem object becomes a distinct compilation key and its executables stay
    alive for the life of the process. A suite that runs a dozen independent
    fits therefore accumulates them all; clearing between tests keeps each fit
    as isolated as it would be in a fresh `python fit_parameters.py` run.
    """
    yield
    try:
        import gc

        import jax

        jax.clear_caches()
        gc.collect()
    except Exception:  # never let cleanup fail a test
        pass


@pytest.fixture
def make_session(tmp_path):
    """Factory: copy a committed fixture session into tmp_path and patch its XML.

    Usage:
        session = make_session("decay_session",
                               population={"NUM_ITERS": 2},
                               gradient={"GRADIENT_OPTIMIZER": "adam"})
    """

    def _make(name: str, population: dict | None = None,
              gradient: dict | None = None, dest_name: str | None = None) -> Path:
        source = FIXTURES / name
        if not source.is_dir():
            raise ValueError(f"no such fixture: {source}")
        dest = tmp_path / (dest_name or name)
        shutil.copytree(source, dest)

        xml_path = dest / "inputs" / "user_input.xml"
        for key, value in (population or {}).items():
            set_xml_setting(xml_path, "POPULATION_OPT", key, value)
        for key, value in (gradient or {}).items():
            set_xml_setting(xml_path, "GRADIENT_OPT", key, value)
        return dest

    return _make


@pytest.fixture
def run_fit():
    """Run the full two-stage fit on a session directory and return the result."""

    def _run(session_dir: Path):
        from fit_parameters import run_driver
        from lib.utils.helper_functions import get_input_reader

        reader = get_input_reader(session_dir / "inputs" / "user_input.xml")
        return run_driver(session_dir, reader)

    return _run
