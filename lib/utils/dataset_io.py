"""
Dataset CSV loading, shared by the fit pipeline and the input checker.

This module is deliberately free of jax and of any project import, so
`tools/check_dataset.py` can load a dataset exactly as a fit would without
pulling in the solver stack. Both callers MUST go through `load_dataset` --
the checker's whole value is that what it sees is what the fit will see, and
two hand-rolled genfromtxt calls would eventually drift apart.

Header support
--------------
The framework consumes the CSV as a pure numeric matrix: column 0 is time and
columns 1.. are observables, positionally. It has no way to know what those
columns MEAN, and a user handing over a fresh CSV has no way to say so.

A real data file usually already carries that statement as a header row, so one
is accepted here and skipped on load. It is metadata for the setup skills, not
input to the solve: `read_header` returns it so `/pfit-new` can seed the
`columns` block and `/pfit-check` can verify the two still agree.

A header is detected rather than declared -- a first row that does not parse as
numbers is a header. That is unambiguous because column 0 is time and a time
value is never blank or non-numeric.
"""

import numpy as np

__all__ = ["has_header", "read_header", "load_dataset"]


def _fields(line: str) -> list[str]:
    return [field.strip() for field in line.strip().split(",")]


def _is_header_line(line: str) -> bool:
    """
    True when `line` cannot be read as a row of numbers.

    Blank fields are skipped rather than rejected: they are legitimate missing
    measurements under the staggered-data workflow. A line of nothing but blanks
    is not a header.
    """
    for field in _fields(line):
        if not field:
            continue
        try:
            float(field)
        except ValueError:
            return True
    return False


def _first_line(path) -> str:
    with open(path, "r", encoding="utf-8-sig") as handle:
        return handle.readline()


def has_header(path) -> bool:
    """True when the file's first row is a header rather than data."""
    return _is_header_line(_first_line(path))


def read_header(path) -> list[str] | None:
    """The header fields, or None when the file has no header."""
    line = _first_line(path)
    return _fields(line) if _is_header_line(line) else None


def load_dataset(path) -> np.ndarray:
    """
    Load one dataset CSV as a float array, skipping a header row if present.

    Missing values become NaN, which is meaningful: deliberate blanks are the
    staggered-data workflow, accidental ones are a defect. Distinguishing them
    is the checker's job, not this function's.
    """
    with open(path, "r", encoding="utf-8-sig") as handle:
        skip = 1 if _is_header_line(handle.readline()) else 0
        handle.seek(0)
        return np.genfromtxt(handle, dtype=float, delimiter=",", skip_header=skip)
