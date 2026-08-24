"""
Record, in the generated script, which sources it was generated from.

`fit_parameters.py` imports `generated/generated_script.py` and never reads
`user_model.py`. A script left behind by an earlier translation therefore fits
the previous version of the equations, completes normally, and reports
parameters for a model that no longer exists. Nothing at run time notices.

Comparing modification times detects the common case but is a poor instrument:
it is fooled by `touch`, by a script regenerated incorrectly, and by any clone,
copy or archive extract that gives every file the same timestamp. It also
reports a comment-only edit as a mismatch.

So `/pfit-jax` stamps the script with a hash of each source, taken after
stripping comments and blank lines. The comparison is then about content:
immune to timestamps, and quiet about edits that cannot change the translation.

Deliberately dependency-free (stdlib only) so the fit entry points can check
this before importing jax, and so `tools/` can use it without the solver stack.
"""

import hashlib
import re
from pathlib import Path

__all__ = ["normalized_hash", "build_stamp", "read_stamp", "verify_stamp",
           "write_stamp", "STAMP_PREFIX"]

STAMP_PREFIX = "# pfit-sources:"

# The two files /pfit-jax reads: the model it translates, and the config it takes
# the parameter order, integrator and step limit from.
SOURCES = ("user_model.py", "user_input.yaml")


def _normalize(text: str) -> str:
    """
    Strip what cannot affect the translation: comments, blank lines, trailing
    whitespace.

    Both sources use `#` comments, so one rule covers them. A `#` inside a
    string literal would be over-stripped; that costs a spurious mismatch, never
    a missed one, which is the right direction for a guard.
    """
    lines = []
    for line in text.splitlines():
        line = re.sub(r"\s+$", "", line.split("#")[0])
        if line.strip():
            lines.append(line)
    return "\n".join(lines)


def normalized_hash(path) -> str | None:
    """Short content hash of one source, or None when it is absent."""
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256(_normalize(path.read_text()).encode("utf-8"))
    return digest.hexdigest()[:16]


def build_stamp(session_dir) -> str:
    """The stamp line for a session's current sources."""
    session_dir = Path(session_dir)
    parts = []
    for name in SOURCES:
        location = ("inputs" if name.endswith(".yaml") else "generated")
        parts.append(f"{name}={normalized_hash(session_dir / location / name)}")
    return f"{STAMP_PREFIX} " + " ".join(parts)


def read_stamp(script_path) -> dict | None:
    """
    The stamp recorded in a generated script, or None when it carries none.

    Only the first few lines are scanned: the stamp is written at the top, and a
    hash-like string appearing later in the file should not be mistaken for one.
    """
    script_path = Path(script_path)
    if not script_path.is_file():
        return None
    with open(script_path, "r", encoding="utf-8") as handle:
        for _ in range(20):
            line = handle.readline()
            if not line:
                break
            if line.startswith(STAMP_PREFIX):
                fields = line[len(STAMP_PREFIX):].split()
                return dict(
                    field.split("=", 1) for field in fields if "=" in field)
    return None


def verify_stamp(session_dir) -> tuple[bool | None, str]:
    """
    Compare a script's stamp against the current sources.

    Returns `(None, reason)` when there is no stamp to compare -- an older
    script, or one written before stamping existed -- so the caller can fall
    back to modification times and say that it did.
    """
    session_dir = Path(session_dir)
    script = session_dir / "generated" / "generated_script.py"
    if not script.is_file():
        return False, "generated_script.py is missing"

    recorded = read_stamp(script)
    if recorded is None:
        return None, ("generated_script.py carries no source stamp, so its "
                      "agreement with the model cannot be verified by content")

    changed = []
    for name in SOURCES:
        location = "inputs" if name.endswith(".yaml") else "generated"
        current = normalized_hash(session_dir / location / name)
        if recorded.get(name) != current:
            changed.append(name)
    if changed:
        return False, (f"{sorted(changed)} changed since generated_script.py was "
                       f"written; the fit would use the previous translation")
    return True, "generated_script.py matches the sources it was generated from"


def write_stamp(session_dir) -> str:
    """
    Write or replace the stamp at the top of a session's generated script.

    Returns the stamp written. Idempotent: re-running replaces the existing line
    rather than accumulating them.
    """
    session_dir = Path(session_dir)
    script = session_dir / "generated" / "generated_script.py"
    if not script.is_file():
        raise FileNotFoundError(script)

    stamp = build_stamp(session_dir)
    lines = script.read_text().splitlines()
    kept = [line for line in lines if not line.startswith(STAMP_PREFIX)]

    # after a shebang and any leading module docstring delimiter, so the file
    # still reads normally
    insert_at = 1 if kept and kept[0].startswith("#!") else 0
    kept.insert(insert_at, stamp)
    script.write_text("\n".join(kept) + "\n")
    return stamp
