"""Guard the single-source-of-truth property of the instruction files.

The instruction layer is prose, so nothing about it is checked by running the
code. Before this file existed, the same rules were restated in CLAUDE.md, the
command files and the lib/LLM instructions, and they had already drifted: the
old user_file_check_instructions.txt listed `Euler` as a valid INTEGRATOR
(it is not adaptive, so it cannot work with the PIDController the framework
always uses) and omitted six solvers that do work — while the command file it
was loaded alongside said the opposite.

These tests hold the structure in place:

- every rule has exactly one home (no library facts restated outside the
  generated digests, no rule restated in the index),
- every cross-reference resolves,
- every reference file declares what it owns.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE = REPO_ROOT / "lib" / "LLM" / "reference"
API = REPO_ROOT / "lib" / "LLM" / "api"
COMMANDS = REPO_ROOT / ".claude" / "commands"
CLAUDE_MD = REPO_ROOT / ".claude" / "CLAUDE.md"

REFERENCE_FILES = sorted(REFERENCE.glob("*.md"))
COMMAND_FILES = sorted(COMMANDS.glob("*.md"))
PROSE_FILES = REFERENCE_FILES + COMMAND_FILES + [CLAUDE_MD]


def test_reference_directory_is_populated():
    assert len(REFERENCE_FILES) >= 8, "expected the canonical reference set"


def test_no_legacy_instruction_txt_files_remain():
    """The .txt instruction files were replaced by lib/LLM/reference/*.md."""
    leftovers = list((REPO_ROOT / "lib" / "LLM").glob("*.txt"))
    assert not leftovers, f"legacy instruction files still present: {leftovers}"


@pytest.mark.parametrize("path", REFERENCE_FILES, ids=lambda p: p.name)
def test_reference_files_declare_ownership(path):
    """Frontmatter records what the file owns and who reads it.

    This is what makes "where does rule X live?" answerable without grepping
    the whole tree.
    """
    text = path.read_text()
    assert text.startswith("---\n"), f"{path.name} has no frontmatter"
    frontmatter = text.split("---", 2)[1]
    for key in ("topic:", "consumed_by:", "generated:", "owns:"):
        assert key in frontmatter, f"{path.name} frontmatter is missing {key}"


# ---------------------------------------------------------------------------
# library facts live only in the generated digests
# ---------------------------------------------------------------------------

# Solver names that exist in the pinned diffrax. Naming one in hand-written
# prose duplicates the generated table and is exactly how the old stale list
# came to contradict reality.
SOLVER_NAMES = [
    "Kvaerno3", "Kvaerno4", "Kvaerno5", "KenCarp3", "KenCarp4", "KenCarp5",
    "Dopri5", "Dopri8", "Tsit5", "Bosh3", "Ralston", "ImplicitEuler",
    "SemiImplicitEuler", "LeapfrogMidpoint", "ReversibleHeun",
]

# Files allowed to name solvers, and why.
SOLVER_NAME_ALLOWED = {
    # the default must be stated where the XML default is documented
    "xml_format.md": {"Kvaerno5"},
    "jax_translation.md": {"Kvaerno5"},
    "validation_rules.md": {"Kvaerno5"},
}


@pytest.mark.parametrize("path", PROSE_FILES, ids=lambda p: p.name)
def test_solver_names_are_not_duplicated_outside_the_digests(path):
    """Hand-written prose must point at lib/LLM/api/diffrax.md, not list solvers.

    The generated table is the single source of truth for which solvers exist
    and which are usable. Only the framework default may be named, and only
    where the default itself is documented.
    """
    text = path.read_text()
    allowed = SOLVER_NAME_ALLOWED.get(path.name, set())
    found = {name for name in SOLVER_NAMES if re.search(rf"\b{name}\b", text)}
    unexpected = found - allowed
    assert not unexpected, (
        f"{path.name} names solvers {sorted(unexpected)} in prose. The solver "
        f"list is generated — reference lib/LLM/api/diffrax.md instead."
    )


@pytest.mark.parametrize("path", PROSE_FILES, ids=lambda p: p.name)
def test_results_codes_are_not_enumerated_outside_their_owner(path):
    """The RESULTS table is generated; only the failure-mask rule may name codes."""
    codes = set(re.findall(r"RESULTS\.(\w+)", path.read_text()))
    if path.name == "jax_translation.md":
        # the mask rule must show the expression, and names the codes it warns about
        assert "successful" in codes
        return
    assert not codes, (
        f"{path.name} enumerates RESULTS codes {sorted(codes)}; that table is "
        f"generated (lib/LLM/api/diffrax.md) and the mask rule lives in "
        f"jax_translation.md"
    )


# ---------------------------------------------------------------------------
# the index states no rules
# ---------------------------------------------------------------------------

def test_claude_md_is_an_index_not_a_rulebook():
    """CLAUDE.md must point at the canonical files, not restate them.

    Length is the practical proxy: it grew to 523 lines by accumulating copies
    of rules that live elsewhere.
    """
    lines = CLAUDE_MD.read_text().splitlines()
    assert len(lines) < 120, (
        f"CLAUDE.md is {len(lines)} lines — it is drifting back into restating "
        f"rules that belong in lib/LLM/reference/"
    )


def test_claude_md_links_every_reference_file():
    """A canonical file nobody can find is not a source of truth."""
    text = CLAUDE_MD.read_text()
    for path in REFERENCE_FILES:
        rel = f"lib/LLM/reference/{path.name}"
        assert rel in text, f"CLAUDE.md does not link {rel}"


# ---------------------------------------------------------------------------
# cross-references resolve
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", PROSE_FILES, ids=lambda p: p.name)
def test_referenced_repo_paths_exist(path):
    """Every lib/... or tools/... path named in prose must exist.

    Catches the dangling pointers a refactor leaves behind — the failure mode
    that turns "read the canonical file" into a dead end.
    """
    text = path.read_text()
    referenced = set(re.findall(r"`((?:lib|tools|tests)/[\w./-]+\.(?:md|py|xml|txt))`", text))
    missing = [r for r in referenced if not (REPO_ROOT / r).exists()]
    assert not missing, f"{path.name} references non-existent paths: {sorted(missing)}"


@pytest.mark.parametrize("path", REFERENCE_FILES, ids=lambda p: p.name)
def test_sibling_references_resolve(path):
    """Bare `foo.md` inside a reference file must name a real sibling."""
    body = path.read_text().split("---", 2)[-1]
    siblings = {p.name for p in REFERENCE_FILES}
    for name in set(re.findall(r"`(\w+\.md)`", body)):
        assert name in siblings, (
            f"{path.name} references sibling `{name}`, which does not exist in "
            f"lib/LLM/reference/"
        )


@pytest.mark.parametrize("path", COMMAND_FILES, ids=lambda p: p.name)
def test_commands_point_at_reference_files(path):
    """Each skill must name the reference files it depends on."""
    text = path.read_text()
    assert "lib/LLM/reference/" in text or "lib/LLM/api/" in text, (
        f"{path.name} names no reference file — its rules would have to come "
        f"from recall"
    )


def test_rules_are_general_not_case_studies():
    """The instruction layer must state rules, not anecdotes about one session.

    A rule justified by "in examples/foo this happened" couples the guidance to a
    session that can change or be deleted, and reads as a special case rather
    than something to apply generally. State the mechanism instead; keep the
    evidence in the commit message.

    CLAUDE.md is exempt: its code map legitimately names what lives in
    examples/, which is navigation rather than guidance.
    """
    session_names = set()
    for parent in ("examples", "sessions"):
        d = REPO_ROOT / parent
        if d.is_dir():
            session_names |= {p.name for p in d.iterdir() if p.is_dir()}
    assert session_names, "expected some committed sessions to check against"

    for path in REFERENCE_FILES + COMMAND_FILES:
        text = path.read_text()
        named = sorted(n for n in session_names if n in text)
        assert not named, (
            f"{path.name} cites specific sessions {named}. Instructions state "
            f"general rules; put the supporting measurements in the commit "
            f"message or a session README instead."
        )
