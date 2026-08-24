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
    # the default must be stated where the config default is documented
    "yaml_format.md": {"Kvaerno5"},
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
    referenced = set(re.findall(r"`((?:lib|tools|tests)/[\w./-]+\.(?:md|py|yaml|txt))`", text))
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

    A rule justified by "in sessions/foo this happened" couples the guidance to a
    session that can change or be deleted, and reads as a special case rather
    than something to apply generally. State the mechanism instead; keep the
    evidence in the commit message.

    CLAUDE.md is exempt: its code map legitimately names what lives in
    sessions/, which is navigation rather than guidance.
    """
    session_names = set()
    d = REPO_ROOT / "sessions"
    if d.is_dir():
        session_names = {p.name for p in d.iterdir() if p.is_dir()}
    assert session_names, "expected some committed sessions to check against"

    for path in REFERENCE_FILES + COMMAND_FILES:
        text = path.read_text()
        named = sorted(n for n in session_names if n in text)
        assert not named, (
            f"{path.name} cites specific sessions {named}. Instructions state "
            f"general rules; put the supporting measurements in the commit "
            f"message or a session README instead."
        )


# ---------------------------------------------------------------------------
# the solver-family rule weighs two axes, not one
# ---------------------------------------------------------------------------

# Phrasings that make smoothness a gate which short-circuits the stiffness
# question. Each of these was in the instruction layer at some point and each
# produces the same failure: on a system that is BOTH non-smooth and stiff, the
# rule confidently recommends an explicit solver, whose step is then pinned by
# the fastest eigenvalue for the whole interval so that no solve completes at
# any MAX_STEPS.
GATING_PHRASES = [
    "check smoothness before stiffness",
    "smoothness before stiffness",
    "but only for a smooth right-hand side",
    "the rule reverses for a non-smooth rhs",
    "asymmetry reverses for a non-smooth",
]

# The generated digests and the config they are generated FROM are checked too.
# The first version of this test scanned only hand-written prose, and a stale
# copy of the gated rule survived in lib/LLM/api/diffrax.md for exactly that
# reason -- an agent reading the digest would have got the superseded rule with
# no indication it was superseded.
GENERATED_AND_SOURCES = (
    sorted(API.glob("*.md")) + [REPO_ROOT / "tools" / "gen_api_context.yaml"]
)


@pytest.mark.parametrize("path", PROSE_FILES + GENERATED_AND_SOURCES,
                         ids=lambda p: p.name)
def test_solver_choice_is_not_gated_on_smoothness_alone(path):
    """Smoothness must not be stated as a gate on the stiffness question.

    Stiffness and smoothness fail in different shapes: stiffness constrains
    every step over the whole interval with no adaptive escape, whereas a
    discontinuity only breaks the implicit inner solve AT the switching
    instants. They are therefore weighed by magnitude, and a rule that checks
    only smoothness inverts the correct answer for a stiff, non-smooth system.
    """
    text = path.read_text().lower()
    found = [p for p in GATING_PHRASES if p in text]
    assert not found, (
        f"{path.name} states smoothness as a gate on solver choice ({found}). "
        f"R1 in tuning_rules.md weighs stiffness and smoothness together; see "
        f"its slaved-vs-active distinction. If this is a generated digest, fix "
        f"the text in tools/gen_api_context.yaml and regenerate — editing the "
        f"digest by hand will be overwritten."
    )


def test_r1_covers_both_axes_and_the_slaved_mode_distinction():
    """tuning_rules.md owns the solver decision and must state both axes.

    The slaved/active distinction is the load-bearing part: only a fast mode
    that has saturated is stiffness an implicit method can convert into larger
    steps. A fast mode that is still active is genuine fast dynamics that every
    method must resolve, so measuring the raw ratio of rate constants conflates
    the two and overstates the case for an implicit solver.
    """
    text = (REFERENCE / "tuning_rules.md").read_text().lower()
    for concept in ("slaved", "chattering", "one-way", "stability limit",
                    "accuracy limit"):
        assert concept in text, (
            f"tuning_rules.md no longer mentions '{concept}' — R1 needs both "
            f"axes and the mechanism that makes them weighable."
        )


# ---------------------------------------------------------------------------
# the cold-start invariant
# ---------------------------------------------------------------------------

# Artifacts that only exist because a fit already ran. Naming one as an input to
# a SETUP skill means that skill's guidance was written for a problem somebody
# has already solved, which is the one case that never occurs in deployment.
SOLUTION_ARTIFACTS = [
    "final_design_point.csv",
    "result_solution_exp",
    "sloppiness_report.txt",
    "fit_diagnosis.txt",
    "NODE_fitting.log",
    "de_fitting.log",
    "pso_fitting.log",
]

# The skills that make setup choices. /pfit-diagnose is deliberately absent: it
# runs after a fit and reads that fit's own record, which is evidence about the
# optimizer's behaviour rather than a reference answer. /pfit-run is absent for
# the same reason -- it produces that record, and reporting what it just wrote
# is not deriving a setup choice from a stored answer.
SETUP_COMMANDS = [
    "pfit-new.md",
    "pfit-check.md",
    "pfit-jax.md",
]


@pytest.mark.parametrize("name", SETUP_COMMANDS)
def test_setup_skills_never_read_solution_artifacts(name):
    """Setup happens without the answer, so setup procedures cannot read one.

    The failure this prevents is quiet. Guidance derived from a stored solution
    still reads as sound and still passes on any session that has one; it breaks
    only on a genuinely new problem, where there is nothing to derive it from
    and no way to notice that it was.
    """
    text = (COMMANDS / name).read_text()
    found = [a for a in SOLUTION_ARTIFACTS if a in text]
    assert not found, (
        f"{name} is a setup skill but names solution artifacts {found}. "
        f"See lib/LLM/reference/cold_start.md: setup may read the equations, "
        f"the dataset and the bounds only."
    )


def test_every_skill_loads_the_cold_start_invariant():
    """The invariant binds all five skills, so all five must reference it."""
    for name in SETUP_COMMANDS + ["pfit-diagnose.md"]:
        text = (COMMANDS / name).read_text()
        assert "cold_start.md" in text, (
            f"{name} does not reference cold_start.md. The invariant applies to "
            f"every skill, including diagnosis."
        )


def test_cold_start_is_owned_by_one_file():
    """The invariant has a single home; others point at it rather than restate."""
    owner = REFERENCE / "cold_start.md"
    assert owner.exists(), "cold_start.md is the canonical home of the invariant"
    text = owner.read_text().lower()
    for concept in ("worst case over the bounds", "reachab", "setup", "diagnosis"):
        assert concept in text, f"cold_start.md no longer covers '{concept}'"

    # the substitutes are defined once, in the owner
    others = [p for p in REFERENCE_FILES if p.name != "cold_start.md"]
    restating = [
        p.name for p in others
        if "reachability check" in p.read_text().lower()
    ]
    assert not restating, (
        f"{restating} restate the reachability check; cold_start.md owns it."
    )


def test_tuning_rules_defers_to_the_cold_start_owner():
    """tuning_rules.md applies the invariant but must not restate it.

    Recommendations are the place the invariant bites hardest, so the file has
    to bind itself to it — while leaving the definition, the artifact list and
    the substitutes in cold_start.md, so there is one place to change them.
    """
    text = (REFERENCE / "tuning_rules.md").read_text()
    assert "cold_start.md" in text, (
        "tuning_rules.md must bind its recommendations to the cold-start "
        "invariant; every rule in it is meant to be computable without a "
        "reference solution."
    )
