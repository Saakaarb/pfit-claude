"""Tests for files users and remote agents rely on after cloning the repo."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGES = ("new", "check", "jax", "run", "diagnose")


def test_readme_documents_install_and_both_agents():
    readme = (ROOT / "README.md").read_text()
    assert "pip install -r requirements.txt" in readme
    assert "Claude Code" in readme
    assert "OpenAI Codex" in readme
    for stage in STAGES:
        assert f"pfit-{stage}" in readme


def test_claude_and_codex_expose_every_workflow_stage():
    for stage in STAGES:
        assert (ROOT / ".claude" / "commands" / f"pfit-{stage}.md").is_file()
        skill = ROOT / ".agents" / "skills" / f"pfit-{stage}" / "SKILL.md"
        text = skill.read_text()
        assert f"name: pfit-{stage}" in text
        assert f".claude/commands/pfit-{stage}.md" in text


def test_agent_index_release_references_exist():
    required = (
        "docs/tunable_choices.md",
        "lib/LLM/reference/study_input_example.md",
    )
    for relative_path in required:
        assert (ROOT / relative_path).is_file(), relative_path
