# pfit remote-agent instructions

This repository fits ODE parameters through a five-stage agent workflow:

`pfit new -> pfit check -> pfit jax -> pfit run -> pfit diagnose`

Before doing pfit work, read `.claude/CLAUDE.md`. It is the project index and
points to the canonical reference for each rule. The files under
`.claude/commands/` are the canonical procedures for both Claude Code and
Codex; do not create a second implementation of those procedures.

Preserve these invariants:

- Use YAML and CSV inputs only. Do not add XML inputs or parsers.
- Never invent equations, experimental data, initial conditions, or scientific
  meaning for ambiguous columns.
- Generate `user_input.yaml` and `user_model.py` together so parameter and state
  ordering cannot drift.
- `pfit jax` includes an LLM translation step. Validate and source-stamp its
  output deterministically before accepting it.
- Never begin a fit when readiness checks fail or generated code is stale.
- Keep each run immutable and diagnose a specific run from its snapshots.
- Prefer existing `lib/` and `tools/` functionality over duplicate layers.
- Keep workflow tests focused on user-visible behavior and core fitting logic.
- Do not modify unrelated session results or untracked research artifacts.

For a named workflow stage, follow its repository skill or the corresponding
`.claude/commands/pfit-<stage>.md` file exactly.
