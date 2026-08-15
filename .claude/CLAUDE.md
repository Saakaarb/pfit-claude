# pfit-claude

Fits unknown parameters of a system of ODEs to time-series data, using a
population search followed by gradient refinement. Claude Code skills generate
and validate the code; the optimization pipeline is plain Python.

**This file is an index. It states no rules — every rule lives in exactly one
file below. Open the relevant one rather than working from memory.**

## Where the rules live

| Topic | Canonical file |
|---|---|
| **The cold-start invariant — no solution is available when setup choices are made**, which artifacts are off-limits during setup, and the substitutes for answer-derived evidence | `lib/LLM/reference/cold_start.md` |
| What the project is, the pipeline, session layout, the multi-experiment execution model, post-fit diagnostics, reproducibility, **the Python environment** | `lib/LLM/reference/project_context.md` |
| `user_input.yaml` schema — every section, field, default and valid value | `lib/LLM/reference/yaml_format.md` |
| Hard constraints on the dataset CSV and the config | `lib/LLM/reference/input_constraints.md` |
| The three `user_model.py` functions, and how to generate a skeleton | `lib/LLM/reference/user_model_contract.md` |
| Converting pseudocode to `generated_script.py` | `lib/LLM/reference/jax_translation.md` |
| What `/pfit-check` validates, its thresholds, the report format | `lib/LLM/reference/validation_rules.md` |
| How to auto-correct inputs from a validation report | `lib/LLM/reference/correction_rules.md` |
| How to recommend settings before a fit, from the model and the data | `lib/LLM/reference/tuning_rules.md` |
| How to diagnose a completed fit from its outputs | `lib/LLM/reference/diagnosis_rules.md` |
| The catalogue of every choice that affects solve and fit quality (human-facing) | `docs/tunable_choices.md` |
| Observables sampled at different time points | `lib/LLM/reference/staggered_data.md` |

**Read `project_context.md` before running any Python command** — it holds the
venv rule and the regeneration policy for the digests below.

## Generated API digests — never hand-edit

Version-pinned to the packages installed in `./venv`, produced by
`tools/gen_api_context.py` and guarded by `tests/test_api_digest.py`. They are
the only authority on library APIs; never write a diffrax/jax/optax call or name
a solver from memory.

| Digest | Covers |
|---|---|
| `lib/LLM/api/diffrax.md` | usable solver classes, the full `RESULTS` table, `diffeqsolve`/`SaveAt`/`PIDController`/adjoint signatures |
| `lib/LLM/api/jax.md` | available `jnp` functions and the tracing rules |
| `lib/LLM/api/optax.md` | optimizer and schedule signatures with real defaults |
| `lib/LLM/api/population_optimizers.md` | scipy DE and pyswarms options |
| `lib/LLM/api/MANIFEST.md` | the pinned version stamp |

## Skills

| Command | Does | Procedure file |
|---|---|---|
| `/pfit-new` | **the entry point.** equations (from a paper, or from the user) -> config + populated `user_model.py`, written together | `.claude/commands/pfit-new.md` |
| `/pfit-skeleton` | EXISTING config -> `user_model.py` skeleton; not an entry point | `.claude/commands/pfit-skeleton.md` |
| `/pfit-check` | validate + auto-correct the inputs, and recommend settings | `.claude/commands/pfit-check.md` |
| `/pfit-jax` | `user_model.py` -> `generated_script.py` | `.claude/commands/pfit-jax.md` |
| `/pfit-diagnose` | completed fit -> diagnosis + what to change | `.claude/commands/pfit-diagnose.md` |

Each command file is procedure only and names the reference files it requires.

## Code map

| Path | Contains |
|---|---|
| `fit_parameters.py` | entry point: full two-stage fit |
| `fit_gradient_only.py` | entry point: gradient stage only, seeded from a previous fit |
| `analyze_fit.py` | entry point: re-run post-fit diagnostics on a completed session |
| `lib/utils/yamlread.py` | config parsing |
| `lib/utils/helper_functions.py` | problem object, per-experiment dispatch, stage orchestration |
| `lib/algorithms/{PSO,DE,NODE}/` | the three optimizers |
| `lib/utils/sloppiness.py` | post-fit identifiability diagnostic |
| `lib/utils/live_view.py` | the live convergence view: reads the iteration logs, draws the plot, and attaches itself to a fit running in-process |
| `lib/utils/output_sample.py` | template for the generated script |
| `lib/utils/user_model_sample_{un,}populated.py` | skeleton and worked model |
| `lib/utils/user_input_sample.yaml` | worked config |
| `sessions/` | every session, worked and in-progress alike; a fit is run against one of these. **Not agent input** — the skills take their templates from `lib/utils/*_sample*`, never from a session |
| `tools/gen_api_context.py` | regenerates the API digests |
| `tools/check_dataset.py` | measures a session's dataset CSVs against the structural requirements the loader and diffrax impose but do not enforce; `/pfit-check` runs it. Reports facts only — severity is owned by `validation_rules.md` |
| `tools/live_fit_monitor.py` | CLI for the same view, to watch a fit started in another terminal (the fit entry points raise it themselves) |
| `tools/plot_fits.py` | replots every fitted session's simulation against its data, from the stored `result_solution_expN.csv` |
| `tools/stiffness_bench.py` | benchmarks stiffness estimators over a parameter box against systems of known character; the evidence behind R1's choice of the matrix measure |
| `tests/` | suite; `pytest -m "not slow"` skips the full fits |
