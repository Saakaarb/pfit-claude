# pfit-claude

Fits unknown parameters of a system of ODEs to time-series data, using a
population search followed by gradient refinement. Claude Code skills generate
and validate the code; the optimization pipeline is plain Python.

**This file is an index. It states no rules — every rule lives in exactly one
file below. Open the relevant one rather than working from memory.**

## Where the rules live

| Topic | Canonical file |
|---|---|
| What the project is, the pipeline, session layout, the multi-experiment execution model, post-fit diagnostics, reproducibility, **the Python environment** | `lib/LLM/reference/project_context.md` |
| `user_input.xml` schema — every section, field, default and valid value | `lib/LLM/reference/xml_format.md` |
| Hard constraints on the dataset CSV and the XML | `lib/LLM/reference/input_constraints.md` |
| The three `user_model.py` functions, and how to generate a skeleton | `lib/LLM/reference/user_model_contract.md` |
| Converting pseudocode to `generated_script.py` | `lib/LLM/reference/jax_translation.md` |
| What `/pfit-check` validates, its thresholds, the report format | `lib/LLM/reference/validation_rules.md` |
| How to auto-correct inputs from a validation report | `lib/LLM/reference/correction_rules.md` |
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
| `/pfit-skeleton` | XML -> `user_model.py` skeleton | `.claude/commands/pfit-skeleton.md` |
| `/pfit-from-source` | paper -> XML + populated `user_model.py` | `.claude/commands/pfit-from-source.md` |
| `/pfit-check` | validate + auto-correct the inputs | `.claude/commands/pfit-check.md` |
| `/pfit-jax` | `user_model.py` -> `generated_script.py` | `.claude/commands/pfit-jax.md` |

Each command file is procedure only and names the reference files it requires.

## Code map

| Path | Contains |
|---|---|
| `fit_parameters.py` | entry point: full two-stage fit |
| `fit_gradient_only.py` | entry point: gradient stage only, seeded from a previous fit |
| `analyze_fit.py` | entry point: re-run post-fit diagnostics on a completed session |
| `lib/utils/xmlread.py` | XML parsing |
| `lib/utils/helper_functions.py` | problem object, per-experiment dispatch, stage orchestration |
| `lib/algorithms/{PSO,DE,NODE}/` | the three optimizers |
| `lib/utils/sloppiness.py` | post-fit identifiability diagnostic |
| `lib/utils/output_sample.py` | template for the generated script |
| `lib/utils/user_model_sample_{un,}populated.py` | skeleton and worked model |
| `lib/utils/user_input_sample.xml` | worked XML |
| `examples/` | complete worked sessions (robertson, ARC, piezo, sliding_basepoint) |
| `tools/gen_api_context.py` | regenerates the API digests |
| `tests/` | suite; `pytest -m "not slow"` skips the full fits |
