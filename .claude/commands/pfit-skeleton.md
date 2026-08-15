Regenerate a user_model.py skeleton from an EXISTING user_input.yaml.

**This is not the entry point for a new session** — `/pfit-new` is, whether or
not the user has a source document. Use this skill only when a
`user_input.yaml` already exists and the user wants the model stub rebuilt from
it: a config carried over from a previous session, a hand-edited config, or a
model file that needs restoring.

If the user is starting from scratch, do not ask them to write the config first.
Send them to `/pfit-new`, which writes the config and the model together from the
same reading of the equations — the only way to guarantee that the config's
parameter and variable orderings match the model's unpacking, since a mismatch
does not raise.

This file is PROCEDURE only. Every rule it depends on lives in exactly one
reference file; read those rather than relying on recall.

## Reference files (read before generating)

| File | Why |
|---|---|
| `lib/LLM/reference/cold_start.md` | the invariant: setup choices are made without the solution | yes |
| `lib/LLM/reference/input_constraints.md` | the constraints validated in step 3 |
| `lib/LLM/reference/user_model_contract.md` | the three functions and the skeleton-generation rules |
| `lib/LLM/reference/yaml_format.md` | how to read the session config |
| `lib/LLM/reference/staggered_data.md` | ONLY if observables are sampled at different time points |
| `lib/LLM/api/jax.md` | tracing rules — the pseudocode must be translatable later |

Worked examples: `lib/utils/user_model_sample_unpopulated.py` (skeleton
template), `lib/utils/user_model_sample_populated.py` (Robertson),
`lib/utils/user_input_sample.yaml`.

## Steps

1. Ask the user for the session name if not given as an argument. The session
   directory is `sessions/<session_name>/`.

2. Read `sessions/<session_name>/inputs/user_input.yaml` and the first ~20 rows of
   each experiment's data CSV.

3. **Validate the inputs against `input_constraints.md` before generating
   anything.**
   - Check **every** item; do not stop at the first failure.
   - Report all findings grouped into **Critical** and **Non-critical**, naming
     the specific file/field for each. End with
     `Number of critical input issues: N`.
   - **Hard gate:** if N > 0, STOP. Write nothing. Present the full list and ask
     the user to fix the inputs (or run `/pfit-check`) and re-run.
   - Non-critical issues are warnings only; report them and continue.

4. Generate the skeleton per `user_model_contract.md`, creating
   `sessions/<session_name>/generated/` if needed.

5. Write it to `sessions/<session_name>/generated/user_model.py`.

6. Cross-check the result against the config once more: parameter names, variable
   names and their order must match.

7. Tell the user what was generated and that they must fill in the ODE logic,
   the loss computation and the writeout before running `/pfit-check`.
