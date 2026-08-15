Generate a user_model.py skeleton for the current session from its user_input.xml.

This file is PROCEDURE only. Every rule it depends on lives in exactly one
reference file; read those rather than relying on recall.

## Reference files (read before generating)

| File | Why |
|---|---|
| `lib/LLM/reference/cold_start.md` | the invariant: setup choices are made without the solution | yes |
| `lib/LLM/reference/input_constraints.md` | the constraints validated in step 3 |
| `lib/LLM/reference/user_model_contract.md` | the three functions and the skeleton-generation rules |
| `lib/LLM/reference/xml_format.md` | how to read the session XML |
| `lib/LLM/reference/staggered_data.md` | ONLY if observables are sampled at different time points |
| `lib/LLM/api/jax.md` | tracing rules — the pseudocode must be translatable later |

Worked examples: `lib/utils/user_model_sample_unpopulated.py` (skeleton
template), `lib/utils/user_model_sample_populated.py` (Robertson),
`lib/utils/user_input_sample.xml`.

## Steps

1. Ask the user for the session name if not given as an argument. The session
   directory is `sessions/<session_name>/`.

2. Read `sessions/<session_name>/inputs/user_input.xml` and the first ~20 rows of
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

6. Cross-check the result against the XML once more: parameter names, variable
   names and their order must match.

7. Tell the user what was generated and that they must fill in the ODE logic,
   the loss computation and the writeout before running `/pfit-check`.
