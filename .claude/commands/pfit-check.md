Validate and auto-correct the user_model.py and user_input.xml for the current session.

This file is PROCEDURE only. The checks, thresholds, report format and
correction policy live in the reference files below — apply them from there, not
from memory.

## Reference files (all required)

| File | Why |
|---|---|
| `lib/LLM/reference/cold_start.md` | the invariant: setup choices are made without the solution |
| `lib/LLM/reference/validation_rules.md` | every check, its severity, and the report format |
| `lib/LLM/reference/correction_rules.md` | how to fix what you find |
| `lib/LLM/reference/tuning_rules.md` | how to derive the Recommendations section, and the apply-on-confirmation policy |
| `lib/LLM/reference/input_constraints.md` | the hard input constraints being checked |
| `lib/LLM/reference/xml_format.md` | field schema, defaults and required fields |
| `lib/LLM/reference/user_model_contract.md` | what the three user functions must do |
| `lib/LLM/api/diffrax.md` | the ONLY authority on valid `INTEGRATOR` names |
| `lib/LLM/api/optax.md` | valid gradient optimizers and their real defaults |
| `lib/LLM/api/population_optimizers.md` | valid `ALGORITHM` values |
| `lib/LLM/api/jax.md` | judging whether the pseudocode is JAX-convertible |
| `lib/LLM/reference/staggered_data.md` | ONLY if the data is staggered/ragged |

## Steps

1. Ask the user for the session name if not given as an argument.

2. Read `sessions/<session_name>/inputs/user_input.xml`,
   `sessions/<session_name>/generated/user_model.py`, and the reference files
   above.

3. **Validate.** Apply every check in `validation_rules.md`. For the optimizer
   settings, read the actual numbers out of the XML and evaluate each threshold
   explicitly rather than eyeballing them.

4. **Correct.** If there are critical errors, apply `correction_rules.md`,
   editing `inputs/user_input.xml` and `generated/user_model.py` in place.

5. **Re-validate.** Repeat 3-4 until there are no critical errors, or you have
   iterated 3 times.

6. **Recommend.** Once there are no critical errors, gather the evidence listed
   in `tuning_rules.md` and apply its rules to produce the Recommendations
   section. Read the dataset CSVs and the RHS for this — the recommendations come
   from the model and the data, not from the XML alone. Emit nothing you cannot
   attach evidence to.

7. Write the final report to
   `sessions/<session_name>/generated/user_input_check.txt`.

8. Tell the user the outcome:
   - clean: "Validation passed. Run `/pfit-jax` to generate the JAX optimization code."
   - otherwise: list the unresolved critical errors and ask them to fix those
     manually before re-running `/pfit-check`.

9. If there are recommendations, list them and ask which to apply (by rule id,
   `all`, or `none`). Apply only what the user names, following the apply policy
   in `tuning_rules.md`, then re-run step 3 on anything you changed.
