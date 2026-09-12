Validate and auto-correct the user_model.py and user_input.yaml for the current session.

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
| `lib/LLM/reference/yaml_format.md` | field schema, defaults and required fields |
| `lib/LLM/reference/user_model_contract.md` | what the three user functions must do |
| `lib/LLM/api/diffrax.md` | the ONLY authority on valid `integrator` names |
| `lib/LLM/api/optax.md` | valid gradient optimizers and their real defaults |
| `lib/LLM/api/population_optimizers.md` | valid `algorithm` values |
| `lib/LLM/api/jax.md` | judging whether the pseudocode is JAX-convertible |
| `lib/LLM/reference/staggered_data.md` | ONLY if the data is staggered/ragged |

## Steps

1. Ask the user for the session name if not given as an argument.

2. Read `sessions/<session_name>/inputs/user_input.yaml`,
   `sessions/<session_name>/generated/user_model.py`, and the reference files
   above.

3. **Measure the datasets.** Run

   ```bash
   ./venv/bin/python3 tools/check_dataset.py <session_name>
   ```

   and read its output. This is not optional and reading the CSVs by eye is not a
   substitute — several of the checks (a trailing delimiter, a duplicate
   timestamp, a degenerate time span) are invisible on inspection and produce a
   completed run with a plausible number. Classify each reported id by the
   severity table in `validation_rules.md`; quote the tool's numbers as the
   evidence. A `SKIP` is unresolved, not a pass.

   The same output ends with the **loss review** (L1-L4): scale ratios between
   measurement channels, declared-but-unused uncertainties, how the samples
   distribute across each observable's range, and the spread across experiments.
   Report what bears on this session — a loss whose channels differ by orders of
   magnitude, or an observable whose samples all sit in one decile of its range,
   is worth raising even when every check passes.

4. **Validate.** Apply every check in `validation_rules.md`. For the optimizer
   settings, read the actual numbers out of the config and evaluate each threshold
   explicitly rather than eyeballing them.

5. **Correct.** If there are critical errors, apply `correction_rules.md`,
   editing `inputs/user_input.yaml` and `generated/user_model.py` in place. Never
   edit a dataset CSV — see the dataset policy in that file.

6. **Re-validate.** Repeat 3-5 until there are no critical errors, or you have
   iterated 3 times.

7. **Recommend.** Once there are no critical errors, gather the evidence listed
   in `tuning_rules.md` and apply its rules to produce the Recommendations
   section. Read the dataset CSVs and the RHS for this — the recommendations come
   from the model and the data, not from the config alone.

8. Write the final report to
   `sessions/<session_name>/generated/user_input_check.txt`.

9. Tell the user the outcome:
   - clean: "Validation passed. Run `/pfit-jax` to generate the JAX optimization code."
   - otherwise: list the unresolved critical errors and ask them to make decisions on those before re-running `/pfit-check`.

10. If there are recommendations, list them and ask which to apply (by rule id,
    `all`, or `none`). Apply only what the user names, following the apply policy
    in `tuning_rules.md`, then re-run steps 3-4 on anything you changed.
