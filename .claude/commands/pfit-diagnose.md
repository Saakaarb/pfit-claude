Diagnose a completed fit and recommend what to change before re-running.

This file is PROCEDURE only. The evidence sources, the symptom rules, the report
format and the apply policy live in the reference files below — apply them from
there, not from memory.

## Reference files

| File | Why | Required |
|---|---|---|
| `lib/LLM/reference/cold_start.md` | the invariant, and which artifacts diagnosis may read | yes |
| `lib/LLM/reference/diagnosis_rules.md` | every evidence source, symptom rule and the report format | yes |
| `lib/LLM/reference/tuning_rules.md` | the evidence rule, the entry format, the do-not-recommend list, the apply policy | yes |
| `lib/LLM/reference/yaml_format.md` | field schema and defaults, to propose a valid edit | yes |
| `lib/LLM/api/diffrax.md` | the ONLY authority on `integrator` names | when the diagnosis touches the solver |
| `lib/LLM/reference/user_model_contract.md` | the loss contract | when the diagnosis touches the loss |
| `lib/LLM/reference/project_context.md` | session layout, the sloppiness report's meaning | yes |
| `lib/LLM/reference/staggered_data.md` | ONLY if the data is staggered/ragged | conditional |
| `docs/tunable_choices.md` | what is tunable in the config vs. only in library code | yes |

## Steps

1. Ask the user for the session name if not given as an argument.

2. Confirm the session has been fitted: `sessions/<session>/outputs/` must
   contain at least a stage-1 log or `NODE_fitting.log`. If it does not, stop and
   tell the user to run the fit first — there is nothing to diagnose. If only
   `final_design_point.csv` exists, say which artifacts are missing and that the
   diagnosis is correspondingly limited.

3. Read `inputs/user_input.yaml`, `generated/user_model.py`, and **every**
   artifact in `outputs/` that `diagnosis_rules.md` lists. Do not stop at the
   first one that explains the outcome — S6 in particular is only separable by
   combining the NODE log with the bounds.

4. Compute, do not eyeball:
   - first and last `best_cost` per stage, and the iteration at which each
     stopped improving materially;
   - each fitted value from `final_design_point.csv` as a fraction of its
     `min_val`/`max_val` range, to detect a pinned parameter;
   - per-column and per-time-region residuals from `result_solution_expN.csv`
     when it exists;
   - the exit-gradient ratio `|grad|_inf / loss` from `sloppiness_report.txt`
     (S7). Do this for every session, including one that looks converged — it is
     the only evidence that separates convergence from an exhausted iteration
     budget, and no log records it.
   Quote the actual numbers in the report.

5. Apply the symptom rules in `diagnosis_rules.md` **in order**, stopping the
   causal chain where that file says to. Reach one verdict, not a list of
   possibilities. Where the evidence genuinely cannot separate two causes, say so
   and name the cheapest discriminating experiment.

6. Write the report to `sessions/<session>/outputs/fit_diagnosis.txt` in the
   format `diagnosis_rules.md` defines.

7. Summarise the verdict in chat, then ask which findings to apply (by id, `all`,
   or `none`). Apply only what the user names, per the apply policy in
   `tuning_rules.md`.

8. If anything was applied, tell the user what to re-run:
   - config-only changes to `gradient_opt`, or a gradient-stage finding, and stage 1
     already found a good basin → `fit_gradient_only.py <session>` reuses the
     existing seed point.
   - a changed `integrator`, bounds, `logscale`, or population settings →
     `fit_parameters.py <session>` (a full re-fit; the old basin is no longer
     valid).
   - a changed `user_model.py` → `/pfit-check`, then `/pfit-jax`, then
     `fit_parameters.py`.
