Run a fit for the current session: pre-flight the inputs, choose the right entry point, start it, and report where the results are.

This file is PROCEDURE only. The readiness thresholds and their severity live in
`lib/LLM/reference/validation_rules.md`; the entry points and outputs live in
`lib/LLM/reference/project_context.md`. Apply them from there, not from memory.

The skill exists for one reason above the others: `fit_parameters.py` imports
`generated/generated_script.py` and **never reads `user_model.py`**. A model
edited since the last translation is therefore not the model that gets fitted,
and nothing at run time notices — the run completes, the log looks ordinary, and
the parameters belong to the previous version of the equations. That is checked
here, before anything is spent.

## Reference files

| File | Why | Required |
|---|---|---|
| `lib/LLM/reference/project_context.md` | the entry points, the session layout, what lands in `outputs/`, the live view, reproducibility | yes |
| `lib/LLM/reference/validation_rules.md` | the severity of each readiness id | yes |
| `lib/LLM/reference/cold_start.md` | this step starts the fit; it does not read another session's results | yes |
| `lib/LLM/reference/diagnosis_rules.md` | ONLY to hand off at the end — do not diagnose here | conditional |

## Invocation

```
/pfit-run <session_name>                    full two-stage fit (default)
/pfit-run <session_name> gradient-only      refinement only, from the stored point
```

The mode is the second argument. If it is absent, use the full fit — but say so,
and mention that `gradient-only` exists when the session already holds a design
point and the user's last change was confined to the gradient settings.

## Steps

1. Ask the user for the session name if not given as an argument. Read the mode
   from the second argument, defaulting to the full fit.

2. **Settle the mode before pre-flighting**, because the checks differ:

   - **full** (`fit_parameters.py`) — both stages. Required after any change to
     the model, the bounds, `logscale`, the integrator or the population
     settings, because the previous basin is no longer valid. Overwrites
     `outputs/`.
   - **gradient-only** (`fit_gradient_only.py`) — refinement alone, seeded from
     `outputs/final_design_point.csv`. Correct only when the global search
     already landed in a good basin and the change was confined to the gradient
     settings. Preserves the existing stage-1 logs and the seed point.

   State which you are using and why. If the user asked for gradient-only on a
   session with no stored design point, do not silently fall back to a full fit —
   the two cost very different amounts of time. Say that the seed is missing and
   let them choose.

3. **Pre-flight in that mode.** Run

   ```bash
   ./venv/bin/python3 tools/check_ready.py <session_name> --mode full
   ./venv/bin/python3 tools/check_ready.py <session_name> --mode gradient-only
   ```

   The mode matters: under `gradient-only` the stored design point becomes a
   **required** artifact rather than a note, because the entry point raises
   without it.

   Classify each reported id by `validation_rules.md`. Do not start a fit while
   any blocking id fails — say what failed and which command fixes it
   (`/pfit-new`, `/pfit-check` or `/pfit-jax`). A stale generated script is the
   one to be strict about: re-running `/pfit-jax` costs seconds, and skipping it
   invalidates the entire run.

4. **Start it** — the entry point chosen in step 3, not always the first one.
   Run in the background so the session is not blocked:

   ```bash
   ./venv/bin/python3 fit_parameters.py <session_name>        # full two-stage
   ./venv/bin/python3 fit_gradient_only.py <session_name>     # refinement only
   ```

   `fit_gradient_only.py` seeds from `outputs/final_design_point.csv` and refuses
   to start without it, so R6 must report that file present before you choose it.
   It preserves the previous stage-1 logs and the seed point; the full fit
   overwrites the outputs directory.

   State up front that this can take minutes for a small system and hours for a
   stiff one, and that the first iteration includes JIT compilation and is not
   representative of the rest. A gradient-only re-run skips the global search
   entirely, so it is typically minutes where the full fit was hours.

5. **While it runs**, do not speculate about the outcome and do not predict a
   result. If asked for progress, read the iteration logs in `outputs/` and
   report the actual numbers.

6. **When it finishes**, report, from the files rather than from expectation:

   - the final cost from each stage, and the iteration at which each stopped
     improving materially;
   - the fitted parameters from `outputs/final_design_point.csv`;
   - where the per-experiment trajectories were written.

   Flag either of these if present, without diagnosing further:

   - **a cost flat at a large value from the first iteration** — candidates are
     not integrating at all, so nothing was fitted; the causes are in
     `diagnosis_rules.md`;
   - **`outputs/fitting_error.txt`** — the run raised; quote the error.

7. **Plot before judging.** Run `tools/plot_fits.py` if the session is listed in
   its config, or say that it needs adding. A loss value is not a verdict, and
   for a multi-experiment session the averaged loss hides which record fitted
   badly.

8. Hand off: "Run `/pfit-diagnose <session_name>` for a verdict on this fit."
   Do not diagnose here — that procedure reads evidence this one does not gather.

## Rules

- Never start a fit with a failing blocking check to "see what happens".
- Never edit `user_model.py`, the config or the data to make a run start. Report
  what is wrong and let the user decide.
- Never report a result you have not read out of a file.
