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
| `lib/LLM/reference/runtime_intervention.md` | monitoring and user-approved intervention during a slow global search | yes |
| `lib/LLM/reference/run_history.md` | run allocation, snapshots and seed selection | yes |
| `lib/LLM/reference/live_dashboard.md` | local browser view, stage handoff and viewer lifecycle | yes |
| `lib/LLM/reference/result_plotting.md` | mandatory saved plots after both run modes | yes |
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
     settings, because the previous basin is no longer valid. Creates a new
     `outputs/<run_id>/`; all earlier runs remain intact.
   - **gradient-only** (`fit_gradient_only.py`) — refinement alone, seeded from
     `outputs/<seed_run_id>/final_design_point.csv`. Correct only when the global search
     already landed in a good basin and the change was confined to the gradient
     settings. Copies the seed into a new run and records its source; earlier
     stage-1 logs and design points stay in their original run.

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

   Also check `output.write_results: true`, required for post-fit plots. If
   disabled, report and resolve that setting before running, then regenerate
   and repeat readiness checks.

4. **Start it** — the entry point chosen in step 3, not always the first one.
   Run in the background so the session is not blocked:

   ```bash
   ./venv/bin/python3 fit_parameters.py <session_name> --live-web        # full two-stage
   ./venv/bin/python3 fit_gradient_only.py <session_name> --live-web     # refinement only
   ```

   `fit_gradient_only.py` seeds from `outputs/<seed_run_id>/final_design_point.csv` and refuses
   to start without it, so R6 must report that file present before you choose it.
   It preserves the previous stage-1 logs and the seed point; the full fit
   creates a separate timestamped directory, preserving previous runs.

   State up front that this can take minutes for a small system and longer for a
   stiff one, and that the first iteration includes JIT compilation and is not
   representative of the rest. A gradient-only re-run skips the global search
   entirely, so it is typically minutes where the full fit was hours.

   Record the exact `Run artifacts:` directory printed at launch. Use it for
   every subsequent read, plot and report; do not reselect latest mid-workflow.
   For gradient-only, accept an optional `--seed-run <run_id-or-path>`, resolve
   it during pre-flight, and pass the same selection to the entry point.

   **Share the live browser view.** Follow `live_dashboard.md`: give the user
   the printed local URL, pinned to this run. If hosting fails, inspect
   `live_server.log` and start the standalone viewer against the active run;
   do not rerun the fit to recover its display. Honour a user request to disable
   the browser view by omitting `--live-web`.

5. **While it runs**, actively monitor process status and logs according to
   `runtime_intervention.md`; do not wait for the user to request progress.
   If excessive failed-solve work is suspected, offer the documented stop,
   archive, adjust, regenerate, and restart procedure with a concrete proposed
   limit. Act only after authorization. Report actual progress and distinguish
   suspected causes from established facts; do not predict the fit outcome.

6. **When it finishes**, report, from the files rather than from expectation:

   - the final cost from each stage, and the iteration at which each stopped
     improving materially;
   - the fitted parameters from `outputs/<run_id>/final_design_point.csv`;
   - where the per-experiment trajectories were written.

   The separate viewer remains available after completion. Include its URL
   and the exact viewer PID's stop command; do not confuse it with the fit PID.

   Flag either of these if present, without diagnosing further:

   - **a cost flat at a large value from the first iteration** — candidates are
     not integrating at all, so nothing was fitted; the causes are in
     `diagnosis_rules.md`;
   - **`outputs/<run_id>/fitting_error.txt`** — the run raised; quote the error.

7. **Generate, save, and inspect plots — mandatory.** Follow
   `result_plotting.md` for every experiment and fitted observable. Create the
   session's plotting config if absent, save `outputs/<run_id>/<session>_fit.png`,
   check the plotting log and open the figure. Link it in the result summary.
   If plotting fails, report the blocker and the workflow as incomplete.

8. Hand off: "Run `/pfit-diagnose <session_name> <run_id>` for a verdict on this fit."
   Do not diagnose here — that procedure reads evidence this one does not gather.

## Rules

- Never start a fit with a failing blocking check.
- Never edit `user_model.py`, the config or the data to bypass a failing
  pre-flight. Config changes during the user-approved runtime intervention are
  allowed only after the current process has stopped, followed by regeneration
  and readiness checks as specified in `runtime_intervention.md`.
- Never report a result you have not read out of a file.
