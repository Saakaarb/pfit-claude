---
topic: Intervention while a global search is still running
consumed_by: [pfit-run]
generated: false
owns: Progress monitoring, intervention offers, stopping and restarting with an approved step limit.
---

# Intervention during a slow global search

Consumed by `/pfit-run`. This procedure handles an active run; it does not
require a completed fit or replace `/pfit-diagnose`.

## Monitor and offer intervention

Record the launched process/job identity, start time, and exact run directory
printed by the driver (`run_history.md`). While supervising the
run, check process status and available iteration/console logs at intervals no
longer than 60 seconds. Report elapsed time and completed iterations, including
when none have completed. Do not promise unattended monitoring after the agent
stops supervising.

Use five minutes without a completed global-search iteration as a default
prompt to review progress, not as a failure threshold. Honour a user-specified
waiting budget instead. After iterations have completed, also review an interval
that exceeds three times the median completed post-first-iteration duration.
No new log row alone proves neither a hang nor exhausted solver steps.

Check whether the process is still running, whether compilation may account for
the delay, the solver and `max_steps`, and whether the equations/bounds admit
singular or rapidly growing trajectories. Read only this run's logs as runtime
evidence. Iteration logs do not establish which candidates exhausted their step
budget. Label the explanation as suspected unless direct evidence establishes it.

If excessive failed-solve work is plausible, review a lower ceiling using R10
in `tuning_rules.md`. Present the current and proposed `max_steps`, the evidence,
and the elapsed time. For example, `100000 -> 5000` is a possible intervention,
not a universal default. Explain that valid trajectories may also need the
larger ceiling, and that the current implementation shares this limit between
population search and gradient refinement. Do not promise a particular speedup
or unchanged fit quality.

Offer the user a concrete choice: keep waiting, or stop this run, preserve its
artifacts, apply the proposed limit, regenerate, and restart. Explicitly say
that this is a restart: the framework has no checkpoint/resume workflow for
the in-progress population. Do not stop or edit a running fit without permission
unless the user has already authorized this intervention. While awaiting an
answer, continue monitoring; elapsed time is not approval. Do not repeat the
same offer every polling interval after the user chooses to wait.

## Apply an accepted intervention

1. Interrupt only the identified fitting process/job and verify that it has
   exited, including any workers belonging to that run. If it does not stop,
   report this and resolve termination before editing or restarting. Suspending
   a process with a stop signal does not make it use new settings on resumption.
2. Preserve the stopped run in place: its snapshots and partial artifacts are
   already isolated in `outputs/<run_id>/`. Record the reason, elapsed time,
   observed progress and proposed change in `intervention.txt` within that run.
   The restart creates a new run directory; never delete or edit the snapshots.
3. Apply the approved `gradient_opt.max_steps` change. Keep population size
   unchanged for this attempt. Reducing it is a separate option that also
   reduces exploration and needs its own stated rationale and authorization.
4. Revalidate the changed setting, then run `/pfit-jax <session>` to regenerate,
   import-check, and stamp the script. Do not just hand-edit its literal or
   restamp an unverified translation. Run the full-mode readiness check.
5. Restart the full fit and monitor it under the same policy. Distinguish new
   compilation time from subsequent iteration timings. Report actual progress;
   a faster iteration alone does not establish preserved fit quality.
6. If the attempt is still unacceptably slow or losses remain at `error_loss`,
   report that the intervention has not resolved the problem. Do not repeatedly
   lower the ceiling or automatically restart. Offer the user the evidence and
   a next decision. Preserve the archive for reverting the settings.
