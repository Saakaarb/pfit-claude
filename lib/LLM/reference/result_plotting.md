---
topic: Required saved post-fit plots
consumed_by: [pfit-run, pfit-diagnose]
generated: false
owns: Plot preparation, saved figure verification, and visual evidence in diagnosis.
---

# Required post-fit plots

Consumed by `/pfit-run` and `/pfit-diagnose`. After both full and gradient-only
fits, measured-versus-fitted plots must be saved and inspected before declaring
the workflow complete. Python entry points do not do this automatically.

1. Require `output.write_results: true` before running. After successful
   optimization, verify a result CSV exists for every configured experiment.
2. Resolve and pin the run per `run_history.md`. Create `<run_dir>/plot_fits.yaml` using `tools/plot_fits.yaml` as the
   style/schema template, including only this session. Set `name` to the
   session name, `root` to its session path, and `output_dir` to
   `sessions/<session>/outputs/<run_id>/`. Include every fitted observable as a panel;
   the tool covers every experiment. Derive measured/simulated column indices
   from that run's snapshot writeout implementation and confirm their meanings against
   the input column declarations. Include labels and units. An uncertainty
   column is not itself a fitted observable.
3. Run:

   ```bash
   ./venv/bin/python3 tools/plot_fits.py --config <run_dir>/plot_fits.yaml --session <session> --run <run_id> --log-file <run_dir>/plot_fits.log
   ```

4. Verify `<run_dir>/<session>_fit.png` was newly generated from the current
   result CSVs. Check the plotting log and open the image; inspect every
   experiment/observable panel for wrong mappings, missing curves or omitted
   measurements. Check exit status as well as figure contents; a successful process alone
   does not establish correct scientific column mappings. Correct plotting errors and regenerate.
5. Link the saved figure in the completion response. A missing entry in the
   shared plotting config is never a reason to skip this step. If plotting
   cannot finish, report optimization as finished but the workflow as incomplete,
   with the specific blocker. A loss-history or sloppiness plot does not replace
   measured-versus-fitted curves.

The plotting tool resolves the selected run under the configured output root.
Do not fabricate missing measurements. Displayed residual metrics must use
finite paired data/predictions; the plotting tool masks non-finite pairs.

## Diagnosis

Open these figures before reaching a verdict. Generate missing or stale plots
first. In `<run_dir>/fit_diagnosis.txt`, record figure paths and visual observations for
each experiment and observable, supported by numerical residuals from the CSVs.
Assess offsets, timing errors, missed peaks and agreement. Do not infer
identifiability or convergence from curves alone. If unavailable trajectories
or plotting failures prevent inspection, give a limited diagnosis with the
blocker; never declare a successful fit without inspecting its curves.
