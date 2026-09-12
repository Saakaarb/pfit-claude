---
topic: Per-run output directories and audit records
consumed_by: [pfit-run, pfit-diagnose, ad-hoc work]
generated: false
owns: Run allocation, snapshot contents, provenance manifests, seed lineage and historical run selection.
---

# Run history and audit records

Every invocation of `fit_parameters.py` or `fit_gradient_only.py` creates a new
directory under the session's configured output directory. Neither deletes or
replaces an earlier run, including old flat output files. The run ID is a UTC
timestamp with microseconds and a random suffix, for example
`20260912T183045.123456Z_a12b34cd`. The driver prints its exact path at launch.

```
sessions/<session>/outputs/<run_id>/
├── run_manifest.json
├── snapshot/
│   ├── inputs/
│   │   ├── user_input.yaml       # exact original configuration
│   │   ├── run_config.yaml       # execution config with snapshot-relative paths
│   │   └── dataset_1.csv         # copies of every experiment's dataset
│   ├── generated/               # generated_script.py, user_model.py, reports, etc.
│   └── framework/               # library, entry points, tools, requirements
├── seed_design_point.csv        # gradient-only: exact seed used
├── final_design_point.csv
├── result_solution_exp1.csv     # one per experiment, when write_results is true
├── pso_fitting.log              # or de_fitting.log; full runs only
├── NODE_fitting.log
├── run_stdout.log              # terminal live view only
├── sloppiness_report.txt
├── sloppiness_spectrum.png
├── <session>_fit.png            # saved by the post-fit plotting step
└── fit_diagnosis.txt            # saved by /pfit-diagnose
```

The fit reads the copied config, datasets and generated code. Editing working
session files afterward cannot change those copies. The manifest records the
mode, timestamps, process ID, status, seed source, Python/platform, installed
package versions, Git revision and SHA-256 hashes of the snapshots. Snapshots
also preserve uncommitted framework edits. Treat snapshot files as immutable.
An abrupt kill or machine crash may leave status `running`; verify the process
before treating that status as current. Caught exceptions and interrupts record
`failed` or `interrupted` and preserve partial outputs.

## Select a run explicitly

Resolve the run once and keep using that exact directory through monitoring,
plotting and diagnosis. Do not repeatedly resolve “latest” while another run
could start. No run ID selects the latest run for readers, including a failed
attempt; it must not silently substitute an older successful fit.

- `fit_gradient_only.py <session> --seed-run <run_id-or-path>` seeds from the
  selected completed run. Without this flag, it selects the latest completed
  run with a design point, falling back to legacy flat results. It copies the
  seed and records its source; the old population logs stay in the parent run.
  For repeated gradient-only runs, follow the recorded seed-source chain to
  locate the originating population search; label that earlier evidence.
  A parameter-name/order mismatch is rejected for runs with snapshots.
- `tools/check_ready.py <session> --mode gradient-only --seed-run <run_id-or-path>`
  checks the same seed selection. Pass the same selection to the fit command.
- `analyze_fit.py <session> --run <run_id-or-path>` reads that run's snapshots
  and parameters and writes diagnostics into that run.
- `tools/plot_fits.py --config <plot-config> --session <session> --run <run_id-or-path>`
  saves the figure and a copy of its plotting configuration/script in that run.
  `tools/plot_diagnostics.py` accepts the same session/run selectors and uses
  snapshots for reintegration. Its existing panels describe experiment 1;
  use `plot_fits.py` for mandatory coverage of all experiments.
- `tools/live_fit_monitor.py --session <session> --run <run_id-or-path>` watches
  that run. The in-process live view is pinned to the newly allocated directory.

Existing flat `outputs/` remain readable for backward compatibility, but lack
snapshots and must be labelled as legacy when interpreting their provenance.
For completed runs, inspect the saved inputs/model, never the mutable working
copies. Apply proposed changes to the working session and start a new run;
never edit an old snapshot to implement a recommendation. Post-processing may
refresh figures/reports only in the explicitly selected run.
