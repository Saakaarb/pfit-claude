---
topic: Locally hosted live loss and parameter view
consumed_by: [pfit-run, pfit-diagnose, ad-hoc work]
generated: false
owns: Browser-view launch, telemetry semantics, continuous stage handoff, and viewer lifecycle.
---

# Live browser view

`/pfit-run` launches the selected entry point with `--live-web` and shares its
printed URL with the user. The existing fitting commands remain the entry
points; the optional viewer runs separately and never controls optimization.

```bash
./venv/bin/python3 fit_parameters.py <session> --live-web
./venv/bin/python3 fit_gradient_only.py <session> --seed-run <run_id> --live-web
```

The server binds only to `127.0.0.1`, using an available port by default.
`--web-port <port>` requests a particular port. Open the printed
`http://127.0.0.1:<port>/` link. No browser is opened automatically, and no
external web service, CDN, or new package is needed.

If the viewer fails to launch, the fit continues. Read `live_server.log` and
start the viewer against that exact run; never restart optimization just to
open the page. This also works for a completed run:

```bash
./venv/bin/python3 tools/live_fit_server.py --run-dir <run_dir>
# Or select a session and run explicitly:
./venv/bin/python3 tools/live_fit_server.py --session <session> --run <run_id>
```

## What the user sees

- Current stage, stage-best evaluated loss, elapsed time and freshness of the
  last parameter update. No point is invented while compilation or a solve is
  still running. The browser refreshes once per second.
- Best trainable parameters in physical units, with bounds from the run's
  snapshotted config. During the gradient's first evaluation, show the starting
  seed and a pending loss, not an unevaluated claim of improvement.
- One continuous iteration axis. Global iterations are numbered from 1; the
  gradient seed evaluation is iteration 0 at the final global x-coordinate.
  Subsequent gradient evaluations continue from there. A gradient-only run
  starts at x=0 and does not splice in an unrelated historical loss curve.
- Separate stage scales by default: global loss on the left, gradient loss on
  the right (logarithmic by default). Each scale can be switched to linear/log;
  a shared-scale option supports direct comparisons. In separate-scale mode,
  the gradient seed is visually aligned with the global endpoint using the
  labelled right-axis range; no loss values are changed. Tighter refinement
  tolerances can produce a different loss at the same parameter vector, so
  tooltips retain both actual values and the handoff is explicitly marked.
- Zero/negative values cannot appear on log axes. The page states when points
  are hidden and offers linear scales; it never replaces them with an epsilon.
- A download of the currently displayed progress JSON.

PSO, DE and both gradient optimizers append `live_progress.jsonl` after existing
evaluations. This includes the gradient seed evaluation that the older NODE
text log omitted. Parameters are unscaled before saving. No extra ODE solve,
gradient evaluation or optimizer update is performed for the display.
Telemetry write failures warn once and cannot stop fitting. A partial last
JSONL record is ignored until its newline arrives.

The display is not a convergence verdict and does not replace mandatory saved
measured-versus-fitted plots (`result_plotting.md`). Older runs without this
progress stream cannot supply historical best parameter vectors retroactively.

## Lifecycle and audit

Use the run directory printed by the driver; the server resolves it once and
never switches to a newer run. `live_server.json` records the local URL and
viewer PID; `live_server.log` records startup errors. The progress stream stays
with that run for audit and later viewing. The HTML and server code are also
included in the framework snapshot.

The viewer remains available after fitting completes, until explicitly stopped.
For a viewer launched manually, Ctrl-C stops only the viewer. For an automatic
viewer, stop the exact viewer PID in `live_server.json` with `kill <viewer_pid>`;
verify it is that viewer process before signalling it. Closing a browser tab
does not stop the server or the fit. Report a disconnected page as stale, not
as fit completion. Run completion comes from `run_manifest.json`, including
the existing caveat about abrupt process/machine termination.
