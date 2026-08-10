# diffrax API digest

**Pinned to: diffrax==0.7.2, jax==0.6.2, equinox==0.13.8, optimistix==0.1.0**

Auto-generated from the installed packages by `tools/gen_api_context.py`. Do not edit by hand — edit `tools/gen_api_context.yaml` and regenerate.

If the versions above differ from the installed ones, this digest is stale: regenerate it before trusting it.

## ODE solvers

The `INTEGRATOR` field in `GRADIENT_OPT/SETTINGS` must name one of these classes exactly. It is substituted for `diffrax.SOLVER_CLASS()` in `_integrate_system`. Anything not in this table does not exist in this version and will raise `AttributeError` at import time.

`validated` marks solvers already exercised by this framework's examples; the others exist and are usable but are untested here.

Only **adaptive** solvers are listed. `_integrate_system` always drives the solve with `diffrax.PIDController(rtol, atol)`, which requires an error estimate, so non-adaptive solvers are unusable and are omitted rather than listed as options.

When in doubt, pick an **implicit (stiff)** solver: on a non-stiff system it merely costs more per step, whereas an explicit solver on a stiff system fails outright. See the gotchas below.

| Solver | Kind | Validated | Summary |
|---|---|---|---|
| `Bosh3` | explicit RK (non-stiff) |  | Bogacki--Shampine's 3/2 method. |
| `Dopri5` | explicit RK (non-stiff) | yes | Dormand-Prince's 5/4 method. |
| `Dopri8` | explicit RK (non-stiff) | yes | Dormand--Prince's 8/7 method. |
| `Heun` | explicit RK (non-stiff), SDE-capable |  | Heun's method. |
| `ImplicitEuler` | implicit (stiff) |  | Implicit Euler method. |
| `KenCarp3` | implicit (stiff) |  | Kennedy--Carpenter's 3/2 IMEX method. |
| `KenCarp4` | implicit (stiff) |  | Kennedy--Carpenter's 4/3 IMEX method. |
| `KenCarp5` | implicit (stiff) |  | Kennedy--Carpenter's 5/4 IMEX method. |
| `Kvaerno3` | implicit (stiff) | yes | Kvaerno's 3/2 method. |
| `Kvaerno4` | implicit (stiff) | yes | Kvaerno's 4/3 method. |
| `Kvaerno5` | implicit (stiff) | yes | Kvaerno's 5/4 method. |
| `Midpoint` | explicit RK (non-stiff), SDE-capable |  | Midpoint method. |
| `Ralston` | explicit RK (non-stiff), SDE-capable |  | Ralston's method. |
| `Sil3` | implicit (stiff) |  | Whitaker--Kar's fast-slow IMEX method. |
| `Tsit5` | explicit RK (non-stiff) | yes | Tsitouras' 5/4 method. |

**Excluded as unusable with this framework** (17 solvers). These exist in diffrax but must never be named as an `INTEGRATOR`:

- _SDE only_: `ALIGN`, `ReversibleHeun`, `SPaRK`
- _needs constructor arguments_: `HalfSolver`
- _not adaptive_: `Euler`, `EulerHeun`, `GeneralShARK`, `ItoMilstein`, `LeapfrogMidpoint`, `QUICSORT`, `SEA`, `SRA1`, `SemiImplicitEuler`, `ShARK`, `ShOULD`, `SlowRK`, `StratonovichMilstein`

## `RESULTS` codes

`_integrate_system` returns `sol.result`. Every code below other than `successful` means the trajectory is NOT trustworthy. The loss function must map failed solves to `constants['error_loss']`.

| Code | Meaning |
|---|---|
| `RESULTS.successful` | Solve completed normally. The ONLY code that means the trajectory is usable. |
| `RESULTS.max_steps_reached` | The maximum number of solver steps was reached. Try increasing `max_steps`. |
| `RESULTS.singular` | A linear solver returned non-finite (NaN or inf) output. This usually means that an operator was not well-posed, and that its solver does not support this. If you are trying solve a linear least-sq... |
| `RESULTS.breakdown` | A form of iterative breakdown has occured in a linear solve. Try using a different solver for this problem or increase `restart` if using GMRES. |
| `RESULTS.stagnation` | A stagnation in an iterative linear solve has occurred. Try increasing `stagnation_iters` or `restart`. |
| `RESULTS.conlim` | Condition number of A seems to be larger than `conlim`. |
| `RESULTS.nonfinite_input` | A linear solver received non-finite (NaN or inf) input and cannot determine a solution. This means that you have a bug upstream of Lineax and should check the inputs to `lineax.linear_solve` for no... |
| `RESULTS.nonlinear_max_steps_reached` | The maximum number of steps was reached in the nonlinear solver. The problem may not be solveable (e.g., a root-find on a function that has no roots), or you may need to increase `max_steps`. |
| `RESULTS.nonlinear_divergence` | Nonlinear solve diverged. |
| `RESULTS.nonfinite` | Nonfinite (inf or nan) values detected during solve. |
| `RESULTS.dt_min_reached` | The minimum step size was reached in the differential equation solver. |
| `RESULTS.event_occurred` | Terminating differential equation solve because an event occurred. |
| `RESULTS.max_steps_rejected` | Maximum number of rejected steps was reached. Consider increasing `diffrax.ClipStepSizeController(store_rejected_steps==...)`. |
| `RESULTS.internal_error` | An internal error occurred in Diffrax. This is a bug! Please open a GitHub issue with a minimum working example. (<50 lines of code is ideal) |

## Core call signatures

```python
diffrax.diffeqsolve(terms, solver, t0, t1, dt0, y0, args=None, saveat=<default>, stepsize_controller=ConstantStepSize(), adjoint=RecursiveCheckpointAdjoint(), event=None, max_steps=4096, throw=True, progress_meter=NoProgressMeter(), solver_state=None, controller_state=None, made_jump=None, discrete_terminating_event=None)
```

```python
diffrax.SaveAt(t0=False, t1=False, ts=None, steps=False, fn=save_y, subs=None, dense=False, solver_state=False, controller_state=False, made_jump=False)
```

```python
diffrax.ODETerm(vector_field)
```

## Step size controllers

| Class | Signature |
|---|---|
| `ClipStepSizeController` | `(controller, step_ts=None, jump_ts=None, store_rejected_steps=None, _callback_on_reject=None)` |
| `ConstantStepSize` | `()` |
| `PIDController` | `(rtol, atol, norm=rms_norm, pcoeff=0, icoeff=1, dcoeff=0, dtmin=None, dtmax=None, force_dtmin=True, factormin=0.2, factormax=10.0, safety=0.9, error_order=None)` |
| `StepTo` | `(ts)` |

## Adjoints (gradient propagation through the solve)

| Class | Summary |
|---|---|
| `BacksolveAdjoint` | Backpropagate through [`diffrax.diffeqsolve`][] by solving the continuous |
| `DirectAdjoint` | A variant of [`diffrax.RecursiveCheckpointAdjoint`][] that is also able to |
| `ForwardMode` | Enables support for forward-mode automatic differentiation (like `jax.jvp` or |
| `ImplicitAdjoint` | Backpropagate via the [implicit function theorem](https://en.wikipedia.org/wiki/Implicit_function_theorem#Statement_of_the_theorem). |
| `RecursiveCheckpointAdjoint` | Enables support for backpropagating through [`diffrax.diffeqsolve`][] by |

`diffeqsolve` defaults to `RecursiveCheckpointAdjoint` when `adjoint` is not passed; the framework does not override it.

## Gotchas (curated — these are the ones that bite)

- `_integrate_system` in `lib/utils/output_sample.py` is copied verbatim into every generated script. The ONLY line that may be substituted is `diffrax.SOLVER_CLASS()` and the `max_steps=` literal.
- Use `diffrax.SaveAt(ts=t_eval)` so the saved rows line up with the data rows they are differenced against. `SaveAt(t0=True, ts=t_eval[1:])` saves at `[init_time, t_eval[1:]]`, which silently misaligns every residual when `INITIAL_TIME` differs from `t_eval[0]`. (An earlier note here claimed `ts=t_eval` raises `_EquinoxRuntimeError` because t0 is implicitly prepended — that is NOT true of diffrax 0.7.2: t0 is saved only when `t0=True` is passed.)
- `diffeqsolve(..., throw=False)` is required: it returns a `RESULTS` code instead of raising, so a failed solve can be mapped to `error_loss` inside jit. Never set `throw=True` in generated code.
- `max_steps` must be a Python int literal (it is a static/compile-time argument). Passing a traced value or a dict lookup fails to compile.
- The stiff/non-stiff choice is ASYMMETRIC *for a smooth RHS*. An implicit solver on a smooth non-stiff system is merely slower per step (roughly 3-10x) and still converges. An explicit solver on a stiff system does not converge at all: the step size collapses, `max_steps` is exhausted, and every candidate scores `error_loss`. So when stiffness is uncertain and the RHS is smooth, choose implicit.
- That asymmetry REVERSES for a non-smooth RHS. An implicit method solves a nonlinear system at every step, and across a discontinuity that solve cannot converge — the integration then fails for ANY `max_steps`, so raising it does not help. If the RHS uses `sign`, `abs`, `floor`, `clip`, or a state-dependent `where` (dry friction, contact, saturation, hysteresis, on/off control), choose an explicit solver. Because a failed solve scores `error_loss`, the cost is not just runtime: whole regions of parameter space become invisible to the optimizer.
- The integration-failure mask must be `failed = jnp.invert(result == RESULTS.successful)`. Enumerating individual failure codes lets inf/NaN trajectories (dt_min_reached, nonlinear_divergence, nonfinite, max_steps_rejected) be scored as a genuine fit, which produces NaN gradients in the NODE stage.
