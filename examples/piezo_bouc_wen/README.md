# Piezoelectric Actuator with Bouc–Wen Hysteresis — worked example

A driven mechanical oscillator whose restoring force carries a **Bouc–Wen
hysteresis** term. It exercises the framework on a smooth-but-nonlinear system
where the difficulty is (a) an internal hysteretic state that is never measured
and (b) parameters spanning many orders of magnitude (stiffness `kp ~ 10⁷`
alongside shape parameters of order 10⁻²).

## System

State `y = (xp, vp, h)` — displacement, velocity, and the hysteretic variable —
driven by an applied voltage `V(t) = 24 + 24·sin(16π t)` (an 8 Hz drive), with
`V̇(t)` its time derivative:

```
dxp/dt = vp
dvp/dt = ( kp·(de·V − h) − cp·vp − kp·xp ) / mp
dh/dt  = alpha·de·V̇ − beta·|V̇|·|h| − gamma·V̇·|h|      (Bouc–Wen, voltage-rate driven)
```

with fixed mass `mp = 0.1`. Six trainable parameters:

| Param | Meaning | Search range | Scale |
|-------|---------|--------------|-------|
| alpha | Bouc–Wen amplitude   | `[0, 1]`            | linear |
| beta  | Bouc–Wen shape       | `[0, 0.1]`          | linear |
| gamma | Bouc–Wen shape       | `[0, 0.1]`          | linear |
| cp    | damping              | `[20, 30]`          | linear |
| kp    | stiffness            | `[1.0×10⁷, 1.5×10⁷]`| log |
| de    | piezo coupling       | `[1.0×10⁻⁷, 1.5×10⁻⁷]`| log |

## Data & loss

Data (`inputs/correct_units_piezo_electric_actuator.csv`) are the actuator
**displacement** over a 0.1–0.5 s window (a single observable; velocity and the
hysteretic state `h` are unmeasured). The loss is a peak-normalised error on the
displacement channel.

## Optimizer configuration

- Global: PSO, population 200, 50 iterations (default algorithm).
- Gradient: L-BFGS, 10 iterations, through the `Kvaerno5` solve.
- **Tolerances differ by stage**: the PSO stage uses a loose `atol = 10⁻³`,
  the gradient stage a tight `atol = 10⁻⁸…10⁻¹⁰`. This matters (see below).
- No `RANDOM_SEED` set → the PSO stage is stochastic run-to-run.

## Result (this redo)

**Combined loss `L = 0.116`** (peak-normalised displacement error), a slight
improvement over the previously committed fit (`0.123` re-evaluated under the
same accurate tolerances).

| Param | Fitted |
|-------|--------|
| alpha | 0.579 |
| beta  | 3.89×10⁻⁴ |
| gamma | 1.48×10⁻² |
| cp    | 24.3 |
| kp    | 1.011×10⁷ |
| de    | 1.293×10⁻⁷ |

Identifiability (`outputs/sloppiness_report.txt`): **well-determined** — spectrum
spans 4.9 decades with **no** flat directions. Best-constrained combination is
dominated by `beta`; least-constrained couples `kp` and `cp`.

The fit tracks all three displacement cycles in shape and amplitude, with a
small phase lead and slight peak clipping (`outputs/fit_result.png`).

### Note on the two-tolerance gap

The PSO stage reports a lower loss (`~0.09`) than the gradient stage
(`~0.116`). This is **not** the gradient stage diverging — it is the loose PSO
integration (`atol = 10⁻³`) under-resolving the dynamics and reporting an
optimistic loss. Re-evaluated at the accurate gradient-stage tolerance, the
PSO-best point is genuinely `~0.116`; `~0.116` is the true achievable loss here.

## Files

```
inputs/user_input.xml          fit configuration
inputs/correct_units_piezo_electric_actuator.csv   displacement data
generated/user_model.py        ODE RHS + loss (numpy pseudocode)
generated/generated_script.py  JAX-jittable version actually run
outputs/                       results (cleared and rewritten on each run)
plot_fit.py, plot_fit_config.yaml   regenerate outputs/fit_result.png
```

## Re-run

```bash
venv/bin/python3 fit_parameters.py examples/piezo_bouc_wen
venv/bin/python3 examples/piezo_bouc_wen/plot_fit.py   # refresh fit_result.png
```
