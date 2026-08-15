# Population (zero-order) optimizer digest

**Pinned to: scipy==1.15.3, pyswarms==1.3.0**

Auto-generated from the installed packages by `tools/gen_api_context.py`. Do not edit by hand — edit `tools/gen_api_context.yaml` and regenerate.

If the versions above differ from the installed ones, this digest is stale: regenerate it before trusting it.

## `scipy.optimize.differential_evolution`

Driven by `lib/algorithms/DE/classes.py`. Only the arguments the framework sets are exposed through user_input.yaml today; the rest are scipy defaults shown here.

```python
differential_evolution(func, bounds, args=(), strategy='best1bin', maxiter=1000, popsize=15, tol=0.01, mutation=(0.5, 1), recombination=0.7, rng=None, callback=None, disp=False, polish=True, init='latinhypercube', atol=0, updating='immediate', workers=1, constraints=(), x0=None, integrality=None, vectorized=False)
```

Valid `strategy` values in this scipy version:

`best1bin`, `best1exp`, `best2bin`, `best2exp`, `currenttobest1bin`, `currenttobest1exp`, `rand1bin`, `rand1exp`, `rand2bin`, `rand2exp`, `randtobest1bin`, `randtobest1exp`

## pyswarms handler strategies

| Handler | Valid strategies |
|---|---|
| `BoundaryHandler` | `intermediate`, `nearest`, `periodic`, `random`, `reflective`, `shrink` |
| `VelocityHandler` | `adjust`, `invert`, `unmodified`, `zero` |
| `OptionsHandler` | `exp_decay`, `lin_variation`, `nonlin_mod`, `random` |

## Gotchas (curated — these are the ones that bite)

- The framework calls `differential_evolution(..., vectorized=True)`, which passes the whole population as an (n_params, m) array; it is transposed to (m, n_params) before `_compute_all_losses`. See `lib/algorithms/DE/classes.py:114`.
- `polish=False` is deliberate: gradient refinement is the NODE stage's job.
- The signature above shows `rng=`, not `seed=`. scipy renamed this argument; `seed=` (which `lib/algorithms/DE/classes.py:124` still passes) is accepted in 1.15.3 but is deprecated and will be removed. Do not 'fix' it to `rng=` without bumping the scipy pin, and do not introduce new `seed=` uses.
- pyswarms draws its per-iteration r1/r2 terms from NumPy's GLOBAL RNG, so `np.random.seed()` controls it. scipy.stats.qmc (the LHS sampler) uses its own Generator and must be seeded via the `seed=` argument.
