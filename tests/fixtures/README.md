# Test fixtures

Purpose-built sessions for the test suite. They exist so the suite can exercise
every optimizer / solver / experiment configuration without paying the cost of
the stiff Robertson or long-horizon van der Pol fits.

## The model

Both fixtures fit the same two-step linear decay chain:

    dA/dt = -k1*A
    dB/dt =  k1*A - k2*B

with **ground truth k1 = 1.0, k2 = 0.3**.

Every CSV is the *exact analytic solution*

    A(t) = A0*exp(-k1*t)
    B(t) = B0*exp(-k2*t) + A0*k1/(k2-k1) * (exp(-k1*t) - exp(-k2*t))

sampled at 21 points on t in [0, 10], with no noise. The formula was verified to
1e-13 against an independent tight-tolerance integration. Because the data is
noiseless and the parameters are identifiable, a converged fit must recover
(1.0, 0.3) closely — which is what lets the integration tests assert accuracy
instead of merely asserting "it did not crash".

| Fixture | Experiments | Initial conditions |
|---|---|---|
| `decay_session` | 1 (`decay_data.csv`) | A=1.0, B=0.0 |
| `decay_multiexp` | 2 (`decay_run_A.csv`, `decay_run_B.csv`) | run A: global (1.0, 0.0); run B: overridden to (2.0, 0.5) |

The system is non-stiff and 2-dimensional, so a full PSO + NODE fit takes a few
seconds rather than minutes.

## Regenerating the data

    ./venv/bin/python3 - <<'PY'
    import numpy as np
    K1, K2 = 1.0, 0.3
    def sol(t, A0, B0):
        A = A0*np.exp(-K1*t)
        B = B0*np.exp(-K2*t) + A0*K1/(K2-K1)*(np.exp(-K1*t)-np.exp(-K2*t))
        return A, B
    t = np.linspace(0.0, 10.0, 21)
    for path, (A0, B0) in {
        "tests/fixtures/decay_session/inputs/decay_data.csv": (1.0, 0.0),
        "tests/fixtures/decay_multiexp/inputs/decay_run_A.csv": (1.0, 0.0),
        "tests/fixtures/decay_multiexp/inputs/decay_run_B.csv": (2.0, 0.5),
    }.items():
        A, B = sol(t, A0, B0)
        np.savetxt(path, np.column_stack([t, A, B]), delimiter=",", fmt="%.10e")
    PY

If you change the ground-truth constants here, update `DECAY_TRUE_PARAMS` in
`tests/conftest.py` to match.

## Note on `generated_script.py`

These are committed, hand-checked equivalents of what `/pfit-jax` produces. They
follow `lib/utils/output_sample.py` exactly, including the exhaustive failure
mask `jnp.invert(result == RESULTS.successful)`, which
`tests/test_api_digest.py::test_failure_mask_is_exhaustive_everywhere` enforces.
The solver class is baked in as `diffrax.Dopri5()`; the integrator test rewrites
that line to cover other solvers.
