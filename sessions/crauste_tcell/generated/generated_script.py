# pfit-sources: user_model.py=e8a291af696d6f5d user_input.yaml=04314b63c4e9b79b
import jax
import jax.numpy as jnp
import diffrax
from diffrax import RESULTS

jax.config.update("jax_enable_x64", True)


# fixed
@jax.jit
def unscale_value(val, min_val, max_val, is_logscale):

    lin_unscaled = ((val + 1.0) / 2.0) * (max_val - min_val) + min_val
    unscaled = jnp.where(is_logscale, 10.0**lin_unscaled, lin_unscaled)

    return unscaled


# fixed
@jax.jit
def scale_value(unscaled_val, min_val, max_val, is_logscale):
    # If logscaled, take log10 first
    lin_val = jnp.where(is_logscale, jnp.log10(unscaled_val), unscaled_val)
    # Linearly map [min_val, max_val] -> [-1, 1]
    scaled = 2.0 * (lin_val - min_val) / (max_val - min_val) - 1.0
    return scaled


# CD8 T-cell differentiation during an acute infection (Crauste et al. 2017):
# naive cells are recruited by the pathogen into early effectors, which
# proliferate, differentiate into late effectors and then into memory cells,
# while both effector compartments clear the pathogen. All death terms are
# density dependent.
@jax.jit
def user_defined_system(t, y, other_args):

    # fixed
    # ----------------------
    trainable_variables = other_args["trainable_variables"]
    constants = other_args["constants"]
    dataset = constants["dataset"]
    t_eval = constants["t_eval"]
    fixed_parameters = constants["fixed_parameters"]
    min_val = constants["min_limits"]
    max_val = constants["max_limits"]
    is_logscale = constants["is_logscale"]
    # ----------------------

    # Order: ['delta_EL', 'delta_LM', 'delta_NE', 'mu_EE', 'mu_LE', 'mu_LL',
    #         'mu_N', 'mu_P', 'mu_PE', 'mu_PL', 'rho_E', 'rho_P']
    (delta_EL, delta_LM, delta_NE, mu_EE, mu_LE, mu_LL,
     mu_N, mu_P, mu_PE, mu_PL, rho_E, rho_P) = unscale_value(
        trainable_variables, min_val, max_val, is_logscale
    )

    N = y[0]
    E = y[1]
    L = y[2]
    M = y[3]
    P = y[4]

    # this part is user entered
    # ---------------------------------------------------
    # Naive: constant-rate loss, plus pathogen-driven recruitment into E
    dN_dt = -mu_N * N - delta_NE * N * P

    # Early effector: recruitment + pathogen-driven proliferation,
    # quadratic self-limitation, differentiation into L
    dE_dt = delta_NE * N * P + rho_E * E * P - mu_EE * E * E - delta_EL * E

    # Late effector: differentiation in, quadratic and E-mediated death,
    # differentiation into memory
    dL_dt = delta_EL * E - mu_LL * L * L - mu_LE * E * L - delta_LM * L

    # Memory: accumulates only, no loss on this timescale
    dM_dt = delta_LM * L

    # Pathogen: quadratic growth, linear clearance, killing by E and L
    dP_dt = rho_P * P * P - mu_P * P - mu_PE * E * P - mu_PL * L * P
    # ---------------------------------------------------

    return jnp.array([dN_dt, dE_dt, dL_dt, dM_dt, dP_dt])


# fixed
@jax.jit
def _integrate_system(constants, trainable_variables):
    term = diffrax.ODETerm(user_defined_system)

    solver = diffrax.Kvaerno5()
    t_eval = constants["t_eval"]
    init_cond = constants["init_cond"]
    init_time = constants["init_time"]
    dataset = constants["dataset"]
    # The saved rows are differenced against `dataset` row-for-row, so the save
    # times MUST equal the data times. `ts=t_eval` guarantees that for any
    # `init_time`. Do NOT use `SaveAt(t0=True, ts=t_eval[1:])`.
    saveat = diffrax.SaveAt(ts=t_eval)

    other_args = {"constants": constants, "trainable_variables": trainable_variables}
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=init_time,
        t1=t_eval[-1],
        max_steps=5000,
        dt0=constants['init_timestep'],
        y0=init_cond,
        args=other_args,
        saveat=saveat,
        throw=False,
        stepsize_controller=diffrax.PIDController(rtol=constants['stepsize_rtol'], atol=constants['stepsize_atol']),
    )
    return sol.ts, sol.ys, sol.result


@jax.jit
def _compute_loss_problem(constants, trainable_variables):

    # fixed
    # ---------------------------------------------------
    dataset = constants["dataset"]
    solution_time, solution, result = _integrate_system(constants, trainable_variables)
    # Any code other than RESULTS.successful means the trajectory is not
    # trustworthy (it may contain inf/NaN). See lib/LLM/api/diffrax.md.
    failed = jnp.invert(result == RESULTS.successful)
    # ---------------------------------------------------

    # this part is user entered
    # ---------------------------------------------------
    # dataset columns: 0-3 the four measured populations, 4-7 their per-point
    # standard deviations, in the same order. Cells not measured at a given
    # time are NaN in both halves.
    measured = dataset[:, 0:4]
    sigma = dataset[:, 4:8]

    # The four observables are the first four states, in the same order.
    model_obs = solution[:, 0:4]

    # Sanitise BEFORE any arithmetic so NaN never enters the autodiff graph.
    mask = jnp.invert(jnp.isnan(measured))
    data_safe = jnp.where(mask, measured, 0.0)
    sigma_safe = jnp.where(mask, sigma, 1.0)

    # Residuals in units of the measurement noise. The populations span roughly
    # four orders of magnitude across observables and time, so noise weighting -
    # not peak scaling - is what makes the four channels comparable. A fit at
    # the noise level scores ~1.
    resid = jnp.where(mask, (model_obs - data_safe) / sigma_safe, 0.0)
    loss_value = jnp.sqrt(jnp.sum(resid * resid) / jnp.sum(mask))
    # ---------------------------------------------------

    # fixed
    # --------------------------------------
    loss = jnp.where(failed,
    constants["error_loss"],
    loss_value
    )

    return loss


# the purpose of this function is to write out a CSV containing info
# that is to be plotted
def _write_problem_result(constants, trainable_variables):

    # fixed
    # ---------------------------------------------------
    dataset = constants["dataset"]

    solution_time, solution, result = _integrate_system(constants, trainable_variables)
    # ---------------------------------------------------

    # the rest is user entered
    # ---------------------------------------------------
    Nts = solution_time.shape[0]

    # time | 4 measured | 4 simulated observables | simulated Pathogen
    writeout_array = jnp.zeros([Nts, 10])
    writeout_array = writeout_array.at[:, 0].set(solution_time)
    writeout_array = writeout_array.at[:, 1].set(dataset[:, 0])   # measured Naive
    writeout_array = writeout_array.at[:, 2].set(dataset[:, 1])   # measured EarlyEffector
    writeout_array = writeout_array.at[:, 3].set(dataset[:, 2])   # measured LateEffector
    writeout_array = writeout_array.at[:, 4].set(dataset[:, 3])   # measured Memory
    writeout_array = writeout_array.at[:, 5].set(solution[:, 0])  # simulated Naive
    writeout_array = writeout_array.at[:, 6].set(solution[:, 1])  # simulated EarlyEffector
    writeout_array = writeout_array.at[:, 7].set(solution[:, 2])  # simulated LateEffector
    writeout_array = writeout_array.at[:, 8].set(solution[:, 3])  # simulated Memory
    writeout_array = writeout_array.at[:, 9].set(solution[:, 4])  # simulated Pathogen
    # ---------------------------------------------------

    return writeout_array
