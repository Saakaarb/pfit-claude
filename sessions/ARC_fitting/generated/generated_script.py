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


# Two-reaction Arrhenius model of lithium-ion battery thermal runaway. Reaction 1
# consumes c1; reaction 2 produces c2 and ignites only above T_ignite.
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

    # Order: ['Ea1', 'A1', 'n1', 'h1', 'Ea2', 'A2', 'm2', 'h2']
    Ea1, A1, n1, h1, Ea2, A2, m2, h2 = unscale_value(
        trainable_variables, min_val, max_val, is_logscale
    )

    T_ignite = fixed_parameters['T_ignite']
    kb = fixed_parameters['kb']

    c1 = y[0]
    c2 = y[1]
    T = y[2]

    # this part is user entered
    # ---------------------------------------------------
    # Arrhenius kinetics; c1 is consumed, c2 is produced towards 1.
    # jnp.abs() guards the fractional powers: a solver trial step can push an
    # extent marginally negative, and (negative)**non-integer is NaN, which
    # would poison the whole trajectory rather than merely cost a step.
    dc1dt = -A1 * jnp.exp(-Ea1 / (kb * T)) * jnp.abs(c1)**n1
    dc2dt = A2 * jnp.exp(-Ea2 / (kb * T)) * jnp.abs(1.0 - c2)**m2

    # the second exotherm contributes only once the cell is hot enough
    second = jnp.where(T > T_ignite, jnp.abs(h2 * dc2dt), 0.0)
    dTdt = jnp.abs(h1 * dc1dt) + second
    # ---------------------------------------------------

    return jnp.array([dc1dt, dc2dt, dTdt])


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
        max_steps=10000,
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
    trainable_vals = unscale_value(trainable_variables, constants["min_limits"],
                                   constants["max_limits"], constants["is_logscale"])
    Ea1, A1, n1, h1, Ea2, A2, m2, h2 = trainable_vals
    fixed_parameters = constants["fixed_parameters"]
    T_ignite = fixed_parameters['T_ignite']
    kb = fixed_parameters['kb']

    c1 = solution[:, 0]
    c2 = solution[:, 1]
    T = solution[:, 2]

    T_exp = dataset[:, 0]
    rate_exp = dataset[:, 1]

    # the model heating rate, reconstructed from the saved states
    dc1dt = -A1 * jnp.exp(-Ea1 / (kb * T)) * jnp.abs(c1)**n1
    dc2dt = A2 * jnp.exp(-Ea2 / (kb * T)) * jnp.abs(1.0 - c2)**m2
    rate_sim = jnp.abs(h1 * dc1dt) + jnp.where(T > T_ignite, jnp.abs(h2 * dc2dt), 0.0)

    # L1: the heating rate spans six decades, so it is matched in log space. A
    # linear residual would let the runaway endpoint own the objective and
    # ignore the low-temperature kinetics that fix the activation energies.
    floor = 1e-12
    log_sim = jnp.log10(jnp.maximum(rate_sim, floor))
    log_exp = jnp.log10(jnp.maximum(rate_exp, floor))
    log_span = jnp.max(log_exp) - jnp.min(log_exp)
    loss_rate = jnp.mean(jnp.abs(log_sim - log_exp)) / log_span

    # L2: the temperature trajectory, normalised by its peak
    loss_temp = jnp.mean(jnp.abs(T_exp - T)) / jnp.max(jnp.abs(T_exp))

    # L3: one-sided physical priors the under-determined data cannot enforce.
    # Vanishes once the fit reaches a valid runaway, so it is inactive at the
    # optimum and only steers the search away from non-igniting solutions.
    c1_end = c1[-1]
    c2_end = c2[-1]
    T_end = T[-1]
    loss_prior = 100.0 * (
        jnp.maximum(0.0, c1_end - 0.1)
        + jnp.maximum(0.0, 0.9 - c2_end)
        + (1.0 / 600.0) * jnp.maximum(0.0, 600.0 - T_end)
    )

    # An endpoint term was trialled here and reverted; see user_model.py.
    loss_value = loss_rate + loss_temp + loss_prior
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
    trainable_vals = unscale_value(trainable_variables, constants["min_limits"],
                                   constants["max_limits"], constants["is_logscale"])
    Ea1, A1, n1, h1, Ea2, A2, m2, h2 = trainable_vals
    fixed_parameters = constants["fixed_parameters"]
    T_ignite = fixed_parameters['T_ignite']
    kb = fixed_parameters['kb']

    c1 = solution[:, 0]
    c2 = solution[:, 1]
    T = solution[:, 2]

    dc1dt = -A1 * jnp.exp(-Ea1 / (kb * T)) * jnp.abs(c1)**n1
    dc2dt = A2 * jnp.exp(-Ea2 / (kb * T)) * jnp.abs(1.0 - c2)**m2
    rate_sim = jnp.abs(h1 * dc1dt) + jnp.where(T > T_ignite, jnp.abs(h2 * dc2dt), 0.0)

    Nts = solution_time.shape[0]
    writeout_array = jnp.zeros([Nts, 5])
    writeout_array = writeout_array.at[:, 0].set(solution_time)
    writeout_array = writeout_array.at[:, 1].set(dataset[:, 0])   # measured temperature (K)
    writeout_array = writeout_array.at[:, 2].set(dataset[:, 1])   # measured heat rate (K/s)
    writeout_array = writeout_array.at[:, 3].set(T)               # simulated temperature (K)
    writeout_array = writeout_array.at[:, 4].set(rate_sim)        # simulated heat rate (K/s)
    # ---------------------------------------------------

    return writeout_array
