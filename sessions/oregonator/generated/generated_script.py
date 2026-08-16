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


# Oregonator (Field-Noyes) model of the Belousov-Zhabotinsky reaction in
# Tyson's dimensionless scaling. X is HBrO2, autocatalytic and fast; Y is
# bromide, faster still by the ratio eps1/eps2 and the variable that switches
# the autocatalysis off; Z is the oxidised catalyst, relaxing on the slow O(1)
# timescale and feeding bromide back. Only X and Z are recorded.
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

    # Order: ['eps1', 'eps2', 'q', 'f']
    eps1, eps2, q, f = unscale_value(
        trainable_variables, min_val, max_val, is_logscale
    )

    X = y[0]
    Y = y[1]
    Z = y[2]

    # this part is user entered
    # ---------------------------------------------------
    # Autocatalytic production of X, quenched by Y and self-limited
    dX_dt = (q * Y - X * Y + X * (1.0 - X)) / eps1

    # Bromide: consumed by X, regenerated from the oxidised catalyst
    dY_dt = (-q * Y - X * Y + f * Z) / eps2

    # Catalyst relaxation on the slow timescale
    dZ_dt = X - Z
    # ---------------------------------------------------

    return jnp.array([dX_dt, dY_dt, dZ_dt])


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
        max_steps=1000000,
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
    # dataset columns: 0-1 the measured X and Z, 2-3 their measurement
    # standard deviations.
    measured = dataset[:, 0:2]
    sigma = dataset[:, 2:4]

    # X is state 0 and Z is state 2; Y (state 1) is not observed.
    sim = jnp.stack([solution[:, 0], solution[:, 2]], axis=1)

    # Residuals in units of the measurement noise. Both channels carry constant
    # noise, so this is the same as a peak-scaled RMSE up to the factor 0.02,
    # and a fit at the noise level scores ~1.
    resid = (sim - measured) / sigma
    loss_value = jnp.sqrt(jnp.mean(jnp.square(resid)))
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

    # time | 2 measured | 2 simulated observables | simulated Y
    writeout_array = jnp.zeros([Nts, 6])
    writeout_array = writeout_array.at[:, 0].set(solution_time)
    writeout_array = writeout_array.at[:, 1].set(dataset[:, 0])   # measured X
    writeout_array = writeout_array.at[:, 2].set(dataset[:, 1])   # measured Z
    writeout_array = writeout_array.at[:, 3].set(solution[:, 0])  # simulated X
    writeout_array = writeout_array.at[:, 4].set(solution[:, 2])  # simulated Z
    writeout_array = writeout_array.at[:, 5].set(solution[:, 1])  # simulated Y
    # ---------------------------------------------------

    return writeout_array
