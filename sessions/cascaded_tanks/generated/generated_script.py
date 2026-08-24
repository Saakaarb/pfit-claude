# pfit-sources: user_model.py=156dc44a49fff1a7 user_input.yaml=5116bb2cc22f5e7c
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


# Cascaded tanks with overflow (Schoukens et al. 2016). A pump drives water into
# an upper tank, which drains through a small opening into a lower tank, which
# drains back to the reservoir. Both outflows follow Bernoulli, hence the square
# roots. The pump voltage is an exogenous recorded signal, interpolated onto the
# solver's own time points. The overflow is deliberately not modelled - this is
# the benchmark's own baseline.
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

    # Order: ['k1', 'k2', 'k3', 'k4']
    k1, k2, k3, k4 = unscale_value(
        trainable_variables, min_val, max_val, is_logscale
    )

    x1 = y[0]
    x2 = y[1]

    # this part is user entered
    # ---------------------------------------------------
    # dataset excludes the time column, so column 0 is the measured level and
    # column 1 is the pump voltage. Interpolate the input at t.
    u = jnp.interp(t, t_eval, dataset[:, 1])

    # Clamp before the square root. The measured levels stay well above zero,
    # but a candidate parameter set can drive a level negative, and
    # sqrt(negative) would put NaN into the loss for every later time point.
    sqrt_x1 = jnp.sqrt(jnp.maximum(x1, 0.0))
    sqrt_x2 = jnp.sqrt(jnp.maximum(x2, 0.0))

    dx1_dt = -k1 * sqrt_x1 + k4 * u
    dx2_dt = k2 * sqrt_x1 - k3 * sqrt_x2
    # ---------------------------------------------------

    return jnp.array([dx1_dt, dx2_dt])


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
    # One observable: the lower tank level, measured directly as x2.
    measured = dataset[:, 0]
    model_obs = solution[:, 1]

    # Peak-normalised RMSE. There is no uncertainty column in this record - the
    # benchmark quotes an output SNR of ~40 dB rather than per-point sigmas - so
    # the residual is scaled by the column's own peak (10.0). Multiply the
    # reported loss by that peak to recover the benchmark's e_RMS in volts.
    scale = jnp.max(jnp.abs(measured))
    scale = jnp.where(scale == 0, 1.0, scale)

    loss_value = jnp.sqrt(jnp.mean(jnp.square((model_obs - measured) / scale)))
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

    # time | measured level | simulated level | simulated upper tank | input
    writeout_array = jnp.zeros([Nts, 5])
    writeout_array = writeout_array.at[:, 0].set(solution_time)
    writeout_array = writeout_array.at[:, 1].set(dataset[:, 0])   # measured lower tank
    writeout_array = writeout_array.at[:, 2].set(solution[:, 1])  # simulated lower tank
    writeout_array = writeout_array.at[:, 3].set(solution[:, 0])  # simulated upper tank
    writeout_array = writeout_array.at[:, 4].set(dataset[:, 1])   # pump voltage
    # ---------------------------------------------------

    return writeout_array
