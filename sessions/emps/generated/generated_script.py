# pfit-sources: user_model.py=2d55ef7192a8cacf user_input.yaml=72edc8653d8fe0a8
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


# Electro-Mechanical Positioning System (Janot, Gautier & Brunot 2019): a mass
# driven along a linear axis by a motor, against viscous and dry friction. The
# motor voltage is an exogenous recorded signal, interpolated onto the solver's
# own time points.
#
# The dry-friction switch is REGULARISED: tanh(v/v_eps) in place of sign(v),
# with v_eps the encoder's own velocity resolution. That trades a chattering
# discontinuity for a fast smooth mode, which is why the solver is implicit.
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

    # Order: ['M', 'Fv', 'Fc', 'OF']
    M, Fv, Fc, OF = unscale_value(
        trainable_variables, min_val, max_val, is_logscale
    )

    gtau = fixed_parameters['gtau']
    v_eps = fixed_parameters['v_eps']

    q = y[0]
    v = y[1]

    # this part is user entered
    # ---------------------------------------------------
    # dataset excludes the time column, so column 0 is the measured position and
    # column 1 is the motor voltage. Interpolate the input at t.
    vir = jnp.interp(t, t_eval, dataset[:, 1])

    # Applied force at the load side, less viscous drag, dry friction and a
    # constant offset. tanh(v/v_eps) is the regularised sign: it saturates to
    # +/-1 within a few v_eps of zero and is smooth through it.
    force = gtau * vir - Fv * v - Fc * jnp.tanh(v / v_eps) - OF

    dq_dt = v
    dv_dt = force / M
    # ---------------------------------------------------

    return jnp.array([dq_dt, dv_dt])


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
        max_steps=100000,
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
    # One observable: the axis position, measured directly as q.
    measured = dataset[:, 0]
    model_obs = solution[:, 0]

    # Peak-normalised RMSE. The record carries no uncertainty column, so the
    # residual is scaled by the position column's own peak (0.2464 m). Multiply
    # the reported loss by that peak to recover an RMSE in metres.
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

    # time | measured position | simulated position | simulated velocity | input
    writeout_array = jnp.zeros([Nts, 5])
    writeout_array = writeout_array.at[:, 0].set(solution_time)
    writeout_array = writeout_array.at[:, 1].set(dataset[:, 0])   # measured position
    writeout_array = writeout_array.at[:, 2].set(solution[:, 0])  # simulated position
    writeout_array = writeout_array.at[:, 3].set(solution[:, 1])  # simulated velocity
    writeout_array = writeout_array.at[:, 4].set(dataset[:, 1])   # motor voltage
    # ---------------------------------------------------

    return writeout_array
