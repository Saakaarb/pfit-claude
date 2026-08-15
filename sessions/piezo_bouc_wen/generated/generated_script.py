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


# Piezoelectric actuator with Bouc-Wen hysteresis
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

    # Order: ['alpha', 'beta', 'gamma', 'cp', 'kp', 'de']
    alpha, beta, gamma, cp, kp, de = unscale_value(trainable_variables, min_val, max_val, is_logscale)
    mp = fixed_parameters['mp']

    xp = y[0]
    vp = y[1]
    h = y[2]

    # this part is user entered
    # ---------------------------------------------------
    V = 24.0 + 24.0 * jnp.sin(16.0 * jnp.pi * t)
    V_dot = 24.0 * 16.0 * jnp.pi * jnp.cos(16.0 * jnp.pi * t)

    dxp_dt = vp
    dvp_dt = (kp * (de * V - h) - cp * vp - kp * xp) / mp
    dh_dt = alpha * de * V_dot - beta * jnp.abs(V_dot) * jnp.abs(h) - gamma * V_dot * jnp.abs(h)
    # ---------------------------------------------------

    return jnp.array([dxp_dt, dvp_dt, dh_dt])


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
    # times MUST equal the data times.
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
    failed = jnp.invert(result == RESULTS.successful)
    # ---------------------------------------------------

    # this part is user entered
    # ---------------------------------------------------
    exp_disp = dataset[:, 0]
    sim_disp = solution[:, 0]

    scale_factor = jnp.max(jnp.abs(exp_disp))
    loss_value = jnp.sqrt(jnp.mean(jnp.square((exp_disp - sim_disp) / scale_factor)))
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
    # the pseudocode fills V_applied with a Python loop; vectorised here because
    # JAX arrays are immutable and the expression is elementwise in time
    V_applied = 24.0 + 24.0 * jnp.sin(16.0 * jnp.pi * solution_time)

    writeout_array = jnp.zeros([Nts, 6])
    writeout_array = writeout_array.at[:, 0].set(solution_time)
    writeout_array = writeout_array.at[:, 1].set(dataset[:, 0])   # experimental displacement
    writeout_array = writeout_array.at[:, 2].set(solution[:, 0])  # simulated xp
    writeout_array = writeout_array.at[:, 3].set(solution[:, 2])  # simulated h
    writeout_array = writeout_array.at[:, 4].set(V_applied)
    writeout_array = writeout_array.at[:, 5].set(solution[:, 1])  # simulated vp
    # ---------------------------------------------------

    return writeout_array
