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


# Sliding-basepoint impact oscillator with stick-slip friction and
# power-driven evolution of the stiffness and damping
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

    # Order: ['c2', 'Dk', 'Dc', 'm1', 'm2']
    c2, Dk, Dc, m1, m2 = unscale_value(trainable_variables, min_val, max_val, is_logscale)
    vf = fixed_parameters['vf']

    x1 = y[0]
    x2 = y[1]
    v1 = y[2]
    v2 = y[3]
    k = y[4]
    c1 = y[5]

    # this part is user entered
    # ---------------------------------------------------
    # coupling spring between the impacting mass and the sliding base
    Fs = k * (x2 - x1)

    # impacting mass: spring force less velocity-proportional damping
    dv1dt = (Fs - c1 * jnp.abs(v1) * jnp.sign(v1)) / m1

    # sliding base: stick-slip Coulomb friction
    F = -Fs
    stuck = jnp.logical_and(jnp.abs(F) < c2, jnp.abs(v2) < vf)
    F2 = jnp.where(stuck, 0.0, F - c2 * jnp.sign(v2))
    dv2dt = F2 / m2

    # stiffness and damping accumulate with dissipated mechanical power
    P = jnp.abs(m1 * v1 * dv1dt)
    dkdt = Dk * P
    dc1dt = Dc * P
    # ---------------------------------------------------

    return jnp.array([v1, v2, dv1dt, dv2dt, dkdt, dc1dt])


# fixed
@jax.jit
def _integrate_system(constants, trainable_variables):
    term = diffrax.ODETerm(user_defined_system)

    solver = diffrax.Dopri8()
    t_eval = constants["t_eval"]
    init_cond = constants["init_cond"]
    init_time = constants["init_time"]
    dataset = constants["dataset"]
    # The saved rows are differenced against `dataset` row-for-row, so the save
    # times MUST equal the data times. Here INITIAL_TIME = 0 while the data
    # starts at 9.94e-4, so this is exactly the case ts=t_eval exists for.
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
    x1 = solution[:, 0]
    x2 = solution[:, 1]
    v1 = solution[:, 2]
    k = solution[:, 4]
    c1 = solution[:, 5]

    # reconstruct the two measured channels from the trajectory
    Fs = k * (x2 - x1)
    F_sim = jnp.abs(Fs - c1 * jnp.abs(v1) * jnp.sign(v1))
    s_sim = x1

    # column 0 of the data is force in kN; the model works in newtons
    F_exp = 1000.0 * dataset[:, 0]
    s_exp = dataset[:, 1]

    # peak-normalised mean absolute error, equal weight on each channel
    loss_F = jnp.mean(jnp.abs(F_exp - F_sim)) / jnp.max(jnp.abs(F_exp))
    loss_s = jnp.mean(jnp.abs(s_exp - s_sim)) / jnp.max(jnp.abs(s_exp))
    loss_value = loss_F + loss_s
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
    x1 = solution[:, 0]
    x2 = solution[:, 1]
    v1 = solution[:, 2]
    k = solution[:, 4]
    c1 = solution[:, 5]

    Fs = k * (x2 - x1)
    F_sim = jnp.abs(Fs - c1 * jnp.abs(v1) * jnp.sign(v1))
    s_sim = x1

    F_exp = 1000.0 * dataset[:, 0]
    s_exp = dataset[:, 1]

    Nts = solution_time.shape[0]
    writeout_array = jnp.zeros([Nts, 5])
    writeout_array = writeout_array.at[:, 0].set(solution_time)
    writeout_array = writeout_array.at[:, 1].set(F_exp)     # measured force (N)
    writeout_array = writeout_array.at[:, 2].set(s_exp)     # measured displacement
    writeout_array = writeout_array.at[:, 3].set(F_sim)     # simulated force (N)
    writeout_array = writeout_array.at[:, 4].set(s_sim)     # simulated displacement
    # ---------------------------------------------------

    return writeout_array
