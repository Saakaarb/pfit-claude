import jax
import jax.numpy as jnp
import diffrax
from diffrax import RESULTS

jax.config.update("jax_enable_x64", True)


@jax.jit
def unscale_value(val, min_val, max_val, is_logscale):
    lin_unscaled = ((val + 1.0) / 2.0) * (max_val - min_val) + min_val
    unscaled = jnp.where(is_logscale, 10.0**lin_unscaled, lin_unscaled)
    return unscaled


@jax.jit
def scale_value(unscaled_val, min_val, max_val, is_logscale):
    lin_val = jnp.where(is_logscale, jnp.log10(unscaled_val), unscaled_val)
    scaled = 2.0 * (lin_val - min_val) / (max_val - min_val) - 1.0
    return scaled


@jax.jit
def user_defined_system(t, y, other_args):
    trainable_variables = other_args["trainable_variables"]
    constants = other_args["constants"]
    dataset = constants["dataset"]
    t_eval = constants["t_eval"]
    fixed_parameters = constants["fixed_parameters"]
    min_val = constants["min_limits"]
    max_val = constants["max_limits"]
    is_logscale = constants["is_logscale"]

    # order: c2, Dk, Dc, m1, m2
    c2, Dk, Dc, m1, m2 = unscale_value(trainable_variables, min_val, max_val, is_logscale)
    vf = fixed_parameters["vf"]

    x1 = y[0]
    x2 = y[1]
    v1 = y[2]
    v2 = y[3]
    k  = y[4]
    c1 = y[5]

    def F1(Fs, c1val, v1val):
        return Fs - c1val * jnp.abs(v1val) * jnp.sign(v1val)

    def F2(Fs, c2val, v2val):
        cond = jnp.logical_and(jnp.abs(Fs) < c2val, jnp.abs(v2val) < vf)
        return jnp.where(cond, 0.0, Fs - c2val * jnp.sign(v2val))

    Fs    = k * (x2 - x1)
    dv1dt = (1.0 / m1) * F1(Fs, c1, v1)
    dv2dt = (1.0 / m2) * F2(-Fs, c2, v2)
    P     = jnp.abs(m1 * v1 * dv1dt)

    return jnp.array([v1, v2, dv1dt, dv2dt, Dk * P, Dc * P])


@jax.jit
def _integrate_system(constants, trainable_variables):
    term = diffrax.ODETerm(user_defined_system)
    solver = diffrax.Kvaerno5()
    t_eval = constants["t_eval"]
    init_cond = constants["init_cond"]
    init_time = constants["init_time"]
    dataset = constants["dataset"]
    saveat = diffrax.SaveAt(t0=True, ts=t_eval[1:])
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
    dataset = constants["dataset"]
    solution_time, solution, result = _integrate_system(constants, trainable_variables)
    failed = jnp.logical_or(result == RESULTS.max_steps_reached, result == RESULTS.singular)

    F_exp = dataset[:, 0:1] * 1000.0  # kN -> N
    s_exp = dataset[:, 1:2]

    Fm = jnp.max(jnp.abs(F_exp))
    sm = jnp.max(jnp.abs(s_exp))

    x1  = solution[:, 0]
    x2  = solution[:, 1]
    v1  = solution[:, 2]
    k   = solution[:, 4]
    c1  = solution[:, 5]

    Fs    = k * (x2 - x1)
    F_sim = jnp.abs(Fs - c1 * jnp.abs(v1) * jnp.sign(v1))
    s_sim = x1

    # paper loss (eq. 12): mean absolute relative error, equal weight on F and s
    F_loss = jnp.mean(jnp.abs(F_exp[:, 0] - F_sim) / Fm)
    s_loss = jnp.mean(jnp.abs(s_exp[:, 0] - s_sim) / sm)
    loss_value = F_loss + s_loss

    loss = jnp.where(failed, constants["error_loss"], loss_value)
    return loss


def _write_problem_result(constants, trainable_variables):
    dataset = constants["dataset"]
    solution_time, solution, result = _integrate_system(constants, trainable_variables)

    F_exp = dataset[:, 0:1] * 1000.0  # kN -> N
    s_exp = dataset[:, 1:2]

    x1  = solution[:, 0]
    x2  = solution[:, 1]
    v1  = solution[:, 2]
    k   = solution[:, 4]
    c1  = solution[:, 5]

    Fs    = k * (x2 - x1)
    F_sim = jnp.abs(Fs - c1 * jnp.abs(v1) * jnp.sign(v1))
    s_sim = x1

    Nts = solution_time.shape[0]
    writeout_array = jnp.zeros((Nts, 5), dtype=jnp.float64)
    writeout_array = writeout_array.at[:, 0].set(solution_time)
    writeout_array = writeout_array.at[:, 1].set(F_exp[:, 0])
    writeout_array = writeout_array.at[:, 2].set(s_exp[:, 0])
    writeout_array = writeout_array.at[:, 3].set(F_sim)
    writeout_array = writeout_array.at[:, 4].set(s_sim)

    return writeout_array
