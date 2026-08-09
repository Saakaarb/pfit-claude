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
    # Linearly map [min_val, max_val] → [-1, 1]
    scaled = 2.0 * (lin_val - min_val) / (max_val - min_val) - 1.0
    return scaled


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

    # Order: ['Ea1', 'h1', 'A1', 'A2', 'Ea2', 'h2', 'm2', 'n1']
    Ea1, h1, A1, A2, Ea2, h2, m2, n1 = unscale_value(trainable_variables, min_val, max_val, is_logscale)

    kb = fixed_parameters['kb']

    c1 = y[0]
    c2 = y[1]
    T  = y[2]

    dc1_dt = -A1 * jnp.exp(-Ea1 / (kb * T)) * c1**n1
    dc2_dt = A2 * jnp.exp(-Ea2 / (kb * T)) * (1 - c2)**m2
    dT_dt = jnp.abs(h1 * dc1_dt)
    dT_dt = jnp.where(T > 485.0, dT_dt + jnp.abs(h2 * dc2_dt), dT_dt)

    return jnp.array([dc1_dt, dc2_dt, dT_dt])


# fixed
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

    # fixed
    # ---------------------------------------------------
    dataset = constants["dataset"]
    solution_time, solution, result = _integrate_system(constants, trainable_variables)
    # Any code other than RESULTS.successful means the trajectory is untrustworthy
    # (it may contain inf/NaN). See lib/LLM/api/diffrax.md for the full code table.
    failed = jnp.invert(result == RESULTS.successful)
    # ---------------------------------------------------

    def compute_dTdt_at_t(t, y):
        other_args = {"constants": constants, "trainable_variables": trainable_variables}
        return user_defined_system(t, y, other_args)[-1]

    heat_rate_pred = jax.vmap(compute_dTdt_at_t)(solution_time, solution)
    eps = 1e-12
    log_range = jnp.log10(jnp.max(dataset[:, -1] + eps)) - jnp.log10(jnp.min(dataset[:, -1] + eps))
    loss1 = jnp.mean(jnp.abs(jnp.log10(heat_rate_pred + eps) - jnp.log10(dataset[:, -1] + eps))) / log_range

    loss2 = jnp.mean(jnp.abs((dataset[:, 0] - solution[:, 2]) / jnp.max(jnp.abs(dataset[:, 0]))))

    c1_end = solution[-1, 0]
    c2_end = solution[-1, 1]
    T_end  = solution[-1, 2]
    loss3 = 100.0 * (jnp.maximum(0.0, c1_end - 0.1) +
                     jnp.maximum(0.0, 0.9 - c2_end) +
                     jnp.maximum(0.0, 600.0 - T_end) / 600.0)

    loss_value = loss1 + loss2 + loss3

    # fixed
    # --------------------------------------
    loss = jnp.where(failed,
        constants["error_loss"],
        loss_value
    )

    return loss


def _write_problem_result(constants, trainable_variables):

    # fixed
    # ---------------------------------------------------
    dataset = constants["dataset"]
    solution_time, solution, result = _integrate_system(constants, trainable_variables)
    # ---------------------------------------------------

    Nts = solution_time.shape[0]
    writeout_array = jnp.zeros([Nts, 5])
    writeout_array = writeout_array.at[:, 0].set(solution_time)
    writeout_array = writeout_array.at[:, 1].set(dataset[:, 0])
    writeout_array = writeout_array.at[:, 2].set(dataset[:, 1])
    writeout_array = writeout_array.at[:, 3].set(solution[:, 2])

    def compute_dTdt_at_t(t, y):
        other_args = {"constants": constants, "trainable_variables": trainable_variables}
        derivs = user_defined_system(t, y, other_args)
        return derivs[-1]

    heat_rate_pred = jax.vmap(compute_dTdt_at_t)(solution_time, solution)
    writeout_array = writeout_array.at[:, 4].set(heat_rate_pred)

    return writeout_array
