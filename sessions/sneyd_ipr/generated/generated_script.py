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

    # Order: [k1, k2, k3, k4, k_1, k_2, k_3, k_4, l2, l4, l6, l_2, l_4, l_6]
    (k1, k2, k3, k4, k_1, k_2, k_3, k_4,
     l2, l4, l6, l_2, l_4, l_6) = unscale_value(trainable_variables, min_val, max_val, is_logscale)

    IPR_O  = y[0]
    IPR_R  = y[1]
    IPR_I1 = y[2]
    IPR_S  = y[3]
    IPR_A  = y[4]
    IPR_I2 = y[5]
    IP3    = y[6]
    Ca     = y[7]

    L1 = (k_1 * l2) / (k1 * l_2)
    L3 = (k_2 * l4) / (k2 * l_4)
    L5 = (k_4 * l6) / (k4 * l_6)

    v0 = ((k_2 + l_4 * Ca) / (1.0 + Ca / L5)) * IPR_O
    v1 = ((k2 * L3 + l4 * Ca) / (L3 + Ca * (1.0 + L3 / L1))) * IP3 * IPR_R
    v2 = (((k1 * L1 + l2) * Ca) / (L1 + Ca * (1.0 + L1 / L3))) * IPR_R
    v3 = (k_1 + l_2) * IPR_I1
    v4 = (((k4 * L5 + l6) * Ca) / (L5 + Ca)) * IPR_O
    v5 = ((L1 * (k_4 + l_6)) / (L1 + Ca)) * IPR_A
    v6 = (((k1 * L1 + l2) * Ca) / (L1 + Ca)) * IPR_A
    v7 = (k_1 + l_2) * IPR_I2
    v8 = ((k3 * L5) / (L5 + Ca)) * IPR_O
    v9 = k_3 * IPR_S

    dIPR_O_dt  = -v0 + v1 - v4 + v5 - v8 + v9
    dIPR_R_dt  =  v0 - v1 - v2 + v3
    dIPR_I1_dt =  v2 - v3
    dIPR_S_dt  =  v8 - v9
    dIPR_A_dt  =  v4 - v5 - v6 + v7
    dIPR_I2_dt =  v6 - v7
    dip3_in_dt = 0.0
    dca_in_dt  = 0.0

    return jnp.array([dIPR_O_dt, dIPR_R_dt, dIPR_I1_dt, dIPR_S_dt, dIPR_A_dt,
                      dIPR_I2_dt, dip3_in_dt, dca_in_dt])


# fixed
@jax.jit
def _integrate_system(constants, trainable_variables):
    term = diffrax.ODETerm(user_defined_system)

    solver = diffrax.Kvaerno5()
    t_eval = constants["t_eval"]
    init_cond = constants["init_cond"]
    init_time = constants["init_time"]
    dataset = constants["dataset"]
    # save times must equal the data times (rows are differenced against dataset)
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
    # Any code other than RESULTS.successful means the trajectory is untrustworthy
    # (it may contain inf/NaN). See lib/LLM/api/diffrax.md for the full code table.
    failed = jnp.invert(result == RESULTS.successful)
    # ---------------------------------------------------

    # Observable: channel open probability Po = (0.9*A + 0.1*O)^4
    IPR_O = solution[:, 0]
    IPR_A = solution[:, 4]
    Po = jnp.power(0.9 * IPR_A + 0.1 * IPR_O, 4)

    Po_data = dataset[:, 0]
    loss_value = jnp.sqrt(jnp.mean(jnp.square(Po - Po_data)))

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

    IPR_O = solution[:, 0]
    IPR_A = solution[:, 4]
    Po = jnp.power(0.9 * IPR_A + 0.1 * IPR_O, 4)

    Nts = solution_time.shape[0]
    writeout_array = jnp.zeros([Nts, 3])
    writeout_array = writeout_array.at[:, 0].set(solution_time)
    writeout_array = writeout_array.at[:, 1].set(dataset[:, 0])
    writeout_array = writeout_array.at[:, 2].set(Po)

    return writeout_array
