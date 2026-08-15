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


# Six-state IP3-receptor gating model of Sneyd & Dufour (2002). IP3 and Ca are
# clamped per experiment and carried as extra states with zero derivative.
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

    # Order: ['k1','k2','k3','k4','k_1','k_2','k_3','k_4','l2','l4','l6','l_2','l_4','l_6']
    k1, k2, k3, k4, k_1, k_2, k_3, k_4, l2, l4, l6, l_2, l_4, l_6 = unscale_value(
        trainable_variables, min_val, max_val, is_logscale
    )

    O = y[0]
    R = y[1]
    I1 = y[2]
    S = y[3]
    A = y[4]
    I2 = y[5]
    IP3 = y[6]
    Ca = y[7]

    # this part is user entered
    # ---------------------------------------------------
    # equilibrium constants derived from the fitted rates
    L1 = k_1 * l2 / (k1 * l_2)
    L3 = k_2 * l4 / (k2 * l_4)
    L5 = k_4 * l6 / (k4 * l_6)

    # the ten transition fluxes
    v0 = (k_2 + l_4 * Ca) / (1.0 + Ca / L5) * O                          # O  -> R
    v1 = (k2 * L3 + l4 * Ca) / (L3 + Ca * (1.0 + L3 / L1)) * IP3 * R     # R  -> O
    v2 = (k1 * L1 + l2) * Ca / (L1 + Ca * (1.0 + L1 / L3)) * R           # R  -> I1
    v3 = (k_1 + l_2) * I1                                                # I1 -> R
    v4 = (k4 * L5 + l6) * Ca / (L5 + Ca) * O                             # O  -> A
    v5 = L1 * (k_4 + l_6) / (L1 + Ca) * A                                # A  -> O
    v6 = (k1 * L1 + l2) * Ca / (L1 + Ca) * A                             # A  -> I2
    v7 = (k_1 + l_2) * I2                                                # I2 -> A
    v8 = k3 * L5 / (L5 + Ca) * O                                         # O  -> S
    v9 = k_3 * S                                                         # S  -> O

    dOdt = -v0 + v1 - v4 + v5 - v8 + v9
    dRdt = v0 - v1 - v2 + v3
    dI1dt = v2 - v3
    dSdt = v8 - v9
    dAdt = v4 - v5 - v6 + v7
    dI2dt = v6 - v7

    # clamped inputs: held constant over the solve
    dIP3dt = 0.0
    dCadt = 0.0
    # ---------------------------------------------------

    return jnp.array([dOdt, dRdt, dI1dt, dSdt, dAdt, dI2dt, dIP3dt, dCadt])


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
    O = solution[:, 0]
    A = solution[:, 4]

    # the measured observable is the channel open probability
    Po_sim = (0.9 * A + 0.1 * O) ** 4
    Po_exp = dataset[:, 0]

    # Po is already a probability in [0, 1], so a plain RMSE is normalized
    loss_value = jnp.sqrt(jnp.mean(jnp.square(Po_sim - Po_exp)))
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
    O = solution[:, 0]
    A = solution[:, 4]

    Po_sim = (0.9 * A + 0.1 * O) ** 4
    Po_exp = dataset[:, 0]

    Nts = solution_time.shape[0]
    writeout_array = jnp.zeros([Nts, 3])
    writeout_array = writeout_array.at[:, 0].set(solution_time)
    writeout_array = writeout_array.at[:, 1].set(Po_exp)     # measured open probability
    writeout_array = writeout_array.at[:, 2].set(Po_sim)     # simulated open probability
    # ---------------------------------------------------

    return writeout_array
