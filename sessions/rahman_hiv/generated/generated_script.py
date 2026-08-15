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


# Immunity-based HIV transmission model of Rahman, Vaidya & Zou (2016): seven
# compartments across an infected and a treated arm, coupled through a force of
# infection damped by a behavioural-change response. Time in years.
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

    # Order: ['rel_n', 'beta_m', 'rel_w', 'tau_w', 'w_n', 'w_m', 'g_m', 'g_w', 'bcr']
    rel_n, beta_m, rel_w, tau_w, w_n, w_m, g_m, g_w, bcr = unscale_value(
        trainable_variables, min_val, max_val, is_logscale
    )

    Lambda = fixed_parameters['Lambda']
    f_t = fixed_parameters['f_t']
    tau_n = fixed_parameters['tau_n']
    tau_m = fixed_parameters['tau_m']
    mu_S = fixed_parameters['mu_S']
    mu_In = fixed_parameters['mu_In']
    mu_Im = fixed_parameters['mu_Im']
    mu_Iw = fixed_parameters['mu_Iw']
    mu_Tn = fixed_parameters['mu_Tn']
    mu_Tm = fixed_parameters['mu_Tm']
    mu_Tw = fixed_parameters['mu_Tw']

    S = y[0]
    In = y[1]
    Im = y[2]
    Iw = y[3]
    Tn = y[4]
    Tm = y[5]
    Tw = y[6]

    # this part is user entered
    # ---------------------------------------------------
    # the stage transmission rates are all tied to the single reference rate
    beta_n = rel_n * beta_m
    beta_w = rel_w * beta_m
    beta_t = f_t * beta_m

    N = S + In + Im + Iw + Tn + Tm + Tw
    infected_total = In + Im + Iw + Tn + Tm + Tw

    # force of infection, damped by a behavioural-change response that
    # weakens transmission as the infected population grows
    lam = ((beta_n * In + beta_m * Im + beta_w * Iw
            + beta_t * (Tn + Tm + Tw)) / N) * jnp.exp(-bcr * infected_total)

    dSdt = Lambda - lam * S - mu_S * S

    dIndt = lam * S - w_n * In - tau_n * In - mu_In * In
    dImdt = w_n * In - w_m * Im - tau_m * Im - mu_Im * Im
    dIwdt = w_m * Im - tau_w * Iw - mu_Iw * Iw

    # the treated arm mirrors the infected arm with the arrows reversed:
    # individuals enter at tau and then improve back up the stages at g
    dTndt = tau_n * In + g_m * Tm - mu_Tn * Tn
    dTmdt = tau_m * Im + g_w * Tw - g_m * Tm - mu_Tm * Tm
    dTwdt = tau_w * Iw - g_w * Tw - mu_Tw * Tw
    # ---------------------------------------------------

    return jnp.array([dSdt, dIndt, dImdt, dIwdt, dTndt, dTmdt, dTwdt])


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
    S = solution[:, 0]
    N = (solution[:, 0] + solution[:, 1] + solution[:, 2] + solution[:, 3]
         + solution[:, 4] + solution[:, 5] + solution[:, 6])

    # the lone observable: the percentage of the population not susceptible
    prevalence_sim = (1.0 - S / N) * 100.0
    prevalence_exp = dataset[:, 0]

    # peak-normalised RMSE, which keeps the loss on a [0, 1] scale
    scale_factor = jnp.max(jnp.abs(prevalence_exp))
    scale_factor = jnp.where(scale_factor == 0, 1.0, scale_factor)

    loss_value = jnp.sqrt(jnp.mean(jnp.square(
        (prevalence_sim - prevalence_exp) / scale_factor)))
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
    S = solution[:, 0]
    N = (solution[:, 0] + solution[:, 1] + solution[:, 2] + solution[:, 3]
         + solution[:, 4] + solution[:, 5] + solution[:, 6])

    prevalence_sim = (1.0 - S / N) * 100.0
    prevalence_exp = dataset[:, 0]

    Nts = solution_time.shape[0]
    writeout_array = jnp.zeros([Nts, 3])
    writeout_array = writeout_array.at[:, 0].set(solution_time)
    writeout_array = writeout_array.at[:, 1].set(prevalence_exp)   # measured prevalence (%)
    writeout_array = writeout_array.at[:, 2].set(prevalence_sim)   # simulated prevalence (%)
    # ---------------------------------------------------

    return writeout_array
