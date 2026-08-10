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

    # Order: [rel_n, beta_m, rel_w, treat_w, worsen_n, worsen_m, improve_m, improve_w, bcr]
    (rel_n, beta_m, rel_w, treat_w, worsen_n, worsen_m,
     improve_m, improve_w, bcr) = unscale_value(trainable_variables, min_val, max_val, is_logscale)

    recruitment = fixed_parameters['recruitment_rate']
    S_death     = fixed_parameters['susceptible_death_rate']
    In_death    = fixed_parameters['infected_normal_death_rate']
    Im_death    = fixed_parameters['infected_moderate_death_rate']
    Iw_death    = fixed_parameters['infected_weak_death_rate']
    Tn_death    = fixed_parameters['treated_normal_death_rate']
    Tm_death    = fixed_parameters['treated_moderate_death_rate']
    Tw_death    = fixed_parameters['treated_weak_death_rate']
    treat_n     = fixed_parameters['infected_normal_treatment_rate']
    treat_m     = fixed_parameters['infected_moderate_treatment_rate']
    t_factor    = fixed_parameters['treated_transmission_factor']

    susceptible       = y[0]
    infected_normal   = y[1]
    infected_moderate = y[2]
    infected_weak     = y[3]
    treated_normal    = y[4]
    treated_moderate  = y[5]
    treated_weak      = y[6]

    beta_n = rel_n * beta_m
    beta_w = rel_w * beta_m
    beta_t = t_factor * beta_m

    total_pop = (susceptible + infected_normal + infected_moderate + infected_weak
                 + treated_normal + treated_moderate + treated_weak)
    total_infected = (infected_normal + infected_moderate + infected_weak
                      + treated_normal + treated_moderate + treated_weak)

    force_of_infection = ((beta_n * infected_normal + beta_m * infected_moderate
                           + beta_w * infected_weak
                           + beta_t * (treated_normal + treated_moderate + treated_weak))
                          / total_pop) * jnp.exp(-bcr * total_infected)

    dsusceptible_dt      = recruitment - force_of_infection * susceptible - S_death * susceptible
    dinfected_normal_dt  = (force_of_infection * susceptible
                            - worsen_n * infected_normal - treat_n * infected_normal
                            - In_death * infected_normal)
    dinfected_moderate_dt = (worsen_n * infected_normal
                             - worsen_m * infected_moderate - treat_m * infected_moderate
                             - Im_death * infected_moderate)
    dinfected_weak_dt    = (worsen_m * infected_moderate
                            - treat_w * infected_weak - Iw_death * infected_weak)
    dtreated_normal_dt   = (improve_m * treated_moderate + treat_n * infected_normal
                            - Tn_death * treated_normal)
    dtreated_moderate_dt = (improve_w * treated_weak - improve_m * treated_moderate
                            + treat_m * infected_moderate - Tm_death * treated_moderate)
    dtreated_weak_dt     = (treat_w * infected_weak - improve_w * treated_weak
                            - Tw_death * treated_weak)

    return jnp.array([dsusceptible_dt, dinfected_normal_dt, dinfected_moderate_dt,
                      dinfected_weak_dt, dtreated_normal_dt, dtreated_moderate_dt,
                      dtreated_weak_dt])


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

    # Observable: prevalence (%) = (1 - susceptible / total_population) * 100
    total_pop = jnp.sum(solution, axis=1)
    prevalence = (1.0 - solution[:, 0] / total_pop) * 100.0

    prev_data = dataset[:, 0]
    scale = jnp.maximum(jnp.max(jnp.abs(prev_data)), 1e-12)
    loss_value = jnp.sqrt(jnp.mean(jnp.square((prevalence - prev_data) / scale)))

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

    total_pop = jnp.sum(solution, axis=1)
    prevalence = (1.0 - solution[:, 0] / total_pop) * 100.0

    Nts = solution_time.shape[0]
    writeout_array = jnp.zeros([Nts, 3])
    writeout_array = writeout_array.at[:, 0].set(solution_time)
    writeout_array = writeout_array.at[:, 1].set(dataset[:, 0])
    writeout_array = writeout_array.at[:, 2].set(prevalence)

    return writeout_array
