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

    # Order: ['Epo_degradation_BaF3', 'k_exp_hetero', 'k_exp_homo', 'k_imp_hetero', 'k_imp_homo', 'k_phos']
    Epo_degradation_BaF3, k_exp_hetero, k_exp_homo, k_imp_hetero, k_imp_homo, k_phos = \
        unscale_value(trainable_variables, min_val, max_val, is_logscale)

    cyt     = fixed_parameters['cyt']
    nuc     = fixed_parameters['nuc']
    Epo0    = fixed_parameters['Epo0']
    specC17 = fixed_parameters['specC17']

    STAT5A  = y[0]
    STAT5B  = y[1]
    pApB    = y[2]
    pApA    = y[3]
    pBpB    = y[4]
    nucpApA = y[5]
    nucpApB = y[6]
    nucpBpB = y[7]

    # Time-dependent Epo input (SBML assignment rule).
    BaF3_Epo = Epo0 * jnp.exp(-Epo_degradation_BaF3 * t)

    cyt_over_nuc = cyt / nuc
    nuc_over_cyt = nuc / cyt

    phos_AA = BaF3_Epo * k_phos * STAT5A * STAT5A
    phos_AB = BaF3_Epo * k_phos * STAT5A * STAT5B
    phos_BB = BaF3_Epo * k_phos * STAT5B * STAT5B

    dSTAT5A_dt = (-2.0 * phos_AA - phos_AB
                  + 2.0 * nuc_over_cyt * k_exp_homo * nucpApA
                  + nuc_over_cyt * k_exp_hetero * nucpApB)
    dSTAT5B_dt = (-phos_AB - 2.0 * phos_BB
                  + nuc_over_cyt * k_exp_hetero * nucpApB
                  + 2.0 * nuc_over_cyt * k_exp_homo * nucpBpB)

    dpApB_dt = phos_AB - k_imp_hetero * pApB
    dpApA_dt = phos_AA - k_imp_homo * pApA
    dpBpB_dt = phos_BB - k_imp_homo * pBpB

    dnucpApA_dt = cyt_over_nuc * k_imp_homo * pApA - k_exp_homo * nucpApA
    dnucpApB_dt = cyt_over_nuc * k_imp_hetero * pApB - k_exp_hetero * nucpApB
    dnucpBpB_dt = cyt_over_nuc * k_imp_homo * pBpB - k_exp_homo * nucpBpB

    return jnp.array([dSTAT5A_dt, dSTAT5B_dt, dpApB_dt, dpApA_dt, dpBpB_dt,
                      dnucpApA_dt, dnucpApB_dt, dnucpBpB_dt])


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
    # Any code other than RESULTS.successful means the trajectory is untrustworthy
    # (it may contain inf/NaN). See lib/LLM/api/diffrax.md for the full code table.
    failed = jnp.invert(result == RESULTS.successful)
    # ---------------------------------------------------

    specC17 = constants["fixed_parameters"]["specC17"]

    STAT5A = solution[:, 0]
    STAT5B = solution[:, 1]
    pApB   = solution[:, 2]
    pApA   = solution[:, 3]
    pBpB   = solution[:, 4]

    pSTAT5A_rel = (100.0 * pApB + 200.0 * pApA * specC17) / \
                  (pApB + STAT5A * specC17 + 2.0 * pApA * specC17)
    pSTAT5B_rel = -(100.0 * pApB - 200.0 * pBpB * (specC17 - 1.0)) / \
                  ((STAT5B * (specC17 - 1.0) - pApB) + 2.0 * pBpB * (specC17 - 1.0))
    rSTAT5A_rel = (100.0 * pApB + 100.0 * STAT5A * specC17 + 200.0 * pApA * specC17) / \
                  (2.0 * pApB + STAT5A * specC17 + 2.0 * pApA * specC17
                   - STAT5B * (specC17 - 1.0) - 2.0 * pBpB * (specC17 - 1.0))

    model_obs = jnp.stack([pSTAT5A_rel, pSTAT5B_rel, rSTAT5A_rel], axis=1)

    scale_factor = jnp.maximum(jnp.max(dataset, axis=0), 1e-12)
    loss_value = jnp.sqrt(jnp.mean(jnp.square((model_obs - dataset) / scale_factor)))

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

    specC17 = constants["fixed_parameters"]["specC17"]

    STAT5A = solution[:, 0]
    STAT5B = solution[:, 1]
    pApB   = solution[:, 2]
    pApA   = solution[:, 3]
    pBpB   = solution[:, 4]

    pSTAT5A_rel = (100.0 * pApB + 200.0 * pApA * specC17) / \
                  (pApB + STAT5A * specC17 + 2.0 * pApA * specC17)
    pSTAT5B_rel = -(100.0 * pApB - 200.0 * pBpB * (specC17 - 1.0)) / \
                  ((STAT5B * (specC17 - 1.0) - pApB) + 2.0 * pBpB * (specC17 - 1.0))
    rSTAT5A_rel = (100.0 * pApB + 100.0 * STAT5A * specC17 + 200.0 * pApA * specC17) / \
                  (2.0 * pApB + STAT5A * specC17 + 2.0 * pApA * specC17
                   - STAT5B * (specC17 - 1.0) - 2.0 * pBpB * (specC17 - 1.0))

    Nts = solution_time.shape[0]
    writeout_array = jnp.zeros([Nts, 7])
    writeout_array = writeout_array.at[:, 0].set(solution_time)
    writeout_array = writeout_array.at[:, 1].set(dataset[:, 0])
    writeout_array = writeout_array.at[:, 2].set(pSTAT5A_rel)
    writeout_array = writeout_array.at[:, 3].set(dataset[:, 1])
    writeout_array = writeout_array.at[:, 4].set(pSTAT5B_rel)
    writeout_array = writeout_array.at[:, 5].set(dataset[:, 2])
    writeout_array = writeout_array.at[:, 6].set(rSTAT5A_rel)

    return writeout_array
