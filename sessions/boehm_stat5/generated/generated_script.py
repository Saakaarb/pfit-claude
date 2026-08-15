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


# STAT5A/STAT5B dimerisation model of Boehm et al. (2014): cytoplasmic
# phosphorylation and dimerisation, nuclear import, and export back as monomers,
# driven by an exponentially decaying Epo stimulus.
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

    # Order: ['k_deg', 'k_exp_hetero', 'k_exp_homo', 'k_imp_hetero', 'k_imp_homo', 'k_phos']
    k_deg, k_exp_hetero, k_exp_homo, k_imp_hetero, k_imp_homo, k_phos = unscale_value(
        trainable_variables, min_val, max_val, is_logscale
    )

    Epo0 = fixed_parameters['Epo0']
    cyt = fixed_parameters['cyt']
    nuc = fixed_parameters['nuc']

    A = y[0]
    B = y[1]
    ApB = y[2]
    ApA = y[3]
    BpB = y[4]
    nApA = y[5]
    nApB = y[6]
    nBpB = y[7]

    # this part is user entered
    # ---------------------------------------------------
    # Epo stimulation decays exponentially from its initial level
    Epo = Epo0 * jnp.exp(-k_deg * t)

    # mass-action phosphorylation / dimerisation in the cytoplasm
    phos_AA = k_phos * Epo * A * A
    phos_AB = k_phos * Epo * A * B
    phos_BB = k_phos * Epo * B * B

    # The volume ratios convert the transport fluxes between compartments:
    # material leaving the cytoplasm is diluted into the smaller nucleus.
    dAdt = (-2.0 * phos_AA - phos_AB
            + (nuc / cyt) * (2.0 * k_exp_homo * nApA + k_exp_hetero * nApB))
    dBdt = (-2.0 * phos_BB - phos_AB
            + (nuc / cyt) * (2.0 * k_exp_homo * nBpB + k_exp_hetero * nApB))

    dApBdt = phos_AB - k_imp_hetero * ApB
    dApAdt = phos_AA - k_imp_homo * ApA
    dBpBdt = phos_BB - k_imp_homo * BpB

    dnApAdt = (cyt / nuc) * k_imp_homo * ApA - k_exp_homo * nApA
    dnApBdt = (cyt / nuc) * k_imp_hetero * ApB - k_exp_hetero * nApB
    dnBpBdt = (cyt / nuc) * k_imp_homo * BpB - k_exp_homo * nBpB
    # ---------------------------------------------------

    return jnp.array([dAdt, dBdt, dApBdt, dApAdt, dBpBdt,
                      dnApAdt, dnApBdt, dnBpBdt])


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
    fixed_parameters = constants["fixed_parameters"]
    s = fixed_parameters['specC17']

    A = solution[:, 0]
    B = solution[:, 1]
    ApB = solution[:, 2]
    ApA = solution[:, 3]
    BpB = solution[:, 4]

    # the three measured relative percentages, from cytoplasmic species only
    pSTAT5A = (100.0 * ApB + 200.0 * ApA * s) / (ApB + A * s + 2.0 * ApA * s)
    pSTAT5B = (-(100.0 * ApB - 200.0 * BpB * (s - 1.0))
               / ((B * (s - 1.0) - ApB) + 2.0 * BpB * (s - 1.0)))
    rSTAT5A = ((100.0 * ApB + 100.0 * A * s + 200.0 * ApA * s)
               / (2.0 * ApB + A * s + 2.0 * ApA * s
                  - B * (s - 1.0) - 2.0 * BpB * (s - 1.0)))

    sim = jnp.stack([pSTAT5A, pSTAT5B, rSTAT5A], axis=1)

    # column-wise scale-normalised RMSE, so the three percentage channels
    # contribute comparably regardless of their individual ranges
    scale_factor = jnp.max(jnp.abs(dataset), axis=0)
    scale_factor = jnp.where(scale_factor == 0, 1.0, scale_factor)

    loss_value = jnp.sqrt(jnp.mean(jnp.square((sim - dataset) / scale_factor)))
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
    fixed_parameters = constants["fixed_parameters"]
    s = fixed_parameters['specC17']

    A = solution[:, 0]
    B = solution[:, 1]
    ApB = solution[:, 2]
    ApA = solution[:, 3]
    BpB = solution[:, 4]

    pSTAT5A = (100.0 * ApB + 200.0 * ApA * s) / (ApB + A * s + 2.0 * ApA * s)
    pSTAT5B = (-(100.0 * ApB - 200.0 * BpB * (s - 1.0))
               / ((B * (s - 1.0) - ApB) + 2.0 * BpB * (s - 1.0)))
    rSTAT5A = ((100.0 * ApB + 100.0 * A * s + 200.0 * ApA * s)
               / (2.0 * ApB + A * s + 2.0 * ApA * s
                  - B * (s - 1.0) - 2.0 * BpB * (s - 1.0)))

    Nts = solution_time.shape[0]
    writeout_array = jnp.zeros([Nts, 7])
    writeout_array = writeout_array.at[:, 0].set(solution_time)
    writeout_array = writeout_array.at[:, 1].set(dataset[:, 0])   # measured pSTAT5A_rel
    writeout_array = writeout_array.at[:, 2].set(dataset[:, 1])   # measured pSTAT5B_rel
    writeout_array = writeout_array.at[:, 3].set(dataset[:, 2])   # measured rSTAT5A_rel
    writeout_array = writeout_array.at[:, 4].set(pSTAT5A)         # simulated pSTAT5A_rel
    writeout_array = writeout_array.at[:, 5].set(pSTAT5B)         # simulated pSTAT5B_rel
    writeout_array = writeout_array.at[:, 6].set(rSTAT5A)         # simulated rSTAT5A_rel
    # ---------------------------------------------------

    return writeout_array
