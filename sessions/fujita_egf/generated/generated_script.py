# pfit-sources: user_model.py=f21c8c04466d4012 user_input.yaml=a9378ce20dc6225a
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


# EGF/EGFR -> Akt -> S6 signalling (Fujita et al. 2010). EGF binds the receptor,
# the complex autophosphorylates, phospho-EGFR recruits and phosphorylates Akt,
# phospho-Akt in turn recruits and phosphorylates S6, and every phosphorylated
# form relaxes back. The receptor pool itself turns over.
#
# The last three integrated variables are NOT species: they carry this
# experiment's EGF stimulus descriptors, which the framework can only deliver
# through the per-experiment initial conditions, and they have zero derivative.
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

    # Order: ['reaction_1_k1', 'reaction_1_k2', 'reaction_2_k1', 'reaction_2_k2',
    #         'reaction_3_k1', 'reaction_4_k1', 'reaction_5_k1', 'reaction_5_k2',
    #         'reaction_6_k1', 'reaction_7_k1', 'reaction_8_k1', 'reaction_9_k1',
    #         'EGFR_turnover', 'scaling_pEGFR_tot', 'scaling_pAkt_tot',
    #         'scaling_pS6_tot']
    (k1_f, k1_r, k2_f, k2_r, k3, k4, k5_f, k5_r, k6, k7, k8, k9, turn,
     scaling_pEGFR_tot, scaling_pAkt_tot, scaling_pS6_tot) = unscale_value(
        trainable_variables, min_val, max_val, is_logscale
    )

    EGFR_setpoint = fixed_parameters['EGFR_setpoint']

    EGFR = y[0]
    pEGFR = y[1]
    pEGFR_Akt = y[2]
    Akt = y[3]
    pAkt = y[4]
    S6 = y[5]
    pAkt_S6 = y[6]
    pS6 = y[7]
    EGF_EGFR = y[8]
    EGF_0 = y[9]
    EGF_rate = y[10]
    EGF_end = y[11]

    # this part is user entered
    # ---------------------------------------------------
    # Stimulus: a constant dose, a ramp, or either of those truncated at
    # EGF_end. The pulse conditions switch here, at t = 60 s.
    EGF = jnp.where(t <= EGF_end, EGF_0 + EGF_rate * t, 0.0)

    v1 = k1_f * EGF * EGFR - k1_r * EGF_EGFR      # ligand binding
    v2 = k2_f * Akt * pEGFR - k2_r * pEGFR_Akt    # Akt recruitment
    v3 = k3 * pEGFR_Akt                           # Akt phosphorylation
    v4 = k4 * pEGFR                               # pEGFR loss
    v5 = k5_f * S6 * pAkt - k5_r * pAkt_S6        # S6 recruitment
    v6 = k6 * pAkt_S6                             # S6 phosphorylation
    v7 = k7 * pAkt                                # pAkt dephosphorylation
    v8 = k8 * pS6                                 # pS6 dephosphorylation
    v9 = k9 * EGF_EGFR                            # receptor autophosphorylation
    v10 = turn * EGFR                             # receptor degradation
    v11 = turn * EGFR_setpoint                    # receptor synthesis

    dEGFR_dt = -v1 - v10 + v11
    dpEGFR_dt = v9 - v2 + v3 - v4
    dpEGFR_Akt_dt = v2 - v3
    dAkt_dt = -v2 + v7
    dpAkt_dt = v3 - v5 + v6 - v7
    dS6_dt = -v5 + v8
    dpAkt_S6_dt = v5 - v6
    dpS6_dt = v6 - v8
    dEGF_EGFR_dt = v1 - v9
    # ---------------------------------------------------

    # The three stimulus carriers are constant over the solve.
    return jnp.array([dEGFR_dt, dpEGFR_dt, dpEGFR_Akt_dt, dAkt_dt,
                      dpAkt_dt, dS6_dt, dpAkt_S6_dt, dpS6_dt,
                      dEGF_EGFR_dt, 0.0, 0.0, 0.0])


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
    # Any code other than RESULTS.successful means the trajectory is not
    # trustworthy (it may contain inf/NaN). See lib/LLM/api/diffrax.md.
    failed = jnp.invert(result == RESULTS.successful)
    # ---------------------------------------------------

    # this part is user entered
    # ---------------------------------------------------
    # dataset and t_eval represent ONE experiment's data; the framework calls
    # this function once per experiment.
    min_val = constants["min_limits"]
    max_val = constants["max_limits"]
    is_logscale = constants["is_logscale"]
    unscaled = unscale_value(trainable_variables, min_val, max_val, is_logscale)
    scaling_pEGFR_tot = unscaled[13]
    scaling_pAkt_tot = unscaled[14]
    scaling_pS6_tot = unscaled[15]

    pEGFR = solution[:, 1]
    pEGFR_Akt = solution[:, 2]
    pAkt = solution[:, 4]
    pAkt_S6 = solution[:, 6]
    pS6 = solution[:, 7]

    # The blots do not distinguish free from complexed phospho-species.
    obs_pEGFR = scaling_pEGFR_tot * (pEGFR + pEGFR_Akt)
    obs_pAkt = scaling_pAkt_tot * (pAkt + pAkt_S6)
    obs_pS6 = scaling_pS6_tot * pS6

    sim = jnp.stack([obs_pEGFR, obs_pAkt, obs_pS6], axis=1)

    # dataset columns: 0-2 the three measured signals, 3-5 their per-point
    # standard deviations, in the same order.
    measured = dataset[:, 0:3]
    sigma = dataset[:, 3:6]

    # Residuals in units of the reported measurement noise, which varies by a
    # factor of ~30 across the record. A fit at the noise level scores ~1.
    resid = (sim - measured) / sigma
    loss_value = jnp.sqrt(jnp.mean(jnp.square(resid)))
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
    min_val = constants["min_limits"]
    max_val = constants["max_limits"]
    is_logscale = constants["is_logscale"]
    unscaled = unscale_value(trainable_variables, min_val, max_val, is_logscale)
    scaling_pEGFR_tot = unscaled[13]
    scaling_pAkt_tot = unscaled[14]
    scaling_pS6_tot = unscaled[15]

    pEGFR = solution[:, 1]
    pEGFR_Akt = solution[:, 2]
    pAkt = solution[:, 4]
    pAkt_S6 = solution[:, 6]
    pS6 = solution[:, 7]

    obs_pEGFR = scaling_pEGFR_tot * (pEGFR + pEGFR_Akt)
    obs_pAkt = scaling_pAkt_tot * (pAkt + pAkt_S6)
    obs_pS6 = scaling_pS6_tot * pS6

    # Reconstruct the stimulus from the carrier states for plotting.
    EGF_0 = solution[:, 9]
    EGF_rate = solution[:, 10]
    EGF_end = solution[:, 11]
    EGF_applied = jnp.where(solution_time <= EGF_end,
                            EGF_0 + EGF_rate * solution_time, 0.0)

    Nts = solution_time.shape[0]

    # time | 3 measured | 3 simulated observables | EGF stimulus
    writeout_array = jnp.zeros([Nts, 8])
    writeout_array = writeout_array.at[:, 0].set(solution_time)
    writeout_array = writeout_array.at[:, 1].set(dataset[:, 0])   # measured pEGFR_tot
    writeout_array = writeout_array.at[:, 2].set(dataset[:, 1])   # measured pAkt_tot
    writeout_array = writeout_array.at[:, 3].set(dataset[:, 2])   # measured pS6_tot
    writeout_array = writeout_array.at[:, 4].set(obs_pEGFR)       # simulated pEGFR_tot
    writeout_array = writeout_array.at[:, 5].set(obs_pAkt)        # simulated pAkt_tot
    writeout_array = writeout_array.at[:, 6].set(obs_pS6)         # simulated pS6_tot
    writeout_array = writeout_array.at[:, 7].set(EGF_applied)     # applied EGF
    # ---------------------------------------------------

    return writeout_array
