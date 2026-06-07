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

    # Trainable (order matches XML): k1,k2,k3,Km3b,k4,Km4,k5,Km5b,k6,k7,k8,
    #                                 k9,k10,k11,k12,k13,k14,k21f,k21r,k22f,k22r,k24
    p = unscale_value(trainable_variables, min_val, max_val, is_logscale)
    k1   = p[0]
    k2   = p[1]
    k3   = p[2]
    Km3b = p[3]
    k4   = p[4]
    Km4  = p[5]
    k5   = p[6]
    Km5b = p[7]
    k6   = p[8]
    k7   = p[9]
    k8   = p[10]
    k9   = p[11]
    k10  = p[12]
    k11  = p[13]
    k12  = p[14]
    k13  = p[15]
    k14  = p[16]
    k21f = p[17]
    k21r = p[18]
    k22f = p[19]
    k22r = p[20]
    k24  = p[21]

    Km3a = fixed_parameters["Km3a"]
    Km5a = fixed_parameters["Km5a"]
    k23  = fixed_parameters["k23"]
    k25  = fixed_parameters["k25"]
    k26  = fixed_parameters["k26"]
    k27  = fixed_parameters["k27"]
    k28  = fixed_parameters["k28"]
    k29  = fixed_parameters["k29"]
    FIH  = fixed_parameters["FIH"]
    FIHn = fixed_parameters["FIHn"]
    VHL  = fixed_parameters["VHL"]
    VHLn = fixed_parameters["VHLn"]

    # State variables
    HIFa         = y[0]
    HIFa_pOH     = y[1]
    HIFa_aOH     = y[2]
    HIFa_aOHpOH  = y[3]
    HIFan_pOH    = y[4]
    HIFan        = y[5]
    HIFd         = y[6]
    HIFd_HRE     = y[7]
    HIFan_aOH    = y[8]
    HIFan_aOHpOH = y[9]
    PHD          = y[10]
    PHDn         = y[11]
    HIFb         = y[12]
    HRE          = y[13]
    mRNA         = y[14]
    Protein      = y[15]
    Luc          = y[16]
    O2           = y[17]   # per-experiment oxygen level (dummy, dY/dt=0)
    PHD_f        = y[18]   # per-experiment PHD inhibition factor (dummy, dY/dt=0)
    FIH_f        = y[19]   # per-experiment FIH inhibition factor (dummy, dY/dt=0)

    # Reaction rates (Table S1)
    v1  = k1
    v2  = k2 * HIFa
    v3  = PHD_f * k3  * PHD  * O2/(Km3a+O2) * HIFa        / (Km3b+HIFa)
    v4  = k4  * VHL  * HIFa_pOH    / (Km4 + HIFa_pOH)
    v5  = FIH_f * k5  * FIH  * O2/(Km5a+O2) * HIFa        / (Km5b+HIFa)
    v6  = k6  * HIFa_aOH
    v7  = PHD_f * k7  * PHD  * O2/(Km3a+O2) * HIFa_aOH    / (Km3b+HIFa_aOH)
    v8  = k8  * VHL  * HIFa_aOHpOH / (Km4 + HIFa_aOHpOH)
    v9  = k9  * HIFa
    v10 = k10 * HIFan
    v11 = k11 * PHD
    v12 = k12 * PHDn
    v13 = k13 * HIFa_aOH
    v14 = k14 * HIFan_aOH
    # Nuclear reactions (k15=k3, k16=k4, k17=k5, k18=k6, k19=k7, k20=k8)
    v15 = PHD_f * k3  * PHDn * O2/(Km3a+O2) * HIFan       / (Km3b+HIFan)
    v16 = k4  * VHLn * HIFan_pOH   / (Km4 + HIFan_pOH)
    v17 = FIH_f * k5  * FIHn * O2/(Km5a+O2) * HIFan       / (Km5b+HIFan)
    v18 = k6  * HIFan_aOH
    v19 = PHD_f * k7  * PHDn * O2/(Km3a+O2) * HIFan_aOH   / (Km3b+HIFan_aOH)
    v20 = k8  * VHLn * HIFan_aOHpOH / (Km4 + HIFan_aOHpOH)
    v21 = k21f * HIFan * HIFb - k21r * HIFd
    v22 = k22f * HIFd  * HRE  - k22r * HIFd_HRE
    v23 = k23 * HIFd_HRE
    v24 = k24 * HIFd_HRE
    v25 = k25 * PHD
    v26 = k26 * mRNA
    v27 = k27 * mRNA
    v28 = k28 * Protein
    v29 = k29 * Protein

    # ODEs (Table S2, with corrected HIFan_aOH equation: includes v13-v14)
    dHIFa         = v1  - v2 - v3 - v5 + v6 - v9  + v10
    dHIFa_pOH     = v3  - v4
    dHIFa_aOH     = v5  - v6 - v7 - v13 + v14
    dHIFa_aOHpOH  = v7  - v8
    dHIFan_pOH    = v15 - v16
    dHIFan        = v9  - v10 - v17 + v18 - v15 - v21
    dHIFd         = v21 - v22
    dHIFd_HRE     = v22
    dHIFan_aOH    = v13 - v14 + v17 - v18 - v19
    dHIFan_aOHpOH = v19 - v20
    dPHD          = v24 - v25 - v11 + v12
    dPHDn         = v11 - v12
    dHIFb         = -v21
    dHRE          = -v22
    dmRNA         = v23 - v26
    dProtein      = v27 - v28
    dLuc          = v29
    dO2_exp       = 0.0
    dPHD_factor   = 0.0
    dFIH_factor   = 0.0

    return jnp.array([dHIFa, dHIFa_pOH, dHIFa_aOH, dHIFa_aOHpOH,
                      dHIFan_pOH, dHIFan, dHIFd, dHIFd_HRE,
                      dHIFan_aOH, dHIFan_aOHpOH,
                      dPHD, dPHDn, dHIFb, dHRE,
                      dmRNA, dProtein, dLuc,
                      dO2_exp, dPHD_factor, dFIH_factor])


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
        max_steps=50000,
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
    failed = jnp.logical_or(result == RESULTS.max_steps_reached, result == RESULTS.singular)
    # ---------------------------------------------------

    fixed_parameters = constants["fixed_parameters"]
    HIF0 = fixed_parameters["HIF0"]

    # dataset cols (after time): [HIF_stab, LUC, mask_stab, mask_luc]
    HIF_stab_data = dataset[:, 0]
    LUC_data      = dataset[:, 1]
    mask_stab     = dataset[:, 2]
    mask_luc      = dataset[:, 3]

    # Model observables
    HIFtot_sim    = solution[:, 0] + solution[:, 5] + solution[:, 2] + solution[:, 8]
    HIF_stab_sim  = HIFtot_sim / HIF0
    LUC_sim       = solution[:, 16]

    eps = 1e-12

    scale_stab = jnp.max(jnp.abs(HIF_stab_data)) + eps
    resid_stab = mask_stab * (HIF_stab_sim - HIF_stab_data) / scale_stab
    loss_stab  = jnp.sqrt(jnp.mean(jnp.square(resid_stab)))

    scale_luc = jnp.max(jnp.abs(LUC_data)) + eps
    resid_luc = mask_luc * (LUC_sim - LUC_data) / scale_luc
    loss_luc  = jnp.sqrt(jnp.mean(jnp.square(resid_luc)))

    loss_value = loss_stab + loss_luc

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

    fixed_parameters = constants["fixed_parameters"]
    HIF0 = fixed_parameters["HIF0"]

    HIF_stab_data = dataset[:, 0]
    LUC_data      = dataset[:, 1]

    HIFtot_sim   = solution[:, 0] + solution[:, 5] + solution[:, 2] + solution[:, 8]
    HIF_stab_sim = HIFtot_sim / HIF0
    LUC_sim      = solution[:, 16]

    Nts = solution_time.shape[0]
    writeout_array = jnp.zeros((Nts, 5))
    writeout_array = writeout_array.at[:, 0].set(solution_time)
    writeout_array = writeout_array.at[:, 1].set(HIF_stab_data)
    writeout_array = writeout_array.at[:, 2].set(HIF_stab_sim)
    writeout_array = writeout_array.at[:, 3].set(LUC_data)
    writeout_array = writeout_array.at[:, 4].set(LUC_sim)

    return writeout_array
