import jax
import jax.numpy as jnp
import diffrax
from diffrax import RESULTS

jax.config.update("jax_enable_x64", True)


# Workaround: helper_functions.py builds the `constants` dict with a mix of
# np.ndarray (strong-typed) and Python scalars (weak-typed). On diffrax 0.7+,
# the implicit-step closure converter rejects the weak/strong mismatch. Cast
# every leaf to a strong-typed jax array before it reaches diffeqsolve.
def _strongify_constants(constants):
    f = constants["fixed_parameters"]
    return {
        "dataset":          jnp.asarray(constants["dataset"],          dtype=jnp.float64),
        "t_eval":           jnp.asarray(constants["t_eval"],           dtype=jnp.float64),
        "init_cond":        jnp.asarray(constants["init_cond"],        dtype=jnp.float64),
        "init_time":        jnp.asarray(constants["init_time"],        dtype=jnp.float64),
        "final_time":       jnp.asarray(constants["final_time"],       dtype=jnp.float64),
        "init_timestep":    jnp.asarray(constants["init_timestep"],    dtype=jnp.float64),
        "stepsize_rtol":    jnp.asarray(constants["stepsize_rtol"],    dtype=jnp.float64),
        "stepsize_atol":    jnp.asarray(constants["stepsize_atol"],    dtype=jnp.float64),
        "max_steps":        jnp.asarray(constants["max_steps"],        dtype=jnp.int64),
        "num_steps":        jnp.asarray(constants["num_steps"],        dtype=jnp.int64),
        "error_loss":       jnp.asarray(constants["error_loss"],       dtype=jnp.float64),
        "min_limits":       jnp.asarray(constants["min_limits"],       dtype=jnp.float64),
        "max_limits":       jnp.asarray(constants["max_limits"],       dtype=jnp.float64),
        "is_logscale":      jnp.asarray(constants["is_logscale"],      dtype=jnp.int64),
        "fixed_parameters": {k: jnp.asarray(v, dtype=jnp.float64) for k, v in f.items()},
    }


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


# Trainable parameter ordering in the parameter vector:
#   index 0: k1     (basal HIFa synthesis rate, nM/s)
#   index 1: k3     (PHD catalytic rate constant, 1/s)
#   index 2: k5     (FIH catalytic rate constant, 1/s)
#   index 3: k22f   (HIFd + HRE association rate, 1/(nM s))
#   index 4: k24    (PHD production via HRE feedback, 1/s)
#   index 5: k29    (Luciferase export rate, 1/s)
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

    k1, k3, k5, k22f, k24, k29 = unscale_value(trainable_variables, min_val, max_val, is_logscale)

    O2_uM = fixed_parameters["O2_uM"]
    k2    = fixed_parameters["k2"]
    Km3a  = fixed_parameters["Km3a"]
    Km3b  = fixed_parameters["Km3b"]
    k4    = fixed_parameters["k4"]
    Km4   = fixed_parameters["Km4"]
    Km5a  = fixed_parameters["Km5a"]
    Km5b  = fixed_parameters["Km5b"]
    k6    = fixed_parameters["k6"]
    k7    = fixed_parameters["k7"]
    Km7a  = fixed_parameters["Km7a"]
    Km7b  = fixed_parameters["Km7b"]
    k8    = fixed_parameters["k8"]
    Km8   = fixed_parameters["Km8"]
    k9    = fixed_parameters["k9"]
    k10   = fixed_parameters["k10"]
    k11   = fixed_parameters["k11"]
    k12   = fixed_parameters["k12"]
    k13   = fixed_parameters["k13"]
    k14   = fixed_parameters["k14"]
    k21f  = fixed_parameters["k21f"]
    k21r  = fixed_parameters["k21r"]
    k22r  = fixed_parameters["k22r"]
    k23   = fixed_parameters["k23"]
    k25   = fixed_parameters["k25"]
    k26   = fixed_parameters["k26"]
    k27   = fixed_parameters["k27"]
    k28   = fixed_parameters["k28"]
    FIH   = fixed_parameters["FIH"]
    FIHn  = fixed_parameters["FIHn"]
    VHL   = fixed_parameters["VHL"]
    VHLn  = fixed_parameters["VHLn"]

    HIFa         = y[0]
    HIFa_pOH     = y[1]
    HIFa_aOH     = y[2]
    HIFa_aOHpOH  = y[3]
    HIFan        = y[4]
    HIFan_pOH    = y[5]
    HIFan_aOH    = y[6]
    HIFan_aOHpOH = y[7]
    HIFd         = y[8]
    HIFd_HRE     = y[9]
    PHD          = y[10]
    PHDn         = y[11]
    HIFb         = y[12]
    HRE          = y[13]
    mRNA         = y[14]
    Protein      = y[15]
    Luciferase   = y[16]

    # Reaction rates v1..v29 (Table S1).
    # Nuclear copies tied to cytoplasmic via k15=k3, Km15a=Km3a, Km15b=Km3b,
    # k16=k4, Km16=Km4, k17=k5, Km17a=Km5a, Km17b=Km5b, k18=k6, k19=k7,
    # Km19a=Km7a, Km19b=Km7b, k20=k8, Km20=Km8.
    v1  = k1
    v2  = k2 * HIFa
    v3  = k3 * PHD * (O2_uM / (Km3a + O2_uM)) * (HIFa / (Km3b + HIFa))
    v4  = k4 * VHL * (HIFa_pOH / (Km4 + HIFa_pOH))
    v5  = k5 * FIH * (O2_uM / (Km5a + O2_uM)) * (HIFa / (Km5b + HIFa))
    v6  = k6 * HIFa_aOH
    v7  = k7 * PHD * (O2_uM / (Km7a + O2_uM)) * (HIFa_aOH / (Km7b + HIFa_aOH))
    v8  = k8 * VHL * (HIFa_aOHpOH / (Km8 + HIFa_aOHpOH))
    v9  = k9 * HIFa
    v10 = k10 * HIFan
    v11 = k11 * PHD
    v12 = k12 * PHDn
    v13 = k13 * HIFa_aOH
    v14 = k14 * HIFan_aOH
    v15 = k3 * PHDn * (O2_uM / (Km3a + O2_uM)) * (HIFan / (Km3b + HIFan))
    v16 = k4 * VHLn * (HIFan_pOH / (Km4 + HIFan_pOH))
    v17 = k5 * FIHn * (O2_uM / (Km5a + O2_uM)) * (HIFan / (Km5b + HIFan))
    v18 = k6 * HIFan_aOH
    v19 = k7 * PHDn * (O2_uM / (Km7a + O2_uM)) * (HIFan_aOH / (Km7b + HIFan_aOH))
    v20 = k8 * VHLn * (HIFan_aOHpOH / (Km8 + HIFan_aOHpOH))
    v21 = k21f * HIFan * HIFb - k21r * HIFd
    v22 = k22f * HIFd * HRE - k22r * HIFd_HRE
    v23 = k23 * HIFd_HRE
    v24 = k24 * HIFd_HRE
    v25 = k25 * PHD
    v26 = k26 * mRNA
    v27 = k27 * mRNA
    v28 = k28 * Protein
    v29 = k29 * Protein

    # ODE right-hand sides (Table S2). Two corrections vs. printed table:
    # +v1 (synthesis) instead of -v1 in d[HIFa]/dt; +v13 - v14 added to
    # d[HIFan_aOH]/dt for the cyt<->nuc shuttle of HIFa-aOH.
    dHIFa_dt         = v1 - v2 - v9 + v10 - v3 - v5 + v6
    dHIFa_pOH_dt     = v3 - v4
    dHIFa_aOH_dt     = v5 - v6 - v7 - v13 + v14
    dHIFa_aOHpOH_dt  = v7 - v8
    dHIFan_dt        = v9 - v10 - v17 + v18 - v15 - v21
    dHIFan_pOH_dt    = v15 - v16
    dHIFan_aOH_dt    = v17 - v18 - v19 + v13 - v14
    dHIFan_aOHpOH_dt = v19 - v20
    dHIFd_dt         = v21 - v22
    dHIFd_HRE_dt     = v22
    dPHD_dt          = v24 - v25 - v11 + v12
    dPHDn_dt         = v11 - v12
    dHIFb_dt         = -v21
    dHRE_dt          = -v22
    dmRNA_dt         = v23 - v26
    dProtein_dt      = v27 - v28
    dLuciferase_dt   = v29

    return jnp.array([dHIFa_dt, dHIFa_pOH_dt, dHIFa_aOH_dt, dHIFa_aOHpOH_dt,
                      dHIFan_dt, dHIFan_pOH_dt, dHIFan_aOH_dt, dHIFan_aOHpOH_dt,
                      dHIFd_dt, dHIFd_HRE_dt,
                      dPHD_dt, dPHDn_dt,
                      dHIFb_dt, dHRE_dt,
                      dmRNA_dt, dProtein_dt, dLuciferase_dt])


# fixed
@jax.jit
def _integrate_system(constants, trainable_variables):
    term = diffrax.ODETerm(user_defined_system)

    solver = diffrax.Tsit5()
    t_eval = constants["t_eval"]
    init_cond = constants["init_cond"]
    init_time = constants["init_time"]
    dataset = constants["dataset"]
    saveat = diffrax.SaveAt(ts=t_eval)

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
    constants = _strongify_constants(constants)
    dataset = constants["dataset"]
    solution_time, solution, result = _integrate_system(constants, trainable_variables)
    # Any code other than RESULTS.successful means the trajectory is untrustworthy
    # (it may contain inf/NaN). See lib/LLM/api/diffrax.md for the full code table.
    failed = jnp.invert(result == RESULTS.successful)
    # ---------------------------------------------------

    # Build model observables matching dataset columns.
    # dataset[:, 0] = HIF-1a stabilisation (fold change vs t=0)   (Fig 2B)
    # dataset[:, 1] = cumulative luciferase (max-normalised to 1) (Fig 2C)
    # State indices 0..9 are the 10 HIF-a-containing species:
    # HIFa, HIFa_pOH, HIFa_aOH, HIFa_aOHpOH, HIFan, HIFan_pOH, HIFan_aOH,
    # HIFan_aOHpOH, HIFd, HIFd_HRE.
    total_hif = jnp.sum(solution[:, 0:10], axis=1)
    total_hif_t0 = jnp.maximum(total_hif[0], 1e-12)
    model_hif_stab = total_hif / total_hif_t0

    luc = solution[:, 16]
    luc_max = jnp.maximum(jnp.max(luc), 1e-12)
    model_luc = luc / luc_max

    model_obs = jnp.column_stack([model_hif_stab, model_luc])

    scale_factor = jnp.maximum(jnp.max(dataset, axis=0), 1e-12)
    loss_value = jnp.sqrt(jnp.mean(jnp.square(jnp.divide(model_obs - dataset, scale_factor))))

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
    constants = _strongify_constants(constants)
    dataset = constants["dataset"]

    solution_time, solution, result = _integrate_system(constants, trainable_variables)
    # ---------------------------------------------------

    # Recompute the model observables (same as in _compute_loss_problem).
    total_hif = jnp.sum(solution[:, 0:10], axis=1)
    total_hif_t0 = jnp.maximum(total_hif[0], 1e-12)
    model_hif_stab = total_hif / total_hif_t0

    luc = solution[:, 16]
    luc_max = jnp.maximum(jnp.max(luc), 1e-12)
    model_luc = luc / luc_max

    writeout_array = jnp.column_stack([solution_time,
                                       dataset[:, 0],
                                       model_hif_stab,
                                       dataset[:, 1],
                                       model_luc])

    return writeout_array
