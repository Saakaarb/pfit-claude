import numpy as np


def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):
    # Trainable (order matches XML): k1,k2,k3,Km3b,k4,Km4,k5,Km5b,k6,k7,k8,
    #                                 k9,k10,k11,k12,k13,k14,k21f,k21r,k22f,k22r,k24
    k1   = trainable_parameters["k1"]
    k2   = trainable_parameters["k2"]
    k3   = trainable_parameters["k3"]
    Km3b = trainable_parameters["Km3b"]
    k4   = trainable_parameters["k4"]
    Km4  = trainable_parameters["Km4"]
    k5   = trainable_parameters["k5"]
    Km5b = trainable_parameters["Km5b"]
    k6   = trainable_parameters["k6"]
    k7   = trainable_parameters["k7"]
    k8   = trainable_parameters["k8"]
    k9   = trainable_parameters["k9"]
    k10  = trainable_parameters["k10"]
    k11  = trainable_parameters["k11"]
    k12  = trainable_parameters["k12"]
    k13  = trainable_parameters["k13"]
    k14  = trainable_parameters["k14"]
    k21f = trainable_parameters["k21f"]
    k21r = trainable_parameters["k21r"]
    k22f = trainable_parameters["k22f"]
    k22r = trainable_parameters["k22r"]
    k24  = trainable_parameters["k24"]

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

    # Linked constraints: nuclear enzymes share kinetics with cytoplasmic counterparts
    # k15=k3, k16=k4, k17=k5, k18=k6, k19=k7, k20=k8 (Table S3 note)

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

    return np.array([dHIFa, dHIFa_pOH, dHIFa_aOH, dHIFa_aOHpOH,
                     dHIFan_pOH, dHIFan, dHIFd, dHIFd_HRE,
                     dHIFan_aOH, dHIFan_aOHpOH,
                     dPHD, dPHDn, dHIFb, dHRE,
                     dmRNA, dProtein, dLuc,
                     dO2_exp, dPHD_factor, dFIH_factor])


def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    # dataset cols (after time): [HIF_stab, LUC, mask_stab, mask_luc]
    # col 0 = HIF_stab (normalised), col 1 = LUC (nM), col 2 = mask_stab, col 3 = mask_luc
    HIF0 = fixed_parameters["HIF0"]

    HIF_stab_data = dataset[:, 0]
    LUC_data      = dataset[:, 1]
    mask_stab     = dataset[:, 2]
    mask_luc      = dataset[:, 3]

    # Model observables
    # HIFtot = non-proline-hydroxylated forms only (matching data generation)
    HIFtot_sim    = solution[:, 0] + solution[:, 5] + solution[:, 2] + solution[:, 8]
    HIF_stab_sim  = HIFtot_sim / HIF0
    LUC_sim       = solution[:, 16]

    # Linear RMSE normalised by max(data) per observable
    eps = 1e-12

    scale_stab = np.max(np.abs(HIF_stab_data)) + eps
    resid_stab = mask_stab * (HIF_stab_sim - HIF_stab_data) / scale_stab
    loss_stab  = np.sqrt(np.mean(np.square(resid_stab)))

    scale_luc = np.max(np.abs(LUC_data)) + eps
    resid_luc = mask_luc * (LUC_sim - LUC_data) / scale_luc
    loss_luc  = np.sqrt(np.mean(np.square(resid_luc)))

    return loss_stab + loss_luc


def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    HIF0 = fixed_parameters["HIF0"]

    HIF_stab_data = dataset[:, 0]
    LUC_data      = dataset[:, 1]

    HIFtot_sim   = solution[:, 0] + solution[:, 5] + solution[:, 2] + solution[:, 8]
    HIF_stab_sim = HIFtot_sim / HIF0
    LUC_sim      = solution[:, 16]

    Nts = solution_time.shape[0]
    out = np.zeros((Nts, 5))
    out[:, 0] = solution_time
    out[:, 1] = HIF_stab_data
    out[:, 2] = HIF_stab_sim
    out[:, 3] = LUC_data
    out[:, 4] = LUC_sim
    return out
