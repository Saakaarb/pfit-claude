import numpy as np

# Key:
# Nts: number of time steps in dataset
# Ny: number of state variables defined in the user_input.xml file
# N_col: number of columns in dataset provided (including the first columna as time)

# Trainable parameter ordering in the parameter vector:
#   index 0: k1     (basal HIFa synthesis rate, nM/s)
#   index 1: k3     (PHD catalytic rate constant, 1/s)
#   index 2: k5     (FIH catalytic rate constant, 1/s)
#   index 3: k22f   (HIFd + HRE association rate, 1/(nM s))
#   index 4: k24    (PHD production via HRE feedback, 1/s)
#   index 5: k29    (Luciferase export rate, 1/s)

def user_defined_system(t: float, y: np.ndarray, trainable_parameters: dict, fixed_parameters: dict, dataset: np.ndarray, t_eval: np.ndarray):

        # Arguments:
        # t: time (float)
        # y: state vector (np.ndarray of size [Ny])
        # trainable_parameters: dictionary of trainable parameters. keys identical to the names in the user_input.xml file
        # fixed_parameters: dictionary of fixed parameters. keys identical to the names in the user_input.xml file
        # dataset: dataset (np.ndarray of size [Nts,N_col-1]) (first dimension in dataset is time)
        # t_eval: evaluation times (np.ndarray of size [Nts])



        # this part is to be populated by the model
        #--------------------------------
        k1   = trainable_parameters['k1']
        k3   = trainable_parameters['k3']
        k5   = trainable_parameters['k5']
        k22f = trainable_parameters['k22f']
        k24  = trainable_parameters['k24']
        k29  = trainable_parameters['k29']

        O2_uM = fixed_parameters['O2_uM']
        k2    = fixed_parameters['k2']
        Km3a  = fixed_parameters['Km3a']
        Km3b  = fixed_parameters['Km3b']
        k4    = fixed_parameters['k4']
        Km4   = fixed_parameters['Km4']
        Km5a  = fixed_parameters['Km5a']
        Km5b  = fixed_parameters['Km5b']
        k6    = fixed_parameters['k6']
        k7    = fixed_parameters['k7']
        Km7a  = fixed_parameters['Km7a']
        Km7b  = fixed_parameters['Km7b']
        k8    = fixed_parameters['k8']
        Km8   = fixed_parameters['Km8']
        k9    = fixed_parameters['k9']
        k10   = fixed_parameters['k10']
        k11   = fixed_parameters['k11']
        k12   = fixed_parameters['k12']
        k13   = fixed_parameters['k13']
        k14   = fixed_parameters['k14']
        k21f  = fixed_parameters['k21f']
        k21r  = fixed_parameters['k21r']
        k22r  = fixed_parameters['k22r']
        k23   = fixed_parameters['k23']
        k25   = fixed_parameters['k25']
        k26   = fixed_parameters['k26']
        k27   = fixed_parameters['k27']
        k28   = fixed_parameters['k28']
        FIH   = fixed_parameters['FIH']
        FIHn  = fixed_parameters['FIHn']
        VHL   = fixed_parameters['VHL']
        VHLn  = fixed_parameters['VHLn']

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


        #--------------------------------

        # this part is to be populated by the user
        #--------------------------------
        # Reaction rates v1..v29 (Table S1).
        # Nuclear copies use the same kinetic constants as cytoplasmic ones
        # (Table S3 footer): k15=k3, Km15a=Km3a, Km15b=Km3b, k16=k4, Km16=Km4,
        # k17=k5, Km17a=Km5a, Km17b=Km5b, k18=k6, k19=k7, k20=k8. For Km19a,
        # Km19b, Km20 (not listed in Table S3) we tie to Km7a, Km7b, Km8.
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

        # ODE right-hand sides (Table S2). Two corrections vs. the printed table:
        #   d[HIFa]/dt: the printed -v1 must be +v1 (v1 is the synthesis rate).
        #   d[HIFan_aOH]/dt: the printed equation is missing +v13 - v14
        #   (the cytoplasmic partner has -v13 + v14, so the nuclear side
        #   needs the opposite signs to conserve mass).
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

        #--------------------------------
        derivatives = np.array([dHIFa_dt, dHIFa_pOH_dt, dHIFa_aOH_dt, dHIFa_aOHpOH_dt,
                                dHIFan_dt, dHIFan_pOH_dt, dHIFan_aOH_dt, dHIFan_aOHpOH_dt,
                                dHIFd_dt, dHIFd_HRE_dt,
                                dPHD_dt, dPHDn_dt,
                                dHIFb_dt, dHRE_dt,
                                dmRNA_dt, dProtein_dt, dLuciferase_dt]) #
        return derivatives # of shape [Ny]. Each derivative term must be user defined

def _compute_loss_problem(solution_time: np.ndarray, solution: np.ndarray, dataset: np.ndarray, trainable_parameters: dict, fixed_parameters: dict):

        # Arguments:
        # solution_time: time (np.ndarray of size [Nts])
        # solution: state vector (np.ndarray of size [Nts,Ny])
        # dataset: dataset (np.ndarray of size [Nts,N_col-1])
        # trainable_parameters: dictionary of trainable parameters. keys identical to the names in the user_input.xml file
        # fixed_parameters: dictionary of fixed parameters. keys identical to the names in the user_input.xml file

        loss=0.0

        # this part is to be populated by the model
        #--------------------------------
        k1   = trainable_parameters['k1']
        k3   = trainable_parameters['k3']
        k5   = trainable_parameters['k5']
        k22f = trainable_parameters['k22f']
        k24  = trainable_parameters['k24']
        k29  = trainable_parameters['k29']

        O2_uM = fixed_parameters['O2_uM']
        k2    = fixed_parameters['k2']
        Km3a  = fixed_parameters['Km3a']
        Km3b  = fixed_parameters['Km3b']
        k4    = fixed_parameters['k4']
        Km4   = fixed_parameters['Km4']
        Km5a  = fixed_parameters['Km5a']
        Km5b  = fixed_parameters['Km5b']
        k6    = fixed_parameters['k6']
        k7    = fixed_parameters['k7']
        Km7a  = fixed_parameters['Km7a']
        Km7b  = fixed_parameters['Km7b']
        k8    = fixed_parameters['k8']
        Km8   = fixed_parameters['Km8']
        k9    = fixed_parameters['k9']
        k10   = fixed_parameters['k10']
        k11   = fixed_parameters['k11']
        k12   = fixed_parameters['k12']
        k13   = fixed_parameters['k13']
        k14   = fixed_parameters['k14']
        k21f  = fixed_parameters['k21f']
        k21r  = fixed_parameters['k21r']
        k22r  = fixed_parameters['k22r']
        k23   = fixed_parameters['k23']
        k25   = fixed_parameters['k25']
        k26   = fixed_parameters['k26']
        k27   = fixed_parameters['k27']
        k28   = fixed_parameters['k28']
        FIH   = fixed_parameters['FIH']
        FIHn  = fixed_parameters['FIHn']
        VHL   = fixed_parameters['VHL']
        VHLn  = fixed_parameters['VHLn']
        #--------------------------------


        # this part is to be populated by the user
        #--------------------------------
        # Build model observables to match the dataset columns.
        # dataset[:, 0] = HIF-1a stabilisation (fold change vs t=0)   (Fig 2B)
        # dataset[:, 1] = cumulative luciferase (max-normalised to 1) (Fig 2C)
        #
        # Total HIF-a is the sum of all HIF-a-containing species
        # (state indices 0..9 inclusive: HIFa, HIFa_pOH, HIFa_aOH,
        #  HIFa_aOHpOH, HIFan, HIFan_pOH, HIFan_aOH, HIFan_aOHpOH,
        #  HIFd, HIFd_HRE).
        total_hif = np.sum(solution[:, 0:10], axis=1)
        total_hif_t0 = np.maximum(total_hif[0], 1e-12)
        model_hif_stab = total_hif / total_hif_t0

        luc = solution[:, 16]
        luc_max = np.maximum(np.max(luc), 1e-12)
        model_luc = luc / luc_max

        model_obs = np.column_stack([model_hif_stab, model_luc])

        # Per-column scale-normalised RMSE. dataset HIF column max is ~3, the
        # luciferase column is already max-normalised to 1, so dividing by
        # np.max(dataset, axis=0) puts both residuals on a comparable scale.
        scale_factor = np.maximum(np.max(dataset, axis=0), 1e-12)
        loss = np.sqrt(np.mean(np.square(np.divide(model_obs - dataset, scale_factor))))
        #--------------------------------

        return loss # scalar


def writeout_description(solution_time: np.ndarray, solution: np.ndarray, dataset: np.ndarray, trainable_parameters: dict, fixed_parameters: dict):

        # Arguments:
        # solution_time: time (np.ndarray of size [Nts])
        # solution: state vector (np.ndarray of size [Nts,Ny])
        # dataset: dataset (np.ndarray of size [Nts,N_col-1])
        # trainable_parameters: dictionary of trainable parameters. keys identical to the names in the user_input.xml file
        # fixed_parameters: dictionary of fixed parameters. keys identical to the names in the user_input.xml file
        # Change the size of writeout_array as per your requirement

        # this part is to be populated by the model
        #--------------------------------
        k1   = trainable_parameters['k1']
        k3   = trainable_parameters['k3']
        k5   = trainable_parameters['k5']
        k22f = trainable_parameters['k22f']
        k24  = trainable_parameters['k24']
        k29  = trainable_parameters['k29']

        O2_uM = fixed_parameters['O2_uM']
        k2    = fixed_parameters['k2']
        Km3a  = fixed_parameters['Km3a']
        Km3b  = fixed_parameters['Km3b']
        k4    = fixed_parameters['k4']
        Km4   = fixed_parameters['Km4']
        Km5a  = fixed_parameters['Km5a']
        Km5b  = fixed_parameters['Km5b']
        k6    = fixed_parameters['k6']
        k7    = fixed_parameters['k7']
        Km7a  = fixed_parameters['Km7a']
        Km7b  = fixed_parameters['Km7b']
        k8    = fixed_parameters['k8']
        Km8   = fixed_parameters['Km8']
        k9    = fixed_parameters['k9']
        k10   = fixed_parameters['k10']
        k11   = fixed_parameters['k11']
        k12   = fixed_parameters['k12']
        k13   = fixed_parameters['k13']
        k14   = fixed_parameters['k14']
        k21f  = fixed_parameters['k21f']
        k21r  = fixed_parameters['k21r']
        k22r  = fixed_parameters['k22r']
        k23   = fixed_parameters['k23']
        k25   = fixed_parameters['k25']
        k26   = fixed_parameters['k26']
        k27   = fixed_parameters['k27']
        k28   = fixed_parameters['k28']
        FIH   = fixed_parameters['FIH']
        FIHn  = fixed_parameters['FIHn']
        VHL   = fixed_parameters['VHL']
        VHLn  = fixed_parameters['VHLn']
        #--------------------------------

        writeout_array = np.zeros([solution_time.shape[0],5]) # DEFINE THIS as required!

        # this part is to be populated by the user
        #--------------------------------
        # Recompute the model observables (same definitions as in
        # _compute_loss_problem) for side-by-side comparison with the data.
        total_hif = np.sum(solution[:, 0:10], axis=1)
        total_hif_t0 = np.maximum(total_hif[0], 1e-12)
        model_hif_stab = total_hif / total_hif_t0

        luc = solution[:, 16]
        luc_max = np.maximum(np.max(luc), 1e-12)
        model_luc = luc / luc_max

        writeout_array[:, 0] = solution_time
        writeout_array[:, 1] = dataset[:, 0]       # data HIF-1a stabilisation
        writeout_array[:, 2] = model_hif_stab      # model HIF-1a stabilisation
        writeout_array[:, 3] = dataset[:, 1]       # data luciferase (max-normalised)
        writeout_array[:, 4] = model_luc           # model luciferase (max-normalised)
        #--------------------------------

        return writeout_array # of custom shape
