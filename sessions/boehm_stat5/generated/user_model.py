import numpy as np

# Key:
# Nts: number of time steps in dataset
# Ny: number of state variables defined in the user_input.xml file
# N_col: number of columns in dataset provided (including the first column as time)

# Boehm et al. (2014) STAT5A/B dimerisation model (PEtab benchmark).
# 8 species, 9 reactions, two compartments (cyt = 1.4, nuc = 0.45).
# Time is in minutes; Epo input decays as BaF3_Epo(t) = Epo0 * exp(-Epo_degradation_BaF3 * t).
#
# Trainable parameter ordering in the parameter vector:
#   0: Epo_degradation_BaF3   (Epo input decay rate, 1/min)
#   1: k_exp_hetero           (nuclear export of pApB,       1/min)
#   2: k_exp_homo             (nuclear export of pApA/pBpB,  1/min)
#   3: k_imp_hetero           (nuclear import of pApB,       1/min)
#   4: k_imp_homo             (nuclear import of pApA/pBpB,  1/min)
#   5: k_phos                 (phosphorylation/dimerisation rate)
#
# Integrated variable ordering (matches user_input.xml):
#   0: STAT5A  1: STAT5B  2: pApB  3: pApA  4: pBpB  5: nucpApA  6: nucpApB  7: nucpBpB


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
        Epo_degradation_BaF3 = trainable_parameters['Epo_degradation_BaF3']
        k_exp_hetero         = trainable_parameters['k_exp_hetero']
        k_exp_homo           = trainable_parameters['k_exp_homo']
        k_imp_hetero         = trainable_parameters['k_imp_hetero']
        k_imp_homo           = trainable_parameters['k_imp_homo']
        k_phos               = trainable_parameters['k_phos']

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
        #--------------------------------

        # this part is to be populated by the user
        #--------------------------------
        # Time-dependent Epo input (SBML assignment rule).
        BaF3_Epo = Epo0 * np.exp(-Epo_degradation_BaF3 * t)

        # Compartment ratios (species are concentrations; fluxes carry a compartment factor).
        cyt_over_nuc = cyt / nuc
        nuc_over_cyt = nuc / cyt

        # Phosphorylation / dimerisation in the cytoplasm
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
        #--------------------------------

        derivatives = np.array([dSTAT5A_dt, dSTAT5B_dt, dpApB_dt, dpApA_dt, dpBpB_dt,
                                dnucpApA_dt, dnucpApB_dt, dnucpBpB_dt])
        return derivatives  # of shape [Ny]. Each derivative term must be user defined


def _compute_loss_problem(solution_time: np.ndarray, solution: np.ndarray, dataset: np.ndarray, trainable_parameters: dict, fixed_parameters: dict):

        # Arguments:
        # solution_time: time (np.ndarray of size [Nts])
        # solution: state vector (np.ndarray of size [Nts,Ny])
        # dataset: dataset (np.ndarray of size [Nts,N_col-1])
        # trainable_parameters: dictionary of trainable parameters. keys identical to the names in the user_input.xml file
        # fixed_parameters: dictionary of fixed parameters. keys identical to the names in the user_input.xml file

        loss = 0.0

        # this part is to be populated by the model
        #--------------------------------
        specC17 = fixed_parameters['specC17']
        #--------------------------------

        # this part is to be populated by the user
        #--------------------------------
        # Cytoplasmic species needed for the observables.
        STAT5A = solution[:, 0]
        STAT5B = solution[:, 1]
        pApB   = solution[:, 2]
        pApA   = solution[:, 3]
        pBpB   = solution[:, 4]

        # Relative-phosphorylation observables (PEtab observable formulas).
        pSTAT5A_rel = (100.0 * pApB + 200.0 * pApA * specC17) / \
                      (pApB + STAT5A * specC17 + 2.0 * pApA * specC17)
        pSTAT5B_rel = -(100.0 * pApB - 200.0 * pBpB * (specC17 - 1.0)) / \
                      ((STAT5B * (specC17 - 1.0) - pApB) + 2.0 * pBpB * (specC17 - 1.0))
        rSTAT5A_rel = (100.0 * pApB + 100.0 * STAT5A * specC17 + 200.0 * pApA * specC17) / \
                      (2.0 * pApB + STAT5A * specC17 + 2.0 * pApA * specC17
                       - STAT5B * (specC17 - 1.0) - 2.0 * pBpB * (specC17 - 1.0))

        model_obs = np.column_stack([pSTAT5A_rel, pSTAT5B_rel, rSTAT5A_rel])

        # dataset columns: 0 = pSTAT5A_rel, 1 = pSTAT5B_rel, 2 = rSTAT5A_rel.
        # Per-column scale-normalised RMSE so the three percentage channels
        # (peaks ~95, ~82, ~50) contribute on a comparable footing.
        scale_factor = np.maximum(np.max(dataset, axis=0), 1e-12)
        loss = np.sqrt(np.mean(np.square(np.divide(model_obs - dataset, scale_factor))))
        #--------------------------------

        return loss  # scalar


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
        specC17 = fixed_parameters['specC17']
        #--------------------------------

        writeout_array = np.zeros([solution_time.shape[0], 7])

        # this part is to be populated by the user
        #--------------------------------
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

        writeout_array[:, 0] = solution_time
        writeout_array[:, 1] = dataset[:, 0]    # data pSTAT5A_rel
        writeout_array[:, 2] = pSTAT5A_rel       # model pSTAT5A_rel
        writeout_array[:, 3] = dataset[:, 1]    # data pSTAT5B_rel
        writeout_array[:, 4] = pSTAT5B_rel       # model pSTAT5B_rel
        writeout_array[:, 5] = dataset[:, 2]    # data rSTAT5A_rel
        writeout_array[:, 6] = rSTAT5A_rel       # model rSTAT5A_rel
        #--------------------------------

        return writeout_array  # of custom shape
