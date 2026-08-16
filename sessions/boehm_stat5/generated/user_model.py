import numpy as np

# Ordering of parameters in trainable_parameters as provided by the user:
# ['k_deg', 'k_exp_hetero', 'k_exp_homo', 'k_imp_hetero', 'k_imp_homo', 'k_phos']
# Ordering of integrated variables as provided by the user:
# ['A', 'B', 'ApB', 'ApA', 'BpB', 'nApA', 'nApB', 'nBpB']
#
# STAT5A/STAT5B dimerisation model of Boehm et al. (2014). Monomers A and B are
# phosphorylated and dimerise in the cytoplasm, are imported into the nucleus,
# and are exported back as monomers. Epo stimulation decays exponentially.


def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):

        k_deg        = trainable_parameters['k_deg']
        k_exp_hetero = trainable_parameters['k_exp_hetero']
        k_exp_homo   = trainable_parameters['k_exp_homo']
        k_imp_hetero = trainable_parameters['k_imp_hetero']
        k_imp_homo   = trainable_parameters['k_imp_homo']
        k_phos       = trainable_parameters['k_phos']

        Epo0 = fixed_parameters['Epo0']
        cyt  = fixed_parameters['cyt']
        nuc  = fixed_parameters['nuc']

        A    = y[0]
        B    = y[1]
        ApB  = y[2]
        ApA  = y[3]
        BpB  = y[4]
        nApA = y[5]
        nApB = y[6]
        nBpB = y[7]

        # Epo stimulation decays exponentially from its initial level
        Epo = Epo0 * np.exp(-k_deg * t)

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

        return np.array([dAdt, dBdt, dApBdt, dApAdt, dBpBdt,
                         dnApAdt, dnApBdt, dnBpBdt])


def _observables(solution, trainable_parameters, fixed_parameters):
        """The measured quantities, keyed by the names in model.observables."""
        s = fixed_parameters['specC17']

        A   = solution[:, 0]
        B   = solution[:, 1]
        ApB = solution[:, 2]
        ApA = solution[:, 3]
        BpB = solution[:, 4]

        pSTAT5A = (100.0 * ApB + 200.0 * ApA * s) / (ApB + A * s + 2.0 * ApA * s)
        pSTAT5B = (-(100.0 * ApB - 200.0 * BpB * (s - 1.0))
                   / ((B * (s - 1.0) - ApB) + 2.0 * BpB * (s - 1.0)))
        rSTAT5A = ((100.0 * ApB + 100.0 * A * s + 200.0 * ApA * s)
                   / (2.0 * ApB + A * s + 2.0 * ApA * s
                      - B * (s - 1.0) - 2.0 * BpB * (s - 1.0)))

        return {"pSTAT5A": pSTAT5A, "pSTAT5B": pSTAT5B, "rSTAT5A": rSTAT5A}


def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):

        observables = _observables(solution, trainable_parameters, fixed_parameters)
        pSTAT5A = observables['pSTAT5A']
        pSTAT5B = observables['pSTAT5B']
        rSTAT5A = observables['rSTAT5A']
        sim = np.stack([pSTAT5A, pSTAT5B, rSTAT5A], axis=1)

        # column-wise scale-normalised RMSE, so the three percentage channels
        # contribute comparably regardless of their individual ranges
        scale_factor = np.max(np.abs(dataset), axis=0)
        scale_factor = np.where(scale_factor == 0, 1.0, scale_factor)

        return np.sqrt(np.mean(np.square((sim - dataset) / scale_factor)))


def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):

        observables = _observables(solution, trainable_parameters, fixed_parameters)
        pSTAT5A = observables['pSTAT5A']
        pSTAT5B = observables['pSTAT5B']
        rSTAT5A = observables['rSTAT5A']

        Nts = solution_time.shape[0]
        writeout_array = np.zeros([Nts, 7])
        writeout_array[:, 0] = solution_time
        writeout_array[:, 1] = dataset[:, 0]   # measured pSTAT5A_rel
        writeout_array[:, 2] = dataset[:, 1]   # measured pSTAT5B_rel
        writeout_array[:, 3] = dataset[:, 2]   # measured rSTAT5A_rel
        writeout_array[:, 4] = pSTAT5A         # simulated pSTAT5A_rel
        writeout_array[:, 5] = pSTAT5B         # simulated pSTAT5B_rel
        writeout_array[:, 6] = rSTAT5A         # simulated rSTAT5A_rel
        return writeout_array
