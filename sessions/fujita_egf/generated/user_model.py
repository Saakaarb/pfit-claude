import numpy as np

# Ordering of parameters in trainable_parameters as provided by the user:
# ['reaction_1_k1', 'reaction_1_k2', 'reaction_2_k1', 'reaction_2_k2',
#  'reaction_3_k1', 'reaction_4_k1', 'reaction_5_k1', 'reaction_5_k2',
#  'reaction_6_k1', 'reaction_7_k1', 'reaction_8_k1', 'reaction_9_k1',
#  'EGFR_turnover', 'scaling_pEGFR_tot', 'scaling_pAkt_tot', 'scaling_pS6_tot']
# Ordering of integrated variables as provided by the user:
# ['EGFR', 'pEGFR', 'pEGFR_Akt', 'Akt', 'pAkt', 'S6', 'pAkt_S6', 'pS6',
#  'EGF_EGFR', 'EGF_0', 'EGF_rate', 'EGF_end']
#
# EGF/EGFR -> Akt -> S6 signalling (Fujita et al. 2010). EGF binds the
# receptor, the complex autophosphorylates, phospho-EGFR recruits and
# phosphorylates Akt, phospho-Akt in turn recruits and phosphorylates S6, and
# every phosphorylated form relaxes back. The receptor pool itself turns over.
#
# The last three integrated variables are NOT species. They carry this
# experiment's EGF stimulus descriptors, which the framework can only deliver
# through the per-experiment initial conditions, and they have zero derivative.
#
# dataset and t_eval represent ONE experiment's data; the framework calls this
# function once per experiment.


def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):

        k1_f  = trainable_parameters['reaction_1_k1']
        k1_r  = trainable_parameters['reaction_1_k2']
        k2_f  = trainable_parameters['reaction_2_k1']
        k2_r  = trainable_parameters['reaction_2_k2']
        k3    = trainable_parameters['reaction_3_k1']
        k4    = trainable_parameters['reaction_4_k1']
        k5_f  = trainable_parameters['reaction_5_k1']
        k5_r  = trainable_parameters['reaction_5_k2']
        k6    = trainable_parameters['reaction_6_k1']
        k7    = trainable_parameters['reaction_7_k1']
        k8    = trainable_parameters['reaction_8_k1']
        k9    = trainable_parameters['reaction_9_k1']
        turn  = trainable_parameters['EGFR_turnover']

        EGFR_setpoint = fixed_parameters['EGFR_setpoint']

        EGFR      = y[0]
        pEGFR     = y[1]
        pEGFR_Akt = y[2]
        Akt       = y[3]
        pAkt      = y[4]
        S6        = y[5]
        pAkt_S6   = y[6]
        pS6       = y[7]
        EGF_EGFR  = y[8]
        EGF_0     = y[9]
        EGF_rate  = y[10]
        EGF_end   = y[11]

        # Stimulus: a constant dose, a ramp, or either of those truncated at
        # EGF_end. The pulse conditions switch here, at t = 60 s.
        EGF = np.where(t <= EGF_end, EGF_0 + EGF_rate * t, 0.0)

        v1  = k1_f * EGF * EGFR - k1_r * EGF_EGFR   # ligand binding
        v2  = k2_f * Akt * pEGFR - k2_r * pEGFR_Akt  # Akt recruitment
        v3  = k3 * pEGFR_Akt                         # Akt phosphorylation
        v4  = k4 * pEGFR                             # pEGFR loss
        v5  = k5_f * S6 * pAkt - k5_r * pAkt_S6      # S6 recruitment
        v6  = k6 * pAkt_S6                           # S6 phosphorylation
        v7  = k7 * pAkt                              # pAkt dephosphorylation
        v8  = k8 * pS6                               # pS6 dephosphorylation
        v9  = k9 * EGF_EGFR                          # receptor autophos.
        v10 = turn * EGFR                            # receptor degradation
        v11 = turn * EGFR_setpoint                   # receptor synthesis

        dEGFR_dt      = -v1 - v10 + v11
        dpEGFR_dt     = v9 - v2 + v3 - v4
        dpEGFR_Akt_dt = v2 - v3
        dAkt_dt       = -v2 + v7
        dpAkt_dt      = v3 - v5 + v6 - v7
        dS6_dt        = -v5 + v8
        dpAkt_S6_dt   = v5 - v6
        dpS6_dt       = v6 - v8
        dEGF_EGFR_dt  = v1 - v9

        # The three stimulus carriers are constant over the solve.
        return np.array([dEGFR_dt, dpEGFR_dt, dpEGFR_Akt_dt, dAkt_dt,
                         dpAkt_dt, dS6_dt, dpAkt_S6_dt, dpS6_dt,
                         dEGF_EGFR_dt, 0.0, 0.0, 0.0])


def _observables(solution, trainable_parameters, fixed_parameters):
        """The measured quantities, keyed by the names in model.observables."""
        s_pEGFR = trainable_parameters['scaling_pEGFR_tot']
        s_pAkt  = trainable_parameters['scaling_pAkt_tot']
        s_pS6   = trainable_parameters['scaling_pS6_tot']

        pEGFR     = solution[:, 1]
        pEGFR_Akt = solution[:, 2]
        pAkt      = solution[:, 4]
        pAkt_S6   = solution[:, 6]
        pS6       = solution[:, 7]

        # The blots do not distinguish free from complexed phospho-species.
        obs_pEGFR = s_pEGFR * (pEGFR + pEGFR_Akt)
        obs_pAkt  = s_pAkt * (pAkt + pAkt_S6)
        obs_pS6   = s_pS6 * pS6

        return {"obs_pEGFR": obs_pEGFR, "obs_pAkt": obs_pAkt, "obs_pS6": obs_pS6}


def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):

        # dataset and t_eval represent ONE experiment's data; the framework
        # calls this function once per experiment.
        #
        # dataset columns: 0-2 the three measured signals, 3-5 their per-point
        # standard deviations, in the same order. Time is not in dataset.
        observables = _observables(solution, trainable_parameters, fixed_parameters)
        obs_pEGFR = observables['obs_pEGFR']
        obs_pAkt = observables['obs_pAkt']
        obs_pS6 = observables['obs_pS6']

        model_obs = np.stack([obs_pEGFR, obs_pAkt, obs_pS6], axis=1)
        measured = dataset[:, 0:3]
        sigma = dataset[:, 3:6]

        # Residuals in units of the reported measurement noise, which varies by
        # a factor of ~30 across the record. A fit at the noise level scores ~1.
        resid = (model_obs - measured) / sigma
        loss = np.sqrt(np.mean(np.square(resid)))

        return loss


def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):

        # dataset and t_eval represent ONE experiment's data; the framework
        # calls this function once per experiment.
        observables = _observables(solution, trainable_parameters, fixed_parameters)
        obs_pEGFR = observables['obs_pEGFR']
        obs_pAkt = observables['obs_pAkt']
        obs_pS6 = observables['obs_pS6']

        Nts = solution_time.shape[0]

        # time | 3 measured | 3 simulated observables | EGF stimulus
        writeout_array = np.zeros([Nts, 8])
        writeout_array[:, 0] = solution_time
        writeout_array[:, 1] = dataset[:, 0]   # measured pEGFR_tot
        writeout_array[:, 2] = dataset[:, 1]   # measured pAkt_tot
        writeout_array[:, 3] = dataset[:, 2]   # measured pS6_tot
        writeout_array[:, 4] = obs_pEGFR
        writeout_array[:, 5] = obs_pAkt
        writeout_array[:, 6] = obs_pS6

        # Reconstruct the stimulus from the carrier states for plotting.
        EGF_0 = solution[:, 9]
        EGF_rate = solution[:, 10]
        EGF_end = solution[:, 11]
        writeout_array[:, 7] = np.where(solution_time <= EGF_end,
                                        EGF_0 + EGF_rate * solution_time, 0.0)

        return writeout_array
