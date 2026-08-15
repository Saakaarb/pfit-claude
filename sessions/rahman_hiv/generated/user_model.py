import numpy as np

# Ordering of parameters in trainable_parameters as provided by the user:
# ['rel_n', 'beta_m', 'rel_w', 'tau_w', 'w_n', 'w_m', 'g_m', 'g_w', 'bcr']
# Ordering of integrated variables as provided by the user:
# ['S', 'In', 'Im', 'Iw', 'Tn', 'Tm', 'Tw']
#
# Immunity-based HIV transmission model of Rahman, Vaidya & Zou (2016). The
# population is split into susceptibles and three disease stages (normal,
# moderate, weak) within each of an infected and a treated arm. Time is in
# years. Infected individuals worsen down the stages; treated individuals
# improve back up them.


def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):

        rel_n  = trainable_parameters['rel_n']
        beta_m = trainable_parameters['beta_m']
        rel_w  = trainable_parameters['rel_w']
        tau_w  = trainable_parameters['tau_w']
        w_n    = trainable_parameters['w_n']
        w_m    = trainable_parameters['w_m']
        g_m    = trainable_parameters['g_m']
        g_w    = trainable_parameters['g_w']
        bcr    = trainable_parameters['bcr']

        Lambda = fixed_parameters['Lambda']
        f_t    = fixed_parameters['f_t']
        tau_n  = fixed_parameters['tau_n']
        tau_m  = fixed_parameters['tau_m']
        mu_S   = fixed_parameters['mu_S']
        mu_In  = fixed_parameters['mu_In']
        mu_Im  = fixed_parameters['mu_Im']
        mu_Iw  = fixed_parameters['mu_Iw']
        mu_Tn  = fixed_parameters['mu_Tn']
        mu_Tm  = fixed_parameters['mu_Tm']
        mu_Tw  = fixed_parameters['mu_Tw']

        S  = y[0]
        In = y[1]
        Im = y[2]
        Iw = y[3]
        Tn = y[4]
        Tm = y[5]
        Tw = y[6]

        # the stage transmission rates are all tied to the single reference rate
        beta_n = rel_n * beta_m
        beta_w = rel_w * beta_m
        beta_t = f_t * beta_m

        N = S + In + Im + Iw + Tn + Tm + Tw
        infected_total = In + Im + Iw + Tn + Tm + Tw

        # force of infection, damped by a behavioural-change response that
        # weakens transmission as the infected population grows
        lam = ((beta_n * In + beta_m * Im + beta_w * Iw
                + beta_t * (Tn + Tm + Tw)) / N) * np.exp(-bcr * infected_total)

        dSdt  = Lambda - lam * S - mu_S * S

        dIndt = lam * S - w_n * In - tau_n * In - mu_In * In
        dImdt = w_n * In - w_m * Im - tau_m * Im - mu_Im * Im
        dIwdt = w_m * Im - tau_w * Iw - mu_Iw * Iw

        # the treated arm mirrors the infected arm with the arrows reversed:
        # individuals enter at tau and then improve back up the stages at g
        dTndt = tau_n * In + g_m * Tm - mu_Tn * Tn
        dTmdt = tau_m * Im + g_w * Tw - g_m * Tm - mu_Tm * Tm
        dTwdt = tau_w * Iw - g_w * Tw - mu_Tw * Tw

        return np.array([dSdt, dIndt, dImdt, dIwdt, dTndt, dTmdt, dTwdt])


def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):

        S = solution[:, 0]
        N = (solution[:, 0] + solution[:, 1] + solution[:, 2] + solution[:, 3]
             + solution[:, 4] + solution[:, 5] + solution[:, 6])

        # the lone observable: the percentage of the population not susceptible
        prevalence_sim = (1.0 - S / N) * 100.0
        prevalence_exp = dataset[:, 0]

        # peak-normalised RMSE, which keeps the loss on a [0, 1] scale
        scale_factor = np.max(np.abs(prevalence_exp))
        scale_factor = np.where(scale_factor == 0, 1.0, scale_factor)

        return np.sqrt(np.mean(np.square(
            (prevalence_sim - prevalence_exp) / scale_factor)))


def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):

        S = solution[:, 0]
        N = (solution[:, 0] + solution[:, 1] + solution[:, 2] + solution[:, 3]
             + solution[:, 4] + solution[:, 5] + solution[:, 6])

        prevalence_sim = (1.0 - S / N) * 100.0
        prevalence_exp = dataset[:, 0]

        Nts = solution_time.shape[0]
        writeout_array = np.zeros([Nts, 3])
        writeout_array[:, 0] = solution_time
        writeout_array[:, 1] = prevalence_exp   # measured prevalence (%)
        writeout_array[:, 2] = prevalence_sim   # simulated prevalence (%)
        return writeout_array
