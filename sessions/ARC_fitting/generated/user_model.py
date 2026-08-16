import numpy as np

# Ordering of parameters in trainable_parameters as provided by the user:
# ['Ea1', 'A1', 'n1', 'h1', 'Ea2', 'A2', 'm2', 'h2']
# Ordering of integrated variables as provided by the user:
# ['c1', 'c2', 'T']
#
# Two-reaction Arrhenius model of lithium-ion battery thermal runaway, fitted to
# accelerating-rate calorimetry. Reaction 1 consumes reactant c1; reaction 2
# produces c2 and only ignites above T_ignite, which is what makes the RHS
# non-smooth.


def _heat_rate(c1, c2, T, Ea1, A1, n1, h1, Ea2, A2, m2, h2, T_ignite, kb):
        """Both reaction rates and the resulting dT/dt, shared by the RHS and
        the loss (which must reconstruct dT/dt from the saved states)."""

        # Arrhenius kinetics; c1 is consumed, c2 is produced towards 1
        dc1dt = -A1 * np.exp(-Ea1 / (kb * T)) * c1**n1
        dc2dt = A2 * np.exp(-Ea2 / (kb * T)) * (1.0 - c2)**m2

        # the second exotherm contributes only once the cell is hot enough
        second = np.where(T > T_ignite, np.abs(h2 * dc2dt), 0.0)
        dTdt = np.abs(h1 * dc1dt) + second

        return dc1dt, dc2dt, dTdt


def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):

        Ea1 = trainable_parameters['Ea1']
        A1  = trainable_parameters['A1']
        n1  = trainable_parameters['n1']
        h1  = trainable_parameters['h1']
        Ea2 = trainable_parameters['Ea2']
        A2  = trainable_parameters['A2']
        m2  = trainable_parameters['m2']
        h2  = trainable_parameters['h2']

        T_ignite = fixed_parameters['T_ignite']
        kb       = fixed_parameters['kb']

        c1 = y[0]
        c2 = y[1]
        T  = y[2]

        dc1dt, dc2dt, dTdt = _heat_rate(c1, c2, T, Ea1, A1, n1, h1,
                                        Ea2, A2, m2, h2, T_ignite, kb)

        return np.array([dc1dt, dc2dt, dTdt])


def _observables(solution, trainable_parameters, fixed_parameters):
        """The measured quantities, keyed by the names in model.observables.

        Temperature is a state and is measured directly (its column declares
        `observes: T`), so only the self-heating rate is named here. It is
        reconstructed from the saved states via the same _heat_rate the RHS uses,
        which is why that helper stays separate.
        """
        Ea1 = trainable_parameters['Ea1']
        A1  = trainable_parameters['A1']
        n1  = trainable_parameters['n1']
        h1  = trainable_parameters['h1']
        Ea2 = trainable_parameters['Ea2']
        A2  = trainable_parameters['A2']
        m2  = trainable_parameters['m2']
        h2  = trainable_parameters['h2']
        T_ignite = fixed_parameters['T_ignite']
        kb       = fixed_parameters['kb']

        c1 = solution[:, 0]
        c2 = solution[:, 1]
        T  = solution[:, 2]

        _, _, rate_sim = _heat_rate(c1, c2, T, Ea1, A1, n1, h1,
                                    Ea2, A2, m2, h2, T_ignite, kb)
        return {"heat_rate": rate_sim}


def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):

        Ea1 = trainable_parameters['Ea1']
        A1  = trainable_parameters['A1']
        n1  = trainable_parameters['n1']
        h1  = trainable_parameters['h1']
        Ea2 = trainable_parameters['Ea2']
        A2  = trainable_parameters['A2']
        m2  = trainable_parameters['m2']
        h2  = trainable_parameters['h2']

        T_ignite = fixed_parameters['T_ignite']
        kb       = fixed_parameters['kb']

        c1 = solution[:, 0]
        c2 = solution[:, 1]
        T  = solution[:, 2]

        T_exp    = dataset[:, 0]
        rate_exp = dataset[:, 1]

        # the model heating rate, reconstructed from the saved states
        rate_sim = _observables(
            solution, trainable_parameters, fixed_parameters)["heat_rate"]

        # L1: the heating rate spans six decades, so it is matched in log space.
        # A linear residual would let the runaway endpoint own the objective and
        # ignore the low-temperature kinetics that fix the activation energies.
        floor = 1e-12
        log_sim = np.log10(np.maximum(rate_sim, floor))
        log_exp = np.log10(np.maximum(rate_exp, floor))
        log_span = np.max(log_exp) - np.min(log_exp)
        loss_rate = np.mean(np.abs(log_sim - log_exp)) / log_span

        # L2: the temperature trajectory, normalised by its peak
        loss_temp = np.mean(np.abs(T_exp - T)) / np.max(np.abs(T_exp))

        # L3: one-sided physical priors the under-determined data cannot
        # enforce. Vanishes once the fit reaches a valid runaway, so it is
        # inactive at the optimum and only steers the search away from
        # non-igniting solutions.
        c1_end = c1[-1]
        c2_end = c2[-1]
        T_end  = T[-1]
        loss_prior = 100.0 * (
            np.maximum(0.0, c1_end - 0.1)
            + np.maximum(0.0, 0.9 - c2_end)
            + (1.0 / 600.0) * np.maximum(0.0, 600.0 - T_end)
        )

        # An endpoint term |T[-1] - T_exp[-1]| was trialled here to stop the
        # temperature overshooting, and was reverted. It did pin the endpoint
        # (1251 K -> 638 K) and the enthalpies fell ~4x as intended, but the
        # optimum moved to a slow ramp: the transition became ~3 decades too
        # slow in rate and ignited ~3000 s early, so the log-rate term degraded
        # from 0.074 to 0.215. Constraining the endpoint does not constrain the
        # SHARPNESS, and trading the runaway away is the worse failure for this
        # experiment. See the note in generated/user_input_check.txt.
        return loss_rate + loss_temp + loss_prior


def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):

        Ea1 = trainable_parameters['Ea1']
        A1  = trainable_parameters['A1']
        n1  = trainable_parameters['n1']
        h1  = trainable_parameters['h1']
        Ea2 = trainable_parameters['Ea2']
        A2  = trainable_parameters['A2']
        m2  = trainable_parameters['m2']
        h2  = trainable_parameters['h2']

        T_ignite = fixed_parameters['T_ignite']
        kb       = fixed_parameters['kb']

        c1 = solution[:, 0]
        c2 = solution[:, 1]
        T  = solution[:, 2]

        rate_sim = _observables(
            solution, trainable_parameters, fixed_parameters)["heat_rate"]

        Nts = solution_time.shape[0]
        writeout_array = np.zeros([Nts, 5])
        writeout_array[:, 0] = solution_time
        writeout_array[:, 1] = dataset[:, 0]   # measured temperature (K)
        writeout_array[:, 2] = dataset[:, 1]   # measured heat rate (K/s)
        writeout_array[:, 3] = T               # simulated temperature (K)
        writeout_array[:, 4] = rate_sim        # simulated heat rate (K/s)
        return writeout_array
