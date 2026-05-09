# NF-kB Signaling — Lipniacki et al. (2004), J. Theor. Biol. 228:195
# BioModels BIOMD0000000140
#
# 24 ODEs, fitting 10 key parameters that control oscillation dynamics
# Observable: NFkB_nuc (nuclear NF-kB, index 14 in solution)
# IKK(0) = 0.2 represents TNF stimulus at t=0
#
# Published values of fitted params:
#   k1 = 5.4
#   k01 = 0.0048
#   tr1 = 0.2448
#   tr2 = 0.99
#   tr3 = 0.0168
#   deg1 = 0.00678
#   tp1 = 0.018
#   tp2 = 0.012
#   r4 = 1.224
#   k2 = 0.828

# Ordering of trainable_parameters:
# ['k1', 'k01', 'tr1', 'tr2', 'tr3', 'deg1', 'tp1', 'tp2', 'r4', 'k2']
# Ordering of integrated variables:
# ['IkBalpha', 'NFkB', 'IkBalpha_NFkB', 'IkBbeta', 'IkBbeta_NFkB', 'IkBeps', 'IkBeps_NFkB', 'IKK_IkBalpha', 'IKK_IkBalpha_NFkB', 'IKK', 'IKK_IkBbeta', 'IKK_IkBbeta_NFkB', 'IKK_IkBeps', 'IKK_IkBeps_NFkB', 'NFkB_nuc', 'IkBalpha_nuc', 'IkBalpha_nuc_NFkB_nuc', 'IkBbeta_nuc', 'IkBbeta_nuc_NFkB_nuc', 'IkBeps_nuc', 'IkBalpha_transcript', 'IkBbeta_transcript', 'IkBeps_transcript', 'IkBeps_nuc_NFkB_nuc']

def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):
    k1 = trainable_parameters['k1']
    k01 = trainable_parameters['k01']
    tr1 = trainable_parameters['tr1']
    tr2 = trainable_parameters['tr2']
    tr3 = trainable_parameters['tr3']
    deg1 = trainable_parameters['deg1']
    tp1 = trainable_parameters['tp1']
    tp2 = trainable_parameters['tp2']
    r4 = trainable_parameters['r4']
    k2 = trainable_parameters['k2']
    a1 = fixed_parameters['a1']
    a2 = fixed_parameters['a2']
    a3 = fixed_parameters['a3']
    a4 = fixed_parameters['a4']
    a5 = fixed_parameters['a5']
    a6 = fixed_parameters['a6']
    a7 = fixed_parameters['a7']
    a8 = fixed_parameters['a8']
    a9 = fixed_parameters['a9']
    d1 = fixed_parameters['d1']
    d2 = fixed_parameters['d2']
    d3 = fixed_parameters['d3']
    d4 = fixed_parameters['d4']
    d5 = fixed_parameters['d5']
    d6 = fixed_parameters['d6']
    deg4 = fixed_parameters['deg4']
    flag_for_after_trigger = fixed_parameters['flag_for_after_trigger']
    fr_after_trigger = fixed_parameters['fr_after_trigger']
    k02 = fixed_parameters['k02']
    k2_IkBbeta_nuc_NFkB_nuc = fixed_parameters['k2_IkBbeta_nuc_NFkB_nuc']
    k2_eps = fixed_parameters['k2_eps']
    r1 = fixed_parameters['r1']
    r2 = fixed_parameters['r2']
    r3 = fixed_parameters['r3']
    r5 = fixed_parameters['r5']
    r6 = fixed_parameters['r6']
    tr2a = fixed_parameters['tr2a']
    tr2b = fixed_parameters['tr2b']
    tr2e = fixed_parameters['tr2e']
    cytoplasm = 1.0
    nucleus   = 1.0

    IkBalpha = y[0]
    NFkB = y[1]
    IkBalpha_NFkB = y[2]
    IkBbeta = y[3]
    IkBbeta_NFkB = y[4]
    IkBeps = y[5]
    IkBeps_NFkB = y[6]
    IKK_IkBalpha = y[7]
    IKK_IkBalpha_NFkB = y[8]
    IKK = y[9]
    IKK_IkBbeta = y[10]
    IKK_IkBbeta_NFkB = y[11]
    IKK_IkBeps = y[12]
    IKK_IkBeps_NFkB = y[13]
    NFkB_nuc = y[14]
    IkBalpha_nuc = y[15]
    IkBalpha_nuc_NFkB_nuc = y[16]
    IkBbeta_nuc = y[17]
    IkBbeta_nuc_NFkB_nuc = y[18]
    IkBeps_nuc = y[19]
    IkBalpha_transcript = y[20]
    IkBbeta_transcript = y[21]
    IkBeps_transcript = y[22]
    IkBeps_nuc_NFkB_nuc = y[23]

    v1 = (cytoplasm * ((a4 * IkBalpha * NFkB) - (d4 * IkBalpha_NFkB)))
    v2 = (cytoplasm * ((a5 * IkBbeta * NFkB) - (d5 * IkBbeta_NFkB)))
    v3 = (cytoplasm * ((a6 * IkBeps * NFkB) - (d6 * IkBeps_NFkB)))
    v4 = (cytoplasm * ((a4 * IKK_IkBalpha * NFkB) - (d4 * IKK_IkBalpha_NFkB)))
    v5 = (cytoplasm * r4 * IKK_IkBalpha_NFkB)
    v6 = (cytoplasm * ((a5 * IKK_IkBbeta * NFkB) - (d5 * IKK_IkBbeta_NFkB)))
    v7 = (cytoplasm * r5 * IKK_IkBbeta_NFkB)
    v8 = (cytoplasm * ((a6 * IKK_IkBeps * NFkB) - (d6 * IKK_IkBeps_NFkB)))
    v9 = (cytoplasm * r6 * IKK_IkBeps_NFkB)
    v10 = (cytoplasm * deg4 * IkBalpha_NFkB)
    v11 = (cytoplasm * deg4 * IkBbeta_NFkB)
    v12 = (cytoplasm * deg4 * IkBeps_NFkB)
    v13 = ((cytoplasm * k1 * NFkB) - (nucleus * k01 * NFkB_nuc))
    v14 = (nucleus * ((a4 * IkBalpha_nuc * NFkB_nuc) - (d4 * IkBalpha_nuc_NFkB_nuc)))
    v15 = (nucleus * ((a5 * IkBbeta_nuc * NFkB_nuc) - (d5 * IkBbeta_nuc_NFkB_nuc)))
    v16 = (nucleus * ((a6 * IkBeps_nuc * NFkB_nuc) - (d6 * IkBeps_nuc_NFkB_nuc)))
    v17 = (nucleus * tr2a)
    v18 = (nucleus * tr2 * (NFkB_nuc ** 2))
    v19 = (nucleus * tr3 * IkBalpha_transcript)
    v20 = (nucleus * tr2b)
    v21 = (nucleus * tr3 * IkBbeta_transcript)
    v22 = (nucleus * tr2e)
    v23 = (nucleus * tr3 * IkBeps_transcript)
    v24 = (cytoplasm * ((a1 * IkBalpha * IKK) - (d1 * IKK_IkBalpha)))
    v25 = (nucleus * tr1 * IkBalpha_transcript)
    v26 = (cytoplasm * deg1 * IkBalpha)
    v27 = ((cytoplasm * tp1 * IkBalpha) - (nucleus * tp2 * IkBalpha_nuc))
    v28 = (cytoplasm * ((a2 * IkBbeta * IKK) - (d2 * IKK_IkBbeta)))
    v29 = (nucleus * tr1 * IkBbeta_transcript)
    v30 = (cytoplasm * deg1 * IkBbeta)
    v31 = ((cytoplasm * 0.5 * tp1 * IkBbeta) - (nucleus * 0.5 * tp2 * IkBbeta_nuc))
    v32 = (cytoplasm * ((a3 * IkBeps * IKK) - (d3 * IKK_IkBeps)))
    v33 = (nucleus * tr1 * IkBeps_transcript)
    v34 = (cytoplasm * deg1 * IkBeps)
    v35 = ((cytoplasm * 0.5 * tp1 * IkBeps) - (nucleus * 0.5 * tp2 * IkBeps_nuc))
    v36 = (cytoplasm * ((a7 * IKK * IkBalpha_NFkB) - (d1 * IKK_IkBalpha_NFkB)))
    v37 = (nucleus * k2 * IkBalpha_nuc_NFkB_nuc)
    v38 = (cytoplasm * ((a8 * IKK * IkBbeta_NFkB) - (d2 * IKK_IkBbeta_NFkB)))
    v39 = (nucleus * k2_IkBbeta_nuc_NFkB_nuc * (fr_after_trigger + flag_for_after_trigger) * IkBbeta_nuc_NFkB_nuc)
    v40 = (cytoplasm * ((a9 * IKK * IkBeps_NFkB) - (d3 * IKK_IkBeps_NFkB)))
    v41 = (nucleus * 0.5 * k2_eps * IkBeps_nuc_NFkB_nuc)
    v42 = (cytoplasm * r1 * IKK_IkBalpha)
    v43 = (cytoplasm * r2 * IKK_IkBbeta)
    v44 = (cytoplasm * r3 * IKK_IkBeps)
    v45 = (cytoplasm * k02 * IKK)

    dIkBalphadt = - v1 - v24 + v25 - v26 - v27
    dNFkBdt = - v1 - v2 - v3 - v4 + v5 - v6 + v7 - v8 + v9 + v10 + v11 + v12 - v13
    dIkBalpha_NFkBdt = + v1 - v10 - v36 + v37
    dIkBbetadt = - v2 - v28 + v29 - v30 - v31
    dIkBbeta_NFkBdt = + v2 - v11 - v38 + v39
    dIkBepsdt = - v3 - v32 + v33 - v34 - v35
    dIkBeps_NFkBdt = + v3 - v12 - v40 + v41
    dIKK_IkBalphadt = - v4 + v24 - v42
    dIKK_IkBalpha_NFkBdt = + v4 - v5 + v36
    dIKKdt = + v5 + v7 + v9 - v24 - v28 - v32 - v36 - v38 - v40 + v42 + v43 + v44 - v45
    dIKK_IkBbetadt = - v6 + v28 - v43
    dIKK_IkBbeta_NFkBdt = + v6 - v7 + v38
    dIKK_IkBepsdt = - v8 + v32 - v44
    dIKK_IkBeps_NFkBdt = + v8 - v9 + v40
    dNFkB_nucdt = + v13 - v14 - v15 - v16
    dIkBalpha_nucdt = - v14 + v27
    dIkBalpha_nuc_NFkB_nucdt = + v14 - v37
    dIkBbeta_nucdt = - v15 + v31
    dIkBbeta_nuc_NFkB_nucdt = + v15 - v39
    dIkBeps_nucdt = - v16 + v35
    dIkBalpha_transcriptdt = + v17 + v18 - v19
    dIkBbeta_transcriptdt = + v20 - v21
    dIkBeps_transcriptdt = + v22 - v23
    dIkBeps_nuc_NFkB_nucdt = + v16 - v41

    return np.array([dIkBalphadt, dNFkBdt, dIkBalpha_NFkBdt, dIkBbetadt, dIkBbeta_NFkBdt, dIkBepsdt, dIkBeps_NFkBdt, dIKK_IkBalphadt, dIKK_IkBalpha_NFkBdt, dIKKdt, dIKK_IkBbetadt, dIKK_IkBbeta_NFkBdt, dIKK_IkBepsdt, dIKK_IkBeps_NFkBdt, dNFkB_nucdt, dIkBalpha_nucdt, dIkBalpha_nuc_NFkB_nucdt, dIkBbeta_nucdt, dIkBbeta_nuc_NFkB_nucdt, dIkBeps_nucdt, dIkBalpha_transcriptdt, dIkBbeta_transcriptdt, dIkBeps_transcriptdt, dIkBeps_nuc_NFkB_nucdt])


def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    NFkB_nuc_pred = solution[:, 14]
    NFkB_nuc_obs  = dataset[:, 0]
    scale = np.max(np.abs(NFkB_nuc_obs))
    scale = np.where(scale == 0, 1.0, scale)
    return np.sqrt(np.mean(np.square((NFkB_nuc_pred - NFkB_nuc_obs) / scale)))


def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    Nts = solution_time.shape[0]
    out = np.zeros([Nts, 3])
    out[:, 0] = solution_time
    out[:, 1] = dataset[:, 0]
    out[:, 2] = solution[:, 14]
    return out