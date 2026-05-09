# MAPK/ERK Cascade — Kholodenko (2000), Eur. J. Biochem. 267:1583
# BioModels BIOMD0000000026
#
# 11 ODEs, 16 mass-action kinetic parameters
# Observable: Mpp (doubly-phosphorylated MAPK, index 2 in solution)
#
# Reactions:
#   v1a = k1*M*MAPKK - k_1*M_MAPKK       (M+MAPKK <-> M_MAPKK)
#   v1b = k2*M_MAPKK                       (M_MAPKK -> Mp+MAPKK)
#   v2a = k3*Mp*MAPKK - k_3*Mp_MAPKK      (Mp+MAPKK <-> Mp_MAPKK)
#   v2b = k4*Mp_MAPKK                      (Mp_MAPKK -> Mpp+MAPKK)
#   v3a = h1*Mpp*MKP3 - h_1*Mpp_MKP3      (Mpp+MKP3 <-> Mpp_MKP3)
#   v3b = h2*Mpp_MKP3                      (Mpp_MKP3 -> Mp_MKP3_dep)
#   v3c = h3*Mp_MKP3_dep - h_3*Mp*MKP3    (Mp_MKP3_dep <-> Mp+MKP3)
#   v4a = h4*Mp*MKP3 - h_4*Mp_MKP3        (Mp+MKP3 <-> Mp_MKP3)
#   v4b = h5*Mp_MKP3                       (Mp_MKP3 -> M_MKP3)
#   v4c = h6*M_MKP3 - h_6*M*MKP3         (M_MKP3 <-> M+MKP3)
#
# Published values: k1=0.02,k_1=1.0,k2=0.01,k3=0.032,k_3=1.0,k4=15.0
#                   h1=0.045,h_1=1.0,h2=0.092,h3=1.0,h_3=0.01
#                   h4=0.01,h_4=1.0,h5=0.5,h6=0.086,h_6=0.0011

# Ordering of parameters in trainable_parameters:
# ['k1','k_1','k2','k3','k_3','k4','h1','h_1','h2','h3','h_3','h4','h_4','h5','h6','h_6']
# Ordering of integrated variables:
# ['M','Mp','Mpp','MAPKK','MKP3','M_MAPKK','Mp_MAPKK','Mpp_MKP3','Mp_MKP3_dep','Mp_MKP3','M_MKP3']

def user_defined_system(t, y, trainable_parameters, fixed_parameters, dataset, t_eval):
    k1  = trainable_parameters['k1']
    k_1 = trainable_parameters['k_1']
    k2  = trainable_parameters['k2']
    k3  = trainable_parameters['k3']
    k_3 = trainable_parameters['k_3']
    k4  = trainable_parameters['k4']
    h1  = trainable_parameters['h1']
    h_1 = trainable_parameters['h_1']
    h2  = trainable_parameters['h2']
    h3  = trainable_parameters['h3']
    h_3 = trainable_parameters['h_3']
    h4  = trainable_parameters['h4']
    h_4 = trainable_parameters['h_4']
    h5  = trainable_parameters['h5']
    h6  = trainable_parameters['h6']
    h_6 = trainable_parameters['h_6']

    M, Mp, Mpp, MAPKK, MKP3, M_MAPKK, Mp_MAPKK, Mpp_MKP3, Mp_MKP3_dep, Mp_MKP3, M_MKP3 = y

    v1a = k1*M*MAPKK    - k_1*M_MAPKK
    v1b = k2*M_MAPKK
    v2a = k3*Mp*MAPKK   - k_3*Mp_MAPKK
    v2b = k4*Mp_MAPKK
    v3a = h1*Mpp*MKP3   - h_1*Mpp_MKP3
    v3b = h2*Mpp_MKP3
    v3c = h3*Mp_MKP3_dep - h_3*Mp*MKP3
    v4a = h4*Mp*MKP3    - h_4*Mp_MKP3
    v4b = h5*Mp_MKP3
    v4c = h6*M_MKP3     - h_6*M*MKP3

    dM           = -v1a + v4c
    dMp          =  v1b - v2a + v3c - v4a
    dMpp         =  v2b - v3a
    dMAPKK       = -v1a + v1b - v2a + v2b
    dMKP3        = -v3a + v3c - v4a + v4c
    dM_MAPKK     =  v1a - v1b
    dMp_MAPKK    =  v2a - v2b
    dMpp_MKP3    =  v3a - v3b
    dMp_MKP3_dep =  v3b - v3c
    dMp_MKP3     =  v4a - v4b
    dM_MKP3      =  v4b - v4c

    return np.array([dM, dMp, dMpp, dMAPKK, dMKP3,
                     dM_MAPKK, dMp_MAPKK, dMpp_MKP3, dMp_MKP3_dep, dMp_MKP3, dM_MKP3])


def _compute_loss_problem(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    Mpp_pred = solution[:, 2]   # index 2 = Mpp
    Mpp_obs  = dataset[:, 0]    # only observable
    scale = np.max(np.abs(Mpp_obs))
    scale = np.where(scale == 0, 1.0, scale)
    return np.sqrt(np.mean(np.square((Mpp_pred - Mpp_obs) / scale)))


def writeout_description(solution_time, solution, dataset, trainable_parameters, fixed_parameters):
    Nts = solution_time.shape[0]
    out = np.zeros([Nts, 4])
    out[:, 0] = solution_time
    out[:, 1] = dataset[:, 0]   # Mpp_obs
    out[:, 2] = solution[:, 2]  # Mpp_pred
    out[:, 3] = solution[:, 1]  # Mp_pred (singly phosphorylated)
    return out
