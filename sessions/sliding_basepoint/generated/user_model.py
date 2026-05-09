import numpy as np

# Key: 
# Nts: number of time steps in dataset
# Ny: number of state variables defined in the user_input.xml file
# N_col: number of columns in dataset 

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
        c2 = trainable_parameters['c2']
        Dk = trainable_parameters['Dk']
        Dc = trainable_parameters['Dc']
        m1 = trainable_parameters['m1']
        m2 = trainable_parameters['m2']
        vf = fixed_parameters['vf']

        x1 = y[0]
        x2 = y[1]
        v1 = y[2]
        v2 = y[3]
        k = y[4]
        c1 = y[5]
        #--------------------------------

        # this part is to be populated by the user
        #--------------------------------
        def F1(Fs,c1,v1):
                return Fs-c1*np.abs(v1)*np.sign(v1)
        def F2(Fs,c2,v2):

                if np.abs(Fs) < c2 and np.abs(v2) < vf:
                        return 0
                else:
                        return Fs - c2*np.sign(v2)
        
        Fs= k*(x2-x1)
        dx1dt=v1
        dx2dt=v2

        dv1dt= (1/m1)*F1(Fs,c1,v1)

        dv2dt= (1/m2)*F2(-1*Fs,c2,v2)
        P = np.abs(m1*v1*dv1dt)
        dkdt= Dk*P
        dc1dt=Dc*P
        #--------------------------------
        derivatives = np.array([dx1dt, dx2dt, dv1dt, dv2dt, dkdt, dc1dt]) 
        return derivatives # of shape [Ny]. Each derivative term must be user defined

def _compute_loss_problem(solution_time: np.ndarray, solution: np.ndarray, dataset: np.ndarray, trainable_parameters: dict, fixed_parameters: dict):

        # Arguments:
        # solution_time: time (np.ndarray of size [Nts])
        # solution: state vector (np.ndarray of size [Nts,Ny])
        # dataset: dataset (np.ndarray of size [Nts,N_col])
        # trainable_parameters: dictionary of trainable parameters. keys identical to the names in the user_input.xml file
        # fixed_parameters: dictionary of fixed parameters. keys identical to the names in the user_input.xml file

        loss=0.0

        # this part is to be populated by the model
        #--------------------------------
        c2 = trainable_parameters['c2']
        Dk = trainable_parameters['Dk']
        Dc = trainable_parameters['Dc']
        m1 = trainable_parameters['m1']
        m2 = trainable_parameters['m2']
        vf = fixed_parameters['vf']
        #--------------------------------


        # this part is to be populated by the user
        #--------------------------------
        def F1(Fs,c1,v1):
                return Fs-c1*np.abs(v1)*np.sign(v1)
        
        F_exp=dataset[:,0:1]*1000
        s_exp=dataset[:,1:2]
        
        scale_factor_F=np.max(np.log10(np.abs(F_exp)))
        scale_factor_s=np.max(np.abs(s_exp))
        
        Nts=solution_time.shape[0]
        
        x2=solution[:,1]
        x1=solution[:,0]
        k=solution[:,4]
        c1=solution[:,5]
        v1=solution[:,2]
        F_sim=np.zeros(Nts)
        s_sim=np.zeros(Nts)
        Fs=np.multiply(k,x2-x1)
        for i in range(Nts):
                F_sim[i]=np.abs(F1(Fs[i],c1[i],v1[i]))
                s_sim[i]=x1[i]
        F_loss=np.sqrt(np.mean(np.square((np.log10(F_sim[1:])-np.log10(F_exp[1:]))/scale_factor_F)))
        s_loss=np.sqrt(np.mean(np.square((s_exp-s_sim)/scale_factor_s)))
        loss=5*F_loss+s_loss
        #--------------------------------

        return loss # scalar
        

def write_problem_result(solution_time: np.ndarray, solution: np.ndarray, dataset: np.ndarray, trainable_parameters: dict, fixed_parameters: dict):

        # Arguments:
        # solution_time: time (np.ndarray of size [Nts])
        # solution: state vector (np.ndarray of size [Nts,Ny])
        # dataset: dataset (np.ndarray of size [Nts,N_col])
        # trainable_parameters: dictionary of trainable parameters. keys identical to the names in the user_input.xml file
        # fixed_parameters: dictionary of fixed parameters. keys identical to the names in the user_input.xml file
        # Change the size of writeout_array as per your requirement

        # this part is to be populated by the model
        #--------------------------------
        c2 = trainable_parameters['c2']
        Dk = trainable_parameters['Dk']
        Dc = trainable_parameters['Dc']
        m1 = trainable_parameters['m1']
        m2 = trainable_parameters['m2']
        vf = fixed_parameters['vf']
        #--------------------------------

        writeout_array = np.zeros([solution_time.shape[0],6]) # DEFINE THIS as required!

        # this part is to be populated by the user
        #--------------------------------
        Nts = solution_time.shape[0]
        writeout_array = np.zeros([Nts, 5])

        def F1(Fs,c1,v1):
                return Fs-c1*np.abs(v1)*np.sign(v1)
        
        F_exp=dataset[:,0:1]*1000
        s_exp=dataset[:,1:2]
                                
        scale_factor=np.max(np.abs(F_exp))
        Nts=solution_time.shape[0]
        
        x2=solution[:,1]
        x1=solution[:,0]
        k=solution[:,4]
        c1=solution[:,5]
        v1=solution[:,2]
        F_sim=np.zeros(Nts)
        s_sim=np.zeros(Nts)
        Fs=np.multiply(k,x2-x1)
        for i in range(Nts):
                F_sim[i]=np.abs(F1(Fs[i],c1[i],v1[i]))
                s_sim[i]=x1[i]
        writeout_array[:,0]=solution_time
        writeout_array[:,1]=F_exp
        writeout_array[:,2]=s_exp
        writeout_array[:,3]=F_sim
        writeout_array[:,4]=s_sim

        #--------------------------------

        return writeout_array # of custom shape
