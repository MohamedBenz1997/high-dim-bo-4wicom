"""
This is the Runner script for Multi-BO auto swiping
@authors: Mohamed Benzaghta
"""

import os

os.system("export MKL_DEBUG_CPU_TYPE=5")
import tensorflow as tf
import torch
from scipy.io import savemat
from tqdm import tqdm
import numpy as np
import random

from TerrestrialClass import Terrestrial
from SinrClass import SINR
from config import Config
from plot_class import Plot

from botorch import fit_fully_bayesian_model_nuts
from botorch.acquisition import qExpectedImprovement
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf

GPU_mode = 0  # set this value one if you have proper GPU setup in your computer

if GPU_mode:
    num_GPU = 1  # choose among available GPUs
    mem_growth = True
    print('Tensorflow version: ', tf.__version__)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    print('Number of GPUs available :', len(gpus))
    # tf.config.experimental.set_visible_devices(gpus[num_GPU], 'GPU')
    # tf.config.experimental.set_memory_growth(gpus[num_GPU], mem_growth)
    print('Used GPU: {}. Memory growth: {}'.format(num_GPU, mem_growth))

# ------------ Import of the classes
SINR = SINR()
config = Config()
plot = Plot()

# Generating data-sets for BO
""
def generate_initial_data(thresholds_vector, Ptx_thresholds_vector, obj_vector, data_size):

    #If optimizing tilts
    train_x = thresholds_vector[:, :, 0]
    # If optimizing powers
    # train_x = Ptx_thresholds_vector[:, :, 0]
    train_x = torch.from_numpy(train_x.numpy()).double()
    train_obj = obj_vector

    for j in range(data_size):
        if config.Specialized_BO == False:
            ##Setting Random tilts for all BSs creating a data set
            # BS_tilt = tf.random.uniform(thresholds_vector.shape, -18, 32)
            ########################################################
            ##Different sets of samples in the initial data-set
            ########################################################
            if j>=0 and j<=24:
                BS_tilt = tf.random.uniform(thresholds_vector.shape, -10.0, -5.0)
            elif j>=25 and j<=49:
                BS_tilt = tf.random.uniform(thresholds_vector.shape, 10.0, 10.0)
            elif j >= 50 and j <= 74:
                BS_tilt = tf.random.uniform(thresholds_vector.shape, -10.0, 10.0)
            elif j >= 75 and j <= 100:
                BS_tilt = tf.random.uniform(thresholds_vector.shape, -10.0, 10.0)
                # Define the excluded range
                excluded_range = tf.constant([-5.0, 5.0]) #-10,10
                # Mask out values within the excluded range
                condition = tf.logical_and(BS_tilt >= excluded_range[0], BS_tilt <= excluded_range[1])
                replacement_values = tf.random.uniform(tf.shape(BS_tilt), -10.0, -5.0)
                BS_tilt = tf.where(condition, replacement_values, BS_tilt)
            ########################################################
            ##DataSets containing only 1-uptilited BS and other downtilted. 4 different uptilts for each BSs resulting in 4*57=228 samples in the initial dataset
            # idx_1 = j % 57
            # BS_tilt = tf.random.uniform(thresholds_vector[0,:,0].shape, -18.0, -10.0)
            # indx = tf.constant([[idx_1]])
            #
            # Uptilt_value = tf.constant([random.uniform(20.0, 32.0)])
            # BS_tilt = tf.tensor_scatter_nd_update(BS_tilt, indx, Uptilt_value)
            # BS_tilt = tf.expand_dims(tf.expand_dims(BS_tilt,axis=0),axis=2)
            ########################################################
            ##Setting Random noise of +- 5deg to iterative BO opt config
            ########################################################
            # BS_tilt = thresholds_vector + tf.random.uniform(thresholds_vector.shape, -4.0, 4.0)
            # excluded_range = tf.constant([-100.0, -18.0])
            # condition = tf.logical_and(BS_tilt >= excluded_range[0], BS_tilt <= excluded_range[1])
            # replacement_values = tf.random.uniform(tf.shape(BS_tilt), -18.0, -10.0)
            # BS_tilt = tf.where(condition, replacement_values, BS_tilt)
            #
            # excluded_range2 = tf.constant([32.0, 100.0])
            # condition2 = tf.logical_and(BS_tilt >= excluded_range2[0], BS_tilt <= excluded_range2[1])
            # replacement_values2 = tf.random.uniform(tf.shape(BS_tilt), 20.0, 32.0)
            # BS_tilt = tf.where(condition2, replacement_values2, BS_tilt)
            ########################################################
            ##Data-set for just uptilts
            #BS_tilt = tf.random.uniform(thresholds_vector.shape, -12.0, 45.0)
            #excluded_range1 = tf.constant([-12.0, 20.0])
            #condition1 = tf.logical_and(BS_tilt >= excluded_range1[0], BS_tilt <= excluded_range1[1])
            #replacement_values1 = tf.random.uniform(tf.shape(BS_tilt), -12.0, -12.0)
            #BS_tilt = tf.where(condition1, replacement_values1, BS_tilt)
            #excluded_range2 = tf.constant([0.0, 5.0])
            #condition2 = tf.logical_and(BS_tilt >= excluded_range2[0], BS_tilt <= excluded_range2[1])
            #replacement_values2 = tf.random.uniform(tf.shape(BS_tilt), 20.0, 45.0)
            #BS_tilt = tf.where(condition2, replacement_values2, BS_tilt)
            ########################################################

        if config.Specialized_BO == True:
            #Setting all tilts to 0
            #BS_tilt = tf.random.uniform(thresholds_vector[0, :, 0].shape, -10.0, -10.0)
            #Setting the uptilts according to the recommnded config
            #thresholds_vector = tf.expand_dims(tf.constant([[]]), axis=2)
            BS_tilt = thresholds_vector[0, :, 0]
            #Select indices to update for uptilts
            #idxes = [3,10,14,17,21,26,31,32,34,44,46]
            #Select indices to update for downtilts
            idxes = [0,1,2,4,5,6,7,8,9,11,12,13,15,16,18,19,20,22,23,24,25,27,28,29,30,33,35,36,37,38,39,40,41,42,43,45,47,48,49,50,51,52,53,54,55,56]
            Update_indices = [random.uniform(-12.0, -8.0) for _ in range(len(idxes))]
            #Update the selected indices with the new values
            for idx, Update_indice in zip(idxes, Update_indices):
                indx = tf.constant([[idx]])
                BS_tilt = tf.tensor_scatter_nd_update(BS_tilt, indx, tf.constant([Update_indice]))
            BS_tilt = tf.expand_dims(tf.expand_dims(BS_tilt, axis=0), axis=2)

        new_train_x = torch.from_numpy(BS_tilt[:,:,0].numpy()).double()
        # BS_tilt = thresholds_vector  # This is for getting the SINR for the opt thresholds after finishing
        BS_tilt = tf.tile(BS_tilt, [2 * config.batch_num, 1, config.Nuser_drop])

        #Setting Random powers for creating a data set
        # P_Tx_TN = tf.random.uniform(Ptx_thresholds_vector.shape, 36, 46)
        # new_train_x = torch.from_numpy(P_Tx_TN[:,:,0].numpy()).double()
        P_Tx_TN = Ptx_thresholds_vector  # This is for getting the SINR for the opt thresholds after finishing
        P_Tx_TN = tf.tile(P_Tx_TN, [2 * config.batch_num, 1, config.Nuser_drop])

        #Run the simulator
        data = Terrestrial()
        data.alpha_factor = 0.0  # LEO at 90deg
        data.BS_tilt = BS_tilt
        data.call()
        #Import of the UAVs and GUEs LSG and SINR data
        Xuser_GUEs = data.Xuser_GUEs
        Xuser_UAVs = data.Xuser_UAVs
        LSG_UAVs_Corridors = data.LSG_UAVs_Corridors
        LSG_GUEs = data.LSG_GUEs
        sinr_TN_UAVs_Corridors = SINR.sinr_TN(LSG_UAVs_Corridors, P_Tx_TN)
        sinr_TN_GUEs = SINR.sinr_TN(LSG_GUEs, P_Tx_TN)

        # BO objective: Sum of log of the RSS
        SINR_sumOftheLog_Obj, Rate_sumOftheLog_Obj, sinr_total_UAVs, sinr_total_GUEs = SINR.BO_Multi_Obj_Cooridor(sinr_TN_UAVs_Corridors, sinr_TN_GUEs, alpha=0.5)
        Rate_sumOftheLog_Obj, Rate_GUEs, Rate_UAVs = SINR.BO_Obj_Rates_and_Outage(LSG_GUEs, LSG_UAVs_Corridors, P_Tx_TN,alpha=0.5)
        Rate_obj = Rate_sumOftheLog_Obj[0].__float__()
        new_obj = torch.tensor([[Rate_obj]], dtype=torch.double)

        ## Serving BSs indexes and UAVs locations
        BSs_id_UAVs, Xuser_UAVs_x, Xuser_UAVs_y = SINR.Cell_id(LSG_UAVs_Corridors, Xuser_UAVs)
        BSs_id_GUEs, Xuser_GUEs_x, Xuser_GUEs_y = SINR.Cell_id(LSG_GUEs, Xuser_GUEs)

        #Append the thresholds and objectives
        train_x = torch.cat((train_x, new_train_x), dim=0)
        train_obj = torch.cat((train_obj, new_obj), dim=0)

    # Save the torch tensors to a file with .pt extension to be loaded using python later
    #file_name = "2023_09_25_Corr_SAforSB_SAonly_Down.pt"
    #torch.save({"train_x": train_x, "train_obj": train_obj}, file_name)

    return train_x, train_obj

# Run BO loop
""
BO_itertions = 40
data_size = 100

#Initial tilts and powers and obj value
#thresholds_vector = tf.expand_dims(tf.expand_dims(tf.random.uniform((57,), 0.0, 0.0, tf.float32), axis=0),axis=2)-10.0
thresholds_vector = tf.expand_dims(tf.constant([[
    -10.0000, - 10.0000, - 10.0000,   15.4027, - 10.0000, - 10.0000, - 10.0000, - 10.0000, - 10.0000, - 10.0000,
    25.1004, - 10.0000, - 10.0000, - 10.0000,   16.8382, - 10.0000, - 10.0000,   27.1016, - 10.0000, - 10.0000,
    - 10.0000,   23.8081, - 10.0000, - 10.0000, - 10.0000, - 10.0000,   16.0281, - 10.0000, - 10.0000, - 10.0000,
    - 10.0000,   18.3018,    5.4751, - 10.0000,   15.1111, - 10.0000, - 10.0000, - 10.0000, - 10.0000, - 10.0000,
    - 10.0000, - 10.0000, - 10.0000, - 10.0000,   11.3447, - 10.0000,    5.0000, - 10.0000, - 10.0000, - 10.0000,
    - 10.0000, - 10.0000, - 10.0000, - 10.0000, - 10.0000, - 10.0000, - 10.0000]]), axis=2)

Ptx_thresholds_vector = tf.expand_dims(tf.expand_dims(tf.random.uniform((57,), 46.0, 46.0, tf.float32), axis=0),axis=2)
obj_vector = torch.tensor([[1.2445]], dtype=torch.double) #-4.66

# Creat the training data-set
#train_x, train_obj = generate_initial_data(thresholds_vector, Ptx_thresholds_vector, obj_vector,data_size)
#train_x = train_x[1:,:]
#train_obj = train_obj[1:,:]
## Load the training data-set
file_name = "2023_09_25_Corr_SAforSB_SAonly_Down.pt"
loaded_data = torch.load(file_name)
train_x = loaded_data["train_x"]
train_obj = loaded_data["train_obj"]
train_x = train_x[1:,:]
train_obj = train_obj[1:,:]
#train_x_SA = loaded_data["train_x"]
#train_obj_SA = loaded_data["train_obj"]
#train_x_SA = train_x_SA[1:,:]
#train_obj_SA = train_obj_SA[1:,:]
#train_x = torch.cat((train_x, train_x_SA), dim=0)
#train_obj = torch.cat((train_obj, train_obj_SA), dim=0)

#Start BO iterating
for i in tqdm(range(BO_itertions)):

    # Creating the surrogate model
    ########################################################
    WARMUP_STEPS = 512  #256
    NUM_SAMPLES =  256  #128
    THINNING = 16

    if config.Specialized_BO == False:

        train_x_org = train_x

        model = SaasFullyBayesianSingleTaskGP(
            train_X=train_x,
            train_Y=train_obj,
            outcome_transform=Standardize(m=1),)

        fit_fully_bayesian_model_nuts(
            model,
            warmup_steps=WARMUP_STEPS,
            num_samples=NUM_SAMPLES,
            thinning=THINNING,
            disable_progbar=True,)

        # Optimizes the qEI acquisition function, and returns a new candidate
        ########################################################
        DIM = 57
        lower_bound = -10.0
        upper_bound = 10.0
        bounds = torch.cat((torch.zeros(1, DIM)+lower_bound, torch.zeros(1, DIM)+upper_bound))

        EI = qExpectedImprovement(model=model, best_f=train_obj.max())
        candidates, acq_values = optimize_acqf(
            EI,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=1024,)

        BS_tilt = tf.constant(candidates.numpy())
        BS_tilt = tf.expand_dims(BS_tilt,axis=2)
        BS_tilt = tf.tile(BS_tilt, [2 * config.batch_num, 1, config.Nuser_drop])

    elif config.Specialized_BO == True:
        #Up-cells
        #idxes = [3,10,14,17,21,26,31,32,34,44,46]
        ##Down-cells
        idxes = [0,1,2,4,5,6,7,8,9,11,12,13,15,16,18,19,20,22,23,24,25,27,28,29,30,33,35,36,37,38,39,40,41,42,43,45,47,48,49,50,51,52,53,54,55,56]
        
        train_x_org = train_x.to(torch.float32)
        train_x = train_x[:, idxes]

        model = SaasFullyBayesianSingleTaskGP(
            train_X=train_x,
            train_Y=train_obj,
            outcome_transform=Standardize(m=1), )

        fit_fully_bayesian_model_nuts(
            model,
            warmup_steps=WARMUP_STEPS,
            num_samples=NUM_SAMPLES,
            thinning=THINNING,
            disable_progbar=True, )

        # Optimizes the qEI acquisition function, and returns a new candidate
        ########################################################
        DIM = 46 
        lower_bound = -12.0
        upper_bound = -8.0
        bounds = torch.cat((torch.zeros(1, DIM) + lower_bound, torch.zeros(1, DIM) + upper_bound))

        EI = qExpectedImprovement(model=model, best_f=train_obj.max())
        candidates, acq_values = optimize_acqf(
            EI,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=1024, )

        #Combining the optimizng indexes and non-optimized indexes together, for the simulator to run
        combined_opt_nonOpt_train_x = torch.zeros_like(train_x_org, dtype=torch.float32)
        combined_opt_nonOpt_train_x[:, idxes] = candidates
        none_optimized_indexes = list(set(range(train_x_org.shape[1])) - set(idxes))
        combined_opt_nonOpt_train_x[:, none_optimized_indexes] = train_x_org[:, none_optimized_indexes]
        candidates = combined_opt_nonOpt_train_x
        candidates = candidates[0, :].view(1, 57)

        BS_tilt = tf.constant(candidates.numpy())
        BS_tilt = tf.expand_dims(BS_tilt, axis=2)
        BS_tilt = tf.tile(BS_tilt, [2 * config.batch_num, 1, config.Nuser_drop])


    #Run simulator based on new candidates
    ########################################################
    #If tilts are not being optimized
    # BS_tilt = tf.tile(thresholds_vector, [2 * config.batch_num, 1, config.Nuser_drop])
    #If power is not being optimized
    P_Tx_TN = tf.tile(Ptx_thresholds_vector, [2 * config.batch_num, 1, config.Nuser_drop])

    data = Terrestrial()
    data.alpha_factor = 0.0  # LEO at 90deg
    data.BS_tilt = BS_tilt
    data.call()

    #Import of the UAVs and GUEs LSG and SINR data
    LSG_UAVs_Corridors = data.LSG_UAVs_Corridors
    LSG_GUEs = data.LSG_GUEs
    sinr_TN_UAVs_Corridors = SINR.sinr_TN(LSG_UAVs_Corridors, P_Tx_TN)
    sinr_TN_GUEs = SINR.sinr_TN(LSG_GUEs, P_Tx_TN)

    # BO objective
    Rate_sumOftheLog_Obj, Rate_GUEs, Rate_UAVs = SINR.BO_Obj_Rates_and_Outage(LSG_GUEs, LSG_UAVs_Corridors, P_Tx_TN,alpha=0.5)
    Rate_obj = Rate_sumOftheLog_Obj[0].__float__()
    new_obj = torch.tensor([[Rate_obj]], dtype=torch.double)

    # Append the thresholds and objectives
    train_x = torch.cat((train_x_org, candidates), dim=0)
    train_x = train_x.to(torch.float64)
    train_obj = torch.cat((train_obj, new_obj), dim=0)

    #Monitoring the progress of best observed value so far
    best_value = tf.expand_dims(tf.reduce_max(train_obj[data_size:,:], axis=0), axis=0)
    if i == 0:
        best_rate_so_far = tf.zeros(best_value.shape, dtype='float64')
    best_rate_so_far = tf.concat([best_rate_so_far, best_value], axis=0)

    # BO Outputs
    Thresholds = train_x[data_size:,:]
    Obj = train_obj[data_size:,:]
    Thresholds = Thresholds.numpy()
    Thresholds = tf.convert_to_tensor(Thresholds)
    Obj = Obj.numpy()
    Obj = tf.convert_to_tensor(Obj)
    best_observed_objective_value = tf.reduce_max(Obj, axis=0)
    optimum_thresholds = tf.tile(tf.cast(Obj == best_observed_objective_value, "float64"), [1, 1]) * Thresholds
    optimum_thresholds = tf.reduce_sum(optimum_thresholds, axis=0)
    Full_tilts = optimum_thresholds

    # Saving BO data for matlab
    data_BO = {"Thresholds": Thresholds.numpy(),
               "Obj": Obj.numpy(),
               "best_observed_objective_value": best_observed_objective_value.numpy(),
               "optimum_thresholds": optimum_thresholds.numpy(),
               "best_rate_so_far": best_rate_so_far.numpy(),
               "Full_tilts": Full_tilts.numpy()}
    file_name = "2023_09_25_HDBO_Corr_SAforSB_SAonly_Down_iteration{}.mat".format(i)
    savemat(file_name, data_BO)

    # d = {"SINR_UAVs": 10 * np.log10(sinr_total_UAVs.numpy()),
    #       "SINR_GUEs": 10 * np.log10(sinr_total_GUEs.numpy()),
    #       "Rate_UAVs": Rate_UAVs.numpy(),
    #       "Rate_GUEs": Rate_GUEs.numpy()}
    # savemat("2023_08_16_SINR_Rate_LambdaHaf_ProductRateObj_IterativeBO_FixedDownTilts.mat", d)
