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
########################################################
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
            # if j>=0 and j<=24:
            #     BS_tilt = tf.random.uniform(thresholds_vector.shape, -18.0, -10.0)
            # elif j>=25 and j<=49:
            #     BS_tilt = tf.random.uniform(thresholds_vector.shape, 20.0, 32.0)
            # elif j >= 50 and j <= 74:
            #     BS_tilt = tf.random.uniform(thresholds_vector.shape, -18.0, 32.0)
            # elif j >= 75 and j <= 100:
            #     BS_tilt = tf.random.uniform(thresholds_vector.shape, -18.0, 32.0)
            #     # Define the excluded range
            #     excluded_range = tf.constant([-9.9, 19.9])
            #     # Mask out values within the excluded range
            #     condition = tf.logical_and(BS_tilt >= excluded_range[0], BS_tilt <= excluded_range[1])
            #     replacement_values = tf.random.uniform(tf.shape(BS_tilt), -18.0, -10.0)
            #     BS_tilt = tf.where(condition, replacement_values, BS_tilt)
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
            BS_tilt = thresholds_vector + tf.random.uniform(thresholds_vector.shape, -4.0, 4.0)

            excluded_range = tf.constant([-100.0, -18.0])
            condition = tf.logical_and(BS_tilt >= excluded_range[0], BS_tilt <= excluded_range[1])
            replacement_values = tf.random.uniform(tf.shape(BS_tilt), -18.0, -10.0)
            BS_tilt = tf.where(condition, replacement_values, BS_tilt)

            excluded_range2 = tf.constant([32.0, 100.0])
            condition2 = tf.logical_and(BS_tilt >= excluded_range2[0], BS_tilt <= excluded_range2[1])
            replacement_values2 = tf.random.uniform(tf.shape(BS_tilt), 20.0, 32.0)
            BS_tilt = tf.where(condition2, replacement_values2, BS_tilt)
            ########################################################
        if config.Specialized_BO == True:
            #Setting all tilts to -12 as in 3GPP
            BS_tilt = tf.random.uniform(thresholds_vector[0, :, 0].shape, -12.0, -12.0)
            #Randomly select indices to update
            idxes = [3,4,5,6,7,8,12,15,16,17,18,19,20,21,22,23,25,27,32,33,34,38,39,40,41,45,48,50,51,52]
            if j >= 0 and j <= 24:
                # Randomly select corresponding values to update
                Update_indices = [random.uniform(-18.0, -10.0) for _ in range(len(idxes))]
                #Update the selected indices with the new values
                for idx, Update_indice in zip(idxes, Update_indices):
                    indx = tf.constant([[idx]])
                    BS_tilt = tf.tensor_scatter_nd_update(BS_tilt, indx, tf.constant([Update_indice]))
                #Expand dimensions to match the original shape
                BS_tilt = tf.expand_dims(tf.expand_dims(BS_tilt, axis=0), axis=2)
            elif j >= 25 and j <= 49:
                Update_indices = [random.uniform(20.0, 32.0) for _ in range(len(idxes))]
                for idx, Update_indice in zip(idxes, Update_indices):
                    indx = tf.constant([[idx]])
                    BS_tilt = tf.tensor_scatter_nd_update(BS_tilt, indx, tf.constant([Update_indice]))
                BS_tilt = tf.expand_dims(tf.expand_dims(BS_tilt, axis=0), axis=2)
            elif j >= 50 and j <= 74:
                Update_indices = [random.uniform(-18.0, 32.0) for _ in range(len(idxes))]
                for idx, Update_indice in zip(idxes, Update_indices):
                    indx = tf.constant([[idx]])
                    BS_tilt = tf.tensor_scatter_nd_update(BS_tilt, indx, tf.constant([Update_indice]))
                BS_tilt = tf.expand_dims(tf.expand_dims(BS_tilt, axis=0), axis=2)
            elif j >= 75 and j <= 100:
                Update_indices = [random.uniform(-18.0, 32.0) for _ in range(len(idxes))]
                for idx, Update_indice in zip(idxes, Update_indices):
                    indx = tf.constant([[idx]])
                    BS_tilt = tf.tensor_scatter_nd_update(BS_tilt, indx, tf.constant([Update_indice]))
                BS_tilt = tf.expand_dims(tf.expand_dims(BS_tilt, axis=0), axis=2)
                # Define the excluded range
                excluded_range = tf.constant([-9.9, 19.9])
                # Mask out values within the excluded range
                condition = tf.logical_and(BS_tilt >= excluded_range[0], BS_tilt <= excluded_range[1])
                replacement_values = tf.random.uniform(tf.shape(BS_tilt), -18.0, -10.0)
                BS_tilt = tf.where(condition, replacement_values, BS_tilt)

        new_train_x = torch.from_numpy(BS_tilt[:,:,0].numpy()).double()
        BS_tilt = thresholds_vector  # This is for getting the SINR for the opt thresholds after finishing
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
    # file_name = "2023_07_14_AlphaHalf_Mix_Product_Rate_DataSet.pt"
    # file_name = "2023_08_01_dataSet_test.pt"
    # torch.save({"train_x": train_x, "train_obj": train_obj}, file_name)

    return train_x, train_obj

# Run BO loop
########################################################
BO_itertions = 200
data_size = 100

#Initial tilts and powers and obj value
# thresholds_vector = tf.expand_dims(tf.expand_dims(tf.random.uniform((57,), 0.0, 0.0, tf.float32), axis=0),axis=2)
# Alpha 0.5 Best Corridors, Globecom Framework
# thresholds_vector = tf.expand_dims(tf.constant([[
#     -9.8102,  -13.0183, -14.1928,  25.8687,  30.4193,  25.6352,  25.0205, -15.3203,  30.6154, -14.3594,
#     -11.5712, -10.3334,  20.0784, -16.5769, -10.7310,  15.6762,  27.6900,  29.1826,  21.9059, -11.7631,
#     -11.7591, -10.5255,  32.6864, -10.7251, -12.0509, -13.0080, -12.0201, -14.8350, -15.3849,  16.8090,
#     -12.3643,  34.8795, -12.1392, -13.2892, -12.7744, -16.7355, -11.6496, -12.2563, -6.8718, -11.0011,
#     -7.4486, -12.0740, -9.2410, -11.0125,   35.9105, -10.4298, -10.2161, -18.3366, -15.8579, -8.7873,
#     -11.3935,  -9.9674, -14.7851, -10.5096, -10.6762, -11.3624, -14.8511]]), axis=2)

thresholds_vector = tf.expand_dims(tf.constant([[
    -12.0000,  -12.0000, -12.0000,  25.8687,  30.4193,  25.6352,  25.0205, -15.3203,  30.6154, -12.0000,
    -12.0000, -12.0000,  20.0784, -12.0000, -12.0000,  15.6762,  27.6900,  29.1826,  21.9059, -12.0000,
    -12.0000, -12.0000,  32.6864, -12.0000, -12.0000, -12.0000, -12.0000, -12.0000, -12.0000,  16.8090,
    -12.0000,  34.8795, -12.0000, -12.0000, -12.0000, -12.0000, -12.0000, -12.0000, -12.0000, -12.0000,
    -12.0000, -12.0000, -12.0000, -12.0000,   35.9105, -12.0000, -12.0000, -12.0000, -12.0000, -12.0000,
    -12.0000,  -12.0000, -12.0000, -12.0000, -12.0000, -12.0000, -12.0000]]), axis=2)

# # Alpha 0.5 Best Corridors, High-dim Framework, only problamatic cells
# thresholds_vector = tf.expand_dims(tf.constant([[
#     -12.0000, - 12.0000, - 12.0000,   23.6723,   27.9053,   32.0000,   25.7613, - 12.0000,   32.0000, - 12.0000,
#     - 12.0000, - 12.0000,   22.4880, - 12.0000, - 12.0000,   32.0000,   32.0000,   26.0659,   24.4629, - 12.0000,
#     - 12.0000, - 12.0000,   28.3232, - 12.0000, - 12.0000, - 12.0000, - 12.0000, - 12.0000, - 12.0000,   32.0000,
#     - 12.0000,   32.0000, - 12.0000, - 12.0000, - 12.0000, - 12.0000, - 12.0000, - 12.0000, - 12.0000, - 12.0000,
#     - 12.0000, - 12.0000, - 12.0000, - 12.0000,   32.0000, - 12.0000, - 12.0000, - 12.0000, - 12.0000, - 12.0000,
#     - 12.0000, - 12.0000, - 12.0000, - 12.0000, - 12.0000, - 12.0000, - 12.0000]]), axis=2)

# # Alpha 0.5 Best Corridors, High-dim Framework, Alpha=1, 100Samples, Noisy Iterative BO Config
# thresholds_vector = tf.expand_dims(tf.constant([[
#     -12.1667, - 13.3691, - 18.0000,   26.1501,   32.0000,   27.8266,   24.5694, - 14.5654,   32.0000, - 13.9548,
#     - 10.8122, - 11.2349,   20.8837, - 18.0000, - 11.0949,   15.7942,   29.7828,   25.2739,   25.3110, - 15.4718,
#     - 14.3984, - 12.7908,   32.0000, - 12.9391, - 10.7055, - 9.5984, - 18.0000, - 10.7880, - 13.5484,   18.8432,
#     - 14.5677,   32.0000, - 14.0476, - 8.0352, - 9.4126, - 13.5271, - 11.4619, - 11.1426, - 6.8492, - 9.9888,
#     - 7.6224, - 13.9718, - 10.1444, - 13.9342,   32.0000, - 12.0637, - 16.3331, - 15.2217, - 14.4253, - 10.6167,
#     - 12.1975, - 9.9580, - 14.9875, - 13.3141, - 14.6836, - 15.5389, - 12.0180]]), axis=2)

Ptx_thresholds_vector = tf.expand_dims(tf.expand_dims(tf.random.uniform((57,), 46.0, 46.0, tf.float32), axis=0),axis=2)
obj_vector = torch.tensor([[2.50]], dtype=torch.double) #-4.66

# Creat the training data-set
train_x, train_obj = generate_initial_data(thresholds_vector, Ptx_thresholds_vector, obj_vector,data_size)
train_x = train_x[1:,:]
train_obj = train_obj[1:,:]

# # Load the training data-set
# file_name = "2023_07_05_Alpha0_GUEs_Product_Rate_DataSet.pt"
# loaded_data = torch.load(file_name)
# train_x = loaded_data["train_x"]
# train_obj = loaded_data["train_obj"]

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
            train_Yvar=torch.full_like(train_obj, 1e-6),
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
        lower_bound = -18.0
        upper_bound = 32.0
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

        idxes = [3, 4, 5, 6, 7, 8, 12, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 27, 32, 33, 34, 38, 39, 40, 41, 45, 48,50, 51, 52]
        train_x_org = train_x.to(torch.float32)
        train_x = train_x[:, idxes]

        model = SaasFullyBayesianSingleTaskGP(
            train_X=train_x,
            train_Y=train_obj,
            train_Yvar=torch.full_like(train_obj, 1e-6),
            outcome_transform=Standardize(m=1), )

        fit_fully_bayesian_model_nuts(
            model,
            warmup_steps=WARMUP_STEPS,
            num_samples=NUM_SAMPLES,
            thinning=THINNING,
            disable_progbar=True, )

        # Optimizes the qEI acquisition function, and returns a new candidate
        ########################################################
        DIM = 30
        lower_bound = -18.0
        upper_bound = 32.0
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
    file_name = "2023_08_11_HighDim_BO_LambdaHalf_Mix_Corr_ProductRate_Alpha1_100Samples_NoisyIterativeDataSet_iteration{}.mat".format(i)
    savemat(file_name, data_BO)

    d = {"SINR_UAVs": 10 * np.log10(sinr_total_UAVs.numpy()),
          "SINR_GUEs": 10 * np.log10(sinr_total_GUEs.numpy()),
          "Rate_UAVs": Rate_UAVs.numpy(),
          "Rate_GUEs": Rate_GUEs.numpy()}
    savemat("2023_08_16_SINR_Rate_LambdaHaf_ProductRateObj_IterativeBO_FixedDownTilts.mat", d)