"""
This is the Runner script for BO auto swiping over 2-thresholds over the entire tier
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

# Genrating data-sets for BO
########################################################
def generate_initial_data(tilts_vector, HPBW_v_vector, Ptx_thresholds_vector, obj_vector, data_size):

    train_x1 = tilts_vector[:, :, 0]
    train_x2 = HPBW_v_vector[:, :, 0]
    train_x = tf.concat([train_x1, train_x2], axis=1)
    train_x = torch.from_numpy(train_x.numpy()).double()
    train_obj = obj_vector

    for j in range(data_size):

        if config.Specialized_BO == False:
            ##Setting Random tilts for all BSs creating a data set
            # BS_tilt = tf.random.uniform(tilts_vector.shape, -18, 32)
            ########################################################
            ##Different sets of samples in the initial data-set
            ########################################################
            if j>=0 and j<=24:
                BS_tilt = tf.random.uniform(tilts_vector.shape, -15.0, -5.0)
            elif j>=25 and j<=49:
                BS_tilt = tf.random.uniform(tilts_vector.shape, 10.0, 45.0)
            elif j >= 50 and j <= 74:
                BS_tilt = tf.random.uniform(tilts_vector.shape, -15.0, 45.0)
            elif j >= 75 and j <= 100:
                BS_tilt = tf.random.uniform(tilts_vector.shape, -15.0, 45.0)
                # Define the excluded range
                excluded_range = tf.constant([-5.0, 10.0]) #-10,10
                # Mask out values within the excluded range
                condition = tf.logical_and(BS_tilt >= excluded_range[0], BS_tilt <= excluded_range[1])
                replacement_values = tf.random.uniform(tf.shape(BS_tilt), -10.0, -5.0)
                BS_tilt = tf.where(condition, replacement_values, BS_tilt)

            # Setting Random vHPBW for creating a data set
            BS_HPBW_v = tf.random.uniform(HPBW_v_vector.shape, 5, 30)
            ########################################################

        if config.Specialized_BO == True:
            # Setting all tilts to 0
            # BS_tilt = tf.random.uniform(tilts_vector[0, :, 0].shape, -10.0, -10.0)
            # Setting the uptilts according to the recommnded config
            # tilts_vector = tf.expand_dims(tf.constant([[]]), axis=2)
            BS_tilt = tilts_vector[0, :, 0]
            BS_HPBW_v = HPBW_v_vector[0, :, 0]
            # Select indices to update for uptilts
            idxes = [3,4,5,10,12,21,24,29,31,33]
            # Select indices to update for downtilts
            # idxes = [0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 18, 19, 20, 22, 23]
            Update_indices = [random.uniform(15.0, 45.0) for _ in range(len(idxes))]
            # Update the selected indices with the new values
            for idx, Update_indice in zip(idxes, Update_indices):
                indx = tf.constant([[idx]])
                BS_tilt = tf.tensor_scatter_nd_update(BS_tilt, indx, tf.constant([Update_indice]))
            BS_tilt = tf.expand_dims(tf.expand_dims(BS_tilt, axis=0), axis=2)

            # Setting Random vHPBW for creating a data set
            Update_indices_vHPBW = [random.uniform(5.0, 30.0) for _ in range(len(idxes))]
            # Update the selected indices with the new values
            for idx, Update_indices_vHPBW in zip(idxes, Update_indices_vHPBW):
                indx = tf.constant([[idx]])
                BS_HPBW_v = tf.tensor_scatter_nd_update(BS_HPBW_v, indx, tf.constant([Update_indices_vHPBW]))
            BS_HPBW_v = tf.expand_dims(tf.expand_dims(BS_HPBW_v, axis=0), axis=2)

        #Setting Random tilts for creating a data set
        new_train_x1 = torch.from_numpy(BS_tilt[:,:,0].numpy()).double()
        # BS_tilt = tilts_vector  # This is for getting the SINR for the opt thresholds after finishing
        BS_tilt = tf.tile(BS_tilt, [2 * config.batch_num, 1, config.Nuser_drop])

        new_train_x2 = torch.from_numpy(BS_HPBW_v[:,:,0].numpy()).double()
        # BS_HPBW_v = HPBW_v_vector  # This is for getting the SINR for the opt thresholds after finishing
        BS_HPBW_v = tf.tile(BS_HPBW_v, [2 * config.batch_num, 1, config.Nuser_drop])

        P_Tx_TN = Ptx_thresholds_vector  # This is for getting the SINR for the opt thresholds after finishing
        P_Tx_TN = tf.tile(P_Tx_TN, [2 * config.batch_num, 1, config.Nuser_drop])

        #Run the simulator
        data = Terrestrial()
        data.alpha_factor = 0.0  # LEO at 90deg
        data.BS_tilt = BS_tilt
        data.BS_HPBW_v = BS_HPBW_v
        data.call()

        # Import distances
        D = data.D
        D_2d = data.D_2d

        #Import of the UAVs and GUEs LSG and SINR data
        Xuser_GUEs = data.Xuser_GUEs
        Xuser_UAVs = data.Xuser_UAVs
        LSG_UAVs_Corridors = data.LSG_UAVs_Corridors
        LSG_GUEs = data.LSG_GUEs
        sinr_TN_UAVs_Corridors = SINR.sinr_TN(LSG_UAVs_Corridors, P_Tx_TN)
        sinr_TN_GUEs = SINR.sinr_TN(LSG_GUEs, P_Tx_TN)

        # BO objective: Sum of log of the RSS
        Rate_sumOftheLog_Obj, UAVs_Coverage_ratio, Rate_GUEs, Rate_UAVs = SINR.BO_Obj_Rates_and_Outage(LSG_GUEs,
                                                                                                       LSG_UAVs_Corridors,
                                                                                                       P_Tx_TN, D, D_2d,
                                                                                                       alpha=0.5)
        Rate_obj = Rate_sumOftheLog_Obj[0].__float__()
        new_obj = torch.tensor([[Rate_obj]], dtype=torch.double)

        ## Serving BSs indexes and UAVs locations
        #BSs_id_UAVs, Xuser_UAVs_x, Xuser_UAVs_y = SINR.Cell_id(LSG_UAVs_Corridors, Xuser_UAVs)
        #BSs_id_GUEs, Xuser_GUEs_x, Xuser_GUEs_y = SINR.Cell_id(LSG_GUEs, Xuser_GUEs)

        #Append the thresholds and objectives
        new_train_x = torch.cat((new_train_x1, new_train_x2), dim=1)
        train_x = torch.cat((train_x, new_train_x), dim=0)
        train_obj = torch.cat((train_obj, new_obj), dim=0)

    # Save the torch tensors to a file with .pt extension to be loaded using python later
    # file_name = "2023_11_02_HDBOEK_Tilt_vHPBW_Corr_DataSet.pt"
    # torch.save({"train_x": train_x, "train_obj": train_obj}, file_name)

    return train_x, train_obj


# Run BO loop
########################################################
BO_itertions = 300
data_size = 40

#Initial tilts and powers and obj value
tilts_vector = tf.expand_dims(tf.expand_dims(tf.random.uniform((57,), 0.0, 0.0, tf.float32), axis=0),axis=2)
HPBW_v_vector = tf.expand_dims(tf.expand_dims(tf.random.uniform((57,), 10.0, 10.0, tf.float32), axis=0),axis=2)

if config.Specialized_BO == True:
    tilts_vector = tf.expand_dims(tf.constant([[
        -15.0000, - 11.3741, - 3.4778,   39.3572,   32.0589,   25.3663, - 0.8104, - 15.0000 ,  16.8360,    5.9341,
        18.3154, - 15.0000,   39.3708,   18.9589, - 15.0000, - 15.0000, - 15.0000, - 15.0000, - 10.7326, - 11.1183,
        - 13.4971,   25.8740, - 15.0000, - 6.3592,    3.6819, - 15.0000, - 12.7818, - 13.9944, - 15.0000,    0.8050,
        - 15.0000,   28.8697, - 13.9976,   33.2211, - 15.0000, - 15.0000, - 1.2510,   15.2366,    9.0837, - 15.0000,
        10.6616,   16.3738, - 12.1902, - 15.0000, - 7.1984,    5.7107, - 0.8688, - 14.8606,    3.9533, - 15.0000,
        - 14.7396, - 15.0000,   26.1245, - 0.4450, - 15.0000,   21.9139,    7.1188]]), axis=2)

    HPBW_v_vector = tf.expand_dims(tf.constant([[
        26.6705,   30.0000,    5.0000,   20.5620,    5.2504,   30.0000,    5.0000,   19.0124,   24.8147,   29.1420,
        30.0000,   16.2765,   13.5567,   30.0000,   22.2120,    9.7450,    8.4363,   30.0000,   30.0000,    5.6024,
        18.8135,   13.0910,   10.6228,   20.7671,   13.7134,    7.1497,   10.0776,    6.5165,   30.0000,   30.0000,
        15.8721,    5.0000,   29.4219,   30.0000,   15.7236,   20.4311,    5.0000,   16.3816,   30.0000,   26.9746,
        23.9609,    8.6423,   20.9576,   30.0000,    9.4576,    5.3242,   30.0000,   30.0000,   30.0000,   13.0928,
        30.0000,   29.0906,    5.0000,    5.0000,   30.0000,   15.2951,   26.3101]]), axis=2)

Ptx_thresholds_vector = tf.expand_dims(tf.expand_dims(tf.random.uniform((57,), 46.0, 46.0, tf.float32), axis=0),axis=2)
obj_vector = torch.tensor([[-0.1940]], dtype=torch.double)

# Creat the training data-set
train_x, train_obj = generate_initial_data(tilts_vector, HPBW_v_vector, Ptx_thresholds_vector, obj_vector,data_size)
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
    NUM_SAMPLES = 256  #128
    THINNING = 16

    if config.Specialized_BO == False:
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

        # Optimizes the qEI acquisition function, and returns a new candidate and observation
        ########################################################
        # Setting tilts bounds
        DIM = 57
        lower_bound = -15.0
        upper_bound = 45.0
        bounds1 = torch.cat((torch.zeros(1, DIM)+lower_bound, torch.zeros(1, DIM)+upper_bound))
        # Setting vHPBW bounds
        DIM = 57
        lower_bound = 5.0
        upper_bound = 30.0
        bounds2 = torch.cat((torch.zeros(1, DIM)+lower_bound, torch.zeros(1, DIM)+upper_bound))
        # Combine both bounds
        bounds = torch.cat((bounds1, bounds2), dim=1)

        EI = qExpectedImprovement(model=model, best_f=train_obj.max())
        candidates, acq_values = optimize_acqf(
            EI,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=1024,)

        #Recommeded tilts
        BS_tilt = tf.constant(candidates[:,0:57].numpy())
        BS_tilt = tf.expand_dims(BS_tilt,axis=2)
        BS_tilt = tf.tile(BS_tilt, [2 * config.batch_num, 1, config.Nuser_drop])
        # Recommeded powers
        BS_HPBW_v = tf.constant(candidates[:,57:].numpy())
        BS_HPBW_v = tf.expand_dims(BS_HPBW_v,axis=2)
        BS_HPBW_v = tf.tile(BS_HPBW_v, [2 * config.batch_num, 1, config.Nuser_drop])

        P_Tx_TN = tf.tile(Ptx_thresholds_vector, [2 * config.batch_num, 1, config.Nuser_drop])

    elif config.Specialized_BO == True:
        #Up-cells
        idxes = [3,4,5,10,12,21,24,29,31,33,3+57,4+57,5+57,10+57,12+57,21+57,24+57,29+57,31+57,33+57]
        #Down-cells
        # idxes = []

        train_x_org = train_x.to(torch.float64)
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
        # Setting tilts bounds
        DIM = 10
        lower_bound = 15.0
        upper_bound = 45.0
        bounds1 = torch.cat((torch.zeros(1, DIM)+lower_bound, torch.zeros(1, DIM)+upper_bound))
        # Setting vHPBW bounds
        DIM = 10
        lower_bound = 5.0
        upper_bound = 30.0
        bounds2 = torch.cat((torch.zeros(1, DIM)+lower_bound, torch.zeros(1, DIM)+upper_bound))
        # Combine both bounds
        bounds = torch.cat((bounds1, bounds2), dim=1)

        EI = qExpectedImprovement(model=model, best_f=train_obj.max())
        candidates, acq_values = optimize_acqf(
            EI,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=1024, )
        candidates = candidates.to(dtype=torch.float64)
        #Combining the optimizng indexes and non-optimized indexes together, for the simulator to run
        combined_opt_nonOpt_train_x = torch.zeros_like(train_x_org, dtype=torch.float64)
        combined_opt_nonOpt_train_x[:, idxes] = candidates
        none_optimized_indexes = list(set(range(train_x_org.shape[1])) - set(idxes))
        combined_opt_nonOpt_train_x[:, none_optimized_indexes] = train_x_org[:, none_optimized_indexes]
        candidates = combined_opt_nonOpt_train_x
        candidates = candidates[0, :].view(1, 114)

        #Recommeded tilts
        BS_tilt = tf.constant(candidates[:,0:57].numpy())
        BS_tilt = tf.expand_dims(BS_tilt,axis=2)
        BS_tilt = tf.tile(BS_tilt, [2 * config.batch_num, 1, config.Nuser_drop])
        # Recommeded powers
        BS_HPBW_v = tf.constant(candidates[:,57:].numpy())
        BS_HPBW_v = tf.expand_dims(BS_HPBW_v,axis=2)
        BS_HPBW_v = tf.tile(BS_HPBW_v, [2 * config.batch_num, 1, config.Nuser_drop])

        P_Tx_TN = tf.tile(Ptx_thresholds_vector, [2 * config.batch_num, 1, config.Nuser_drop])

    # Run simulator based on new candidates
    data = Terrestrial()
    data.alpha_factor = 0.0  # LEO at 90deg
    data.BS_tilt = tf.constant(BS_tilt.numpy(), dtype=tf.float32)
    data.BS_HPBW_v = tf.constant(BS_HPBW_v.numpy(), dtype=tf.float32)
    data.call()

    # Import distances
    D = data.D
    D_2d = data.D_2d

    # Import of the UAVs and GUEs LSG and SINR data
    LSG_UAVs_Corridors = data.LSG_UAVs_Corridors
    LSG_GUEs = data.LSG_GUEs
    sinr_TN_UAVs_Corridors = SINR.sinr_TN(LSG_UAVs_Corridors, P_Tx_TN)
    sinr_TN_GUEs = SINR.sinr_TN(LSG_GUEs, P_Tx_TN)

    # BO objective
    Rate_sumOftheLog_Obj, UAVs_Coverage_ratio, Rate_GUEs, Rate_UAVs = SINR.BO_Obj_Rates_and_Outage(LSG_GUEs,
                                                                                                   LSG_UAVs_Corridors,
                                                                                                   P_Tx_TN, D, D_2d,
                                                                                                   alpha=0.5)
    Rate_obj = Rate_sumOftheLog_Obj[0].__float__()
    new_obj = torch.tensor([[Rate_obj]], dtype=torch.double)

    # Append the thresholds and objectives
    if config.Specialized_BO == False:
        train_x = torch.cat((train_x, candidates), dim=0)
    elif config.Specialized_BO == True:
        train_x = torch.cat((train_x_org, candidates), dim=0)
        train_obj = torch.cat((train_obj, new_obj), dim=0)


    # Monitoring the progress of best observed value so far
    # best_value = tf.expand_dims(tf.reduce_max(train_obj[data_size + 1:, :], axis=0), axis=0)
    best_value = tf.expand_dims(tf.reduce_max(train_obj[data_size :, :], axis=0), axis=0)
    if i == 0:
        best_rate_so_far = tf.zeros(best_value.shape, dtype='float64')
    best_rate_so_far = tf.concat([best_rate_so_far, best_value], axis=0)

    # BO Outputs
    Thresholds = train_x[data_size:, :]
    Obj = train_obj[data_size:, :]
    Thresholds = Thresholds.numpy()
    Thresholds = tf.convert_to_tensor(Thresholds)
    Obj = Obj.numpy()
    Obj = tf.convert_to_tensor(Obj)
    best_observed_objective_value = tf.reduce_max(Obj, axis=0)
    optimum_thresholds = tf.tile(tf.cast(Obj == best_observed_objective_value, "float64"), [1, 1]) * Thresholds
    optimum_thresholds = tf.reduce_sum(optimum_thresholds, axis=0)
    Full_tilts = optimum_thresholds[0:57]
    Full_HPBW_h = optimum_thresholds[57:]
    # Saving BO data for matlab
    data_BO = {"Thresholds": Thresholds.numpy(),
               "Obj": Obj.numpy(),
               "best_observed_objective_value": best_observed_objective_value.numpy(),
               "optimum_thresholds": optimum_thresholds.numpy(),
               "best_rate_so_far": best_rate_so_far.numpy(),
               "Full_tilts": Full_tilts.numpy(),
               "Full_HPBW_h": Full_HPBW_h.numpy()}
    file_name = "2024_04_25_SAASBO_EK_Corr_tilts_vHPBW_Up_iteration{}.mat".format(i)
    savemat(file_name, data_BO)

    # d = {"SINR_UAVs": 10 * np.log10(sinr_total_UAVs.numpy()),
    #       "SINR_GUEs": 10 * np.log10(sinr_total_GUEs.numpy()),
    #       "Rate_UAVs": Rate_UAVs.numpy(),
    #       "Rate_GUEs": Rate_GUEs.numpy()}
    # savemat("2023_06_20_SINR_Rate_Alpha0_ProductRateObj.mat", d)