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


def generate_initial_data(tilts_vector, HPBW_v_vector, Ptx_thresholds_vector, obj_vector, data_size):

    train_x = HPBW_v_vector[:, :, 0]
    train_x = torch.from_numpy(train_x.numpy()).double()
    train_obj = obj_vector

    for j in range(data_size):
        if config.Specialized_BO == False:
            ##Setting Random tilts for all BSs creating a data set
            BS_HPBW_v = tf.random.uniform(HPBW_v_vector.shape, 10, 30)
            ########################################################
            ##Different sets of samples in the initial data-set
            ########################################################
            # if j >= 0 and j <= 24:
            #     BS_tilt = tf.random.uniform(tilts_vector.shape, -10.0, -5.0)
            # elif j >= 25 and j <= 49:
            #     BS_tilt = tf.random.uniform(tilts_vector.shape, 10.0, 10.0)
            # elif j >= 50 and j <= 74:
            #     BS_tilt = tf.random.uniform(tilts_vector.shape, -10.0, 10.0)
            # elif j >= 75 and j <= 100:
            #     BS_tilt = tf.random.uniform(tilts_vector.shape, -10.0, 10.0)
            #     # Define the excluded range
            #     excluded_range = tf.constant([-5.0, 5.0])  # -10,10
            #     # Mask out values within the excluded range
            #     condition = tf.logical_and(BS_tilt >= excluded_range[0], BS_tilt <= excluded_range[1])
            #     replacement_values = tf.random.uniform(tf.shape(BS_tilt), -10.0, -5.0)
            #     BS_tilt = tf.where(condition, replacement_values, BS_tilt)
            ########################################################

        if config.Specialized_BO == True:
            # Setting the uptilts according to the recommnded config
            # tilts_vector = tf.expand_dims(tf.constant([[]]), axis=2)
            BS_tilt = tilts_vector[0, :, 0]
            # Select indices to update for uptilts
            # idxes = [3,10,14,17,21,26,31,32,34,44,46]
            # Select indices to update for downtilts
            idxes = [0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 18, 19]
            Update_indices = [random.uniform(-12.0, -8.0) for _ in range(len(idxes))]
            # Update the selected indices with the new values
            for idx, Update_indice in zip(idxes, Update_indices):
                indx = tf.constant([[idx]])
                BS_tilt = tf.tensor_scatter_nd_update(BS_tilt, indx, tf.constant([Update_indice]))
            BS_tilt = tf.expand_dims(tf.expand_dims(BS_tilt, axis=0), axis=2)

        new_train_x = torch.from_numpy(BS_HPBW_v[:, :, 0].numpy()).double()

        # BS_HPBW_v = HPBW_v_vector  # This is for getting the SINR for the opt thresholds after finishing
        BS_HPBW_v = tf.tile(BS_HPBW_v, [2 * config.batch_num, 1, config.Nuser_drop])

        BS_tilt = tilts_vector  # This is for getting the SINR for the opt thresholds after finishing
        BS_tilt = tf.tile(BS_tilt, [2 * config.batch_num, 1, config.Nuser_drop])

        P_Tx_TN = Ptx_thresholds_vector  # This is for getting the SINR for the opt thresholds after finishing
        P_Tx_TN = tf.tile(P_Tx_TN, [2 * config.batch_num, 1, config.Nuser_drop])


        # Run the simulator
        data = Terrestrial()
        data.alpha_factor = 0.0  # LEO at 90deg
        data.BS_tilt = BS_tilt
        data.BS_HPBW_v = BS_HPBW_v
        data.call()
        # Import of the UAVs and GUEs LSG and SINR data
        Xuser_GUEs = data.Xuser_GUEs
        Xuser_UAVs = data.Xuser_UAVs
        LSG_UAVs_Corridors = data.LSG_UAVs_Corridors
        LSG_GUEs = data.LSG_GUEs
        sinr_TN_UAVs_Corridors = SINR.sinr_TN(LSG_UAVs_Corridors, P_Tx_TN)
        sinr_TN_GUEs = SINR.sinr_TN(LSG_GUEs, P_Tx_TN)

        # BO objective: Sum of log of the RSS
        SINR_sumOftheLog_Obj, Rate_sumOftheLog_Obj, sinr_total_UAVs, sinr_total_GUEs = SINR.BO_Multi_Obj_Cooridor(
            sinr_TN_UAVs_Corridors, sinr_TN_GUEs, alpha=0.5)
        Rate_sumOftheLog_Obj, Rate_GUEs, Rate_UAVs = SINR.BO_Obj_Rates_and_Outage(LSG_GUEs, LSG_UAVs_Corridors, P_Tx_TN,
                                                                                  alpha=0.5)
        Rate_obj = Rate_sumOftheLog_Obj[0].__float__()
        new_obj = torch.tensor([[Rate_obj]], dtype=torch.double)

        ## Serving BSs indexes and UAVs locations
        # BSs_id_UAVs, Xuser_UAVs_x, Xuser_UAVs_y = SINR.Cell_id(LSG_UAVs_Corridors, Xuser_UAVs)
        # BSs_id_GUEs, Xuser_GUEs_x, Xuser_GUEs_y = SINR.Cell_id(LSG_GUEs, Xuser_GUEs)

        # Append the thresholds and objectives
        train_x = torch.cat((train_x, new_train_x), dim=0)
        train_obj = torch.cat((train_obj, new_obj), dim=0)

    # Save the torch tensors to a file with .pt extension to be loaded using python later
    # file_name = "2023_09_25_Corr_SAforSB_SAonly_Down.pt"
    # torch.save({"train_x": train_x, "train_obj": train_obj}, file_name)

    return train_x, train_obj


# Run BO loop
""
BO_itertions = 100
data_size = 2

# Initial tilts and powers and obj value
# tilts_vector = tf.expand_dims(tf.expand_dims(tf.random.uniform((57,), 0.0, 0.0, tf.float32), axis=0),axis=2)-12.0
tilts_vector = tf.expand_dims(tf.constant([[
    -13.8631, - 9.8419, - 12.8234,   33.5664, - 9.9551,   29.5257, - 12.0700,   17.4982, - 13.0527, - 11.9808,
    24.8476, - 13.2057,   35.8744, - 12.1665, - 8.0000, - 10.5512,   22.0820, - 12.5277, - 14.1143,   23.1800,
    25.8866,   29.8430,   37.1068, - 13.1227, - 13.5094, - 8.9340, - 12.4654, - 12.8602, - 13.1440,   35.7825,
    - 13.4270, - 12.0730, - 13.0696,   27.5800, - 11.6572, - 12.9629, - 12.6227, - 9.4583, - 11.2001, - 13.3936,
    - 10.7517, - 12.6729, - 10.1889, - 11.8640, - 9.7783, - 12.0929,   43.2390, - 12.6582, - 14.1254,   23.8554,
    - 12.1073, - 11.6695, - 13.6616,   43.4639, - 12.4544, - 12.4960, - 10.4618]]), axis=2)
HPBW_v_vector = tf.expand_dims(tf.expand_dims(tf.random.uniform((57,), 10.0, 10.0, tf.float32), axis=0),axis=2)
Ptx_thresholds_vector = tf.expand_dims(tf.expand_dims(tf.random.uniform((57,), 46.0, 46.0, tf.float32), axis=0),axis=2)
obj_vector = torch.tensor([[2.0075]], dtype=torch.double)

# Creat the training data-set
train_x, train_obj = generate_initial_data(tilts_vector, HPBW_v_vector, Ptx_thresholds_vector, obj_vector,data_size)
# train_x = train_x[1:,:]
# train_obj = train_obj[1:,:]
## Load the training data-set
# file_name = "2023_09_25_Corr_SAforSB_SAonly_Down.pt"
# loaded_data = torch.load(file_name)
# train_x = loaded_data["train_x"]
# train_obj = loaded_data["train_obj"]
# train_x = train_x[1:, :]
# train_obj = train_obj[1:, :]
# train_x_SA = loaded_data["train_x"]
# train_obj_SA = loaded_data["train_obj"]
# train_x_SA = train_x_SA[1:,:]
# train_obj_SA = train_obj_SA[1:,:]
# train_x = torch.cat((train_x, train_x_SA), dim=0)
# train_obj = torch.cat((train_obj, train_obj_SA), dim=0)

# Start BO iterating
for i in tqdm(range(BO_itertions)):

    # Creating the surrogate model
    ########################################################
    WARMUP_STEPS = 512  # 256
    NUM_SAMPLES = 256  # 128
    THINNING = 16

    if config.Specialized_BO == False:

        train_x_org = train_x

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
        DIM = 57
        lower_bound = 10.0
        upper_bound = 30.0
        bounds = torch.cat((torch.zeros(1, DIM) + lower_bound, torch.zeros(1, DIM) + upper_bound))

        EI = qExpectedImprovement(model=model, best_f=train_obj.max())
        candidates, acq_values = optimize_acqf(
            EI,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=1024, )

        BS_HPBW_v = tf.constant(candidates.numpy())
        BS_HPBW_v = tf.expand_dims(BS_HPBW_v, axis=2)
        BS_HPBW_v = tf.tile(BS_HPBW_v, [2 * config.batch_num, 1, config.Nuser_drop])

    elif config.Specialized_BO == True:
        # Up-cells
        # idxes = [3,10,14,17,21,26,31,32,34,44,46]
        ##Down-cells
        idxes = [0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 18, 19, 20, 22, 23, 24, 25, 27, 28, 29, 30, 33, 35, 36,
                 37, 38, 39, 40, 41, 42, 43, 45, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56]

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

        # Combining the optimizng indexes and non-optimized indexes together, for the simulator to run
        combined_opt_nonOpt_train_x = torch.zeros_like(train_x_org, dtype=torch.float32)
        combined_opt_nonOpt_train_x[:, idxes] = candidates
        none_optimized_indexes = list(set(range(train_x_org.shape[1])) - set(idxes))
        combined_opt_nonOpt_train_x[:, none_optimized_indexes] = train_x_org[:, none_optimized_indexes]
        candidates = combined_opt_nonOpt_train_x
        candidates = candidates[0, :].view(1, 57)

        BS_HPBW_v = tf.constant(candidates.numpy())
        BS_HPBW_v = tf.expand_dims(BS_tilt, axis=2)
        BS_HPBW_v = tf.tile(BS_tilt, [2 * config.batch_num, 1, config.Nuser_drop])

    # Run simulator based on new candidates
    ########################################################

    # If power and tilts are not being optimized
    P_Tx_TN = tf.tile(Ptx_thresholds_vector, [2 * config.batch_num, 1, config.Nuser_drop])
    BS_tilt = tf.tile(tilts_vector, [2 * config.batch_num, 1, config.Nuser_drop])

    data = Terrestrial()
    data.alpha_factor = 0.0  # LEO at 90deg
    data.BS_tilt = BS_tilt
    data.BS_HPBW_v = BS_HPBW_v
    data.call()

    # Import of the UAVs and GUEs LSG and SINR data
    LSG_UAVs_Corridors = data.LSG_UAVs_Corridors
    LSG_GUEs = data.LSG_GUEs
    sinr_TN_UAVs_Corridors = SINR.sinr_TN(LSG_UAVs_Corridors, P_Tx_TN)
    sinr_TN_GUEs = SINR.sinr_TN(LSG_GUEs, P_Tx_TN)

    # BO objective
    Rate_sumOftheLog_Obj, Rate_GUEs, Rate_UAVs = SINR.BO_Obj_Rates_and_Outage(LSG_GUEs, LSG_UAVs_Corridors, P_Tx_TN,
                                                                              alpha=0.5)
    Rate_obj = Rate_sumOftheLog_Obj[0].__float__()
    new_obj = torch.tensor([[Rate_obj]], dtype=torch.double)

    # Append the thresholds and objectives
    train_x = torch.cat((train_x_org, candidates), dim=0)
    train_x = train_x.to(torch.float64)
    train_obj = torch.cat((train_obj, new_obj), dim=0)

    # Monitoring the progress of best observed value so far
    best_value = tf.expand_dims(tf.reduce_max(train_obj[data_size:, :], axis=0), axis=0)
    if i == 0:
        best_rate_so_far = tf.zeros(best_value.shape, dtype='float64')
    best_rate_so_far = tf.concat([best_rate_so_far, best_value], axis=0)

    # BO Outputs
    Thresholds = train_x
    Obj = train_obj
    Thresholds = Thresholds.numpy()
    Thresholds = tf.convert_to_tensor(Thresholds)
    Obj = Obj.numpy()
    Obj = tf.convert_to_tensor(Obj)
    best_observed_objective_value = tf.reduce_max(Obj, axis=0)
    optimum_thresholds = tf.tile(tf.cast(Obj == best_observed_objective_value, "float64"), [1, 1]) * Thresholds
    optimum_thresholds = tf.reduce_sum(optimum_thresholds, axis=0)
    Full_tilts = tilts_vector[:,:,0]
    Full_powers = Ptx_thresholds_vector[:, :, 0]
    Full_HPBW_h = optimum_thresholds

    # Saving BO data for matlab
    data_BO = {"Thresholds": Thresholds.numpy(),
               "Obj": Obj.numpy(),
               "best_observed_objective_value": best_observed_objective_value.numpy(),
               "optimum_thresholds": optimum_thresholds.numpy(),
               "best_rate_so_far": best_rate_so_far.numpy(),
               "Full_tilts": Full_tilts.numpy(),
               "Full_HPBW_h": Full_HPBW_h.numpy()}
    file_name = "2023_11_08_HDBOEK_Corr_vHPBW_iteration{}.mat".format(i)
    savemat(file_name, data_BO)

    # d = {"SINR_UAVs": 10 * np.log10(sinr_total_UAVs.numpy()),
    #       "SINR_GUEs": 10 * np.log10(sinr_total_GUEs.numpy()),
    #       "Rate_UAVs": Rate_UAVs.numpy(),
    #       "Rate_GUEs": Rate_GUEs.numpy()}
    # savemat("2023_08_16_SINR_Rate_LambdaHaf_ProductRateObj_IterativeBO_FixedDownTilts.mat", d)

tilts_vector = tf.expand_dims(tf.constant([[
    -10.7028, - 13.2502, - 10.8893, - 12.9726, - 11.8176, - 11.4146, - 10.0879, - 10.6533, - 12.6312, - 12.0264,
    - 12.1142, - 12.4132, - 11.1537, - 10.8106, - 12.4998, - 11.4740, - 15.0189, - 11.0204, - 12.9443, - 12.8542,
    - 12.7713, - 12.3419, - 11.4416, - 13.2092, - 11.1103, - 10.3184, - 12.7448, - 12.5280, - 13.7621, - 13.2310,
    - 12.9288, - 10.5971, - 12.1113, - 13.5342, - 12.0637, - 12.9327, - 11.9668, - 14.3249, - 11.6192, - 12.7023,
    - 12.5337, - 12.3194, - 11.2269, - 11.1509, - 14.1286, - 10.3987, - 12.8748, - 10.2173, - 11.7782, - 12.5399,
    - 10.7587, - 10.1067, - 12.4050, - 12.6742, - 13.5555, - 11.6321, - 11.5568]]), axis=2)
#
# HPBW_v_vector = tf.expand_dims(tf.constant([[
#     10.0000   10.0000   10.0000   10.0000   10.0000   10.0000   10.0000   10.0000   10.0000   10.0000
#     10.0000   10.0000   22.1171   10.0000   10.0000   10.0000   10.0000   10.0000   10.0000   10.0000
#     10.0000   13.0620   21.3984   10.0000   10.0000   10.0000   10.0000   10.0000   10.0000   17.5808
#     10.0000   10.0000   10.0000   10.0000   10.0000   10.0000   10.0000   10.0000   10.0000   10.0000
#     10.0000   10.0000   10.0000   10.0000   10.0000   10.0000   10.0000   10.0000   10.0000   10.0000
#     10.0000   10.0000   10.0000   10.0000   10.0000   10.0000   10.0000]]), axis=2)
#
# obj_vector = torch.tensor([[2.0059]], dtype=torch.double)
