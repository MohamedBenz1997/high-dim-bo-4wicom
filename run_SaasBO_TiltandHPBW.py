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
            ########################################################

        if config.Specialized_BO == True:
            # Setting all tilts to 0
            # BS_tilt = tf.random.uniform(tilts_vector[0, :, 0].shape, -10.0, -10.0)
            # Setting the uptilts according to the recommnded config
            # tilts_vector = tf.expand_dims(tf.constant([[]]), axis=2)
            BS_tilt = tilts_vector[0, :, 0]
            # Select indices to update for uptilts
            idxes = [3,10,14,17,21,26,31,32,34,44,46]
            # Select indices to update for downtilts
            # idxes = [0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 18, 19, 20, 22, 23]
            Update_indices = [random.uniform(-12.0, -8.0) for _ in range(len(idxes))]
            # Update the selected indices with the new values
            for idx, Update_indice in zip(idxes, Update_indices):
                indx = tf.constant([[idx]])
                BS_tilt = tf.tensor_scatter_nd_update(BS_tilt, indx, tf.constant([Update_indice]))
            BS_tilt = tf.expand_dims(tf.expand_dims(BS_tilt, axis=0), axis=2)

        #Setting Random tilts for creating a data set
        new_train_x1 = torch.from_numpy(BS_tilt[:,:,0].numpy()).double()
        # BS_tilt = tilts_vector  # This is for getting the SINR for the opt thresholds after finishing
        BS_tilt = tf.tile(BS_tilt, [2 * config.batch_num, 1, config.Nuser_drop])

        #Setting Random powers for creating a data set
        BS_HPBW_v = tf.random.uniform(HPBW_v_vector.shape, 5, 30)
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
BO_itertions = 4
data_size = 2

#Initial tilts and powers and obj value
tilts_vector = tf.expand_dims(tf.expand_dims(tf.random.uniform((57,), 0.0, 0.0, tf.float32), axis=0),axis=2)
HPBW_v_vector = tf.expand_dims(tf.expand_dims(tf.random.uniform((57,), 10.0, 10.0, tf.float32), axis=0),axis=2)
Ptx_thresholds_vector = tf.expand_dims(tf.expand_dims(tf.random.uniform((57,), 46.0, 46.0, tf.float32), axis=0),axis=2)
obj_vector = torch.tensor([[-4.4779]], dtype=torch.double)

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

    #Run simulator based on new candidates
    data = Terrestrial()
    data.alpha_factor = 0.0  # LEO at 90deg
    data.BS_tilt = BS_tilt
    data.BS_HPBW_v = BS_HPBW_v
    data.call()

    # Import distances
    D = data.D
    D_2d = data.D_2d

    #Import of the UAVs and GUEs LSG and SINR data
    LSG_UAVs_Corridors = data.LSG_UAVs_Corridors
    LSG_GUEs = data.LSG_GUEs
    sinr_TN_UAVs_Corridors = SINR.sinr_TN(LSG_UAVs_Corridors, P_Tx_TN)
    sinr_TN_GUEs = SINR.sinr_TN(LSG_GUEs, P_Tx_TN)

    # BO objective
    Rate_sumOftheLog_Obj, UAVs_Coverage_ratio, Rate_GUEs, Rate_UAVs = SINR.BO_Obj_Rates_and_Outage(LSG_GUEs, LSG_UAVs_Corridors, P_Tx_TN, D, D_2d, alpha=0.5)
    Rate_obj = Rate_sumOftheLog_Obj[0].__float__()
    new_obj = torch.tensor([[Rate_obj]], dtype=torch.double)

    # Append the thresholds and objectives
    train_x = torch.cat((train_x, candidates), dim=0)
    train_obj = torch.cat((train_obj, new_obj), dim=0)

    #Monitoring the progress of best observed value so far
    best_value = tf.expand_dims(tf.reduce_max(train_obj[data_size+1:,:], axis=0), axis=0)
    if i == 0:
        best_rate_so_far = tf.zeros(best_value.shape, dtype='float64')
    best_rate_so_far = tf.concat([best_rate_so_far, best_value], axis=0)

    # BO Outputs
    Thresholds = train_x[data_size+1:,:]
    Obj = train_obj[data_size+1:,:]
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
    file_name = "2024_04_16_SAASBO_Corr_tilts_vHPBW_iteration{}.mat".format(i)
    savemat(file_name, data_BO)

    # d = {"SINR_UAVs": 10 * np.log10(sinr_total_UAVs.numpy()),
    #       "SINR_GUEs": 10 * np.log10(sinr_total_GUEs.numpy()),
    #       "Rate_UAVs": Rate_UAVs.numpy(),
    #       "Rate_GUEs": Rate_GUEs.numpy()}
    # savemat("2023_06_20_SINR_Rate_Alpha0_ProductRateObj.mat", d)