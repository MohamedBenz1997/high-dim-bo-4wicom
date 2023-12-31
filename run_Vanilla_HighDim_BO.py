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

        ##Setting Random tilts for all BSs creating a data set
        BS_tilt = tf.random.uniform(thresholds_vector.shape, -25, 0)
        ########################################################
        ##Different sets of samples in the initial data-set
        ########################################################
        """
        if j>=0 and j<=24:
            BS_tilt = tf.random.uniform(thresholds_vector.shape, -18.0, -10.0)
        elif j>=25 and j<=49:
            BS_tilt = tf.random.uniform(thresholds_vector.shape, 20.0, 35.0)
        elif j >= 50 and j <= 74:
            BS_tilt = tf.random.uniform(thresholds_vector.shape, -18.0, 35.0)
        elif j >= 75 and j <= 100:
            BS_tilt = tf.random.uniform(thresholds_vector.shape, -18.0, 35.0)
            # Define the excluded range
            excluded_range = tf.constant([-9.9, 19.9])
            # Mask out values within the excluded range
            condition = tf.logical_and(BS_tilt >= excluded_range[0], BS_tilt <= excluded_range[1])
            replacement_values = tf.random.uniform(tf.shape(BS_tilt), -18.0, -10.0)
            BS_tilt = tf.where(condition, replacement_values, BS_tilt)
        """

        new_train_x = torch.from_numpy(BS_tilt[:,:,0].numpy()).double()
        #BS_tilt = thresholds_vector  # This is for getting the SINR for the opt thresholds after finishing
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
        SINR_sumOftheLog_Obj, Rate_sumOftheLog_Obj, sinr_total_UAVs, sinr_total_GUEs = SINR.BO_Multi_Obj_Cooridor(sinr_TN_UAVs_Corridors, sinr_TN_GUEs, alpha=0.0)
        Rate_sumOftheLog_Obj, Rate_GUEs, Rate_UAVs = SINR.BO_Obj_Rates_and_Outage(LSG_GUEs, LSG_UAVs_Corridors, P_Tx_TN,alpha=0.0)
        Rate_obj = Rate_sumOftheLog_Obj[0].__float__()
        new_obj = torch.tensor([[Rate_obj]], dtype=torch.double)

        ## Serving BSs indexes and UAVs locations
        #BSs_id_UAVs, Xuser_UAVs_x, Xuser_UAVs_y = SINR.Cell_id(LSG_UAVs_Corridors, Xuser_UAVs)
        #BSs_id_GUEs, Xuser_GUEs_x, Xuser_GUEs_y = SINR.Cell_id(LSG_GUEs, Xuser_GUEs)

        #Append the thresholds and objectives
        train_x = torch.cat((train_x, new_train_x), dim=0)
        train_obj = torch.cat((train_obj, new_obj), dim=0)

    # Save the torch tensors to a file with .pt extension to be loaded using python later
    #file_name = "2023_09_01_Mix_Corr_ProductRate_100m_DataSet.pt"
    #torch.save({"train_x": train_x, "train_obj": train_obj}, file_name)

    return train_x, train_obj

# Run BO loop
########################################################
BO_itertions = 100
data_size = 100

#Initial tilts and powers and obj value
thresholds_vector = tf.expand_dims(tf.expand_dims(tf.random.uniform((21,), 0.0, 0.0, tf.float32), axis=0),axis=2)

#test
# thresholds_vector = tf.expand_dims(tf.constant([[
#     -10.2166, - 5.6441, - 7.8328, - 10.3622,   28.8273,   24.8715, - 20.5624, - 4.9725, - 5.3742, - 10.2022,
#     - 12.1339, - 4.1592, - 16.1180, - 11.1611, - 13.3262, - 21.8899,   29.3800, - 19.1746,   23.8064, - 11.0432,
#     - 4.8618,   22.1615,   34.2849, - 9.5840,   16.4401, - 16.7987,   18.0865, - 9.7250, - 9.4683,   31.3820,
#     - 10.1557,   34.6480,   24.9927, - 7.3283, - 5.9540, - 15.2060, - 14.8842, - 14.6028,   26.6185, - 4.9412,
#     - 12.8872, - 7.3110, - 5.3984, - 8.0392, - 7.5671, - 6.3992, - 5.4704, - 5.9519, - 6.7554, - 5.6645,
#     - 6.4249, - 4.9947, - 7.2825, - 9.0928, - 7.8785, - 22.5504, - 16.3658,]]), axis=2)


Ptx_thresholds_vector = tf.expand_dims(tf.expand_dims(tf.random.uniform((21,), 46.0, 46.0, tf.float32), axis=0),axis=2)
obj_vector = torch.tensor([[-0.2529]], dtype=torch.double) #-4.66

# Creat the training data-set
train_x, train_obj = generate_initial_data(thresholds_vector, Ptx_thresholds_vector, obj_vector,data_size)
# # Load the training data-set
# file_name = "2023_09_01_Mix_Corr_ProductRate_100m_DataSet.pt"
# loaded_data = torch.load(file_name)
# train_x = loaded_data["train_x"]
# train_obj = loaded_data["train_obj"]

train_x = train_x[1:,:]
train_obj = train_obj[1:,:]

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
        DIM = 21
        lower_bound = -25.0
        upper_bound = 0.0
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
    Rate_sumOftheLog_Obj, Rate_GUEs, Rate_UAVs = SINR.BO_Obj_Rates_and_Outage(LSG_GUEs, LSG_UAVs_Corridors, P_Tx_TN,alpha=0.0)
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
    file_name = "2023_09_28_Vanilla_HighDimBO_OneTier_GUEs_iteration{}.mat".format(i)
    savemat(file_name, data_BO)
