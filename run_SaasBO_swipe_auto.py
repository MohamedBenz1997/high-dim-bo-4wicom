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
def generate_initial_data(thresholds_vector, Ptx_thresholds_vector, obj_vector, data_size):

    train_x = thresholds_vector[:, :, 0]
    train_x = torch.from_numpy(train_x.numpy()).double()
    train_obj = obj_vector

    for j in range(data_size):

        #Setting Random tilts for creating a data set
        BS_tilt = tf.random.uniform(thresholds_vector.shape, -20, 0)
        new_train_x = torch.from_numpy(BS_tilt[:,:,0].numpy()).double()
        # BS_tilt = thresholds_vector  # This is for getting the SINR for the opt thresholds after finishing
        BS_tilt = tf.tile(BS_tilt, [2 * config.batch_num, 1, config.Nuser_drop])

        #Keeping the power fixed
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
        BSs_id_UAVs, Xuser_UAVs_x, Xuser_UAVs_y = SINR.Cell_id(LSG_UAVs_Corridors, Xuser_UAVs)
        BSs_id_GUEs, Xuser_GUEs_x, Xuser_GUEs_y = SINR.Cell_id(LSG_GUEs, Xuser_GUEs)

        #Append the thresholds and objectives
        train_x = torch.cat((train_x, new_train_x), dim=0)
        train_obj = torch.cat((train_obj, new_obj), dim=0)

    # Save the torch tensors to a file with .pt extension to be loaded using python later
    file_name = "2023_07_05_Alpha0_GUEs_Product_Rate_DataSet.pt"
    torch.save({"train_x": train_x, "train_obj": train_obj}, file_name)

    return train_x, train_obj

# Creating the surrogate model
########################################################
def initialize_model(train_x, train_obj):

    WARMUP_STEPS = 256
    NUM_SAMPLES = 128
    THINNING = 16

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

    return model

# Optimizes the qEI acquisition function, and returns a new candidate and observation
########################################################
def optimize_qEI_and_get_observation(model, train_obj):

    DIM = 57
    lower_bound = -20.0
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

    return candidates, BS_tilt

# Run BO loop
########################################################
BO_itertions = 100
data_size = 100

#Initial tilts and powers and obj value
thresholds_vector = tf.expand_dims(tf.expand_dims(tf.random.uniform((57,), 0.0, 0.0, tf.float32), axis=0),axis=2)
# # Alpha0 Best (obj=4.78) Prior 912 Samples
# thresholds_vector = tf.expand_dims(tf.constant([[
#     -11.7359, -12.7983, -10.7167, -11.4129, -10.4888, -11.2637, -11.3429, -12.4476, -11.5424, -10.8629,
#     -10.6428, -11.2177, -11.0399, -11.1755, -11.0855, -11.4278, -11.4093, -12.2184, -10.6219, -12.0062,
#     -12.1743, -12.4214, -13.1057, -12.5554, -11.9632, -12.4047, -12.1101, -12.3828, -12.1266, -13.3123,
#     -12.7162, -10.9860, -11.8068, -13.0786, -12.4863, -12.8726, -12.2590, -13.3431, -11.6485, -11.1929,
#     -11.9571, -11.2089, -11.5908, -10.6035, -11.0119, -11.4163, -11.3212, -12.0278, -11.1372, -10.8733,
#     -10.8399, -10.5011, -10.3708, -10.1331, -12.8212, -11.9515, -11.6880]]), axis=2)
Ptx_thresholds_vector = tf.expand_dims(tf.expand_dims(tf.random.uniform((57,), 46.0, 46.0, tf.float32), axis=0),axis=2)
obj_vector = torch.tensor([[-2.13]], dtype=torch.double)

# Creat the training data-set
train_x, train_obj = generate_initial_data(thresholds_vector, Ptx_thresholds_vector, obj_vector,data_size)

# # Load the training data-set
# file_name = "2023_07_05_Alpha0_GUEs_Product_Rate_DataSet.pt"
# loaded_data = torch.load(file_name)
# train_x = loaded_data["train_x"]
# train_obj = loaded_data["train_obj"]

#Start iterating
for i in tqdm(range(BO_itertions)):

    #Surrogate Model
    model = initialize_model(train_x, train_obj)

    #Obtain candidate via acquetion function
    candidates, BS_tilt = optimize_qEI_and_get_observation(model, train_obj)

    #Run simulator based on new candidates
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
    Full_tilts = optimum_thresholds

    # Saving BO data for matlab
    data_BO = {"Thresholds": Thresholds.numpy(),
               "Obj": Obj.numpy(),
               "best_observed_objective_value": best_observed_objective_value.numpy(),
               "optimum_thresholds": optimum_thresholds.numpy(),
               "best_rate_so_far": best_rate_so_far.numpy(),
               "Full_tilts": Full_tilts.numpy()}
    file_name = "2023_07_07_HighDim_BO_tilt_Alpha0_GUEs_Product_Rate_iteration{}_set1.mat".format(i)
    savemat(file_name, data_BO)

    d = {"SINR_UAVs": 10 * np.log10(sinr_total_UAVs.numpy()),
          "SINR_GUEs": 10 * np.log10(sinr_total_GUEs.numpy()),
          "Rate_UAVs": Rate_UAVs.numpy(),
          "Rate_GUEs": Rate_GUEs.numpy()}
    savemat("2023_06_20_SINRgi_Rate_Alpha0_ProductRateObj.mat", d)