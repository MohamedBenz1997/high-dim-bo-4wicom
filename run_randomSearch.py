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
from torch.utils.tensorboard import SummaryWriter

from TerrestrialClass import Terrestrial
from SinrClass import SINR
from config import Config
from plot_class import Plot


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

#Auto genrating data-sets for BO
""
def generate_initial_data(thresholds_vector, Ptx_thresholds_vector, obj_vector):

    train_x = thresholds_vector
    train_x = torch.from_numpy(train_x.numpy()).double()
    train_obj = obj_vector
    train_obj = torch.from_numpy(train_obj.numpy()).double()

    for j in tqdm(range(200)):
        new_train_x1 = tf.random.uniform(thresholds_vector.shape, -25.0, 0.0, tf.float32)
        BS_tilt = new_train_x1
        BS_tilt = thresholds_vector #This is for getting the SINR for the opt thresholds after finishing
        BS_tilt = tf.tile(BS_tilt, [2 * config.batch_num, 1, config.Nuser_drop])

        new_train_x2 = tf.random.uniform(Ptx_thresholds_vector.shape, 40.0, 46.0, tf.float32)
        P_Tx_TN = new_train_x2
        P_Tx_TN = Ptx_thresholds_vector #This is for getting the SINR for the opt thresholds after finishing
        P_Tx_TN = tf.tile(P_Tx_TN, [2 * config.batch_num, 1, config.Nuser_drop])

        new_train_x = tf.concat([new_train_x1, new_train_x2], axis=1)
        new_train_x = torch.from_numpy(new_train_x.numpy()).double()
        if config.IterativeBO_1Threshold == True:
            new_train_x = new_train_x1
            new_train_x = torch.from_numpy(new_train_x.numpy()).double()

        # Run the simulator
        data = Terrestrial()
        data.alpha_factor = 0.0  # LEO at 90deg
        data.BS_tilt = BS_tilt
        data.call()

        # ------------ Import of the UAVs and GUEs LSG and SINR data
        Xuser_GUEs = data.Xuser_GUEs
        Xuser_UAVs = data.Xuser_UAVs
        LSG_UAVs_Corridors = data.LSG_UAVs_Corridors
        LSG_GUEs = data.LSG_GUEs
        sinr_TN_UAVs_Corridors = SINR.sinr_TN(LSG_UAVs_Corridors, P_Tx_TN)
        sinr_TN_GUEs = SINR.sinr_TN(LSG_GUEs, P_Tx_TN)

        #BO objective: Sum of log of the RSS
        SINR_sumOftheLog_Obj, Rate_sumOftheLog_Obj, sinr_total_UAVs, sinr_total_GUEs = SINR.BO_Multi_Obj_Cooridor(sinr_TN_UAVs_Corridors, sinr_TN_GUEs, alpha=0.0)
        Rate_sumOftheLog_Obj, Rate_GUEs, Rate_UAVs = SINR.BO_Obj_Rates_and_Outage(LSG_GUEs, LSG_UAVs_Corridors, P_Tx_TN, alpha=0.0)
        SINR_obj = Rate_sumOftheLog_Obj[0].__float__()
        Rate_obj = Rate_sumOftheLog_Obj[0].__float__()
        new_obj1 = SINR_obj
        new_obj2 = Rate_obj
        new_obj = torch.tensor([[new_obj1, new_obj2]], dtype=torch.double)

        train_x = torch.cat((train_x, new_train_x), dim=0)
        train_obj = torch.cat((train_obj, new_obj), dim=0)

        #updating best observed KPI
        best_value = tf.expand_dims(tf.reduce_max(train_obj, axis=0), axis=0)
        if j == 0:
            best_value_all = tf.zeros(best_value.shape, dtype='float64')
        best_value_all = tf.concat([best_value_all, best_value], axis=0)

    return train_x, train_obj, best_value_all

#Initial config
thresholds_vector = tf.expand_dims(tf.expand_dims(tf.random.uniform((57,), 0.0, 0.0, tf.float32), axis=0),axis=2)-12.0
Ptx_thresholds_vector = tf.expand_dims(tf.expand_dims(tf.random.uniform((57,), 46.0, 46.0, tf.float32), axis=0),axis=2)
obj_vector = torch.tensor([[-0.2529, -0.2529]], dtype=torch.double)


#Obtain the training data-set
train_x, train_obj, best_value_all = generate_initial_data(thresholds_vector, Ptx_thresholds_vector, obj_vector)

#Outputs
Thresholds = train_x[:,:,0]
Obj = train_obj
best_observed_objective_value = tf.reduce_max(Obj, axis=0)
optimum_thresholds = tf.tile(tf.expand_dims(tf.cast(Obj[:,0] == best_observed_objective_value[0], "float64"),axis=1), [1, 57]) * Thresholds
optimum_thresholds = tf.reduce_sum(optimum_thresholds, axis=0)
opt_values = optimum_thresholds
thresholds_vector = tf.expand_dims(tf.expand_dims(opt_values, axis=0), axis=2)
obj_vector = torch.from_numpy(tf.expand_dims(best_observed_objective_value, axis=0).numpy())
Full_tilts = thresholds_vector[:,:,0]

# Saving for matlab
data_BO = {"Thresholds": Thresholds.numpy(),
                        "Obj": Obj.numpy(),
                        "best_observed_objective_value": best_observed_objective_value.numpy(),
                        "optimum_thresholds": optimum_thresholds.numpy(),
                        "best_rate_so_far": best_value_all.numpy(),
                        "Full_tilts": Full_tilts.numpy()}
file_name = "2023_09_29_RandomSearch_2Tier_GUEs.mat"
savemat(file_name, data_BO)


