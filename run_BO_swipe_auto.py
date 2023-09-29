"""
This is the Runner script for BO auto swiping over 2-thresholds over the entire tier
@authors: Mohamed Benzaghta
"""

import os
os.system("export MKL_DEBUG_CPU_TYPE=5")
import tensorflow as tf
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from TerrestrialClass import Terrestrial
from SinrClass import SINR
from config import Config
from plot_class import Plot

from botorch.models import FixedNoiseGP, ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_model
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.sampling import SobolQMCNormalSampler

from botorch.models.transforms.outcome import Standardize
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.multi_objective.box_decompositions.dominated import (DominatedPartitioning,)

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
def generate_initial_data(idx_1, thresholds_vector, Ptx_thresholds_vector, obj_vector):

    T_1 = tf.expand_dims(thresholds_vector[:, idx_1, 0], axis=1)
    T_2 = tf.expand_dims(Ptx_thresholds_vector[:, idx_1, 0], axis=1)
    train_x = tf.concat([T_1, T_2], axis=1)
    if config.IterativeBO_1Threshold == True:
        train_x = T_1
    train_x = torch.from_numpy(train_x.numpy()).double()
    train_obj = obj_vector

    for j in range(10):
        BS_tilt = thresholds_vector[0,:,0]
        P_Tx_TN = Ptx_thresholds_vector[0, :, 0]
        indx = tf.constant([[idx_1]])

        rand_values = tf.constant([random.uniform(-25.0, 0.0)])
        new_train_x1 = torch.from_numpy( tf.expand_dims(rand_values, axis=0).numpy()).double()
        BS_tilt = tf.tensor_scatter_nd_update(BS_tilt, indx, rand_values)
        BS_tilt = tf.expand_dims(tf.expand_dims(BS_tilt,axis=0),axis=2)
        #BS_tilt = thresholds_vector #This is for getting the SINR for the opt thresholds after finishing
        BS_tilt = tf.tile(BS_tilt, [2 * config.batch_num, 1, config.Nuser_drop])

        rand_values1 = tf.constant([random.uniform(40.0, 46.0)])
        new_train_x2 = torch.from_numpy( tf.expand_dims(rand_values1, axis=0).numpy()).double()
        P_Tx_TN = tf.tensor_scatter_nd_update(P_Tx_TN, indx, rand_values1)
        P_Tx_TN = tf.expand_dims(tf.expand_dims(P_Tx_TN,axis=0),axis=2)
        P_Tx_TN = Ptx_thresholds_vector #This is for getting the SINR for the opt thresholds after finishing
        P_Tx_TN = tf.tile(P_Tx_TN, [2 * config.batch_num, 1, config.Nuser_drop])

        new_train_x = torch.cat((new_train_x1, new_train_x2), dim=1)
        if config.IterativeBO_1Threshold == True:
            new_train_x = new_train_x1

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

        ## Serving BSs indexes and UAVs locations
        #BSs_id_UAVs, Xuser_UAVs_x, Xuser_UAVs_y = SINR.Cell_id(LSG_UAVs_Corridors, Xuser_UAVs)
        #BSs_id_GUEs, Xuser_GUEs_x, Xuser_GUEs_y = SINR.Cell_id(LSG_GUEs, Xuser_GUEs)

    return train_x, train_obj
""

#Creating the surrogate model
""
def initialize_model(train_x, train_obj):
    models = []
    for i in range(train_obj.shape[-1]):
        train_y = train_obj[..., i:i+1]
        train_yvar = torch.full_like(train_y, NOISE_SE[i] ** 2)
        models.append(
            FixedNoiseGP(train_x, train_y, train_yvar, outcome_transform=Standardize(m=1))
        )
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    return mll, model
""
#Optimizes the qNEHVI acquisition function, and returns a new candidate and observation.
""
def optimize_qnehvi_and_get_observation(model, train_x, sampler, idx_1, thresholds_vector, Ptx_thresholds_vector):
    acq_func = qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        X_baseline=normalize(train_x, bounds),
        prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
        sampler=sampler,)
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,)
    new_x =  unnormalize(candidates.detach(), bounds=bounds)
    new_x_ref = new_x
    #2-Updated thresholds
    thresholds_vector_ref = thresholds_vector[0, :, 0]
    Ptx_thresholds_vector_ref = Ptx_thresholds_vector[0, :, 0]
    indx = tf.constant([[idx_1]])
    if config.IterativeBO_1Threshold == False:
        opt_values = tf.constant([new_x[0][0].__float__()])
        opt_values1 = tf.constant([new_x[0][1].__float__()])
        BS_tilt = tf.tensor_scatter_nd_update(thresholds_vector_ref, indx, opt_values)
        BS_tilt = tf.expand_dims(tf.expand_dims(BS_tilt, axis=0), axis=2)
        BS_tilt = tf.tile(BS_tilt, [2 * config.batch_num, 1, config.Nuser_drop])
        P_Tx_TN = tf.tensor_scatter_nd_update(Ptx_thresholds_vector_ref, indx, opt_values1)
        P_Tx_TN = tf.expand_dims(tf.expand_dims(P_Tx_TN, axis=0), axis=2)
        P_Tx_TN = tf.tile(P_Tx_TN, [2 * config.batch_num, 1, config.Nuser_drop])

    if config.IterativeBO_1Threshold == True:
        opt_values = tf.constant([new_x[0][0].__float__()])
        BS_tilt = tf.tensor_scatter_nd_update(thresholds_vector_ref, indx, opt_values)
        BS_tilt = tf.expand_dims(tf.expand_dims(BS_tilt, axis=0), axis=2)
        BS_tilt = tf.tile(BS_tilt, [2 * config.batch_num, 1, config.Nuser_drop])
        P_Tx_TN = tf.tile(Ptx_thresholds_vector, [2 * config.batch_num, 1, config.Nuser_drop])


    #Run the simulator
    data = Terrestrial()
    data.alpha_factor = 0.0 #LEO at 90deg
    data.BS_tilt = BS_tilt
    data.call()

    # ------------ Import of the UAVs and GUEs LSG and SINR data
    LSG_UAVs_Corridors = data.LSG_UAVs_Corridors
    LSG_GUEs = data.LSG_GUEs
    sinr_TN_UAVs_Corridors = SINR.sinr_TN(LSG_UAVs_Corridors, P_Tx_TN)
    sinr_TN_GUEs = SINR.sinr_TN(LSG_GUEs, P_Tx_TN)

    # BO objective: Sum of log of the RSS
    Rate_sumOftheLog_Obj, Rate_GUEs, Rate_UAVs = SINR.BO_Obj_Rates_and_Outage(LSG_GUEs, LSG_UAVs_Corridors, P_Tx_TN, alpha=0.0)
    SINR_obj = Rate_sumOftheLog_Obj[0].__float__()
    Rate_obj = Rate_sumOftheLog_Obj[0].__float__()
    new_obj1 = SINR_obj
    new_obj2 = Rate_obj
    new_obj = torch.tensor([[new_obj1, new_obj2]], dtype=torch.double)
    return new_x_ref, new_obj
""
#Start BO swiping experiment
#############################################################
# BO initial variables
N_BATCH = 30
MC_SAMPLES = 128
BATCH_SIZE = 1
NUM_RESTARTS = 10
RAW_SAMPLES = 512
NOISE_SE = torch.tensor([0.001, 0.001], dtype=torch.double)

#2Thresholds bounds
bounds_lower = torch.tensor([[-20.0, 44.0]], dtype=torch.double)
bounds_higher = torch.tensor([[30.0, 46.0]], dtype=torch.double)
bounds = torch.cat((bounds_lower, bounds_higher), 0)
standard_bounds_lower = torch.tensor([[0.0]], dtype=torch.double)
standard_bounds_lower = standard_bounds_lower.tile((2,))
standard_bounds_higher = torch.tensor([[1.0]], dtype=torch.double)
standard_bounds_higher = standard_bounds_higher.tile((2,))
standard_bounds = torch.cat((standard_bounds_lower, standard_bounds_higher), 0)

if config.IterativeBO_1Threshold == True:
    bounds_lower = torch.tensor([[-25.0]], dtype=torch.double)
    bounds_higher = torch.tensor([[0.0]], dtype=torch.double)
    bounds = torch.cat((bounds_lower, bounds_higher), 0)
    standard_bounds_lower = torch.tensor([[0.0]], dtype=torch.double)
    standard_bounds_higher = torch.tensor([[1.0]], dtype=torch.double)
    standard_bounds = torch.cat((standard_bounds_lower, standard_bounds_higher), 0)

#ref_point for EHVI
ref_point = torch.tensor([-1.0, -1.0], dtype=torch.double)

BS_id = list(range(200))
random.shuffle(BS_id)

for iteration, i in enumerate(tqdm(BS_id)):

    idx_1 = i%57

    #For 1st iteration
    if i == BS_id[0]:
        thresholds_vector = tf.expand_dims(tf.expand_dims(tf.random.uniform((57,), 0.0, 0.0, tf.float32), axis=0),axis=2)
        Ptx_thresholds_vector = tf.expand_dims(tf.expand_dims(tf.random.uniform((57,), 46.0, 46.0, tf.float32), axis=0),axis=2)

        #test
        obj_vector = torch.tensor([[-0.2529, -0.2529]], dtype=torch.double)


    #Obtain the training data-set
    train_x, train_obj = generate_initial_data(idx_1, thresholds_vector, Ptx_thresholds_vector, obj_vector)
    train_x_qnehvi, train_obj_qnehvi = train_x, train_obj

    # call functions to generate initial training data and initialize model
    mll_qnehvi, model_qnehvi = initialize_model(train_x_qnehvi, train_obj_qnehvi)
    hvs_qnehvi = []

    # compute hypervolume
    bd = DominatedPartitioning(ref_point=ref_point, Y=train_obj_qnehvi)
    volume = bd.compute_hypervolume().item()
    hvs_qnehvi.append(volume)

    # The BO loop
    for k in range(N_BATCH):
        # fit the models
        fit_gpytorch_model(mll_qnehvi)
        # define the qEI and qNEI acquisition modules using a QMC sampler
        qnehvi_sampler = SobolQMCNormalSampler(MC_SAMPLES)
        # optimize acquisition functions and get new observations
        new_x_qnehvi, new_obj_qnehvi = optimize_qnehvi_and_get_observation(model_qnehvi, train_x_qnehvi, qnehvi_sampler, idx_1, thresholds_vector, Ptx_thresholds_vector)
        # update training points
        train_x_qnehvi = torch.cat([train_x_qnehvi, new_x_qnehvi])
        train_obj_qnehvi = torch.cat([train_obj_qnehvi, new_obj_qnehvi])
        # update progress
        for hvs_list, train_obj in zip(
                (hvs_qnehvi),
                (train_obj_qnehvi,
                 ),
        ):
            # compute hypervolume
            bd = DominatedPartitioning(ref_point=ref_point, Y=train_obj)
            volume = bd.compute_hypervolume().item()
            hvs_qnehvi.append(volume)
        # reinitialize the models so they are ready for fitting on next iteration
        # Note: we find improved performance from not warm starting the model hyperparameters
        # using the hyperparameters from the previous iteration
        mll_qnehvi, model_qnehvi = initialize_model(train_x_qnehvi, train_obj_qnehvi)
        # update progress
        best_value_qNEHVI = tf.expand_dims(tf.reduce_max(train_obj_qnehvi, axis=0), axis=0)
        if k == 0:
            best_value_qNEHVI_all = tf.zeros(best_value_qNEHVI.shape, dtype='float64')
        best_value_qNEHVI_all = tf.concat([best_value_qNEHVI_all, best_value_qNEHVI], axis=0)

        # BO Outputs
        Thresholds = train_x_qnehvi
        Obj = train_obj_qnehvi
        Thresholds = Thresholds.numpy()
        Thresholds = tf.convert_to_tensor(Thresholds)
        Obj = Obj.numpy()
        Obj = tf.convert_to_tensor(Obj)
        best_observed_objective_value = tf.reduce_max(Obj, axis=0)
        optimum_thresholds = tf.tile(tf.cast(Obj == best_observed_objective_value, "float64"), [1, 1]) * Thresholds
        optimum_thresholds = tf.reduce_sum(optimum_thresholds, axis=0)

    #Update the threshold vector and obj vector with BO optimum value
    thresholds_vector_ref = thresholds_vector[0, :, 0]
    indx = tf.constant([[idx_1]])
    # opt_values = tf.constant([optimum_thresholds[0].__float__(), optimum_thresholds[1].__float__()])
    opt_values = tf.constant([optimum_thresholds[0].__float__()])
    thresholds_vector = tf.tensor_scatter_nd_update(thresholds_vector_ref, indx, opt_values)
    thresholds_vector = tf.expand_dims(tf.expand_dims(thresholds_vector, axis=0), axis=2)



    obj_vector = torch.from_numpy(tf.expand_dims(best_observed_objective_value, axis=0).numpy())
    Full_tilts = thresholds_vector[:,:,0]
    Full_powers = Ptx_thresholds_vector[:, :, 0]

    # Saving BO for matlab
    data_BO = {"Thresholds": Thresholds.numpy(),
                "Obj": Obj.numpy(),
                "best_observed_objective_value": best_observed_objective_value.numpy(),
                "optimum_thresholds": optimum_thresholds.numpy(),
                "best_rate_so_far": best_value_qNEHVI_all.numpy(),
                "Full_tilts": Full_tilts.numpy(),
                "Full_powers": Full_powers.numpy()}
    file_name = "2023_09_30_IterativeBO_2Tier_GUEs_iteration{}_Cell{}.mat".format(iteration,idx_1)
    savemat(file_name, data_BO)
    #############################################################
    # d = {"SINR_UAVs": 10 * np.log10(sinr_total_UAVs.numpy()),
    #      "SINR_GUEs": 10 * np.log10(sinr_total_GUEs.numpy()),
    #      "Cell_id_GUEs":BSs_id_GUEs.numpy(),
    #      "GUEs_x": Xuser_GUEs_x.numpy(),
    #      "GUEs_y": Xuser_GUEs_y.numpy(),
    #      "Cell_id_UAVs": BSs_id_UAVs.numpy(),
    #      "UAVs_x": Xuser_UAVs_x.numpy(),
    #      "UAVs_y": Xuser_UAVs_y.numpy()}
    # # savemat("2023_04_08_SINR_Cell_ID_AlphaHalf_Mix.mat", d)
    # savemat("2023_04_25_Alpha0_GUEs_Cell_ID.mat", d)

    #d = {"SINR_UAVs": 10 * np.log10(sinr_total_UAVs.numpy()),
          #"SINR_GUEs": 10 * np.log10(sinr_total_GUEs.numpy()),
          #"Rate_UAVs": Rate_UAVs.numpy(),
          #"Rate_GUEs": Rate_GUEs.numpy()}
    #savemat("2023_09_28_IterativeBO_OneTier_GUEs_iteration{}.mat", d)
    #
    # d = {"Cell_id_GUEs":BSs_id_GUEs.numpy(),
    #       "GUEs_x": Xuser_GUEs_x.numpy(),
    #       "GUEs_y": Xuser_GUEs_y.numpy(),
    #       "Cell_id_UAVs": BSs_id_UAVs.numpy(),
    #       "UAVs_x": Xuser_UAVs_x.numpy(),
    #       "UAVs_y": Xuser_UAVs_y.numpy()}
    # savemat("2023_06_21_Cell_ID_Alpha0_ProductRateObj_LOS.mat", d)
