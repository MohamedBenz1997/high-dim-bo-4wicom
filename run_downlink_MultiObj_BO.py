"""
This is the Runner script of the simulator
@authors: Mohamed Benzaghta
"""

import os
os.system("export MKL_DEBUG_CPU_TYPE=5")
import tensorflow as tf
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import math
import sys
import time
import random

from TerrestrialClass import Terrestrial
from SinrClass import SINR
from config import Config
from DeploymentClass import Deployment
from plot_class import Plot
from NTN_LSG_Class import NTN_Large_Scale_Gain

from botorch.models import SingleTaskGP, FixedNoiseGP, ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.acquisition.objective import ConstrainedMCObjective
from botorch.optim import optimize_acqf
# from botorch import fit_gpytorch_model
from botorch.fit import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.sampling import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning

from botorch.models.transforms.outcome import Standardize
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.multi_objective.box_decompositions.dominated import (DominatedPartitioning,)
from botorch.utils.multi_objective.pareto import is_non_dominated

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

# ------------ Set seeds
# tf.random.set_seed(43)
# np.random.seed(43)

# ------------ Import of the classes
SINR = SINR()
config = Config()
plot = Plot()

a = np.random.randint(1, 10)
l = []
n = []
j = []

best = []
best_rate_so_far = []
tb = SummaryWriter(f"runs/loss_evolution_a{a}")

##BO initial variables
#############################################################
N_BATCH = 1
MC_SAMPLES = 128
BATCH_SIZE = 1
NUM_RESTARTS = 10
RAW_SAMPLES = 512

NOISE_SE = torch.tensor([0.001, 0.001], dtype=torch.double)

#4 Thresholds bounds
bounds_lower = torch.tensor([[-90.0]])
bounds_lower = bounds_lower.tile((2,))
bounds_higher = torch.tensor([[90.0]])
bounds_higher = bounds_higher.tile((2,))
bounds = torch.cat((bounds_lower, bounds_higher), 0)

standard_bounds_lower = torch.tensor([[0.0]])
standard_bounds_lower = standard_bounds_lower.tile((2,))
standard_bounds_higher = torch.tensor([[1.0]])
standard_bounds_higher = standard_bounds_higher.tile((2,))
standard_bounds = torch.cat((standard_bounds_lower, standard_bounds_higher), 0)

# ref_point =
ref_point = torch.tensor([0.0, 2.0])


#############################################################

# def generate_initial_data():
#
#     # generate training data for 2-thresholds
#     # train_x = torch.tensor([
#     #     [8.350014, -14.972653],
#     #     [-19.261618,  -2.0534477],
#     #     [-3.6468983, -10.100101 ],
#     #     [1.4039326, 7.905136],
#     #     [3.4445267, -8.048817 ],
#     #     [6.5807037, -22.7563],
#     #     [-0.64222336,  8.407013],
#     #     [-27.027107, -23.71307],
#     #     [-1.7742119, -8.250412],
#     #     [4.0738373, -18.049835]], dtype=torch.double)
#     #
#     # train_obj = torch.tensor([[-326.1353,   67.0168],
#     #                                [-325.5621,   67.0529],
#     #                                [-325.4689,   67.0608],
#     #                                [-327.9120,   66.9078],
#     #                                [-326.2430,   67.0126],
#     #                                [-326.4424,   66.9979],
#     #                                [-327.5977,   66.9270],
#     #                                [-326.3738,   66.9995],
#     #                                [-325.7533,   67.0440],
#     #                                [-325.6115,   67.0479]], dtype=torch.double)
#
#     train_x = torch.tensor([
#         [0.0, 0.0],
#         [-72.48674, -79.2049],
#         [-65.93138 ,   7.964592],
#         [-76.390495, -25.766716],
#         [38.789505, -57.288746]], dtype=torch.double)
#
#     train_obj = torch.tensor([[0.3121, 2.9946],
#                                    [0.3286, 3.0042],
#                                    [0.2770, 2.9751],
#                                    [0.3306, 3.0062],
#                                    [0.3335, 3.0073]], dtype=torch.double)
#
#     best_observed_value = train_obj.max().item()
#
#     return train_x, train_obj, best_observed_value

def initialize_model(train_x, train_obj):
    # define models for objective and constraint
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

def optimize_qnehvi_and_get_observation(model, train_x, sampler):
    #Optimizes the qNEHVI acquisition function, and returns a new candidate and observation.
    # partition non-dominated space into disjoint rectangles
    acq_func = qNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        X_baseline=normalize(train_x, bounds),
        prune_baseline=True,  # prune baseline points that have estimated zero probability of being Pareto optimal
        sampler=sampler,
    )
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=standard_bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )
    # observe new values
    new_x =  unnormalize(candidates.detach(), bounds=bounds)
    new_x_ref = new_x

    #4-thresholds
    candidate_1 = torch.tile(new_x[0][0], (1, 1))
    candidate_2 = torch.tile(new_x[0][1], (1, 1))
    # candidate_fixed2 = torch.tensor([[-45.0, -8.78, -45.0, -11.10, -11.68, -36.59, -15.15, -45.0, -24.36, 45.0, 45.0, -24.16]])
    candidate_fixed1 = torch.tensor([[9.09, -72.67, 8.95, 18.60, 24.69, 25.82, 28.86, -26.50, 23.41, -19.77, 19.56, -57.36, 18.45, -67.89, -90.0, 16.14, 21.92, 90.0, 23.77, -49.23, -53.88, 37.38, 35.81, 45.92,
                                      -75.21, 45.03, -56.70, 62.62, 48.00, -90.0, 58.64, 38.01, 25.65, 12.39, 67.67, 90.0]])
    candidate_Zeros = torch.tile(new_x[0][0], (1, 19)) * 0.0
    candidate = torch.cat((candidate_fixed1, candidate_1, candidate_2, candidate_Zeros), 1)
    new_x = candidate

    # new_x = new_x.expand(config.batch_num,3, 1)

    #After getting the new candidate we need to get the simulator output [This is for the simulator output]
    #############################################################

    # This is for NTN offloading
    #############################################################
    # T1_TN_ref = tf.constant([[60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0,
    #                           60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0,
    #                           60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0,
    #                           60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0]]) - 60.0 - 600.0
    #
    # T2_NTN_ref = 600.0
    # T1_TN = tf.expand_dims(T1_TN_ref, axis=0)
    # # Uncomment these if you want to not do random thresholds initializing
    # T1_TN = tf.tile(T1_TN, [config.batch_num, 1, 1])
    # T2_NTN = tf.tile(tf.expand_dims(tf.expand_dims(tf.expand_dims(T2_NTN_ref, axis=0), axis=1), axis=2),
    #                  [config.batch_num, 1, 1])
    # T1_TN = tf.concat([T1_TN, T2_NTN], axis=2)
    #############################################################


    # This is for the new tilt advised by the BO
    #############################################################
    BS_tilt = new_x
    ## Convert thresholds to numpy then to tensorflow (I cannot use torch on my GPU!!)
    BS_tilt = BS_tilt.numpy()
    BS_tilt = tf.convert_to_tensor(BS_tilt)
    BS_tilt = tf.expand_dims(BS_tilt, axis=2)
    BS_tilt = tf.tile(BS_tilt, [2 * config.batch_num, 1, config.Nuser_drop])  # This is for 19 thresholds

    #Random tilts (2 thresholds)
    # # BS_tilt_fixed2 = tf.constant([[[-7.41], [16.49], [13.08], [-18.47], [-38.29], [45.00], [-28.30], [-45.0], [-8.78], [-45.0], [-11.10], [-11.68], [-36.59], [-15.15], [-45.0], [-24.36], [45.0], [45.0], [-24.16], ]])
    BS_tilt_fixed1 = tf.constant([[[9.09], [-72.67], [8.95], [18.60], [24.69], [25.82], [28.86], [-26.50], [23.41], [-19.77], [19.56], [-57.36], [18.45], [-67.89], [-90.0], [16.14], [21.92], [90.0], [23.77], [-49.23], [-53.88], [37.38], [35.81], [45.92],
                                   [-75.21], [45.03], [-56.70], [62.62], [48.00], [-90.0], [58.64], [38.01], [25.65], [12.39], [67.67], [90.0]]])
    # BS_tilt_rand1 = tf.expand_dims(tf.expand_dims(tf.random.uniform((1,), -90.0, 90.0, tf.float32), axis=0), axis=2)
    # BS_tilt_rand2 = tf.expand_dims(tf.expand_dims(tf.random.uniform((1,), -90.0, 90.0, tf.float32), axis=0), axis=2)
    # BS_tilt_zeros = tf.expand_dims(tf.expand_dims(tf.random.uniform((19,), 0.0, 0.0, tf.float32), axis=0), axis=2)
    # BS_tilt = tf.concat([BS_tilt_fixed1, BS_tilt_rand1, BS_tilt_rand2, BS_tilt_zeros], axis=1)
    # BS_tilt = tf.tile(BS_tilt, [2 * config.batch_num, 1, config.Nuser_drop])
    #############################################################

    ## Iterating over several LEO elevations
    #############################################################
    # Getting the LEO constellation Destribution
    N_sat = 512
    phi = np.arange(0, 90.1, 0.1)
    F = 1 - np.exp(-N_sat / 2. * (1 - np.cos(np.deg2rad(phi))))
    n_sample = 1
    U = np.random.uniform(0, 1, n_sample)
    def cdf2rand(u):
        indx = np.where(u <= F)
        indx = indx[0][0].__int__()
        return phi[indx]
    samples = [cdf2rand(u) for u in U]
    alpha_deg = [90.0 - samples for samples in samples]
    # Converting the sampling angles to distances to be used in the simulator
    alpha_deg = np.deg2rad(alpha_deg)
    alpha_deg = np.array(alpha_deg)
    a = 1
    b = - (2 * config.RE * ((np.cos(alpha_deg) ** 2))) / (config.RE + config.Zleo)
    c = ((config.RE ** 2) * ((np.cos(alpha_deg) ** 2))) / ((config.RE + config.Zleo) ** 2) + (
                np.cos(alpha_deg) ** 2) - 1.0
    d = (b ** 2) - (4 * a * c)
    sol1 = (-b - np.sqrt(d)) / (2 * a)
    sol2 = (-b + np.sqrt(d)) / (2 * a)
    n_alpha_D2D = (40030.0e3) * (np.rad2deg(np.arccos(sol2))) / 360.0
    n_alpha_D2D = n_alpha_D2D.tolist()
    n_alpha_D2D = [0.0]
    # 66317 #19148.006144643623
    SINR_allElv = []
    Rate_allElv = []
    RSS_allElv = []
    for a in n_alpha_D2D:
        data = Terrestrial()
        data.alpha_factor = a
        # data.T1_TN = T1_TN
        # data.toes = toes
        data.BS_tilt = BS_tilt
        data.call()

        # ------------ Import of the UAVs LSG data (all deployed UAVs)
        LSG_UAVs_Corridors = data.LSG_UAVs_Corridors
        LSG_GUEs = data.LSG_GUEs

        # ------------ Import of the LSG data
        #
        # LSG_assign_UAVs_Offloaded = data.LSG_assign_GUEs_Offloaded
        # LSG_assign_UAVs_Not_Offloaded = data.LSG_assign_GUEs_Not_Offloaded
        #


        #
        # Keeping track of the offloaded UAVs in all of the elevations
        # Offloaded_UEs_perc = data.Offloaded_UEs_perc
        # BSs_load_all = data.BSs_load_all
        # BSs_load_tracking = data.BSs_load_tracking
        #

        # SINR calculations
        # sinr_NTN_UAVs_Offloaded = SINR.sinr_LEO(LSG_assign_UAVs_Offloaded)
        sinr_TN_UAVs_Corridors = SINR.sinr_TN(LSG_UAVs_Corridors)
        sinr_TN_GUEs = SINR.sinr_TN(LSG_GUEs)

        # BO objective: Sum of log of the RSS
        SINR_sumOftheLog_Obj, Rate_sumOftheLog_Obj, sinr_total_UAVs, sinr_total_GUEs = SINR.BO_Multi_Obj_Cooridor(sinr_TN_UAVs_Corridors, sinr_TN_GUEs, alpha=1)
        SINR_allElv.append(SINR_sumOftheLog_Obj)
        Rate_allElv.append(Rate_sumOftheLog_Obj)
        # BO objective: Sum of log of the SINRS
        # SINR_log = tf.math.log(sinr_TN_UAVs_Not_Offloaded) / tf.math.log(10.0)
        # SINR_sumOftheLog = tf.reduce_sum(SINR_log, axis=0)
        # SINR_sumOftheLog = tf.expand_dims(tf.convert_to_tensor(SINR_sumOftheLog), axis=0)
        # SINR_allElv.append(SINR_sumOftheLog)
        # Rate calculations
        # Rate_perElv, Rate_TNandNTN = SINR.rate_TN_NTN(LSG_assign_UAVs_Not_Offloaded, LSG_assign_UAVs_Offloaded,Offloaded_UEs_perc, BSs_load_all)
        # Rate_allElv.append(Rate_perElv)
        # Keeping track of SINR over all elevations
        # if a == n_alpha_D2D[0]:
        #     sinr_NTN_UAVs_Offloaded_all = tf.zeros([sinr_NTN_UAVs_Offloaded.shape[0]])
        #     sinr_TN_UAVs_Not_Offloaded_all = tf.zeros([sinr_TN_UAVs_Not_Offloaded.shape[0]])
        #     Rate_TNandNTN_all = tf.ones(
        #         [Rate_TNandNTN.shape[0]])  # It is 1 here when we want to keep the zero rates for outage users
        # sinr_NTN_UAVs_Offloaded_all = tf.concat([sinr_NTN_UAVs_Offloaded_all, sinr_NTN_UAVs_Offloaded], axis=0)
        # sinr_TN_UAVs_Not_Offloaded_all = tf.concat([sinr_TN_UAVs_Not_Offloaded_all, sinr_TN_UAVs_Not_Offloaded], axis=0)
        # Rate_TNandNTN_all = tf.concat([Rate_TNandNTN_all, Rate_TNandNTN], axis=0)

    # Reporting SINR and Rates over all elevations
    #############################################################
    # Remove the zeros that are embedded in the beggining
    # bool_mask_1 = tf.not_equal(sinr_NTN_UAVs_Offloaded_all, 0)
    # sinr_NTN_UAVs_Offloaded_all = tf.boolean_mask(sinr_NTN_UAVs_Offloaded_all, bool_mask_1)
    # bool_mask_2 = tf.not_equal(sinr_TN_UAVs_Not_Offloaded_all, 0)
    # sinr_TN_UAVs_Not_Offloaded_all = tf.boolean_mask(sinr_TN_UAVs_Not_Offloaded_all, bool_mask_2)
    # bool_mask_3 = tf.not_equal(Rate_TNandNTN_all,1.0)  # It is 1 here when we want to keep the zero rates for outage users
    # Rate_TNandNTN_all = tf.boolean_mask(Rate_TNandNTN_all, bool_mask_3)
    # sinr_NTN_UAVs_Offloaded = sinr_NTN_UAVs_Offloaded_all
    # sinr_TN_UAVs_Not_Offloaded = sinr_TN_UAVs_Not_Offloaded_all
    # Rate_TNandNTN = Rate_TNandNTN_all
    # Offloaded_UEs = sinr_NTN_UAVs_Offloaded.shape[0] / (sinr_NTN_UAVs_Offloaded.shape[0] + sinr_TN_UAVs_Not_Offloaded.shape[0])
    #############################################################
    # Rate and RSS Objectives over all Elvations
    #############################################################
    SINR_obj = (sum(SINR_allElv))
    Rate_obj = (sum(Rate_allElv))
    SINR_allElv = []
    Rate_allElv = []
    SINR_obj = SINR_obj[0].__float__()
    Rate_obj = Rate_obj[0].__float__()
    new_obj1 = SINR_obj
    new_obj2 = Rate_obj
    new_obj = torch.tensor([[new_obj1, new_obj2]], dtype=torch.double)
    # SINR Objective over all Elvations
    #############################################################
    # SINR_obj = (sum(SINR_allElv))
    # SINR_allElv = []
    # SINR_obj = SINR_obj[0].__float__()
    # new_obj = SINR_obj
    #############################################################

    return new_x_ref, new_obj, sinr_total_UAVs, sinr_total_GUEs

# # call functions to generate initial training data and initialize model
# #############################################################
# train_x_qnehvi, train_obj_qnehvi, best_observed_value = generate_initial_data()
# mll_qnehvi, model_qnehvi = initialize_model(train_x_qnehvi, train_obj_qnehvi)
#
# Model_mean = []
# Model_var = []
# Model_test = []
# hvs_qnehvi = []
#
# # compute hypervolume
# bd = DominatedPartitioning(ref_point=ref_point, Y=train_obj_qnehvi)
# volume = bd.compute_hypervolume().item()
# hvs_qnehvi.append(volume)
#
# #############################################################
#
# for i in tqdm(range(N_BATCH)):
#
#     # fit the models
#     fit_gpytorch_model(mll_qnehvi)
#
#     # define the qEI and qNEI acquisition modules using a QMC sampler
#     qnehvi_sampler = SobolQMCNormalSampler(MC_SAMPLES)
#
#     # optimize acquisition functions and get new observations
#     new_x_qnehvi, new_obj_qnehvi, sinr_total_UAVs, sinr_total_GUEs = optimize_qnehvi_and_get_observation(
#         model_qnehvi, train_x_qnehvi, qnehvi_sampler)
#
#     # update training points
#     train_x_qnehvi = torch.cat([train_x_qnehvi, new_x_qnehvi])
#     train_obj_qnehvi = torch.cat([train_obj_qnehvi, new_obj_qnehvi])
#
#     # update progress
#     for hvs_list, train_obj in zip(
#         (hvs_qnehvi),
#         (train_obj_qnehvi,
#         ),
#     ):
#         # compute hypervolume
#         bd = DominatedPartitioning(ref_point=ref_point, Y=train_obj)
#         volume = bd.compute_hypervolume().item()
#         hvs_qnehvi.append(volume)
#
#     # reinitialize the models so they are ready for fitting on next iteration
#     # Note: we find improved performance from not warm starting the model hyperparameters
#     # using the hyperparameters from the previous iteration
#     mll_qnehvi, model_qnehvi = initialize_model(train_x_qnehvi, train_obj_qnehvi)
#
#
#     # update progress
#     best_value_qNEHVI = tf.expand_dims(tf.reduce_max(train_obj_qnehvi, axis=0),axis=0)
#     if i == 0:
#         best_value_qNEHVI_all = tf.zeros(best_value_qNEHVI.shape,dtype='float64')
#
#     best_value_qNEHVI_all = tf.concat([best_value_qNEHVI_all, best_value_qNEHVI], axis=0)
#
#     # Model Outputs
#     #############################################################
#     #For 1-threshold test set
#     # test_batch_size = 1000
#     # test = torch.linspace(-90.0,90.0, test_batch_size).reshape(-1, 1)
#
#     #For 2-threshold test set
#     # test_batch_size = 90
#     # test_1 = torch.linspace(-45.0,45.0, test_batch_size)
#     # test_2 = test_1
#     # test_1, test_2 = torch.meshgrid(test_1, test_2)
#     # test_1 = test_1.reshape(-1, 1)
#     # test_2 = test_2.reshape(-1, 1)
#     # test_grid = torch.stack([test_1, test_2], dim=1)
#     # test = test_grid[:, :, 0]
#
#     #Mean and Variance from posterior
#     # posterior = model_qnehvi.posterior(test)
#     # mean = posterior.mean
#     # var = posterior.variance
#     # mean = mean.tolist()
#     # var = var.tolist()
#     # test = test.tolist()
#     # Model_mean.append(mean)
#     # Model_var.append(var)
#     # Model_test.append(test)
#     #############################################################
#
#     # BO Outputs
#     #############################################################
#     Thresholds = train_x_qnehvi
#     Obj = train_obj_qnehvi
#     Thresholds = Thresholds.numpy()
#     Thresholds = tf.convert_to_tensor(Thresholds)
#     Obj = Obj.numpy()
#     Obj = tf.convert_to_tensor(Obj)
#
#     # best_observed_objective_value = min_Rates.max().item()
#     best_observed_objective_value = tf.reduce_max(Obj, axis=0)
#     # optimum_thresholds = tf.cast(Obj == best_observed_objective_value, "double") * Thresholds
#     optimum_thresholds =tf.tile(tf.cast(Obj == best_observed_objective_value, "double"),[1,1]) * Thresholds
#     optimum_thresholds = tf.reduce_sum(optimum_thresholds, axis=0)
#     # optimum_thresholds = optimum_thresholds[1]
#     #############################################################
#     # Saving BO for matlab
#     data_BO = {"Thresholds": Thresholds.numpy(),
#                "Obj": Obj.numpy(),
#                "best_observed_objective_value": best_observed_objective_value.numpy(),
#                "optimum_thresholds": optimum_thresholds.numpy(),
#                "best_rate_so_far": best_value_qNEHVI_all.numpy()}
#     savemat("2023_03_21_BO_tilt_4Cooridors_2thresholds_iteration19_36and37_set1.mat", data_BO)
#     #############################################################
#     # Saving BO for matlab with Model paramters
#     # data_BO = {"Thresholds": Thresholds.numpy(),
#     #            "Obj": Obj.numpy(),
#     #            "best_observed_objective_value": best_observed_objective_value.numpy(),
#     #            "optimum_thresholds": optimum_thresholds.numpy(),
#     #            "best_rate_so_far": best_value_qNEHVI_all.numpy(),
#     #            "model_mean": Model_mean,
#     #            "model_var": Model_var,
#     #            "test_points": Model_test}
#     # savemat("2023_02_23_BO_tilt_GUEsOnly_6thresholds_ISD500m_fixedLOSandShadowing_MultiOutput_model_set1.mat", data_BO)
#     #############################################################
#     # Saving for Python reuse
#     # np.save('TN_thresholds', TN_thresholds)
#     # np.save('Rates', min_Rates)
#     #############################################################
#
#     # Saving SINR and rates for matlab
#     # d = {"SINR_UAVs": 10 * np.log10(sinr_total_UAVs.numpy()),
#     #      "SINR_GUEs": 10 * np.log10(sinr_total_GUEs.numpy())}
#     # savemat("SINR_UAVs_GUEs.mat", d)
#
#
# # Saving SINR and rates for matlab
# # d = {"TN_NotOffloaded_SINR": 10 * np.log10(sinr_TN_UAVs_Not_Offloaded.numpy()),
# #          "NTN_Offloaded_SINR": 10*np.log10(sinr_NTN_UAVs_Offloaded.numpy()),
# #          "Rates": Rate_TNandNTN.numpy(),
# #          # "Offloaded_UEs": Offloaded_UEs,
# #          # "Rate_obj":Rate_obj,
# #          "BSs_load_tracking": BSs_load_tracking.numpy(),
# #          "SINR_obj":SINR_obj}
# # # savemat("2023_01_14_GUEs_LoadBalancing_3thresholds_NoBias_minus20_0_SINRandRates.mat", d)
# # savemat("test00.mat", d)
#
# ##Pareto front plotting
# #############################################################
# # from matplotlib.cm import ScalarMappable
# #
# # fig, axes = plt.subplots(1, 1, figsize=(8, 6), sharex=True, sharey=True)
# # algos = [ "qNEHVI"]
# # cm = plt.cm.get_cmap('viridis')
# #
# # batch_number = torch.cat([torch.zeros(4), torch.arange(1, N_BATCH+1).repeat(BATCH_SIZE, 1).t().reshape(-1)]).numpy()
# #
# # for  train_obj in train_obj_qnehvi:
# #     sc = axes.scatter(
# #         train_obj_qnehvi[:, 0].cpu().numpy(), train_obj_qnehvi[:,1].cpu().numpy(), c=batch_number, alpha=0.8,
# #     )
# #     axes.set_title(algos[0])
# #     axes.set_xlabel("Objective 1")
# # axes.set_ylabel("Objective 2")
# #
# # norm = plt.Normalize(batch_number.min(), batch_number.max())
# # sm =  ScalarMappable(norm=norm, cmap=cm)
# # sm.set_array([])
# # fig.subplots_adjust(right=0.9)
# # cbar_ax = fig.add_axes([0.93, 0.15, 0.01, 0.7])
# # cbar = fig.colorbar(sm, cax=cbar_ax)
# # cbar.ax.set_title("Iteration")
# # plt.show()
# #############################################################

#Testing

def generate_initial_data(idx_1, idx_2, thresholds_vector, obj_vector):
    train_x = torch.from_numpy(thresholds_vector[:,idx_1:idx_2+1,0].numpy())
    train_obj = obj_vector
    for j in range(4):
        BS_tilt = thresholds_vector[0,:,0]
        indx = tf.constant([[idx_1], [idx_2]])
        rand_values = tf.constant([random.uniform(-90.0, 90.0), random.uniform(-90.0, 90.0)])
        new_train_x = torch.from_numpy( tf.expand_dims(rand_values, axis=0).numpy())
        BS_tilt = tf.tensor_scatter_nd_update(BS_tilt, indx, rand_values)
        BS_tilt = tf.expand_dims(tf.expand_dims(BS_tilt,axis=0),axis=2)
        BS_tilt = tf.tile(BS_tilt, [2 * config.batch_num, 1, config.Nuser_drop])

        # Run the simulator
        data = Terrestrial()
        data.alpha_factor = 0.0  # LEO at 90deg
        data.BS_tilt = BS_tilt
        data.call()

        # ------------ Import of the UAVs and GUEs LSG and SINR data
        LSG_UAVs_Corridors = data.LSG_UAVs_Corridors
        LSG_GUEs = data.LSG_GUEs
        sinr_TN_UAVs_Corridors = SINR.sinr_TN(LSG_UAVs_Corridors)
        sinr_TN_GUEs = SINR.sinr_TN(LSG_GUEs)

        # BO objective: Sum of log of the RSS
        SINR_sumOftheLog_Obj, Rate_sumOftheLog_Obj, sinr_total_UAVs, sinr_total_GUEs = SINR.BO_Multi_Obj_Cooridor(sinr_TN_UAVs_Corridors, sinr_TN_GUEs, alpha=1)
        SINR_obj = SINR_sumOftheLog_Obj[0].__float__()
        Rate_obj = Rate_sumOftheLog_Obj[0].__float__()
        new_obj1 = SINR_obj
        new_obj2 = Rate_obj
        new_obj = torch.tensor([[new_obj1, new_obj2]], dtype=torch.double)

        train_x = torch.cat((train_x, new_train_x), dim=0)
        train_obj = torch.cat((train_obj, new_obj), dim=0)

    return train_x, train_obj

for i in range(29):
    idx_1 = 2*i
    idx_2 = 2*i + 1
    if i == 0:
        thresholds_vector = tf.expand_dims(tf.expand_dims(tf.random.uniform((57,), 0.0, 0.0, tf.float32), axis=0), axis=2)
        obj_vector = torch.tensor([[0.3121, 2.9946]], dtype=torch.double)

    #Obtain the training data-set
    train_x, train_obj = generate_initial_data(idx_1, idx_2, thresholds_vector, obj_vector)
    y = 2.0