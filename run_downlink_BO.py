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
from botorch.sampling import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning


GPU_mode = 1  # set this value one if you have proper GPU setup in your computer

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
MC_SAMPLES = 256
BATCH_SIZE = 1
NUM_RESTARTS = 10
RAW_SAMPLES = 512
NOISE_SE = 0.001
# train_yvar = torch.tensor(NOISE_SE**2)

bounds_lower = torch.tensor([[-45.0]])
bounds_lower = bounds_lower.tile((2,))
bounds_higher = torch.tensor([[45.0]])
bounds_higher = bounds_higher.tile((2,))
bounds = torch.cat((bounds_lower, bounds_higher), 0)

best_observed_nei = []
#############################################################

def generate_initial_data():
    # generate training data for 2-thresholds
    train_x = torch.tensor([
        [-39.050087, -21.939178],

        [34.80735, -20.934471],

        [-74.90397, 70.42882],

        [-20.834732, -48.010338],

        [-40.971107, 13.313961]], dtype=torch.double)

    train_obj = torch.tensor([[-350.9634704589844],
                                   [-363.9451599121094],
                                   [-461.8426513671875],
                                   [-358.501220703125],
                                   [-359.94122314453125]], dtype=torch.double)


    best_observed_value = train_obj.max().item()

    return train_x, train_obj, best_observed_value

def initialize_model(train_x, train_obj, state_dict=None):
    # define models for objective and constraint

    This is for single-output BO
    model_obj = FixedNoiseGP(train_x, train_obj, train_yvar.expand_as(train_obj)).to(train_x)

    # model_obj = SingleTaskGP(train_x, train_obj).to(train_x)
    model = ModelListGP(model_obj)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)

    return mll, model

def optimize_acqf_and_get_observation(acq_func):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
    )
    # observe new values
    new_x = candidates.detach()
    new_x_ref = new_x
    # new_x = new_x.tile((19,))
    candidate_1 = torch.tile(new_x[0][0], (1, 9)) #Change here
    candidate_2 = torch.tile(new_x[0][1], (1, 10))
    candidate = torch.cat((candidate_1, candidate_2), 1)
    new_x = candidate
    new_x = new_x.expand(config.batch_num, 1, 19)

    #After getting the new candidate we need to get the simulator output [This is for the simulator output]
    #############################################################

    # This is for NTN offloading
    #############################################################
    T1_TN_ref = tf.constant([[60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0,
                              60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0,
                              60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0,
                              60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0]]) - 60.0 - 600.0

    T2_NTN_ref = 600.0
    T1_TN = tf.expand_dims(T1_TN_ref, axis=0)
    # Uncomment these if you want to not do random thresholds initializing
    T1_TN = tf.tile(T1_TN, [config.batch_num, 1, 1])
    T2_NTN = tf.tile(tf.expand_dims(tf.expand_dims(tf.expand_dims(T2_NTN_ref, axis=0), axis=1), axis=2),
                     [config.batch_num, 1, 1])
    T1_TN = tf.concat([T1_TN, T2_NTN], axis=2)
    #############################################################

    # This is for RSS+toe load balancing
    #############################################################
    # # Fixing toes to be Zeros
    toes = tf.expand_dims(tf.zeros(T1_TN_ref.shape) + 0.0001, axis=2)
    toes = tf.tile(toes, [2 * config.batch_num, 1, int(config.GUE_ratio * config.Nuser_drop)])
    #############################################################

    # This is for the new tilt advised by the BO
    #############################################################
    BS_tilt = new_x
    ## Fixed BS tilts
    BS_tilt = torch.tensor([20.0])
    BS_tilt = BS_tilt.expand(config.batch_num, 1, 19)
    ## Convert thresholds to numpy then to tensorflow (I cannot use torch on my GPU!!)
    BS_tilt = BS_tilt.numpy()
    BS_tilt = tf.convert_to_tensor(BS_tilt)
    BS_tilt = tf.expand_dims(tf.expand_dims(BS_tilt[0, 0, :], axis=0), axis=2)
    BS_tilt = tf.tile(BS_tilt, [2 * config.batch_num, 3, config.Nuser_drop])  # This is for 19 thresholds

    #Random tilts (2 thresholds)
    # BS_tilt_1 = tf.expand_dims(tf.expand_dims(tf.random.uniform((1,), -45.0, 45.0, tf.float32), axis=0), axis=2)
    # BS_tilt_2 = tf.expand_dims(tf.expand_dims(tf.random.uniform((1,), -45.0, 45.0, tf.float32), axis=0), axis=2)
    # BS_tilt_1 = tf.tile(BS_tilt_1, [1, 9, 1])
    # BS_tilt_2 = tf.tile(BS_tilt_2, [1, 10, 1])
    #
    # BS_tilt = tf.concat([BS_tilt_1, BS_tilt_2], axis=1)
    # BS_tilt = tf.tile(BS_tilt, [2 * config.batch_num, 3, config.Nuser_drop])
    #############################################################

    #Random tilts (1 thresholds)
    # BS_tilt_1 = tf.expand_dims(tf.expand_dims(tf.random.uniform((1,), -45.0, 45.0, tf.float32), axis=0), axis=2)
    # BS_tilt_1 = tf.tile(BS_tilt_1, [1, 19, 1])
    # BS_tilt = tf.tile(BS_tilt_1, [2 * config.batch_num, 3, config.Nuser_drop])
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
        data.T1_TN = T1_TN
        data.toes = toes
        data.BS_tilt = BS_tilt
        data.call()
        # ------------ Import of the LSG data
        LSG_assign_UAVs_Offloaded = data.LSG_assign_GUEs_Offloaded
        LSG_assign_UAVs_Not_Offloaded = data.LSG_assign_GUEs_Not_Offloaded

        # Keeping track of the offloaded UAVs in all of the elevations
        Offloaded_UEs_perc = data.Offloaded_UEs_perc
        BSs_load_all = data.BSs_load_all
        BSs_load_tracking = data.BSs_load_tracking
        # SINR calculations
        sinr_NTN_UAVs_Offloaded = SINR.sinr_LEO(LSG_assign_UAVs_Offloaded)
        sinr_TN_UAVs_Not_Offloaded = SINR.sinr_TN(LSG_assign_UAVs_Not_Offloaded, toes)

        # BO objective: Sum of log of the RSS
        RSS_sumOftheLog_Obj, Rate_sumOftheLog_Obj = SINR.RSS_TN(LSG_assign_UAVs_Not_Offloaded)
        RSS_allElv.append(RSS_sumOftheLog_Obj)
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
    RSS_obj = (sum(RSS_allElv))
    Rate_obj = (sum(Rate_allElv))
    RSS_allElv = []
    Rate_allElv = []
    RSS_obj = RSS_obj[0].__float__()
    Rate_obj = Rate_obj[0].__float__()
    new_obj1 = RSS_obj
    new_obj2 = Rate_obj

    # SINR Objective over all Elvations
    #############################################################
    # SINR_obj = (sum(SINR_allElv))
    # SINR_allElv = []
    # SINR_obj = SINR_obj[0].__float__()
    # new_obj = SINR_obj
    #############################################################

    return new_x_ref, new_obj1

# call functions to generate initial training data and initialize model
#############################################################
train_x_nei, train_obj_nei, best_observed_value_nei = generate_initial_data()
mll_nei, model_nei = initialize_model(train_x_nei, train_obj_nei)
best_observed_nei.append(best_observed_value_nei)

Model_mean = []
Model_var = []
Model_test = []
#############################################################

for i in tqdm(range(N_BATCH)):

    # fit the models
    fit_gpytorch_model(mll_nei)
    # define the qEI and qNEI acquisition modules using a QMC sampler
    qmc_sampler = SobolQMCNormalSampler(MC_SAMPLES)

    # for best_f, we use the best observed noisy values as an approximation
    qNEI = qNoisyExpectedImprovement(
        model=model_nei,
        X_baseline=train_x_nei,
        sampler=qmc_sampler,)

    # optimize and get new observation
    new_x_nei, new_obj_nei = optimize_acqf_and_get_observation(qNEI)

    # update training points
    train_x_nei = torch.cat([train_x_nei, new_x_nei])
    new_obj_nei = torch.tensor(new_obj_nei)
    new_obj_nei = new_obj_nei.expand(1, 1)
    train_obj_nei = torch.cat([train_obj_nei, new_obj_nei])

    # update progress
    best_value_nei = train_obj_nei.max().item()
    best_observed_nei.append(best_value_nei)

    # Model Outputs
    #############################################################

    #For 1-threshold test set
    # test_batch_size = 1000
    # test = torch.linspace(-90.0,90.0, test_batch_size).reshape(-1, 1)

    #For 2-threshold test set
    test_batch_size = 90
    test_1 = torch.linspace(-45.0,45.0, test_batch_size)
    test_2 = test_1
    test_1, test_2 = torch.meshgrid(test_1, test_2)
    test_1 = test_1.reshape(-1, 1)
    test_2 = test_2.reshape(-1, 1)
    test_grid = torch.stack([test_1, test_2], dim=1)
    test_grid = test_grid[:, :, 0]

    posterior = model_nei.posterior(test_grid)
    mean = posterior.mean
    var = posterior.variance

    mean = mean.tolist()
    var = var.tolist()
    test_grid = test_grid.tolist()
    Model_mean.append(mean)
    Model_var.append(var)
    Model_test.append(test_grid)
    #############################################################

    # reinitialize the models so they are ready for fitting on next iteration
    # use the current state dict to speed up fitting
    mll_nei, model_nei = initialize_model(
        train_x_nei,
        train_obj_nei,
        model_nei.state_dict(),
    )



    # BO Outputs
    #############################################################
    Thresholds = train_x_nei
    Obj = train_obj_nei
    Thresholds = Thresholds.numpy()
    Thresholds = tf.convert_to_tensor(Thresholds)
    Obj = Obj.numpy()
    Obj = tf.convert_to_tensor(Obj)

    # best_observed_objective_value = min_Rates.max().item()
    best_observed_objective_value = tf.reduce_max(Obj).__float__()
    optimum_thresholds = tf.cast(Obj == best_observed_objective_value, "double") * Thresholds
    optimum_thresholds = tf.reduce_sum(optimum_thresholds, axis=0)
    #############################################################
    # Saving BO for matlab
    data_BO = {"Thresholds": Thresholds.numpy(),
               "Rates": Obj.numpy(),
               "best_observed_objective_value": best_observed_objective_value,
               "optimum_thresholds": optimum_thresholds.numpy(),
               "best_rate_so_far": best_observed_nei,
               "model_mean": Model_mean,
               "model_var": Model_var,
               "test_points": Model_test}
    savemat("2023_02_13_BO_tilt_GUEsOnly_1threshold_ISD500m_RSS_fixedLOSandShadowing_model_2thresholds_set1.mat", data_BO)
    #############################################################
    # Saving for Python reuse
    # np.save('TN_thresholds', TN_thresholds)
    # np.save('Rates', min_Rates)
    #############################################################


# Saving SINR and rates for matlab
# d = {"TN_NotOffloaded_SINR": 10 * np.log10(sinr_TN_UAVs_Not_Offloaded.numpy()),
#          "NTN_Offloaded_SINR": 10*np.log10(sinr_NTN_UAVs_Offloaded.numpy()),
#          "Rates": Rate_TNandNTN.numpy(),
#          # "Offloaded_UEs": Offloaded_UEs,
#          # "Rate_obj":Rate_obj,
#          "BSs_load_tracking": BSs_load_tracking.numpy(),
#          "SINR_obj":SINR_obj}
# # savemat("2023_01_14_GUEs_LoadBalancing_3thresholds_NoBias_minus20_0_SINRandRates.mat", d)
# savemat("test00.mat", d)

