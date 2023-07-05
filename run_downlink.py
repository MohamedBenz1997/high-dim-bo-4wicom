"""
This is the Runner script of the simulator

@authors: Mohamed Benzaghta and Rasoul Nikbakht
"""

import os
os.system("export MKL_DEBUG_CPU_TYPE=5")
import tensorflow as tf
from TerrestrialClass import Terrestrial
from SinrClass import SINR
from config import Config
from DeploymentClass import Deployment
from plot_class import Plot
import numpy as np
import matplotlib.pyplot as plt
from BO_manual import BO_class
# from BO_multi import BO_multi_class
from BO_multi_qNEI import BO_multi_qNEI_class
from scipy.io import savemat
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import math
from NTN_LSG_Class import NTN_Large_Scale_Gain
import time
import numpy as np
import sys
# for i in range(1, len(sys.argv)):
#     print('argument:', i, 'value:', sys.argv[i])\
    
GPU_mode= 1 # set this value one if you have proper GPU setup in your computer
# #The easiet way for using the GPU is docker
# # tf.debugging.set_log_device_placement(True)
# # gpus = tf.config.list_logical_devices('CPU')
# # strategy = tf.distribute.MirroredStrategy(gpus)
if GPU_mode:
    num_GPU = 1 # choose among available GPUs
    mem_growth = True
    print('Tensorflow version: ', tf.__version__)
    gpus = tf.config.experimental.list_physical_devices("GPU")
    print('Number of GPUs available :', len(gpus))
    # tf.config.experimental.set_visible_devices(gpus[num_GPU], 'GPU')
    # tf.config.experimental.set_memory_growth(gpus[num_GPU], mem_growth)
    print('Used GPU: {}. Memory growth: {}'.format(num_GPU, mem_growth))

#------------ Set seeds
# tf.random.set_seed(43)
# np.random.seed(43)

#------------ Import of the classes
SINR = SINR()
config = Config()
plot = Plot()
BO_class=BO_class()
# BO_multi_class=BO_multi_class()
BO_multi_qNEI_class = BO_multi_qNEI_class()
# deployment=Deployment()


a = np.random.randint(1,10)
N = 5
l = []
n= []
j=[]
avg_window= []
best=[]
best_rate_so_far=[]
tb = SummaryWriter(f"runs/loss_evolution_a{a}")
Rate_allElv=[]
SINR_allElv=[]
Offloaded_UEs_allElv=[]
Offloaded_UEs_allBO=[]

for i in tqdm(range(N)):

    #For TN-NTN Offloading
    #############################################################
    # T1_TN = BO_multi_class.call_forConfig()
    # #Convert thresholds to numpy then to tensorflow (I cannot use torch on my GPU!!)
    # T1_TN = T1_TN.numpy()
    # T1_TN = tf.convert_to_tensor(T1_TN)

    #For BenchMark Oflloading all UAVs (Manual)
    #############################################################
    T1_TN_ref = tf.constant([[60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0,
                          60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0,
                          60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0,
                          60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0]])-60.0-600.0
    # T1_TN_ref = tf.constant([[-4.9153,   -4.8144,   -5.0608,   -4.7286,   -5.1729,   -5.1192,   -4.9223,   -4.8792,   -4.7748,   -4.7414,
    #                           -4.8228, -5.1083, -5.0236, -4.7353, -4.8585, -4.8759, -4.8000, -4.9618, -5.1345, -4.8757,
    #                           -5.0842, -5.0377, -4.7414, -4.7468, -5.1184, -4.9699, -4.7915, -4.9924, -4.6835, -4.7527,
    #                           -4.6159, -4.9236, -4.7695, -4.9476, -4.7589, -4.7011, -4.7232, -4.8833, -4.7460, -4.7568,
    #                           -5.1169, -5.1199, -5.2550, -5.0268, -4.7312, -5.0668, -4.6878, -4.6359, -4.8021, -4.8935,
    #                           -4.7239, -4.7977, -4.8807, -4.6084, -4.8737, -4.9400, -5.0549
    #                           ]])

    T2_NTN_ref = 600.0
    T1_TN=tf.expand_dims(T1_TN_ref,axis=0)
    # # # # Uncomment these if you want to not do random thresholds initializing
    T1_TN= tf.tile(T1_TN,[config.batch_num,1,1] )
    T2_NTN = tf.tile(tf.expand_dims(tf.expand_dims(tf.expand_dims(T2_NTN_ref, axis=0), axis=1), axis=2), [config.batch_num, 1, 1])
    T1_TN = tf.concat([T1_TN, T2_NTN ], axis=2)
    #
    # BSs thresholds Random
    # T1_TN = tf.random.uniform(T1_TN.shape, -10.0, 10.0, tf.float32)
    # T1_TN = tf.expand_dims(T1_TN[0, :, :], axis=0)
    # T1_TN = tf.tile(T1_TN, [config.batch_num, 1, 1])
    #############################################################

    #For RSS+toes Load balancing
    #############################################################
    # toes = BO_multi_class.call_forConfig()
    # #Convert thresholds to numpy then to tensorflow (I cannot use torch on my GPU!!)
    # toes = toes.numpy()
    # toes = tf.convert_to_tensor(toes)
    # toes = tf.expand_dims(tf.expand_dims(toes[0,0,:],axis=0),axis=2)
    # toes = tf.tile(toes, [2 * config.batch_num, 1, int(config.GUE_ratio * config.Nuser_drop)])

    #This is for RSS+toe load balancing
    #############################################################
    # # Fixing toes to be Zeros
    toes = tf.expand_dims(tf.zeros(T1_TN_ref.shape)+0.0001,axis=2)
    toes = tf.tile(toes, [2*config.batch_num, 1, int(config.GUE_ratio * config.Nuser_drop)])
    ## Adding optimized threshold
    # toes_1 = toes[:,0:19,:]-0.0 #+0.0
    # toes_2 = toes[:, 19:38, :]-0.0 #-19.9186
    # toes_3 = toes[:, 38:, :]-0.0 #-0.3318
    # toes = tf.concat([toes_1, toes_2, toes_3], axis=1)
    # #  Random toes
    # toes = tf.expand_dims(tf.expand_dims(tf.random.uniform((config.Nap * 3,), -20.0, 0.0, tf.float32),axis=0),axis=2)
    # toes = tf.tile(toes, [2*config.batch_num, 1, int(config.GUE_ratio * config.Nuser_drop)])

    # # Fixing 3 TN thresholds and repeating them over the 19 BSs deployments (Random toes)
    # toes_1 = tf.expand_dims(tf.expand_dims(tf.random.uniform((1,), -20.0, 0.0, tf.float32),axis=0),axis=2)
    # toes_1 = tf.tile(toes_1, [2*config.batch_num, config.Nap, int(config.GUE_ratio * config.Nuser_drop)])
    # toes_2 = tf.expand_dims(tf.expand_dims(tf.random.uniform((1,), -20.0, 0.0, tf.float32),axis=0),axis=2)
    # toes_2 = tf.tile(toes_2, [2*config.batch_num, config.Nap, int(config.GUE_ratio * config.Nuser_drop)])
    # toes_3 = tf.expand_dims(tf.expand_dims(tf.random.uniform((1,), -20.0, 0.0, tf.float32),axis=0),axis=2)
    # toes_3 = tf.tile(toes_3, [2*config.batch_num, config.Nap, int(config.GUE_ratio * config.Nuser_drop)])
    # toes = tf.concat([toes_1, toes_2, toes_3], axis=1)

    # For tilt optimization
    #############################################################
    # BS_tilt = BO_multi_class.call_forConfig()
    BS_tilt = BO_multi_qNEI_class.call_forConfig()
    #Convert thresholds to numpy then to tensorflow (I cannot use torch on my GPU!!)
    BS_tilt = BS_tilt.numpy()
    BS_tilt = tf.convert_to_tensor(BS_tilt)
    BS_tilt = tf.expand_dims(tf.expand_dims(BS_tilt[0,0,:],axis=0),axis=2)
    # BS_tilt = tf.tile(BS_tilt, [2 * config.batch_num, 1, config.Nuser_drop]) #This is for 57 thresholds
    BS_tilt = tf.tile(BS_tilt,[2 * config.batch_num, 3,  config.Nuser_drop])  # This is for 19 thresholds
    # BS_tilt = BS_tilt[0].__float__()
    # #Fixed
    # BS_tilt = tf.expand_dims(tf.zeros(T1_TN_ref.shape)-12.0,axis=2)
    # #Random
    # BS_tilt = tf.expand_dims(tf.expand_dims(tf.random.uniform((config.Nap,), -13.7, -13.7, tf.float32), axis=0), axis=2)
    # BS_tilt = tf.tile(BS_tilt, [2*config.batch_num, 3,  config.Nuser_drop])

    ## Iterating over several LEO elevations
    #############################################################
    #Getting the LEO constellation Destribution
    N_sat = 512
    phi = np.arange(0,90.1,0.1)
    F = 1 - np.exp(-N_sat / 2. * (1 - np.cos(np.deg2rad(phi))))
    n_sample = 1
    U = np.random.uniform(0, 1,n_sample)
    def cdf2rand(u):
        indx = np.where(u <= F)
        indx = indx[0][0].__int__()
        return phi[indx]
    samples = [cdf2rand(u) for u in U]
    alpha_deg = [90.0 - samples for samples in samples]

    #Converting the sampling angles to distances to be used in the simulator
    alpha_deg = np.deg2rad(alpha_deg)
    alpha_deg = np.array(alpha_deg)
    a = 1
    b = - (2 * config.RE * ((np.cos(alpha_deg) ** 2))) / (config.RE + config.Zleo)
    c = ((config.RE ** 2) * ((np.cos(alpha_deg) ** 2))) / ((config.RE + config.Zleo) ** 2) + (np.cos(alpha_deg) ** 2) - 1.0
    d = (b ** 2) - (4 * a * c)
    sol1 = (-b - np.sqrt(d)) / (2 * a)
    sol2 = (-b + np.sqrt(d)) / (2 * a)
    n_alpha_D2D = (40030.0e3) * (np.rad2deg(np.arccos(sol2))) / 360.0
    n_alpha_D2D=n_alpha_D2D.tolist()
    n_alpha_D2D=[0.0]
    #66317 #19148.006144643623
    for a in n_alpha_D2D:
        data = Terrestrial()
        NTN_LSG=NTN_Large_Scale_Gain()
        deployment = Deployment()

        data.alpha_factor=a
        data.T1_TN = T1_TN
        data.toes = toes
        data.BS_tilt = BS_tilt
        data.call()

        #------------ Import of the LSG data

        LSG_assign_UAVs_Offloaded=data.LSG_assign_GUEs_Offloaded
        LSG_assign_UAVs_Not_Offloaded=data.LSG_assign_GUEs_Not_Offloaded

        ##########################
        # LSG_assign_UAVs_Offloaded=data.LSG_assign_UAVs_Offloaded
        # LSG_assign_UAVs_Not_Offloaded=data.LSG_assign_UAVs_Not_Offloaded
        ##########################

        #Keeping track of the offloaded UAVs in all of the elevations
        Offloaded_UEs_perc = data.Offloaded_UEs_perc
        BSs_load_all = data.BSs_load_all
        BSs_load_tracking = data.BSs_load_tracking
        #SINR calculations
        sinr_NTN_UAVs_Offloaded= SINR.sinr_LEO(LSG_assign_UAVs_Offloaded)
        sinr_TN_UAVs_Not_Offloaded= SINR.sinr_TN(LSG_assign_UAVs_Not_Offloaded,toes)

        # BO objective: Sum of log of the SINRS
        SINR_log=tf.math.log(sinr_TN_UAVs_Not_Offloaded) / tf.math.log(10.0)
        SINR_sumOftheLog=tf.reduce_sum(SINR_log, axis=0)
        SINR_sumOftheLog = tf.expand_dims(tf.convert_to_tensor(SINR_sumOftheLog),axis=0)
        SINR_allElv.append(SINR_sumOftheLog)

        #Rate calculations
        Rate_perElv, Rate_TNandNTN = SINR.rate_TN_NTN(LSG_assign_UAVs_Not_Offloaded, LSG_assign_UAVs_Offloaded, Offloaded_UEs_perc, BSs_load_all)
        Rate_allElv.append(Rate_perElv)

        #Keeping track of offloaded UEs over all elevations
        Offloaded_UEs_allElv.append(Offloaded_UEs_perc)

        #Keeping track of SINR over all elevations
        if a==n_alpha_D2D[0]:
            sinr_NTN_UAVs_Offloaded_all=tf.zeros([sinr_NTN_UAVs_Offloaded.shape[0]])
            sinr_TN_UAVs_Not_Offloaded_all = tf.zeros([sinr_TN_UAVs_Not_Offloaded.shape[0]])
            Rate_TNandNTN_all = tf.ones([Rate_TNandNTN.shape[0]]) #It is 1 here when we want to keep the zero rates for outage users

        sinr_NTN_UAVs_Offloaded_all= tf.concat([sinr_NTN_UAVs_Offloaded_all, sinr_NTN_UAVs_Offloaded], axis=0)
        sinr_TN_UAVs_Not_Offloaded_all=tf.concat([sinr_TN_UAVs_Not_Offloaded_all, sinr_TN_UAVs_Not_Offloaded], axis=0)
        Rate_TNandNTN_all = tf.concat([Rate_TNandNTN_all, Rate_TNandNTN], axis=0)

    # Reporting SINR and Rates over all elevations
    #############################################################
    #Remove the zeros that are embedded in the beggining
    bool_mask_1 = tf.not_equal(sinr_NTN_UAVs_Offloaded_all, 0)
    sinr_NTN_UAVs_Offloaded_all = tf.boolean_mask(sinr_NTN_UAVs_Offloaded_all, bool_mask_1)
    bool_mask_2 = tf.not_equal(sinr_TN_UAVs_Not_Offloaded_all, 0)
    sinr_TN_UAVs_Not_Offloaded_all = tf.boolean_mask(sinr_TN_UAVs_Not_Offloaded_all, bool_mask_2)
    bool_mask_3 = tf.not_equal(Rate_TNandNTN_all, 1.0) #It is 1 here when we want to keep the zero rates for outage users
    Rate_TNandNTN_all = tf.boolean_mask(Rate_TNandNTN_all, bool_mask_3)

    sinr_NTN_UAVs_Offloaded=sinr_NTN_UAVs_Offloaded_all
    sinr_TN_UAVs_Not_Offloaded=sinr_TN_UAVs_Not_Offloaded_all
    Rate_TNandNTN=Rate_TNandNTN_all
    # Offloaded_UEs = sinr_NTN_UAVs_Offloaded.shape[0]/(config.Nap*3*n_sample*config.batch_num)
    Offloaded_UEs = sinr_NTN_UAVs_Offloaded.shape[0] / ( sinr_NTN_UAVs_Offloaded.shape[0]+ sinr_TN_UAVs_Not_Offloaded.shape[0])
    #############################################################

    # Rate Objective over all Elvations
    #############################################################
    Rate_obj=(sum(Rate_allElv))
    Rate_allElv=[]
    Rate_obj=Rate_obj[0].__float__()
    # BO_multi_class.append_minRate(Rate_obj)

    # SINR Objective over all Elvations
    #############################################################
    SINR_obj=(sum(SINR_allElv))
    SINR_allElv=[]
    SINR_obj=SINR_obj[0].__float__()
    # BO_multi_class.append_minRate(SINR_obj)
    BO_multi_qNEI_class.append_minRate(SINR_obj)
    #############################################################

    #Keeping track of the offloaded UEs over all BO  iterations
    #############################################################
    Offloaded_UEs = (sum(Offloaded_UEs_allElv)*config.UAV_ratio * config.Nuser_drop) / (config.UAV_ratio * config.Nuser_drop * n_sample)
    Offloaded_UEs_allBO.append(Offloaded_UEs)
    Offloaded_UEs_allElv = []
    #############################################################

    #BS Load tracking
    #############################################################
    if i == 0:
        BSs_load_tracking_all = tf.zeros(BSs_load_tracking.shape,'float32')
    #
    BSs_load_tracking_all = tf.concat([BSs_load_tracking_all, BSs_load_tracking], axis=0)
    #############################################################

    #------------ DL SINR plotting
    #############################################################
    #
    # fig, ax = plt.subplots(1, 1)
    # ax.plot(*plot.ecdf(10*np.log10(sinr_TN_UAVs_Baseline.numpy())),color='black', linestyle='solid')
    # ax.plot(*plot.ecdf(10*np.log10(sinr_TN_GUEs_Baseline.numpy())),color='blue', linestyle='solid')
    # ax.plot(*plot.ecdf(10*np.log10(sinr_NTN_UAVs_Offloaded.numpy())),color='black', linestyle='dashed')
    # ax.plot(*plot.ecdf(10*np.log10(sinr_TN_UAVs_Not_Offloaded.numpy())),color='blue', linestyle='dashed')
    # ax.set_title('Downlink Analysis of Hybrid TN/NTN networks')
    # ax.set_xlabel("SINR [dB]")
    # ax.set_ylabel('CDF')
    # ax.legend(["TN-connected UAVs Baseline","TN-connected GUEs Baseline","NTN-connected UAVs Offloaded","TN-connected UAVs Not-Offloaded"], loc='best')
    # plt.ylim([0, 1])
    # plt.figure(figsize=(30, 30), dpi=460)
    # plt.show()

    #Real time monitoring of iterations
    #############################################################
    # Compute loss
    # Thresholds=BO_multi_class.TN_thresholds
    # Thresholds=BO_multi_class.min_Rates
    # T_nPlus1 = Thresholds[-1, :]
    # T_n = Thresholds[-2, :]
    # value = tf.norm(T_nPlus1-T_n, ord='euclidean')
    # best_observed_objective_value = BO_multi_class.min_Rates.max().item()
    # best.append(best_observed_objective_value)
    # if not i % 100:
    #     value=BO_multi_class.min_Rates.max()  #.item()
    #     value = value.numpy()
    #     tb.add_scalar("Loss", value, i)
    #     j.append(i)
    #     l.append(value)
    #
    #     tb.close()
    #     plt.figure(figsize=(5, 5))
    #     plt.plot(j, l)
    #     plt.show()
    # Add value to tensorboard event
    # tb.add_scalar("Loss", value, i)
    # # Generate processing time
    # time.sleep(0.5)
    # list for check here on jupyter notebbok for debugging
    # l.append(value)
    # n.append(i+1)

    #Sliding window average
    # k=2
    # def moving_average(a, n):
    #     test = np.cumsum(a, dtype=float)
    #     test[n:] = test[n:] - test[:-n]
    #     return test[n - 1:] / n
    #
    # avg_window=moving_average(l, k)
    # avg_window=avg_window.tolist()
    # if i>k-2:
    #     j.append(i + 1)
    #     tb.close()
    #     plt.figure(figsize=(5, 5))
    #     plt.plot(j, avg_window)
    #     plt.show()
    # tb.close()
    # plt.figure(figsize=(5, 5))
    # plt.plot(n, l)
    # plt.plot(j, l)
    # plt.show()

    #Keeping track of best observed values
    value=BO_multi_class.min_Rates.max()  #.item()
    value = value.numpy()
    best_rate_so_far.append(value)

    # plt.figure(figsize=(5, 5))
    # plt.plot(n, best_rate_so_far)
    # plt.show()
    #############################################################

    #BO Outputs
    #############################################################
    TN_thresholds=BO_multi_class.TN_thresholds
    min_Rates=BO_multi_class.min_Rates
    TN_thresholds = TN_thresholds.numpy()
    TN_thresholds = tf.convert_to_tensor(TN_thresholds)
    min_Rates = min_Rates.numpy()
    min_Rates = tf.convert_to_tensor(min_Rates)

    # best_observed_objective_value = min_Rates.max().item()
    best_observed_objective_value = tf.reduce_max(min_Rates).__float__()
    optimum_thresholds=tf.cast(min_Rates == best_observed_objective_value, "float32") * TN_thresholds
    optimum_thresholds=tf.reduce_sum(optimum_thresholds,axis=0)
    # ############################################################
    # Saving BO for matlab
    # data_BO = {"Thresholds": TN_thresholds.numpy(),
    #            "Rates": min_Rates.numpy(),
    #            "best_observed_objective_value": best_observed_objective_value,
    #            "optimum_thresholds": optimum_thresholds.numpy(),
    #            "best_rate_so_far": best_rate_so_far,
    #            "Offloaded_UEs":Offloaded_UEs_allBO,
    #            "BSs_load_tracking":BSs_load_tracking_all.numpy()}
    # savemat("2023_01_23_BO_Uptilt_GUEsOnly_19threshold_ISD500m_BiasedInitialSet_set1.mat", data_BO)
    #############################################################
    #Saving for Python reuse
    # np.save('TN_thresholds', TN_thresholds)
    # np.save('Rates', min_Rates)
    #############################################################
    #Saving SINR and rates for matlab
    # d = {"TN_NotOffloaded_SINR": 10 * np.log10(sinr_TN_UAVs_Not_Offloaded.numpy()),
    #      "NTN_Offloaded_SINR": 10*np.log10(sinr_NTN_UAVs_Offloaded.numpy()),
    #      "Rates": Rate_TNandNTN.numpy(),
    #      # "Offloaded_UEs": Offloaded_UEs,
    #      # "Rate_obj":Rate_obj,
    #      "BSs_load_tracking": BSs_load_tracking.numpy(),
    #      "SINR_obj":SINR_obj}
    # # savemat("2023_01_14_GUEs_LoadBalancing_3thresholds_NoBias_minus20_0_SINRandRates.mat", d)
    # savemat("test00.mat", d)


#Real time monitoring of iterations
#############################################################
# tb.close()
# plt.figure(figsize=(3,3))
# plt.plot(np.arange(N), l)
# plt.show()
#############################################################

#------------ UL SINR operations and plotting
#############################################################
# sinr_UL_TN_GUEsFromGUEs= SINR.sinr_UL(LSG_assign_GUEs_Baseline)
# sinr_UL_TN_UAVsFromUAVs= SINR.sinr_UL(LSG_assign_UAVs_Baseline)
# sinr_UL_TN_GUEs,sinr_UL_TN_UAVs=SINR.sinr_UL_combined(LSG_assign_GUEs_Baseline,LSG_assign_UAVs_Baseline)
# sinr_UL_NTN_UAVs=SINR.snr_LEO_UL(LSG_assign_UAVs_Offloaded,data.UAVS_maxAG_forUL)
#
# fig, ax = plt.subplots(1, 1)
# ax.plot(*plot.ecdf(10*np.log10(sinr_UL_TN_GUEsFromGUEs.numpy())),color='black', linestyle='solid')
# ax.plot(*plot.ecdf(10*np.log10(sinr_UL_TN_GUEs.numpy())),color='black', linestyle='dashed')
# ax.plot(*plot.ecdf(10*np.log10(sinr_UL_NTN_UAVs.numpy())),color='red', linestyle='solid')
# ax.plot(*plot.ecdf(10*np.log10(sinr_UL_TN_UAVs.numpy())),color='blue', linestyle='dashed')
# ax.plot(*plot.ecdf(10*np.log10(sinr_UL_TN_UAVsFromUAVs.numpy())),color='blue', linestyle='solid')
# ax.set_title('Uplink Analysis of Hybrid TN/NTN networks ')
# ax.set_xlabel("SINR [dB]")
# ax.set_ylabel('CDF')
# ax.legend(["GUEs from GUEs","GUEs from GUEs and UAVs","UAVs from UAVs","UAVs from GUEs and UAVs","UAVs from UAVs TN"], loc='best') #
# plt.ylim([0, 1])
# plt.figure(figsize=(30, 30), dpi=460)
# plt.show()
#############################################################

