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
from plot_class import Plot
import numpy as np
import matplotlib.pyplot as plt
from BO_manual import SINR
from scipy.io import savemat


class Runner_for_BO():
    def __init__(self):
        Config.__init__(self)
        self.data = Terrestrial.call()
        self.SINR = SINR()
        self.SINR=SINR()
        self.plot = Plot()
        return

    def Call(self):
        #------------ Set seeds
        tf.random.set_seed(43)
        np.random.seed(43)

        #------------ Import of the LSG data
        LSG_assign_UAVs_Baseline=self.data.LSG_assign_UAVs_Baseline
        LSG_assign_GUEs_Baseline=self.data.LSG_assign_GUEs_Baseline
        LSG_assign_UAVs_Offloaded=self.data.LSG_assign_UAVs_Offloaded
        LSG_assign_UAVs_Not_Offloaded=self.data.LSG_assign_UAVs_Not_Offloaded

        #------------ Number of simulations (Total number of simulations will be batch_num*i )
        for i in range(2):
            self.data = self.Terrestrial()
            self.data.call()
            LSG_assign_UAVs_Baseline = tf.concat([LSG_assign_UAVs_Baseline, self.data.LSG_assign_UAVs_Baseline], axis=0)
            LSG_assign_GUEs_Baseline = tf.concat([LSG_assign_GUEs_Baseline, self.data.LSG_assign_GUEs_Baseline], axis=0)
            LSG_assign_UAVs_Offloaded = tf.concat([LSG_assign_UAVs_Offloaded, self.data.LSG_assign_UAVs_Offloaded], axis=0)
            LSG_assign_UAVs_Not_Offloaded = tf.concat([LSG_assign_UAVs_Not_Offloaded, self.data.LSG_assign_UAVs_Not_Offloaded], axis=0)


        #------------ DL SINR operations and plotting
        sinr_TN_GUEs_Baseline = self.SINR.sinr_TN(LSG_assign_GUEs_Baseline)
        sinr_TN_UAVs_Baseline = self.SINR.sinr_TN(LSG_assign_UAVs_Baseline)
        sinr_NTN_withIB_UAVs_Offloaded= self.SINR.sinr_LEO_withIB(LSG_assign_UAVs_Offloaded)
        sinr_TN_UAVs_Not_Offloaded= self.SINR.sinr_TN(LSG_assign_UAVs_Not_Offloaded)

        fig, ax = plt.subplots(1, 1)
        ax.plot(*plot.ecdf(10*np.log10(sinr_TN_UAVs_Baseline.numpy())),color='black', linestyle='solid')
        ax.plot(*plot.ecdf(10*np.log10(sinr_TN_GUEs_Baseline.numpy())),color='blue', linestyle='solid')
        ax.plot(*plot.ecdf(10*np.log10(sinr_NTN_withIB_UAVs_Offloaded.numpy())),color='black', linestyle='dashed')
        ax.plot(*plot.ecdf(10*np.log10(sinr_TN_UAVs_Not_Offloaded.numpy())),color='blue', linestyle='dashed')
        ax.set_title('Downlink Analysis of Hybrid TN/NTN networks')
        ax.set_xlabel("SINR [dB]")
        ax.set_ylabel('CDF')
        ax.legend(["TN-connected UAVs Baseline","TN-connected GUEs Baseline","NTN-connected UAVs Offloaded","TN-connected UAVs Not-Offloaded"], loc='best')
        plt.ylim([0, 1])
        plt.figure(figsize=(30, 30), dpi=460)
        plt.show()
        SINR_NTN_min=tf.reduce_min(sinr_NTN_withIB_UAVs_Offloaded)
        SINR_TN_min=tf.reduce_min(sinr_TN_UAVs_Not_Offloaded)

        Offloaded_UEs=self.data.Mean_Offloaded_UEs
        NTN_Rate=(1/(72*Offloaded_UEs))*1e7*(tf.math.log(1+SINR_NTN_min)/tf.math.log(2.0))*(1/1e3)
        TN_Rate=(1/(15.0))*3e7*(tf.math.log(1+SINR_TN_min)/tf.math.log(2.0))*(1/1e3)
        min_Rate=tf.reduce_min(tf.concat([TN_Rate, NTN_Rate], axis=0))

        return min_Rate



















