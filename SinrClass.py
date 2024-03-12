"""
This is the SINR Class of the simulator:
    This is the class in which DL and UL for both TN/NTN SINRs are computed

@authors: Mohamed Benzaghta and Rasoul Nikbakht
"""

import tensorflow as tf
from config import Config
from TerrestrialClass import Terrestrial

class SINR(Config):
    def __init__(self):
        Config.__init__(self)
        self.NF_TN_UL_dB=5.0
        self.NF_TN_UL=tf.math.pow(10.0, self.NF_TN_UL_dB/ 10.0)
        # self.P_Tx_TN = tf.math.pow(10.0, (self.P_Tx_dB - 30.0) / 10.0)
        self.P_Tx_NTN = tf.math.pow(10.0, (self.P_Tx_sat_db - 30.0) / 10.0)
        self.PSD = tf.math.pow(10.0, (-174.0 - 30.0) / 10.0)
        self.BW_TN = self.bandwidth
        self.BW_NTN = self.bandwidth_sat
        self.NF_TN = tf.math.pow(10.0, self.noise_figure_user / 10.0)
        self.NF_NTN = tf.math.pow(10.0, self.noise_figure_user_sat / 10.0)

    data = Terrestrial()

    # ------------ DL SINR
    def sinr_TN(self, LSG, P_Tx_TN):
        if self.N==1:
            P_Tx_TN = tf.slice(P_Tx_TN, [0, 0, 0], [tf.shape(LSG)[0], 21, tf.shape(LSG)[-1]])
        elif self.N==2:
            P_Tx_TN = tf.slice(P_Tx_TN, [0, 0, 0], [tf.shape(LSG)[0], 57, tf.shape(LSG)[-1]])
        P_Tx_TN = tf.math.pow(10.0, (P_Tx_TN - 30.0) / 10.0)  # Conver to linear
        LSL_TN=-LSG
        LSL_min_TN = tf.math.reduce_min(LSL_TN, axis=1, keepdims=True)
        LSL_TN_sorted = tf.sort(LSL_TN, axis=1)

        #Finding the BSs ids of served UAVs
        indexing = tf.cast(-LSG == LSL_min_TN, "float32")
        if self.N==1:
            BSs_id = tf.expand_dims(tf.expand_dims(tf.range(0, 21, dtype=tf.float32), 0), 2) * indexing
        elif self.N==2:
            BSs_id = tf.expand_dims(tf.expand_dims(tf.range(0, 57, dtype=tf.float32), 0), 2) * indexing
        BSs_id = tf.reduce_sum(BSs_id, axis=1)

        LSG_min_TN = tf.math.pow(10, (-LSL_min_TN) / 10)
        LSG_min_TN=LSG_min_TN*tf.cast(LSG_min_TN != 1.0, "float32")
        LSG_linear = tf.math.pow(10, (LSG) / 10)

        LSG_TN_sorted=-LSL_TN_sorted
        LSG_TN_sorted = tf.math.pow(10, (LSG_TN_sorted) / 10)
        LSG_TN_sorted = LSG_TN_sorted * tf.cast(LSG_TN_sorted != 1.0, "float32")

        ##If TN relief SINR is need uncomment this line,13 cells of for UAVs at 150m, 11 cells of for UAVs at 100m, 7 cells of for UAVs at 50m
        # LSG_TN_sorted = LSG_TN_sorted[:, 11:, :] * tf.cast(LSG_TN_sorted[:, 11:, :] != 1.0, "float32") #For TN interference relief
        snr_link = LSG_linear * P_Tx_TN / (self.BW_TN * self.NF_TN)
        P_Tx_TN_assigned = tf.expand_dims(tf.reduce_max(tf.abs(P_Tx_TN * tf.cast(LSG_linear == tf.tile(LSG_min_TN, [1, 57, 1]), "float32")), axis=1), axis=1)
        num_TN = LSG_min_TN * P_Tx_TN_assigned / (self.BW_TN * self.NF_TN)
        denom_TN = tf.expand_dims(tf.reduce_sum(snr_link, axis=1), axis=1) - num_TN + self.PSD
        # denom_TN = tf.expand_dims(tf.reduce_sum(snr_link, axis=1), axis=1) + self.PSD #For TN interference relief
        sinr_TN_withZeros = num_TN / denom_TN
        #To ignore the zeros which corresponds to the GUEs columns when dealing with UAVs and vice versa
        # bool_mask = tf.not_equal(sinr_TN_withZeros, 0)
        # sinr_TN = tf.boolean_mask(sinr_TN_withZeros, bool_mask)

        sinr_TN = sinr_TN_withZeros
        return sinr_TN

    # # ------------ DL SINR
    # def RSS_TN(self, LSG, toes):
    #     LSL_TN=-LSG[:,0:57,:]
    #
    #     if self.RSS_offloading:
    #         toes = toes[0:self.batch_num, :, 0:57]
    #     if self.one_tier:
    #         LSL_TN = -LSG[:, 0:21, :]
    #     LSL_min_TN = tf.math.reduce_min(LSL_TN, axis=1, keepdims=True)
    #     LSL_TN_sorted = tf.sort(LSL_TN, axis=1)
    #
    #     #This is when bias toe is introduced
    #     if self.RSS_offloading:
    #         LSL_min_TN_extended= tf.cast(tf.tile(LSL_min_TN,[1, 57, 1]) == LSL_TN, "float32")
    #         toes_UEs = tf.math.reduce_sum( LSL_min_TN_extended*toes, axis=1, keepdims=True) #This is the toe value corresponding to each assosiated UE based on its assosiated BS
    #         toes_full = tf.tile(tf.expand_dims(toes_UEs[:,0,:],axis=2), [1,1,57])
    #
    #         LSL_min_TN=LSL_min_TN-toes_UEs
    #         LSL_TN_sorted=LSL_TN_sorted-toes_full
    #
    #     LSG_min_TN = tf.math.pow(10, (-LSL_min_TN) / 10)
    #     LSG_min_TN=LSG_min_TN*tf.cast(LSG_min_TN != 1.0, "float32")
    #
    #     ##If TN relief SINR is need uncomment this line,13 cells of for UAVs at 150m, 11 cells of for UAVs at 100m, 7 cells of for UAVs at 50m
    #     # LSG_TN_sorted = LSG_TN_sorted[:, 11:, :] * tf.cast(LSG_TN_sorted[:, 11:, :] != 1.0, "float32") #For TN interference relief
    #     RSS_TN = LSG_min_TN* self.P_Tx_TN
    #
    #     #To ignore the zeros which corresponds to the GUEs columns when dealing with UAVs and vice versa
    #     bool_mask = tf.not_equal(RSS_TN, 0)
    #     RSS_TN = tf.boolean_mask(RSS_TN, bool_mask)
    #
    #     RSS_TN_dB = tf.math.log(RSS_TN) / tf.math.log(10.0)
    #     return RSS_TN

    def sinr_LEO(self, LSG):
        LSL_NTN=-LSG
        LSL_min_NTN = tf.math.reduce_min(LSL_NTN, axis=1, keepdims=True)
        LSG_min_NTN = tf.math.pow(10, (-LSL_min_NTN) / 10)
        LSG_min_NTN = LSG_min_NTN * tf.cast(LSG_min_NTN != 1.0, "float32")

        LSL_NTN_sorted=tf.sort(LSL_NTN,axis=1)
        LSG_NTN_sorted=-LSL_NTN_sorted
        LSG_NTN_sorted = tf.math.pow(10, (LSG_NTN_sorted) / 10)
        LSG_NTN_sorted = LSG_NTN_sorted * tf.cast(LSG_NTN_sorted != 1.0, "float32")
        snr_link_NTN = LSG_NTN_sorted * self.P_Tx_NTN / (self.BW_NTN * self.NF_NTN)
        num_NTN = LSG_min_NTN * self.P_Tx_NTN / (self.BW_NTN * self.NF_NTN)
        denom_NTN = tf.expand_dims(tf.reduce_sum(snr_link_NTN, axis=1), axis=1) - num_NTN + self.PSD
        sinr_NTN_withZeros = num_NTN / denom_NTN
        sinr_NTN_withZeros=tf.expand_dims(sinr_NTN_withZeros[:,:,:],axis=2)
        #To ignore the zeros which corresponds to the GUEs columns when dealing with UAVs and vice versa
        bool_mask = tf.not_equal(sinr_NTN_withZeros, 0)
        sinr_LEO = tf.boolean_mask(sinr_NTN_withZeros, bool_mask)
        # sinr_LEO = tf.boolean_mask(sinr_NTN_withZeros, bool_mask)

        if self.FRF_3:
            sinr_NTN_withZeros = (LSG_min_NTN * self.P_Tx_NTN) / (self.BW_NTN * self.NF_NTN * self.PSD)
            sinr_NTN_withZeros=tf.expand_dims(sinr_NTN_withZeros[:,:,:],axis=2)
            bool_mask = tf.not_equal(sinr_NTN_withZeros, 0)
            sinr_LEO = tf.boolean_mask(sinr_NTN_withZeros, bool_mask)

        return sinr_LEO

    # ------------ UL SINR
    #This function for GUEs UL from GUEs only or UAVs UL from UAVs only
    def sinr_UL(self, LSG):
        LSG_TN = LSG[:, 0:57, :]
        LSL_TN=-LSG[:,0:57,:]
        LSL_TN=LSL_TN+tf.cast(LSL_TN == 0.0, "float32")*10000#the casting and adding a big number is to get rid of zeros when choosing min value for assigment
        LSL_min_TN = tf.expand_dims(tf.math.reduce_min(LSL_TN, axis=1, keepdims=False),axis=2)
        LSG_min_TN = tf.math.pow(10, (-LSL_min_TN) / 10)
        LSG_TN = tf.math.pow(10, (LSG_TN) / 10)
        LSG_TN = LSG_TN * tf.cast(LSG_TN != 1.0, "float32")

        alpha = tf.expand_dims(tf.ones([LSG_TN.shape[0], 3 * self.Nap], 'float32')* self.alpha, axis=1)
        P_all = self.Po + alpha * tf.expand_dims(LSL_min_TN[:,:,0],axis=1)
        P_all = tf.tile(P_all, [1, 57, 1])
        P_all = tf.math.minimum(P_all, self.UE_P_Tx_dbm)
        P_all = tf.math.pow(10, (P_all - 30.0) / 10)

        snr_TN = (P_all * LSG_TN) / (self.UE_bandwidth * self.NF_TN_UL)

        num_TN = (tf.expand_dims(P_all[:,0,:],axis=2)* LSG_min_TN) / (self.UE_bandwidth * self.NF_TN_UL)
        denom_TN= tf.expand_dims(tf.reduce_sum(snr_TN, axis=2),axis=2) - num_TN + self.PSD

        sinr_UL_TN = num_TN / denom_TN

        bool_mask = tf.not_equal(sinr_UL_TN, 0)
        sinr_UL_TN = tf.boolean_mask(sinr_UL_TN, bool_mask)

        return sinr_UL_TN

    #This function for GUEs UL from GUEs+UAVs and UAVs UL from UAVs and GUEs
    def sinr_UL_combined(self, LSG_GUEs, LSG_UAVs):
        LSG_combined=LSG_GUEs[:, 0:57, :]+LSG_UAVs[:, 0:57, :]
        LSL_combined=-LSG_combined
        LSG_combined = tf.math.pow(10, (LSG_combined) / 10)
        LSL_min_combined = tf.expand_dims(tf.math.reduce_min(LSL_combined, axis=1, keepdims=False), axis=2)
        LSG_GUEs = LSG_GUEs[:, 0:57, :]
        LSL_GUEs=-LSG_GUEs[:,0:57,:]
        LSL_GUEs=LSL_GUEs+tf.cast(LSL_GUEs == 0.0, "float32")*10000#the casting and adding a big number is to get rid of zeros when choosing min value for assigment
        LSL_min_GUEs = tf.expand_dims(tf.math.reduce_min(LSL_GUEs, axis=1, keepdims=False),axis=2)
        LSG_min_GUEs = tf.math.pow(10, (-LSL_min_GUEs) / 10)
        LSG_GUEs = tf.math.pow(10, (LSG_GUEs) / 10)
        LSG_GUEs = LSG_GUEs * tf.cast(LSG_GUEs != 1.0, "float32")

        LSG_UAVs = LSG_UAVs[:, 0:57, :]
        LSL_UAVs=-LSG_UAVs[:,0:57,:]
        LSL_UAVs=LSL_UAVs+tf.cast(LSL_UAVs == 0.0, "float32")*10000
        LSL_min_UAVs = tf.expand_dims(tf.math.reduce_min(LSL_UAVs, axis=1, keepdims=False),axis=2)
        LSG_min_UAVs = tf.math.pow(10, (-LSL_min_UAVs) / 10)
        LSG_UAVs = tf.math.pow(10, (LSG_UAVs) / 10)
        LSG_UAVs = LSG_UAVs * tf.cast(LSG_UAVs != 1.0, "float32")

        alpha = tf.expand_dims(tf.ones([LSG_GUEs.shape[0], 3 * self.Nap], 'float32')* self.alpha, axis=1)
        P_GUEs_all = self.Po + alpha * tf.expand_dims(LSL_min_GUEs[:,:,0],axis=1)
        P_GUEs_all = tf.tile(P_GUEs_all, [1, 57, 1])
        P_GUEs_all = tf.math.minimum(P_GUEs_all, self.UE_P_Tx_dbm)
        P_GUEs_all = tf.math.pow(10, (P_GUEs_all - 30.0) / 10)

        P_UAVs_all = self.Po + alpha * tf.expand_dims(LSL_min_UAVs[:,:,0],axis=1)
        P_UAVs_all = tf.tile(P_UAVs_all, [1, 57, 1])
        P_UAVs_all = tf.math.minimum(P_UAVs_all, self.UE_P_Tx_dbm)
        P_UAVs_all = tf.math.pow(10, (P_UAVs_all - 30.0) / 10)

        P_combined = self.Po + alpha * tf.expand_dims(LSL_min_combined[:,:,0],axis=1)
        P_combined = tf.tile(P_combined, [1, 57, 1])
        P_combined = tf.math.minimum(P_combined, self.UE_P_Tx_dbm)
        P_combined = tf.math.pow(10, (P_combined - 30.0) / 10)

        snr_Combined= (P_combined * LSG_combined) / (self.UE_bandwidth * self.NF_TN_UL)

        num_GUEs = (tf.expand_dims(P_GUEs_all[:,0,:],axis=2) * LSG_min_GUEs) / (self.UE_bandwidth * self.NF_TN_UL)
        denom_GUEs= tf.expand_dims(tf.reduce_sum(snr_Combined, axis=2),axis=2)- num_GUEs + self.PSD

        num_UAVs = (tf.expand_dims(P_UAVs_all[:,0,:],axis=2) * LSG_min_UAVs) / (self.UE_bandwidth * self.NF_TN_UL)
        denom_UAVs= tf.expand_dims(tf.reduce_sum(snr_Combined, axis=2),axis=2)- num_UAVs + self.PSD

        sinr_UL_GUEs = num_GUEs / denom_GUEs
        sinr_UL_UAVs = num_UAVs / denom_UAVs

        bool_mask1 = tf.not_equal(sinr_UL_GUEs, 0)
        sinr_UL_GUEs = tf.boolean_mask(sinr_UL_GUEs, bool_mask1)

        bool_mask2 = tf.not_equal(sinr_UL_UAVs, 0)
        sinr_UL_UAVs = tf.boolean_mask(sinr_UL_UAVs, bool_mask2)

        return sinr_UL_GUEs,sinr_UL_UAVs

    def snr_LEO_UL(self,LSG,NTN_AG):
        G_over_T_dB = NTN_AG - 28.9  # 1.1 #AG-28.9 when 30-28.9 for LEO at 90deg

        LSL_UAVs=-LSG
        LSL_UAVs = LSL_UAVs + tf.cast(LSL_UAVs == 0.0, "float32") * 10000
        LSG_UAVs= -tf.reduce_min(LSL_UAVs,axis=1)-NTN_AG+G_over_T_dB- (10 * tf.math.log(1.38e-23 * self.UE_bandwidth) / tf.math.log(10.0))
        LSG = tf.math.pow(10, LSG_UAVs / 10)
        UAVs_Ptx=tf.math.pow(10.0, (self.UE_P_Tx_dbm - 30.0) / 10.0)
        sinr_UL_NTN_UAVs = LSG*UAVs_Ptx

        bool_mask1 = tf.not_equal(sinr_UL_NTN_UAVs, 0)
        sinr_UL_NTN_UAVs = tf.boolean_mask(sinr_UL_NTN_UAVs, bool_mask1)

        return sinr_UL_NTN_UAVs

    # # ------------ DL RSS
    def RSS_TN(self, LSG):

        LSL_TN=-LSG[:,0:57,:]
        LSL_min_TN = tf.math.reduce_min(LSL_TN, axis=1, keepdims=True)
        LSG_min_TN = tf.math.pow(10, (-LSL_min_TN) / 10)
        LSG_min_TN=LSG_min_TN*tf.cast(LSG_min_TN != 1.0, "float32")

        RSS_TN = LSG_min_TN * self.P_Tx_TN
        RSS_TN_withZeros = RSS_TN
        RSS_TN_dB = 10 * (tf.math.log(RSS_TN_withZeros) / tf.math.log(10.0)) + 30.0 #+30 here is to convert the RSS to dBm

        #Sum of the log of RSS, which already in dBm scale, equivalent to mutipling in linear scale
        RSS_sumOftheLog = tf.reduce_sum(tf.math.log(RSS_TN_withZeros) / tf.math.log(10.0), axis=2)
        # Avraging over M.C runs
        RSS_sumOftheLog_Obj = tf.reduce_mean(RSS_sumOftheLog, axis=0)

        SNR_TN = RSS_TN / (self.BW_TN * self.NF_TN * self.PSD)
        Rate_TN = ((1.0 / 15.0) * self.BW_TN * (tf.math.log(1 + SNR_TN) / tf.math.log(2.0)))/(1e6)
        #Sum of the log of Rates, which already in MHz scale, equivalent to mutipling in linear scale
        Rate_sumOftheLog = tf.reduce_sum(tf.math.log(Rate_TN) / tf.math.log(10.0), axis=2)
        # Avraging over M.C runs
        Rate_sumOftheLog_Obj = tf.reduce_mean(Rate_sumOftheLog, axis=0)

        return RSS_sumOftheLog_Obj, Rate_sumOftheLog_Obj

    def RSS_LEO(self, LSG):

        LSL_NTN=-LSG
        LSL_min_NTN = tf.math.reduce_min(LSL_NTN, axis=1, keepdims=True)
        LSG_min_NTN = tf.math.pow(10, (-LSL_min_NTN) / 10)
        LSG_min_NTN = LSG_min_NTN * tf.cast(LSG_min_NTN != 1.0, "float32")

        RSS_NTN = LSG_min_NTN * self.P_Tx_NTN
        RSS_NTN_withZeros = RSS_NTN
        RSS_NTN_withZeros=tf.expand_dims(RSS_NTN_withZeros[:,:,:],axis=2)
        #To ignore the zeros which corresponds to the GUEs columns when dealing with UAVs and vice versa
        bool_mask = tf.not_equal(RSS_NTN_withZeros, 0)
        RSS_LEO = tf.boolean_mask(RSS_NTN_withZeros, bool_mask)

        return RSS_LEO


    # ------------ DL SINR and Rate for sum of the log objective
    def rate_TN_NTN(self, LSG_Not_Offloaded, LSG_Offloaded, Offloaded_UEs_perc, BSs_load):
        # TN Calculation
        LSL_TN=-LSG_Not_Offloaded[:,0:57,:]
        LSL_min_TN = tf.math.reduce_min(LSL_TN, axis=1, keepdims=True)
        LSG_min_TN = tf.math.pow(10, (-LSL_min_TN) / 10)
        LSG_min_TN=LSG_min_TN*tf.cast(LSG_min_TN != 1.0, "float32")
        LSL_TN_sorted=tf.sort(LSL_TN,axis=1)
        LSG_TN_sorted=-LSL_TN_sorted
        LSG_TN_sorted = tf.math.pow(10, (LSG_TN_sorted) / 10)
        LSG_TN_sorted = LSG_TN_sorted * tf.cast(LSG_TN_sorted != 1.0, "float32")
        ##If TN relief SINR is need uncomment this line,13 cells of for UAVs at 150m, 11 cells of for UAVs at 100m, 7 cells of for UAVs at 50m
        # LSG_TN_sorted = LSG_TN_sorted[:, 11:, :] * tf.cast(LSG_TN_sorted[:, 11:, :] != 1.0, "float32") #For TN interference relief
        snr_link = LSG_TN_sorted* self.P_Tx_TN / (self.BW_TN * self.NF_TN)
        num_TN = LSG_min_TN * self.P_Tx_TN / (self.BW_TN * self.NF_TN)
        denom_TN = tf.expand_dims(tf.reduce_sum(snr_link, axis=1), axis=1) - num_TN + self.PSD
        # denom_TN = tf.expand_dims(tf.reduce_sum(snr_link, axis=1), axis=1) + self.PSD #For TN interference relief
        sinr_TN_withZeros = num_TN / denom_TN

        # NTN Calculation
        LSL_NTN=-LSG_Offloaded
        LSL_min_NTN = tf.math.reduce_min(LSL_NTN, axis=1, keepdims=True)
        LSG_min_NTN = tf.math.pow(10, (-LSL_min_NTN) / 10)
        LSG_min_NTN = LSG_min_NTN * tf.cast(LSG_min_NTN != 1.0, "float32")
        LSL_NTN_sorted=tf.sort(LSL_NTN,axis=1)
        LSG_NTN_sorted=-LSL_NTN_sorted
        LSG_NTN_sorted = tf.math.pow(10, (LSG_NTN_sorted) / 10)
        LSG_NTN_sorted = LSG_NTN_sorted * tf.cast(LSG_NTN_sorted != 1.0, "float32")
        snr_link_NTN = LSG_NTN_sorted * self.P_Tx_NTN / (self.BW_NTN * self.NF_NTN)
        num_NTN = LSG_min_NTN * self.P_Tx_NTN / (self.BW_NTN * self.NF_NTN)
        denom_NTN = tf.expand_dims(tf.reduce_sum(snr_link_NTN, axis=1), axis=1) - num_NTN + self.PSD
        sinr_NTN_withZeros = num_NTN / denom_NTN
        # sinr_NTN_withZeros=tf.expand_dims(sinr_NTN_withZeros[:,:,:],axis=2)
        if self.FRF_3:
            sinr_NTN_withZeros = (LSG_min_NTN * self.P_Tx_NTN) / (self.BW_NTN * self.NF_NTN * self.PSD)
            # sinr_NTN_withZeros=tf.expand_dims(sinr_NTN_withZeros[:,:,:],axis=2)

        #Keeping track of the offloaded UAVs in all of the batches
        # Offloaded_UEs_perElv = self.data.Offloaded_UEs_perElv
        # Offloaded_UEs_allElv.append(Offloaded_UEs_perElv)

        # Rate Calculation
        # Shanon
        TN_Rate = (1 / tf.expand_dims(BSs_load+1, axis=1)) * self.BW_TN * (tf.math.log(1 + sinr_TN_withZeros) / tf.math.log(2.0)) #The plus one in the BS Load is to add 1UAV per TN cell to the load and to ensure that the BSs have at least 1 UE associated with them
        # SINR Mapping
        # Rate_factor = (1 / tf.expand_dims(BSs_load+1, axis=1)) * self.BW_TN
        # sinr_TN_dB = 10.0* tf.math.log((tf.cast(sinr_TN_withZeros == 0.0, "float32")+sinr_TN_withZeros)) / tf.math.log(10.0) #Converting the SINR to dB. while keeping the zeros for the NTN assosiated users
        # sinr_TN_dB = (tf.cast(sinr_TN_dB == 0.0, "float32") * -10000) + sinr_TN_dB
        # Rate_1 = tf.cast(sinr_TN_dB < -5.1470, "float32")*Rate_factor*0.0
        # Rate_2 = tf.cast(sinr_TN_dB >= -5.1470, "float32") * tf.cast(sinr_TN_dB < -3.1800, "float32")*Rate_factor*0.2344
        # Rate_3 = tf.cast(sinr_TN_dB >= -3.1800, "float32") *tf.cast(sinr_TN_dB < -1.2530, "float32")*Rate_factor * 0.3770
        # Rate_4 = tf.cast(sinr_TN_dB >= -1.2530, "float32") *tf.cast(sinr_TN_dB < 0.7610, "float32")*Rate_factor * 0.6016
        # Rate_5 = tf.cast(sinr_TN_dB >= 0.7610, "float32") *tf.cast(sinr_TN_dB < 2.6990, "float32") * Rate_factor * 0.8770
        # Rate_6 = tf.cast(sinr_TN_dB >= 2.6990, "float32") * tf.cast(sinr_TN_dB < 4.6940,"float32") * Rate_factor * 1.1758
        # Rate_7 = tf.cast(sinr_TN_dB >= 4.6940, "float32") * tf.cast(sinr_TN_dB < 6.5250,"float32") * Rate_factor * 1.4766
        # Rate_8 = tf.cast(sinr_TN_dB >= 6.5250, "float32") * tf.cast(sinr_TN_dB < 8.5730,"float32") * Rate_factor * 1.9141
        # Rate_9 = tf.cast(sinr_TN_dB >= 8.5730, "float32") * tf.cast(sinr_TN_dB < 10.3660, "float32") * Rate_factor * 2.4063
        # Rate_10 = tf.cast(sinr_TN_dB >= 10.3660, "float32") * tf.cast(sinr_TN_dB < 12.2890,"float32") * Rate_factor * 2.7305
        # Rate_11 = tf.cast(sinr_TN_dB >= 12.2890, "float32") * tf.cast(sinr_TN_dB < 14.1730,"float32") * Rate_factor * 3.3223
        # Rate_12 = tf.cast(sinr_TN_dB >= 14.1730, "float32") * tf.cast(sinr_TN_dB < 15.8880, "float32") * Rate_factor * 3.9023
        # Rate_13 = tf.cast(sinr_TN_dB >= 15.8880, "float32") * tf.cast(sinr_TN_dB < 17.8140,"float32") * Rate_factor * 4.5234
        # Rate_14 = tf.cast(sinr_TN_dB >= 17.8140, "float32") * tf.cast(sinr_TN_dB < 19.8290, "float32") * Rate_factor * 5.1152
        # Rate_15 = tf.cast(sinr_TN_dB >= 19.8290, "float32")*Rate_factor*5.5547
        # TN_Rate = Rate_1+Rate_2+Rate_3+Rate_4+Rate_5+Rate_6+Rate_7+Rate_8+Rate_9+Rate_10+Rate_11+Rate_12+Rate_13+Rate_14+Rate_15

        if Offloaded_UEs_perc!=0:
            sinr_NTN_dB = 10.0 * tf.math.log((tf.cast(sinr_NTN_withZeros == 0.0, "float32") + sinr_NTN_withZeros)) / tf.math.log(10.0)  # Converting the SINR to dB. while keeping the zeros for the TN assosiated users
            sinr_NTN_dB = (tf.cast(sinr_NTN_dB == 0.0, "float32") * -10000) + sinr_NTN_dB
            Rate_NTN_1 = tf.cast(sinr_NTN_dB < -5.1470, "float32") * Rate_factor * 0.0
            Rate_NTN_2 = tf.cast(sinr_NTN_dB >= -5.1470, "float32") * (1 / (720 * Offloaded_UEs_perc)) * 1e7 * (tf.math.log(1 + sinr_NTN_withZeros) / tf.math.log(2.0))
            NTN_Rate = Rate_NTN_1 + Rate_NTN_2
        else:
            NTN_Rate = 0.0
        Rate = TN_Rate + NTN_Rate

        #Reporting the rates of all batches and all users for CDF uses
        Rate_TNandNTN = Rate
        bool_mask0 = tf.not_equal(Rate_TNandNTN, -1.0) #It is -1 here when we want to keep the zero rates for outage users
        Rate_TNandNTN = tf.boolean_mask(Rate_TNandNTN, bool_mask0)

        Rate_forLog=(tf.cast(Rate == 0.0, "float32")*1.000002302587744)+Rate

        # Max-Product [Rasoul Paper, bias variable introduced]
        alpha_1 = 0.01  # This is to avoid the effect of the bad UEs
        alpha_2 = 0.01  # This is to avoid the effect of the very good UEs
        Rate_forLog = (alpha_1 + 1 / (Rate_forLog + alpha_2))

        #Sum of the log Rate objective
        Rate_log=tf.math.log(Rate_forLog) / tf.math.log(10.0)
        Rate_sumOftheLog=tf.reduce_sum(Rate_log, axis=2)
        Rate = tf.reduce_mean(Rate_sumOftheLog, axis=0)


        return Rate, Rate_TNandNTN

    # # ------------ BO Multi-obj SINR
    def BO_Multi_Obj_Cooridor(self, sinr_TN_UAVs_Corridors, sinr_TN_GUEs, alpha):

        sinr_total_UAVs = sinr_TN_UAVs_Corridors
        sinr_total_GUEs = sinr_TN_GUEs

        #Sum of the log of SINRs, which already in dB scale, equivalent to mutipling in linear scale
        SINR_sumOftheLog_UAVs = tf.reduce_sum(10 * (tf.math.log(sinr_TN_UAVs_Corridors) / tf.math.log(10.0)), axis=2)
        SINR_sumOftheLog_GUEs  = tf.reduce_sum(10 * (tf.math.log(sinr_TN_GUEs) / tf.math.log(10.0)), axis=2)
        # Normalizing over number of deployed users
        SINR_sumOftheLog_UAVs = SINR_sumOftheLog_UAVs/(tf.floor(self.UAV_ratio*self.Nuser_drop))
        SINR_sumOftheLog_GUEs = SINR_sumOftheLog_GUEs/(tf.round(self.GUE_ratio*self.Nuser_drop))
        # Avraging over M.C runs
        SINR_sumOftheLog_Obj_UAVs = tf.reduce_mean(SINR_sumOftheLog_UAVs, axis=0)
        SINR_sumOftheLog_Obj_GUEs = tf.reduce_mean(SINR_sumOftheLog_GUEs, axis=0)

        #SINR Objective (alpha=1 UAVs only, alpha=0 GUEs only)
        SINR_sumOftheLog_Obj = (alpha)*(SINR_sumOftheLog_Obj_UAVs)+(1-alpha)*(SINR_sumOftheLog_Obj_GUEs)

        #Rate Calculations and Obj
        Rate_TN_UAVs = ((1.0 / 15.0) * self.BW_TN * (tf.math.log(1 + sinr_TN_UAVs_Corridors) / tf.math.log(2.0)))/(1e3)
        Rate_total_UAVs = Rate_TN_UAVs
        Rate_TN_GUEs = ((1.0 / 15.0) * self.BW_TN * (tf.math.log(1 + sinr_TN_GUEs) / tf.math.log(2.0)))/(1e3)
        Rate_total_GUEs = Rate_TN_GUEs
        #Sum of the log of Rates, which already in KHz scale, equivalent to mutipling in linear scale
        Rate_sumOftheLog_UAVs = tf.reduce_sum(tf.math.log(Rate_total_UAVs) / tf.math.log(10.0), axis=2)
        Rate_sumOftheLog_GUEs = tf.reduce_sum(tf.math.log(Rate_total_GUEs) / tf.math.log(10.0), axis=2)
        # Normalizing over number of deployed users
        Rate_sumOftheLog_UAVs = Rate_sumOftheLog_UAVs/(tf.floor(self.UAV_ratio*self.Nuser_drop))
        Rate_sumOftheLog_GUEs = Rate_sumOftheLog_GUEs/(tf.round(self.GUE_ratio*self.Nuser_drop))
        # Avraging over M.C runs
        Rate_sumOftheLog_Obj_UAVs = tf.reduce_mean(Rate_sumOftheLog_UAVs, axis=0)
        Rate_sumOftheLog_Obj_GUEs = tf.reduce_mean(Rate_sumOftheLog_GUEs, axis=0)
        #Rate Objective (alpha=1 UAVs only, alpha=0 GUEs only)
        Rate_sumOftheLog_Obj = (alpha)*(Rate_sumOftheLog_Obj_UAVs)+(1-alpha)*(Rate_sumOftheLog_Obj_GUEs)

        bool_mask = tf.not_equal(sinr_total_UAVs, 0)
        sinr_total_UAVs = tf.boolean_mask(sinr_total_UAVs, bool_mask)

        bool_mask1 = tf.not_equal(sinr_total_GUEs, 0)
        sinr_total_GUEs = tf.boolean_mask(sinr_total_GUEs, bool_mask1)


        return SINR_sumOftheLog_Obj, Rate_sumOftheLog_Obj, sinr_total_UAVs, sinr_total_GUEs

    ## Serving BSs indexes and UAVs locations
    def Cell_id(self, LSG, Xuser_UAVs):

        LSL_TN=-LSG
        LSL_min_TN = tf.math.reduce_min(LSL_TN, axis=1, keepdims=True)
        LSL_TN_sorted = tf.sort(LSL_TN, axis=1)

        #Finding the BSs ids of served UAVs
        indexing = tf.cast(LSL_TN == LSL_min_TN, "float32")
        BSs_id = tf.expand_dims(tf.expand_dims(tf.range(1, 58, dtype=tf.float32), 0), 2) * indexing
        BSs_id = tf.reduce_sum(BSs_id, axis=1)

        # Min for UAVs, this is when the antenna beawidth is so small and unlucky users see all three sectors the same. Because LOS, shadowing and PL are all the same and antenna backlobe is fixed to -22dB
        # BSs_id = BSs_id + tf.cast(BSs_id == 0.0, "float32")*1e3
        # BSs_id = tf.reduce_min(BSs_id, axis=1)

        # Max for GUEs
        # BSs_id = tf.reduce_max(BSs_id, axis=1)

        bool_mask1 = tf.not_equal(BSs_id, 1e9)
        BSs_id = tf.boolean_mask(BSs_id, bool_mask1)

        Xuser_UAVs_x  = Xuser_UAVs[:,:,0]
        bool_mask2 = tf.not_equal(Xuser_UAVs_x, 1e9)
        Xuser_UAVs_x = tf.boolean_mask(Xuser_UAVs_x, bool_mask2)

        Xuser_UAVs_y  = Xuser_UAVs[:,:,1]
        bool_mask2 = tf.not_equal(Xuser_UAVs_y, 1e9)
        Xuser_UAVs_y = tf.boolean_mask(Xuser_UAVs_y, bool_mask2)

        return BSs_id, Xuser_UAVs_x, Xuser_UAVs_y

    # # ------------ BO Multi-obj RSS
    def BO_Multi_Obj_Cooridor_RSS(self, LSG_UAVs_Corridors, LSG_GUEs, alpha):

        LSL_UAVs = -LSG_UAVs_Corridors
        LSL_min_UAVs = tf.math.reduce_min(LSL_UAVs, axis=1, keepdims=True)
        LSG_min_UAVs = tf.math.pow(10, (-LSL_min_UAVs) / 10)
        RSS_UAVs = LSG_min_UAVs * self.P_Tx_TN
        RSS_UAVs_dBm = 10 * (tf.math.log(RSS_UAVs) / tf.math.log(10.0)) + 30.0 #+30 here is to convert the RSS to dBm

        LSL_GUEs = -LSG_GUEs
        LSL_min_GUEs = tf.math.reduce_min(LSL_GUEs, axis=1, keepdims=True)
        LSG_min_GUEs = tf.math.pow(10, (-LSL_min_GUEs) / 10)
        RSS_GUEs = LSG_min_GUEs * self.P_Tx_TN
        RSS_GUEs_dBm = 10 * (tf.math.log(RSS_GUEs) / tf.math.log(10.0)) + 30.0 #+30 here is to convert the RSS to dBm

        #Sum of the RSS in dBm
        RSS_sum_UAVs = tf.reduce_sum(RSS_UAVs_dBm, axis=2)
        RSS_sum_GUEs  = tf.reduce_sum(RSS_GUEs_dBm, axis=2)
        # Normalizing over number of deployed users
        RSS_sum_UAVs = RSS_sum_UAVs/(tf.floor(self.UAV_ratio*self.Nuser_drop))
        RSS_sum_GUEs = RSS_sum_GUEs/(tf.round(self.GUE_ratio*self.Nuser_drop))
        # Avraging over M.C runs
        RSS_sum_Obj_UAVs = tf.reduce_mean(RSS_sum_UAVs, axis=0)
        RSS_sum_Obj_GUEs = tf.reduce_mean(RSS_sum_GUEs, axis=0)
        #SINR Objective (alpha=1 UAVs only, alpha=0 GUEs only)
        RSS_sum_Obj = (alpha)*(RSS_sum_Obj_UAVs)+(1-alpha)*(RSS_sum_Obj_GUEs)

        #Rate Calculations and Obj
        SNR_UAVs = RSS_UAVs / (self.BW_TN * self.NF_TN * self.PSD)
        Rate_UAVs = ((1.0 / 15.0) * self.BW_TN * (tf.math.log(1 + SNR_UAVs) / tf.math.log(2.0))) / (1e3)
        SNR_GUEs = RSS_GUEs / (self.BW_TN * self.NF_TN * self.PSD)
        Rate_GUEs = ((1.0 / 15.0) * self.BW_TN * (tf.math.log(1 + SNR_GUEs) / tf.math.log(2.0))) / (1e3)
        Rate_total_UAVs = Rate_UAVs
        Rate_total_GUEs = Rate_GUEs
        #Sum of the log of Rates, which already in KHz scale, equivalent to mutipling in linear scale
        Rate_sumOftheLog_UAVs = tf.reduce_sum(tf.math.log(Rate_total_UAVs) / tf.math.log(10.0), axis=2)
        Rate_sumOftheLog_GUEs = tf.reduce_sum(tf.math.log(Rate_total_GUEs) / tf.math.log(10.0), axis=2)
        # Normalizing over number of deployed users
        Rate_sumOftheLog_UAVs = Rate_sumOftheLog_UAVs/(tf.floor(self.UAV_ratio*self.Nuser_drop))
        Rate_sumOftheLog_GUEs = Rate_sumOftheLog_GUEs/(tf.round(self.GUE_ratio*self.Nuser_drop))
        # Avraging over M.C runs
        Rate_sumOftheLog_Obj_UAVs = tf.reduce_mean(Rate_sumOftheLog_UAVs, axis=0)
        Rate_sumOftheLog_Obj_GUEs = tf.reduce_mean(Rate_sumOftheLog_GUEs, axis=0)
        #Rate Objective (alpha=1 UAVs only, alpha=0 GUEs only)
        Rate_sumOftheLog_Obj = (alpha)*(Rate_sumOftheLog_Obj_UAVs)+(1-alpha)*(Rate_sumOftheLog_Obj_GUEs)

        bool_mask = tf.not_equal(RSS_UAVs_dBm, 1e6)
        RSS_UAVs_dBm = tf.boolean_mask(RSS_UAVs_dBm, bool_mask)

        bool_mask1 = tf.not_equal(RSS_GUEs_dBm, 1e6)
        RSS_GUEs_dBm = tf.boolean_mask(RSS_GUEs_dBm, bool_mask1)


        return RSS_sum_Obj, Rate_sumOftheLog_Obj, RSS_UAVs_dBm, RSS_GUEs_dBm

    # ------------ DL SINR
    def BO_GUEs_test_Obj_Rates(self, LSG, LSG_UAVs_Corridors,  P_Tx_TN, alpha):

        #This is for GUEs
        P_Tx_TN_GUEs = tf.slice(P_Tx_TN, [0, 0, 0], [tf.shape(LSG)[0], 57, tf.shape(LSG)[-1]])
        P_Tx_TN_GUEs = tf.math.pow(10.0, (P_Tx_TN_GUEs - 30.0) / 10.0)  # Conver to linear
        LSL_TN_GUEs = -LSG
        LSL_min_TN_GUEs = tf.math.reduce_min(LSL_TN_GUEs, axis=1, keepdims=True)

        #Finding the BSs load
        indexing_GUEs = tf.cast(-LSG == LSL_min_TN_GUEs, "float32")

        #This is to make sure that if user has same LSG to all 3 sectors, we choose one of the sectors randomly
        mask1 = tf.random.uniform(indexing_GUEs.shape, 1, 57)*indexing_GUEs
        random_selection1 = tf.tile(tf.expand_dims(tf.reduce_max(mask1, axis=1),axis=1) , [1,57,1])
        indexing_GUEs = tf.cast(random_selection1 == mask1, "float32")

        BS_load_GUEs =tf.tile(tf.expand_dims(tf.reduce_sum(indexing_GUEs, axis=2),axis=2), [1, 1, tf.round(self.GUE_ratio*self.Nuser_drop)])

        LSG_min_TN_GUEs = tf.math.pow(10, (-LSL_min_TN_GUEs) / 10)
        LSG_min_TN_GUEs = LSG_min_TN_GUEs*tf.cast(LSG_min_TN_GUEs != 1.0, "float32")
        LSG_linear_GUEs = tf.math.pow(10, (LSG) / 10)

        snr_link_GUEs = LSG_linear_GUEs * P_Tx_TN_GUEs / (self.BW_TN * self.NF_TN)
        P_Tx_TN_assigned_GUEs = tf.expand_dims(tf.reduce_max(tf.abs(P_Tx_TN_GUEs * tf.cast(LSG_linear_GUEs == tf.tile(LSG_min_TN_GUEs, [1, 57, 1]), "float32")), axis=1), axis=1)
        num_TN_GUEs = LSG_min_TN_GUEs * P_Tx_TN_assigned_GUEs / (self.BW_TN * self.NF_TN)
        denom_TN_GUEs = tf.expand_dims(tf.reduce_sum(snr_link_GUEs, axis=1), axis=1) - num_TN_GUEs + self.PSD
        sinr_TN_GUEs = num_TN_GUEs / denom_TN_GUEs
        #Making upper bound and lower bound as 25dB and -25dB
        sinr_TN_GUEs =tf.cast(sinr_TN_GUEs < 316.2, "float32")*sinr_TN_GUEs + tf.cast(sinr_TN_GUEs >= 316.2, "float32")*316.2
        sinr_TN_GUEs = tf.cast(sinr_TN_GUEs > 0.003, "float32") * sinr_TN_GUEs + tf.cast(sinr_TN_GUEs <= 0.003,"float32") * 0.003

        # This is for UAVs
        P_Tx_TN_UAVs = tf.slice(P_Tx_TN, [0, 0, 0], [tf.shape(LSG_UAVs_Corridors)[0], 57, tf.shape(LSG_UAVs_Corridors)[-1]])
        P_Tx_TN_UAVs = tf.math.pow(10.0, (P_Tx_TN_UAVs - 30.0) / 10.0)
        LSL_TN_UAVs = -LSG_UAVs_Corridors
        LSL_min_TN_UAVs = tf.math.reduce_min(LSL_TN_UAVs, axis=1, keepdims=True)

        #Finding the BSs load
        indexing_UAVs = tf.cast(-LSG_UAVs_Corridors == LSL_min_TN_UAVs, "float32")

        # This is to make sure that if user has same LSG to all 3 sectors, we choose one of the sectors randomly
        mask = tf.random.uniform(indexing_UAVs.shape, 1, 57)*indexing_UAVs
        random_selection = tf.tile(tf.expand_dims(tf.reduce_max(mask, axis=1),axis=1) , [1,57,1])
        indexing_UAVs = tf.cast(random_selection == mask, "float32")
        BS_load_UAVs =tf.tile(tf.expand_dims(tf.reduce_sum(indexing_UAVs, axis=2),axis=2), [1, 1, tf.round(self.UAV_ratio*self.Nuser_drop)])

        LSG_min_TN_UAVs = tf.math.pow(10, (-LSL_min_TN_UAVs) / 10)
        LSG_min_TN_UAVs = LSG_min_TN_UAVs*tf.cast(LSG_min_TN_UAVs != 1.0, "float32")
        LSG_linear_UAVs = tf.math.pow(10, (LSG_UAVs_Corridors) / 10)

        snr_link_UAVs = LSG_linear_UAVs * P_Tx_TN_UAVs / (self.BW_TN * self.NF_TN)
        P_Tx_TN_assigned_UAVs = tf.expand_dims(tf.reduce_max(tf.abs(P_Tx_TN_UAVs * tf.cast(LSG_linear_UAVs == tf.tile(LSG_min_TN_UAVs, [1, 57, 1]), "float32")), axis=1), axis=1)
        num_TN_UAVs = LSG_min_TN_UAVs * P_Tx_TN_assigned_UAVs / (self.BW_TN * self.NF_TN)
        denom_TN_UAVs = tf.expand_dims(tf.reduce_sum(snr_link_UAVs, axis=1), axis=1) - num_TN_UAVs + self.PSD
        sinr_TN_UAVs = num_TN_UAVs / denom_TN_UAVs
        #Making upper bound and lower bound as 25dB and -25dB
        sinr_TN_UAVs =tf.cast(sinr_TN_UAVs < 316.2, "float32")*sinr_TN_UAVs + tf.cast(sinr_TN_UAVs >= 316.2, "float32")*316.2
        sinr_TN_UAVs = tf.cast(sinr_TN_UAVs > 0.003, "float32") * sinr_TN_UAVs + tf.cast(sinr_TN_UAVs <= 0.003,"float32") * 0.003

        #BS load combined GUEs and UAVs
        BS_load_combined = tf.expand_dims(BS_load_GUEs[:,:,0]+BS_load_UAVs[:,:,0],axis=2)
        BS_load_GUEs = tf.tile(BS_load_combined,[1, 1, tf.round(self.GUE_ratio * self.Nuser_drop)])
        BS_load_UAVs = tf.tile(BS_load_combined, [1, 1, tf.round(self.UAV_ratio*self.Nuser_drop)])
        BS_load_assigned_GUEs = tf.expand_dims(tf.reduce_max(tf.abs(BS_load_GUEs * tf.cast(LSG_linear_GUEs == tf.tile(LSG_min_TN_GUEs, [1, 57, 1]), "float32")), axis=1),axis=1)
        BS_load_assigned_UAVs = tf.expand_dims(tf.reduce_max(tf.abs(BS_load_UAVs * tf.cast(LSG_linear_UAVs == tf.tile(LSG_min_TN_UAVs, [1, 57, 1]), "float32")), axis=1),axis=1)

        #Rate Calculations and Obj GUEs
        Rate_GUEs = ((1.0 / BS_load_assigned_GUEs) * self.BW_TN * (tf.math.log(1 + sinr_TN_GUEs) / tf.math.log(2.0)))/1e6

        #Sum of the log of Rates, which already in MHz scale, equivalent to mutipling in linear scale
        Rate_sumOftheLog_GUEs = tf.reduce_sum(tf.math.log(Rate_GUEs) / tf.math.log(10.0) , axis=2)

        # Normalizing over number of deployed users
        #Rate_sumOftheLog_GUEs = Rate_sumOftheLog_GUEs/(tf.round(self.GUE_ratio*self.Nuser_drop))

        # Avraging over M.C runs
        Rate_sumOftheLog_Obj_GUEs = tf.reduce_mean(Rate_sumOftheLog_GUEs, axis=0)

        bool_mask2 = tf.not_equal(Rate_GUEs, -10.0)
        Rate_GUEs = tf.boolean_mask(Rate_GUEs, bool_mask2)

        #Rate Calculations and Obj UAVs
        Rate_UAVs = ((1.0 / BS_load_assigned_UAVs) * self.BW_TN * (tf.math.log(1 + sinr_TN_UAVs) / tf.math.log(2.0)))/1e6

        #Sum of the log of Rates, which already in KHz scale, equivalent to mutipling in linear scale
        Rate_sumOftheLog_UAVs = tf.reduce_sum(tf.math.log(Rate_UAVs) / tf.math.log(10.0) , axis=2)

        # Normalizing over number of deployed users
        #Rate_sumOftheLog_UAVs = Rate_sumOftheLog_UAVs/(tf.round(self.UAV_ratio*self.Nuser_drop))

        # Avraging over M.C runs
        Rate_sumOftheLog_Obj_UAVs = tf.reduce_mean(Rate_sumOftheLog_UAVs, axis=0)

        bool_mask3_UAVs = tf.not_equal(Rate_UAVs, -10.0)
        Rate_UAVs = tf.boolean_mask(Rate_UAVs, bool_mask3_UAVs)

        #Performance Metric
        Rate_sumOftheLog_Obj = (alpha) * (Rate_sumOftheLog_Obj_UAVs)*0.1 + (1 - alpha) * (Rate_sumOftheLog_Obj_GUEs)*0.1

        return Rate_sumOftheLog_Obj, Rate_GUEs, Rate_UAVs

    def BO_Obj_Rates_and_Outage(self, LSG, LSG_UAVs_Corridors,  P_Tx_TN, D, D_2d, alpha):

        #This is for GUEs
        if self.N==1:
            P_Tx_TN_GUEs = tf.slice(P_Tx_TN, [0, 0, 0], [tf.shape(LSG)[0], 21, tf.shape(LSG)[-1]])
        elif self.N==2:
            P_Tx_TN_GUEs = tf.slice(P_Tx_TN, [0, 0, 0], [tf.shape(LSG)[0], 57, tf.shape(LSG)[-1]])
        P_Tx_TN_GUEs = tf.math.pow(10.0, (P_Tx_TN_GUEs - 30.0) / 10.0)  # Conver to linear
        LSL_TN_GUEs = -LSG
        LSL_min_TN_GUEs = tf.math.reduce_min(LSL_TN_GUEs, axis=1, keepdims=True)

        #Finding the BSs load
        indexing_GUEs = tf.cast(-LSG == LSL_min_TN_GUEs, "float32")

        # #Finding served BS distances
        # D_GUEs = D[:, 0:57, 0:tf.shape(indexing_GUEs)[-1]] * indexing_GUEs
        # D_2d_GUEs = D_2d[:, 0:57, 0:tf.shape(indexing_GUEs)[-1]] * indexing_GUEs
        # D_served = tf.reduce_sum(D_GUEs,axis=1)
        # D_served = tf.reshape(D_served, [-1])
        # D_2d_served = tf.reduce_sum(D_2d_GUEs, axis=1)
        # D_2d_served = tf.reshape(D_2d_served, [-1])

        #This is to make sure that if user has same LSG to all 3 sectors, we choose one of the sectors randomly
        if self.N==1:
            mask1 = tf.random.uniform(indexing_GUEs.shape, 1, 21) * indexing_GUEs
            random_selection1 = tf.tile(tf.expand_dims(tf.reduce_max(mask1, axis=1), axis=1), [1, 21, 1])
        elif self.N==2:
            mask1 = tf.random.uniform(indexing_GUEs.shape, 1, 57)*indexing_GUEs
            random_selection1 = tf.tile(tf.expand_dims(tf.reduce_max(mask1, axis=1),axis=1) , [1,57,1])
        indexing_GUEs = tf.cast(random_selection1 == mask1, "float32")

        BS_load_GUEs =tf.tile(tf.expand_dims(tf.reduce_sum(indexing_GUEs, axis=2),axis=2), [1, 1, tf.round(self.GUE_ratio*self.Nuser_drop)])

        LSG_min_TN_GUEs = tf.math.pow(10, (-LSL_min_TN_GUEs) / 10)
        LSG_min_TN_GUEs = LSG_min_TN_GUEs*tf.cast(LSG_min_TN_GUEs != 1.0, "float32")
        LSG_linear_GUEs = tf.math.pow(10, (LSG) / 10)

        snr_link_GUEs = LSG_linear_GUEs * P_Tx_TN_GUEs / (self.BW_TN * self.NF_TN)
        if self.N==1:
            P_Tx_TN_assigned_GUEs = tf.expand_dims(tf.reduce_max(tf.abs(P_Tx_TN_GUEs * tf.cast(LSG_linear_GUEs == tf.tile(LSG_min_TN_GUEs, [1, 21, 1]), "float32")),axis=1), axis=1)
        elif self.N==2:
            P_Tx_TN_assigned_GUEs = tf.expand_dims(tf.reduce_max(tf.abs(P_Tx_TN_GUEs * tf.cast(LSG_linear_GUEs == tf.tile(LSG_min_TN_GUEs, [1, 57, 1]), "float32")), axis=1), axis=1)
        num_TN_GUEs = LSG_min_TN_GUEs * P_Tx_TN_assigned_GUEs / (self.BW_TN * self.NF_TN)
        denom_TN_GUEs = tf.expand_dims(tf.reduce_sum(snr_link_GUEs, axis=1), axis=1) - num_TN_GUEs + self.PSD
        sinr_TN_GUEs = num_TN_GUEs / denom_TN_GUEs
        #Making upper bound and lower bound as 25dB and -25dB
        sinr_TN_GUEs =tf.cast(sinr_TN_GUEs < 316.2, "float32")*sinr_TN_GUEs + tf.cast(sinr_TN_GUEs >= 316.2, "float32")*316.2
        sinr_TN_GUEs = tf.cast(sinr_TN_GUEs > 0.003, "float32") * sinr_TN_GUEs + tf.cast(sinr_TN_GUEs <= 0.003,"float32") * 0.003

        # This is for UAVs
        if self.N==1:
            P_Tx_TN_UAVs = tf.slice(P_Tx_TN, [0, 0, 0],[tf.shape(LSG_UAVs_Corridors)[0], 21, tf.shape(LSG_UAVs_Corridors)[-1]])
        elif self.N==2:
            P_Tx_TN_UAVs = tf.slice(P_Tx_TN, [0, 0, 0], [tf.shape(LSG_UAVs_Corridors)[0], 57, tf.shape(LSG_UAVs_Corridors)[-1]])
        P_Tx_TN_UAVs = tf.math.pow(10.0, (P_Tx_TN_UAVs - 30.0) / 10.0)
        LSL_TN_UAVs = -LSG_UAVs_Corridors
        LSL_min_TN_UAVs = tf.math.reduce_min(LSL_TN_UAVs, axis=1, keepdims=True)

        #Finding the BSs load
        indexing_UAVs = tf.cast(-LSG_UAVs_Corridors == LSL_min_TN_UAVs, "float32")

        # This is to make sure that if user has same LSG to all 3 sectors, we choose one of the sectors randomly
        if self.N == 1:
            mask = tf.random.uniform(indexing_UAVs.shape, 1, 21) * indexing_UAVs
            random_selection = tf.tile(tf.expand_dims(tf.reduce_max(mask, axis=1), axis=1), [1, 21, 1])
        elif self.N == 2:
            mask = tf.random.uniform(indexing_UAVs.shape, 1, 57)*indexing_UAVs
            random_selection = tf.tile(tf.expand_dims(tf.reduce_max(mask, axis=1),axis=1) , [1,57,1])
        indexing_UAVs = tf.cast(random_selection == mask, "float32")
        BS_load_UAVs =tf.tile(tf.expand_dims(tf.reduce_sum(indexing_UAVs, axis=2),axis=2), [1, 1, tf.round(self.UAV_ratio*self.Nuser_drop)])

        BS_load_UAVs =tf.tile(tf.expand_dims(tf.reduce_sum(indexing_UAVs, axis=2),axis=2), [1, 1, tf.round(self.UAV_ratio*self.Nuser_drop)])

        LSG_min_TN_UAVs = tf.math.pow(10, (-LSL_min_TN_UAVs) / 10)
        LSG_min_TN_UAVs = LSG_min_TN_UAVs*tf.cast(LSG_min_TN_UAVs != 1.0, "float32")
        LSG_linear_UAVs = tf.math.pow(10, (LSG_UAVs_Corridors) / 10)

        snr_link_UAVs = LSG_linear_UAVs * P_Tx_TN_UAVs / (self.BW_TN * self.NF_TN)
        if self.N==1:
            P_Tx_TN_assigned_UAVs = tf.expand_dims(tf.reduce_max(tf.abs(P_Tx_TN_UAVs * tf.cast(LSG_linear_UAVs == tf.tile(LSG_min_TN_UAVs, [1, 21, 1]), "float32")),axis=1), axis=1)
        elif self.N==2:
            P_Tx_TN_assigned_UAVs = tf.expand_dims(tf.reduce_max(tf.abs(P_Tx_TN_UAVs * tf.cast(LSG_linear_UAVs == tf.tile(LSG_min_TN_UAVs, [1, 57, 1]), "float32")), axis=1), axis=1)
        num_TN_UAVs = LSG_min_TN_UAVs * P_Tx_TN_assigned_UAVs / (self.BW_TN * self.NF_TN)
        denom_TN_UAVs = tf.expand_dims(tf.reduce_sum(snr_link_UAVs, axis=1), axis=1) - num_TN_UAVs + self.PSD
        sinr_TN_UAVs = num_TN_UAVs / denom_TN_UAVs
        #Making upper bound and lower bound as 25dB and -25dB
        sinr_TN_UAVs =tf.cast(sinr_TN_UAVs < 316.2, "float32")*sinr_TN_UAVs + tf.cast(sinr_TN_UAVs >= 316.2, "float32")*316.2
        sinr_TN_UAVs = tf.cast(sinr_TN_UAVs > 0.003, "float32") * sinr_TN_UAVs + tf.cast(sinr_TN_UAVs <= 0.003,"float32") * 0.003

        #BS load combined GUEs and UAVs
        BS_load_combined = tf.expand_dims(BS_load_GUEs[:,:,0]+BS_load_UAVs[:,:,0],axis=2)
        BS_load_GUEs = tf.tile(BS_load_combined,[1, 1, tf.round(self.GUE_ratio * self.Nuser_drop)])
        BS_load_UAVs = tf.tile(BS_load_combined, [1, 1, tf.round(self.UAV_ratio*self.Nuser_drop)])
        if self.N==1:
            BS_load_GUEs = tf.tile(BS_load_combined,[1, 1, LSG_linear_GUEs.shape[2]])
            BS_load_UAVs = tf.tile(BS_load_combined, [1, 1, LSG_linear_UAVs.shape[2]])
        if self.N==1:
            BS_load_assigned_GUEs = tf.expand_dims(tf.reduce_max(tf.abs(BS_load_GUEs * tf.cast(LSG_linear_GUEs == tf.tile(LSG_min_TN_GUEs, [1, 21, 1]), "float32")),axis=1), axis=1)
            BS_load_assigned_UAVs = tf.expand_dims(tf.reduce_max(tf.abs(BS_load_UAVs * tf.cast(LSG_linear_UAVs == tf.tile(LSG_min_TN_UAVs, [1, 21, 1]), "float32")),axis=1), axis=1)
        elif self.N==2:
            BS_load_assigned_GUEs = tf.expand_dims(tf.reduce_max(tf.abs(BS_load_GUEs * tf.cast(LSG_linear_GUEs == tf.tile(LSG_min_TN_GUEs, [1, 57, 1]), "float32")), axis=1),axis=1)
            BS_load_assigned_UAVs = tf.expand_dims(tf.reduce_max(tf.abs(BS_load_UAVs * tf.cast(LSG_linear_UAVs == tf.tile(LSG_min_TN_UAVs, [1, 57, 1]), "float32")), axis=1),axis=1)

        #Rate Calculations and Obj GUEs
        Rate_GUEs = ((1.0 / BS_load_assigned_GUEs) * self.BW_TN * (tf.math.log(1 + sinr_TN_GUEs) / tf.math.log(2.0)))/1e6

        #Sum of the log of Rates, which already in KHz scale, equivalent to mutipling in linear scale
        Rate_sumOftheLog_GUEs = tf.reduce_sum(tf.math.log(Rate_GUEs)/ tf.math.log(2.0), axis=2)
        # Avraging over M.C runs
        Rate_sumOftheLog_Obj_GUEs = tf.reduce_mean(Rate_sumOftheLog_GUEs, axis=0)
        bool_mask2 = tf.not_equal(Rate_GUEs, -10.0)
        Rate_GUEs = tf.boolean_mask(Rate_GUEs, bool_mask2)

        #Rate Calculations and Obj UAVs
        Rate_UAVs = ((1.0 / BS_load_assigned_UAVs) * self.BW_TN * (tf.math.log(1 + sinr_TN_UAVs) / tf.math.log(2.0)))/1e6
        #Sum of the log of Rates, which already in MHz scale, equivalent to mutipling in linear scale
        Rate_sumOftheLog_UAVs = tf.reduce_sum(tf.math.log(Rate_UAVs)/ tf.math.log(2.0), axis=2)
        # Avraging over M.C runs
        Rate_sumOftheLog_Obj_UAVs = tf.reduce_mean(Rate_sumOftheLog_UAVs, axis=0)
        bool_mask3_UAVs = tf.not_equal(Rate_UAVs, -10.0)
        Rate_UAVs = tf.boolean_mask(Rate_UAVs, bool_mask3_UAVs)

        #Performance Metric
        Rate_sumOftheLog_Obj1 = (alpha) * (Rate_sumOftheLog_Obj_UAVs) + (1 - alpha) * (Rate_sumOftheLog_Obj_GUEs)

        #Outage term calculations (Jeffrey's paper)
        GUEs_Outage = tf.tile(tf.cast(sinr_TN_GUEs < 0.31622776, "float32"), [1, 57, 1])*indexing_GUEs
        UAVs_Outage = tf.tile(tf.cast(sinr_TN_UAVs < 0.31622776, "float32"), [1, 57, 1]) * indexing_UAVs
        Users_Outage_perBS = tf.reduce_sum(GUEs_Outage, axis=2)  + tf.reduce_sum(UAVs_Outage, axis=2)
        Users_Outage = tf.reduce_sum(tf.reduce_mean(Users_Outage_perBS, axis=0), axis=0)
        Outage_ratio = Users_Outage / 855.0
        Coverage_ratio = 1 - Outage_ratio

        UAVs_Outage_perBS = tf.reduce_sum(UAVs_Outage, axis=2)
        UAVs_Outage = tf.reduce_sum(tf.reduce_mean(UAVs_Outage_perBS, axis=0),axis=0)
        UAVs_Outage_ratio = UAVs_Outage/(0.3333*855.0)
        UAVs_Coverage_ratio = 1-UAVs_Outage_ratio

        #Final Obj of sum log rates
        Rate_sumOftheLog_Obj = Rate_sumOftheLog_Obj1/100.0
        
        return Rate_sumOftheLog_Obj, UAVs_Coverage_ratio, Rate_GUEs, Rate_UAVs