"""
This is the Main Class of the simulator:
    This is the class of the TN/NTN that deploys BSs/LEO and GUEs/UAVs.
    Calculate the distances in 2D and 3D then associate UE to BSs/LEO such that each
    BS will have at least 1 associated user. The call of TN/NTN classes to calculate
    the LSG are performed here as  well.

@authors: Mohamed Benzaghta and Rasoul Nikbakht
"""
import tensorflow as tf
from config import Config
from LargeScaleGainClass import Large_Scale_Gain
from LSG_Class_UAVs import Large_Scale_Gain_drone
from DeploymentClass import  Deployment
import numpy as np
from NTN_LSG_Class import NTN_Large_Scale_Gain
import math

class Terrestrial(Config):

    def __init__(self):
        Config.__init__(self)
        self.Deployment = Deployment()
        self.LSGclass = Large_Scale_Gain()
        self.NTNLSG = NTN_Large_Scale_Gain()
        self.Large_Scale_Gain_drone=Large_Scale_Gain_drone()


    def call(self):

        Xap,Xuser, Xuser_in,Zuser_indoor,Xuser_out,D,D_2d,D_in,D_2d_in,D_2d_building,BS_wrapped_Cord,BS_wrapped_Cord_in,Azi_phi_deg,Elv_thetha_deg,Azi_phi_deg_in, Elv_thetha_deg_in,D_UAV, D_2d_UAV, BS_wrapped_Cord_UAV,Azi_phi_deg_UAV, Elv_thetha_deg_UAV = self.Deployment.Call(self.alpha_factor)

        if self.open_access:
            if self.UAVs:
                if self.one_tier == False:
                    if self.indoor == False:
                        LSL_1, LSG_1, G_Antenna_1, p_LOS_1, pl_1, shadowing_LOS_1, shadowing_NLOS_1 = self.Large_Scale_Gain_drone.call(D[:, 0:19, 0:int(self.GUE_ratio * self.Nuser_drop) + 1],D_2d[:, 0:19, 0:int(self.GUE_ratio * self.Nuser_drop) + 1],Azi_phi_deg[:, 0:19, 0:int(self.GUE_ratio * self.Nuser_drop) + 1],Elv_thetha_deg[:, 0:19, 0:int(self.GUE_ratio * self.Nuser_drop) + 1], self.Zuser,self.BS_tilt, self.BS_HPBW_v)  # GUEs
                        LSL_2, LSG_2, G_Antenna_2, p_LOS_2, pl_2, shadowing_LOS_2, shadowing_NLOS_2 = self.Large_Scale_Gain_drone.call(D[:, 0:19, int(self.GUE_ratio * self.Nuser_drop) + 1:],D_2d[:, 0:19, int(self.GUE_ratio * self.Nuser_drop) + 1:],Azi_phi_deg[:, 0:19, int(self.GUE_ratio * self.Nuser_drop) + 1:],Elv_thetha_deg[:, 0:19, int(self.GUE_ratio * self.Nuser_drop) + 1:],tf.expand_dims(Xuser[:, int(self.GUE_ratio * self.Nuser_drop)+1:, 2], axis=1),self.BS_tilt, self.BS_HPBW_v)  # UAVs

                    if self.indoor == True:
                        LSL_1, LSG_1, G_Antenna_1, p_LOS_1, pl_1, shadowing_LOS_1, shadowing_NLOS_1 = self.LSGclass.call(D[:, 0:19, :],D_2d[:, 0:19, :],D_in[:, 0:19, :], D_2d_in[:, 0:19, :],D_2d_building[:, 0:19, :],Azi_phi_deg[:,0:19, :],Elv_thetha_deg[:,0:19, :],Azi_phi_deg_in[:, 0:19, :],Elv_thetha_deg_in[:, 0:19, :],Zuser_indoor[:, 0:19, :], self.BS_tilt, self.BS_HPBW_v)
                        LSL_2, LSG_2, G_Antenna_2, p_LOS_2, pl_2, shadowing_LOS_2, shadowing_NLOS_2 = self.Large_Scale_Gain_drone.call(D_UAV[:, 0:19, :],D_2d_UAV[:, 0:19, :],Azi_phi_deg_UAV[:, 0:19, :],Elv_thetha_deg_UAV[:, 0:19, :], self.Zuav, self.BS_tilt, self.BS_HPBW_v)
                        #Need to connect them for assigment phase later
                        D = tf.concat([D, D_in,D_UAV], axis=2)
                        D_2d = tf.concat([D_2d, D_2d_in, D_2d_UAV], axis=2)
                        Azi_phi_deg = tf.concat([Azi_phi_deg, Azi_phi_deg_in, Azi_phi_deg_UAV], axis=2)
                        Elv_thetha_deg = tf.concat([Elv_thetha_deg, Elv_thetha_deg_in, Elv_thetha_deg_UAV], axis=2)
                """
                if self.one_tier == True:
                    if self.indoor == False:
                        LSL_1, LSG_1, G_Antenna_1, p_LOS_1, pl_1, shadowing_LOS_1, shadowing_NLOS_1 = self.Large_Scale_Gain_drone.call(D[:,0:7,0:int(self.GUE_ratio*self.Nuser_drop)], D_2d[:,0:7,0:int(self.GUE_ratio*self.Nuser_drop)], Azi_phi_deg[:,0:7,0:int(self.GUE_ratio*self.Nuser_drop)], Elv_thetha_deg[:,0:7,0:int(self.GUE_ratio*self.Nuser_drop)], self.Zuser, self.BS_tilt) #GUEs
                        LSL_2, LSG_2, G_Antenna_2, p_LOS_2, pl_2, shadowing_LOS_2, shadowing_NLOS_2 = self.Large_Scale_Gain_drone.call(D[:, 0:7, int(self.GUE_ratio * self.Nuser_drop):],D_2d[:, 0:7, int(self.GUE_ratio * self.Nuser_drop):],Azi_phi_deg[:, 0:7, int(self.GUE_ratio * self.Nuser_drop):],Elv_thetha_deg[:, 0:7, int(self.GUE_ratio * self.Nuser_drop):], self.Zuav ,self.BS_tilt) #UAVs
                    if self.indoor == True:
                        LSL_1, LSG_1, G_Antenna_1, p_LOS_1, pl_1, shadowing_LOS_1, shadowing_NLOS_1 = self.LSGclass.call(D[:, 0:7, :],D_2d[:, 0:7, :],D_in[:, 0:7, :], D_2d_in[:, 0:7, :],D_2d_building[:, 0:7, :],Azi_phi_deg[:,0:7, :],Elv_thetha_deg[:,0:7, :],Azi_phi_deg_in[:, 0:7, :],Elv_thetha_deg_in[:, 0:7, :],Zuser_indoor[:, 0:7, :], self.BS_tilt)
                        LSL_2, LSG_2, G_Antenna_2, p_LOS_2, pl_2, shadowing_LOS_2, shadowing_NLOS_2 = self.Large_Scale_Gain_drone.call(D_UAV[:, 0:7, :],D_2d_UAV[:, 0:7, :],Azi_phi_deg_UAV[:, 0:7, :],Elv_thetha_deg_UAV[:, 0:7, :], self.Zuav, self.BS_tilt)
                        #Need to connect them for assigment phase later
                        D = tf.concat([D, D_in,D_UAV], axis=2)
                        D_2d = tf.concat([D_2d, D_2d_in, D_2d_UAV], axis=2)
                        Azi_phi_deg = tf.concat([Azi_phi_deg, Azi_phi_deg_in, Azi_phi_deg_UAV], axis=2)
                        Elv_thetha_deg = tf.concat([Elv_thetha_deg, Elv_thetha_deg_in, Elv_thetha_deg_UAV], axis=2)
                """
                LSL = tf.concat([LSL_1, LSL_2], axis=2)
                LSG = tf.concat([LSG_1, LSG_2], axis=2)
                G_Antenna = tf.concat([G_Antenna_1, G_Antenna_2], axis=2)
                p_LOS = tf.concat([p_LOS_1, p_LOS_2], axis=2)
                shadowing_LOS = tf.concat([shadowing_LOS_1, shadowing_LOS_2], axis=2)
                shadowing_NLOS = tf.concat([shadowing_NLOS_1, shadowing_NLOS_2], axis=2)
                pl = tf.concat([pl_1, pl_2], axis=2)

            # from UEs  to sat (all beams)
            if self.one_tier == False:
                if self.N==1:
                    D_2d_Sat = tf.expand_dims(D_2d[:, 3, :],axis=1)  # 3 is the center sat for 1-tier, you can change its coordinates from Deployment class
                    Azi_phi_deg_Sat = tf.expand_dims(Azi_phi_deg[:, 3, :], axis=1)
                    Elv_thetha_deg_Sat = tf.expand_dims(Elv_thetha_deg[:, 3, :], axis=1)
                elif self.N==2:
                    D_2d_Sat = tf.expand_dims(D_2d[:, 19, :],axis=1)  # 19 is the center sat for 2-tier network, you can change its coordinates from Deployment class
                    Azi_phi_deg_Sat = tf.expand_dims(Azi_phi_deg[:, 19, :], axis=1)
                    Elv_thetha_deg_Sat = tf.expand_dims(Elv_thetha_deg[:, 19, :], axis=1)

            if self.one_tier == True:
                D_2d_Sat = tf.expand_dims(D_2d[:, 3, :], axis=1) #3 is the center sat for 1-tier, you can change its coordinates from Deployment class
                Azi_phi_deg_Sat = tf.expand_dims(Azi_phi_deg[:, 3, :], axis=1)
                Elv_thetha_deg_Sat = tf.expand_dims(Elv_thetha_deg[:, 3, :], axis=1)

            NTN_LSL, NTN_LSG, NTN_AntennaGain_dB, NTN_LOS, NTN_PL, NTN_shadowing_LOS, NTN_shadowing_NLOS = self.NTNLSG.call(D_2d_Sat,Azi_phi_deg_Sat,Elv_thetha_deg_Sat,Xuser,self.alpha_factor)

            #Finding the load on each BS
            # LSG_all_TN_GUEs = LSG[:, :, 0:int(self.GUE_ratio * self.Nuser_drop)]
            # BSs_assigment = tf.cast(LSG_all_TN_GUEs == tf.tile(tf.expand_dims(tf.reduce_max(LSG_all_TN_GUEs, axis=1), axis=1), [1, 57, 1]), "float32")
            # BSs_load_all = tf.reduce_sum(BSs_assigment, axis=2)/3.0

            #Contacting LSG of TN and NTN
            if self.LSG_all:
                LSG_all_TN = LSG[:,:,0:int(self.GUE_ratio*self.Nuser_drop)]
                BSs_assigment = tf.cast(LSG_all_TN == tf.tile(tf.expand_dims(tf.reduce_max(LSG_all_TN,axis=1),axis=1),[1,21,1]), "float32")
                BSs_load_all = tf.reduce_sum(BSs_assigment,axis=2) #This is the load of all the BSs
                # BSs_assigment_UAVs = BSs_assigment[:,:, int(self.GUE_ratio * self.Nuser_drop)+1:]
                # BSs_load_UAVs = tf.tile(tf.expand_dims(BSs_load_all,axis=2),[1,1,21]) * BSs_assigment_UAVs
                # BSs_load_UAVs = tf.expand_dims(tf.reduce_sum(BSs_load_UAVs,axis=1),axis=1) #This is the load of the BSs that have UAVs connected to them
            # LSG = tf.concat([LSG + self.P_Tx_dB , NTN_LSG+ self.P_Tx_sat_db+self.sat_bias], axis=1)
            LSL=-LSG

            G_Antenna = tf.concat([G_Antenna, NTN_AntennaGain_dB], axis=1)
            p_LOS = tf.concat([p_LOS, NTN_LOS], axis=1)
            pl = tf.concat([pl, NTN_PL], axis=1)
            shadowing_LOS = tf.concat([shadowing_LOS, NTN_shadowing_LOS], axis=1)
            shadowing_NLOS = tf.concat([shadowing_NLOS, NTN_shadowing_NLOS], axis=1)
            
        if self.indoor==False:
            D_2d=D_2d
            D=D

        if self.sectoring_status:
            if self.open_access:
                if self.one_tier == False:
                    Xap_1 = Xap[:, 0:19, :]
                    D_2d_1 = D_2d[:, 0:19, :]
                    D_1 = D[:, 0:19, :]

                    Xap_1 = tf.tile(Xap_1, [1, 3, 1])
                    D_2d_1 = tf.tile(D_2d_1, [1, 3, 1])
                    D_1 = tf.tile(D_1, [1, 3, 1])

                    Xap_2 = Xap[:, 19:26, :]
                    D_2d_2 = D_2d[:, 19:26, :]
                    D_2 = D[:, 19:26, :]

                    Xap = tf.concat([Xap_1, Xap_2], axis=1)
                    D_2d = tf.concat([D_2d_1, D_2d_2], axis=1)
                    D = tf.concat([D_1, D_2], axis=1)

                if self.one_tier == True:
                    Xap_1 = Xap[:, 0:7, :]
                    D_2d_1 = D_2d[:, 0:7, :]
                    D_1 = D[:, 0:7, :]

                    Xap_1 = tf.tile(Xap_1, [1, 3, 1])
                    D_2d_1 = tf.tile(D_2d_1, [1, 3, 1])
                    D_1 = tf.tile(D_1, [1, 3, 1])

                    Xap_2 = Xap[:, 7:14, :]
                    D_2d_2 = D_2d[:, 7:14, :]
                    D_2 = D[:, 7:14, :]

                    Xap = tf.concat([Xap_1, Xap_2], axis=1)
                    D_2d = tf.concat([D_2d_1, D_2d_2], axis=1)
                    D = tf.concat([D_1, D_2], axis=1)

                self.Nap = 3 * self.Nap
                # self.Nuser = 3 * self.Nuser
                self.Nuser = 3 * self.Nap

            elif self.open_access==False:
                Xap = tf.tile(Xap, [1, 3, 1])
                D_2d = tf.tile(D_2d,[1,3,1])
                D= tf.tile(D,[1,3,1])
                self.Nap =3*self.Nap
                self.Nuser = 3*self.Nuser

        # if self.one_tier == False:
            # LSL_assign_Baseline, AP_assign_user_Baseline, assigned_batch_index_Baseline,pl_assign_Baseline,shadowing_LOS_assign_Baseline,shadowing_NLOS_assign_Baseline,G_Antenna_assign_Baseline,D_assign_Baseline,p_LOS_assign_Baseline,D_2d_assign_Baseline,LSGwithTxPower_TN_before_assigment_Baseline,LSGwithTxPower_LEOcenterBeam_before_assigment_Baseline,LSL_assign_UAVs_Baseline,LSL_assign_GUEs_Baseline,_ = self.Assign_AP(LSL[:,0:57,:],D[:,0:57,:],D_2d[:,0:57,:],pl[:,0:57,:],shadowing_LOS[:,0:57,:],shadowing_NLOS[:,0:57,:],G_Antenna[:,0:57,:],p_LOS[:,0:57,:],LSL[:,57:,:])
            # LSG_assign_UAVs_Baseline=-LSL_assign_UAVs_Baseline
            # LSG_assign_GUEs_Baseline=-LSL_assign_GUEs_Baseline
            #
            # LSG_assign_UAVs_Offloaded,LSG_assign_UAVs_Not_Offloaded=self.Assign_AP_OpenAccess_OnlyLEO(LSL[:, :, int(self.GUE_ratio * self.Nuser_drop)+1:],D_2d[:,:,int(self.GUE_ratio * self.Nuser_drop)+1:],G_Antenna[:,:,int(self.GUE_ratio * self.Nuser_drop)+1:],G_Antenna[:, :, int(self.GUE_ratio * self.Nuser_drop)+1:],Xuser)
            # self.UAVS_maxAG_forUL=tf.reduce_max(G_Antenna[:,57:64,:])

            #This is to ensure that every UAV is connected to a BS, and then BO thresholds are used for offloading discion
            #########################
            # LSL_TN_assigned_UAVs,_,_,_,_,_,_,_,_,_,_,_,_,_,LSL_NTN_assigned_UAVs,BSs_load_all = self.Assign_AP(LSL[:,0:57,int(self.GUE_ratio * self.Nuser_drop):],D[:,0:57,int(self.GUE_ratio * self.Nuser_drop):],D_2d[:,0:57,int(self.GUE_ratio * self.Nuser_drop):],pl[:,0:57,int(self.GUE_ratio * self.Nuser_drop):],shadowing_LOS[:,0:57,int(self.GUE_ratio * self.Nuser_drop):],shadowing_NLOS[:,0:57,int(self.GUE_ratio * self.Nuser_drop):],G_Antenna[:,0:57,int(self.GUE_ratio * self.Nuser_drop):],p_LOS[:,0:57,int(self.GUE_ratio * self.Nuser_drop):],LSL[:,57:,int(self.GUE_ratio * self.Nuser_drop):],BSs_load_all)
            # LSL_assigned_UAVs = tf.concat([LSL_TN_assigned_UAVs,LSL_NTN_assigned_UAVs],axis=1)
            # LSG_assign_UAVs_Offloaded,LSG_assign_UAVs_Not_Offloaded,Offloaded_UEs_perc = self.BO_offloading_Assigned_UAVs(LSL_assigned_UAVs)
            #########################

            #Load Balancing experiment
            #
            # LSL_TN_assigned_GUEs,_,_,_,_,_,_,_,_,_,_,_,_,_,LSL_NTN_assigned_GUEs,BSs_load_all = self.Assign_AP(LSL[:,0:57,0:int(self.GUE_ratio * self.Nuser_drop)],D[:,0:57,0:int(self.GUE_ratio * self.Nuser_drop)],D_2d[:,0:57,0:int(self.GUE_ratio * self.Nuser_drop)],pl[:,0:57,0:int(self.GUE_ratio * self.Nuser_drop)],shadowing_LOS[:,0:57,0:int(self.GUE_ratio * self.Nuser_drop)],shadowing_NLOS[:,0:57,0:int(self.GUE_ratio * self.Nuser_drop)],G_Antenna[:,0:57,0:int(self.GUE_ratio * self.Nuser_drop)],p_LOS[:,0:57,0:int(self.GUE_ratio * self.Nuser_drop)],LSL[:,57:,0:int(self.GUE_ratio * self.Nuser_drop)])
            # LSL_assigned_GUEs = tf.concat([LSL_TN_assigned_GUEs,LSL_NTN_assigned_GUEs],axis=1)
            # LSG_assign_GUEs_Offloaded,LSG_assign_GUEs_Not_Offloaded,Offloaded_UEs_perc = self.BO_offloading_Assigned_UAVs(LSL_assigned_GUEs)
            # BSs_load_tracking = tf.round(tf.expand_dims(tf.reduce_mean(BSs_load_all, axis=0),axis=0))  # This is to keep track of the load on each BSs by averaging over all the M.C batch runs
            #

        # if self.one_tier == True:
            # LSL_assign_Baseline, AP_assign_user_Baseline, assigned_batch_index_Baseline,pl_assign_Baseline,shadowing_LOS_assign_Baseline,shadowing_NLOS_assign_Baseline,G_Antenna_assign_Baseline,D_assign_Baseline,p_LOS_assign_Baseline,D_2d_assign_Baseline,LSGwithTxPower_TN_before_assigment_Baseline,LSGwithTxPower_LEOcenterBeam_before_assigment_Baseline,LSL_assign_UAVs_Baseline,LSL_assign_GUEs_Baseline,_ = self.Assign_AP(LSL[:,0:21,:],D[:,0:21,:],D_2d[:,0:21,:],pl[:,0:21,:],shadowing_LOS[:,0:21,:],shadowing_NLOS[:,0:21,:],G_Antenna[:,0:21,:],p_LOS[:,0:21,:],LSL[:,21:,:])
            # LSG_assign_UAVs_Baseline=-LSL_assign_UAVs_Baseline
            # LSG_assign_GUEs_Baseline=-LSL_assign_GUEs_Baseline
            #
            # LSG_assign_UAVs_Offloaded,LSG_assign_UAVs_Not_Offloaded=self.Assign_AP_OpenAccess_OnlyLEO(LSL[:, :, int(self.GUE_ratio * self.Nuser_drop)+1:],D_2d[:,:,int(self.GUE_ratio * self.Nuser_drop)+1:],G_Antenna[:,:,int(self.GUE_ratio * self.Nuser_drop)+1:],G_Antenna[:, :, int(self.GUE_ratio * self.Nuser_drop)+1:],Xuser[:,int(self.GUE_ratio * self.Nuser_drop)+1:,:])
            # self.UAVS_maxAG_forUL=tf.reduce_max(G_Antenna[:,57:64,:])

            # LSL_TN_assigned_UAVs,_,_,_,_,_,_,_,_,_,_,_,_,_,LSL_NTN_assigned_UAVs = self.Assign_AP(LSL[:,0:21,int(self.GUE_ratio * self.Nuser_drop):],D[:,0:21,int(self.GUE_ratio * self.Nuser_drop):],D_2d[:,0:21,int(self.GUE_ratio * self.Nuser_drop):],pl[:,0:21,int(self.GUE_ratio * self.Nuser_drop):],shadowing_LOS[:,0:21,int(self.GUE_ratio * self.Nuser_drop):],shadowing_NLOS[:,0:21,int(self.GUE_ratio * self.Nuser_drop):],G_Antenna[:,0:21,int(self.GUE_ratio * self.Nuser_drop):],p_LOS[:,0:21,int(self.GUE_ratio * self.Nuser_drop):],LSL[:,21:,int(self.GUE_ratio * self.Nuser_drop):])
            # LSL_assigned_UAVs = tf.concat([LSL_TN_assigned_UAVs,LSL_NTN_assigned_UAVs],axis=1)

            # LSG_assign_UAVs_Offloaded,LSG_assign_UAVs_Not_Offloaded = self.BO_offloading_Assigned_UAVs(LSL_assigned_UAVs)


        #Saving of data to be reported in the Run script
        #
        # self.LSG_assign_GUEs_Offloaded = LSG_assign_GUEs_Offloaded
        # self.LSG_assign_GUEs_Not_Offloaded = LSG_assign_GUEs_Not_Offloaded
        # self.Offloaded_UEs_perc = Offloaded_UEs_perc
        # self.BSs_load_all=tf.round(BSs_load_all)
        # self.LSG_TN_assigned_GUEs = -LSL_TN_assigned_GUEs
        # self.LSG_NTN_assigned_GUEs = -LSL_NTN_assigned_GUEs
        # self.BSs_load_tracking = BSs_load_tracking
        #
        self.LSG_UAVs_Corridors = LSG[:,0:57,int(self.GUE_ratio * self.Nuser_drop):]
        self.LSG_GUEs = LSG[:, 0:57, 0:int(self.GUE_ratio * self.Nuser_drop)]
        self.Xuser_UAVs = Xuser[:, int(self.GUE_ratio * self.Nuser_drop):, 0:2]
        self.Xuser_GUEs = Xuser[:, 0:int(self.GUE_ratio * self.Nuser_drop), 0:2]
        #####################
        # self.LSG_assign_UAVs_Offloaded = LSG_assign_UAVs_Offloaded
        # self.LSG_assign_UAVs_Not_Offloaded = LSG_assign_UAVs_Not_Offloaded
        # self.Offloaded_UEs_perc = Offloaded_UEs_perc
        # self.BSs_load_all=BSs_load_all
        # self.LSG_TN_assigned_UAVs = -LSL_TN_assigned_UAVs
        # self.LSG_NTN_assigned_UAVs = -LSL_NTN_assigned_UAVs
        #####################

        # self.LSG_assign_UAVs_Baseline=LSG_assign_UAVs_Baseline
        # self.LSG_assign_GUEs_Baseline = LSG_assign_GUEs_Baseline
        # self.AP_assign_user_Baseline = AP_assign_user_Baseline
        # self.assigned_batch_index_Baseline = assigned_batch_index_Baseline
        # self.D_2d_assign_Baseline = D_2d_assign_Baseline
        self.D_2d=D_2d
        self.D = D
        # self.Xuser = Xuser
        # self.Elv_thetha_deg=Elv_thetha_deg
        # self.Xap = Xap
        # self.AP_assign_user = AP_assign_user
        # self.assigned_batch_index = assigned_batch_index
        # self.LSG_all_TN=LSG_all_TN
        # self.BSs_load_UAVs=BSs_load_UAVs

        return

    # ------------ Assigment Functions
    def keep_assigned_user_batch(self,X,AP_assign_user,assigned_batch_index):
       X = tf.gather(X, assigned_batch_index, axis=0)
       AP_assign_user = tf.tile(tf.reduce_sum(AP_assign_user, axis=1, keepdims=True), [1, self.Nap, 1])
       AP_assign_user = tf.reshape(tf.transpose(AP_assign_user, [0, 2, 1]), [-1, self.Nap])
       X_assigned = tf.reshape(tf.transpose(X, [0, 2, 1]), [-1, X.shape[1]])
       ind = tf.squeeze(tf.where(AP_assign_user[:, 0] > 0))
       X_assigned = tf.transpose(tf.reshape(tf.gather(X_assigned, ind, axis=0), [-1, self.Nuser, self.Nap]),
                                     [0, 2, 1])
       return X_assigned

    def Assign_AP_OpenAccess(self, LSL, D_2d,G_Antenna,G_Antenna_other):

        self.P_Tx_TN = tf.math.pow(10.0, (self.P_Tx_dB - 30.0) / 10.0)
        self.P_Tx_NTN = tf.math.pow(10.0, (self.P_Tx_sat_db - 30.0) / 10.0)
        self.PSD = tf.math.pow(10.0, (-174.0 - 30.0) / 10.0)
        self.BW_TN = self.bandwidth
        self.BW_NTN = self.bandwidth_sat
        self.NF_TN = tf.math.pow(10.0, self.noise_figure_user / 10.0)
        self.NF_NTN = tf.math.pow(10.0, self.noise_figure_user_sat / 10.0)

        LSL_assign = tf.zeros([D_2d.shape[0], D_2d.shape[1], 1], dtype='float32')
        D_2d_assign = tf.zeros([D_2d.shape[0], D_2d.shape[1], 1], dtype='float32')
        G_Antenna_assign = tf.zeros([D_2d.shape[0], D_2d.shape[1], 1], dtype='float32')

        LSL_other6beams_assign = tf.zeros([D_2d.shape[0], D_2d.shape[1], 1], dtype='float32')
        #This is to find the LSG of the other 6 beams, extra fillling is necessary so that we end up with tensor of 58BSs in axis 1 to follow same aproach as in D2d. In the final step after assigment we will only take the first 6 which refer to the other leo beams
        LSL_other6beams = LSL[:, 58:64, :] + self.P_Tx_sat_db
        LSL_other6beams_fillingBlanks = LSL[:, 0:52, :]
        LSL_other6beams  = tf.concat([LSL_other6beams, LSL_other6beams_fillingBlanks], axis=1)
        #Same logic for AG NTN for other 6 beams
        G_Antenna_other6beams_assign = tf.zeros([D_2d.shape[0], D_2d.shape[1], 1], dtype='float32')
        G_Antenna_other6beams = G_Antenna_other[:, 58:64, :]
        G_Antenna_other6beams_fillingBlanks = G_Antenna_other[:, 0:52, :]
        G_Antenna_other6beams = tf.concat([G_Antenna_other6beams, G_Antenna_other6beams_fillingBlanks], axis=1)

        self.user_index = np.array([])
        AP_assign_user = tf.zeros([D_2d.shape[0], 1, self.Nuser_drop], dtype='float32')

        LSL_TN = LSL[:, 0:57, :] + self.P_Tx_dB
        LSL_NTN =LSL[:, 57:64, :] + self.P_Tx_sat_db+ self.sat_bias
        LSG_TN = tf.math.pow(10, (-LSL_TN) / 10)
        LSG_NTN = tf.math.pow(10, (-LSL_NTN) / 10)
        LSG = tf.concat([LSG_TN, LSG_NTN], axis=1)
        LSG = 10.0 * (tf.math.log(LSG) / tf.math.log(10.0))
        LSL = -LSG

        LSL_min_TN = tf.math.reduce_min(LSL_TN, axis=1, keepdims=True)
        LSG_min_TN = tf.math.pow(10, (-LSL_min_TN) / 10)

        LSL_TN_sorted=tf.sort(LSL_TN,axis=1)
        LSG_TN_sorted=-LSL_TN_sorted
        LSG_TN_sorted = tf.math.pow(10, (LSG_TN_sorted) / 10)
        snr_link = LSG_TN_sorted[:,0:self.k_dominant_BS,:] * self.P_Tx_TN / (self.BW_TN * self.NF_TN)
        num_TN = LSG_min_TN * self.P_Tx_TN / (self.BW_TN * self.NF_TN)
        denom_TN = tf.expand_dims(tf.reduce_sum(snr_link, axis=1), axis=1) - num_TN + self.PSD
        sinr_onlyground = num_TN / denom_TN
        sinr_onlyground = 10.0 * (tf.math.log(sinr_onlyground) / tf.math.log(10.0))

        LSL_min_NTN = tf.math.reduce_min(LSL_NTN, axis=1, keepdims=True)
        LSG_min_NTN = tf.math.pow(10, (-LSL_min_NTN) / 10)
        snr_link_NTN = LSG_NTN * self.P_Tx_NTN / (self.BW_NTN * self.NF_NTN)
        num_NTN = LSG_min_NTN * self.P_Tx_NTN / (self.BW_NTN * self.NF_NTN)
        denom_NTN = tf.expand_dims(tf.reduce_sum(snr_link_NTN, axis=1), axis=1) - num_NTN + self.PSD
        sinr_onlyLEO = num_NTN / denom_NTN
        sinr_onlyLEO = 10.0 * (tf.math.log(sinr_onlyLEO) / tf.math.log(10.0))

        TN_Assign = tf.cast(sinr_onlyground >= sinr_onlyLEO, "float32") #The 2.0 is just a small offset for insurance
        TN_Assign = tf.tile(TN_Assign, [1, 57, 1])
        NTN_Assign = tf.cast(sinr_onlyground < sinr_onlyLEO, "float32")
        self.Offloaded_UEs=tf.reduce_sum(NTN_Assign, axis=2)
        self.Min_Offloaded_UEs=tf.reduce_min(self.Offloaded_UEs)
        self.Max_Offloaded_UEs = tf.reduce_max(self.Offloaded_UEs)
        LSL_Assign_ref = tf.concat([TN_Assign, NTN_Assign], axis=1) * LSL[:, 0:58, :]
        LSL_TN_sort = tf.math.argmin(LSL_Assign_ref[:, 0:57, :], axis=1)
        LSL_NTN_sort = tf.math.argmax(LSL_Assign_ref, axis=1)
        LSL_NTN_sort_cond = tf.cast(LSL_NTN_sort == 57, "int64")
        LSL_NTN_sort = LSL_NTN_sort_cond * LSL_NTN_sort
        LSL_sort = LSL_TN_sort + LSL_NTN_sort

        LSL_TN_min = tf.math.reduce_min(LSL_Assign_ref[:, 0:57, :], axis=1, keepdims=True)
        LSL_NTN_min = tf.math.reduce_max(tf.expand_dims(LSL_Assign_ref[:, 57, :], axis=1), axis=1, keepdims=True)
        LSL_min = LSL_TN_min + LSL_NTN_min

        d_sort = tf.reduce_sum(tf.cast(LSL[:, 0:58, :] == LSL_min, 'float32') * D_2d, axis=1)

        mask = tf.expand_dims(tf.range(1.0, self.Nuser_drop + 1), axis=0)
        mask = tf.tile(mask, [D_2d.shape[0], 1])  # Q
        assigned_batch_index = tf.range(0, D_2d.shape[0])

        if self.open_access:
            self.Nap = self.Nap + 1  # +7
        elif self.open_access == False:
            self.Nap = self.Nap

        for i in range(self.Nap):
            # print(i)
            ind_ap_i = tf.cast(LSL_sort == i, 'float32') * tf.cast(d_sort > self.Dist2D_exclud, 'float32')
            valid_batch = tf.reduce_sum(ind_ap_i, axis=1)
            valid_batch = tf.squeeze(tf.where(valid_batch > 0))
            ind_ap_i = tf.gather(ind_ap_i, valid_batch, axis=0)
            LSL_sort = tf.gather(LSL_sort, valid_batch, axis=0)
            d_sort = tf.gather(d_sort, valid_batch, axis=0)
            LSL = tf.gather(LSL[:, 0:58, :], valid_batch, axis=0)
            D_2d = tf.gather(D_2d, valid_batch, axis=0)
            G_Antenna = tf.gather(G_Antenna, valid_batch, axis=0)
            LSL_other6beams = tf.gather(LSL_other6beams, valid_batch, axis=0)
            G_Antenna_other6beams = tf.gather(G_Antenna_other6beams, valid_batch, axis=0)

            assigned_batch_index = tf.gather(assigned_batch_index, valid_batch, axis=0)
            LSL_assign = tf.gather(LSL_assign, valid_batch, axis=0)
            D_2d_assign = tf.gather(D_2d_assign, valid_batch, axis=0)
            G_Antenna_assign = tf.gather(G_Antenna_assign, valid_batch, axis=0)
            LSL_other6beams_assign = tf.gather(LSL_other6beams_assign, valid_batch, axis=0)
            G_Antenna_other6beams_assign = tf.gather(G_Antenna_other6beams_assign, valid_batch, axis=0)

            mask = tf.gather(mask, valid_batch, axis=0)

            AP_assign_user = tf.gather(AP_assign_user, valid_batch, axis=0)

            mask_i = mask * ind_ap_i
            mask_i = tf.transpose(tf.random.shuffle(tf.transpose(mask_i, [1, 0])), [1, 0])

            ap_assign_user = tf.reduce_max(mask * tf.cast(mask_i > 0.0, "float32"), axis=1, keepdims=True) - 1
            ap_assign_user = tf.gather_nd(mask_i, tf.concat(
                [tf.expand_dims(tf.constant(range(mask_i.shape[0])), axis=1), tf.cast(ap_assign_user, "int32")],
                axis=1))
            self.user_index = np.append(self.user_index, ap_assign_user.numpy() / self.Nuser_drop)

            mask_ap = tf.expand_dims(tf.scatter_nd(tf.concat([tf.expand_dims(tf.constant(range(mask.shape[0])), axis=1),
                                                              tf.cast(tf.expand_dims(ap_assign_user, axis=1),
                                                                      "int32") - 1], axis=1), tf.ones(mask.shape[0]),
                                                   [mask.shape[0], self.Nuser_drop]), axis=1)

            LSL_selected_user = tf.reduce_sum(LSL[:, 0:58, :] * mask_ap, axis=2, keepdims=True)
            D_2d_selected_user = tf.reduce_sum(mask_ap * D_2d, axis=2, keepdims=True)
            G_Antenna_selected_user = tf.reduce_sum(mask_ap * G_Antenna, axis=2, keepdims=True)
            LSL_other6beams_selected_user = tf.reduce_sum(mask_ap * LSL_other6beams, axis=2, keepdims=True)
            G_Antenna_other6beams_selected_user = tf.reduce_sum(mask_ap * G_Antenna_other6beams, axis=2, keepdims=True)

            LSL_assign = tf.concat([LSL_assign, LSL_selected_user], axis=2)
            D_2d_assign = tf.concat([D_2d_assign, D_2d_selected_user], axis=2)
            G_Antenna_assign = tf.concat([G_Antenna_assign, G_Antenna_selected_user], axis=2)
            LSL_other6beams_assign = tf.concat([LSL_other6beams_assign, LSL_other6beams_selected_user], axis=2)
            G_Antenna_other6beams_assign = tf.concat([G_Antenna_other6beams_assign, G_Antenna_other6beams_selected_user], axis=2)

            AP_assign_user = tf.concat([AP_assign_user, mask_ap], axis=1)

        LSL_assign = LSL_assign[0:self.batch_num, :, 1:]
        D_2d_assign = D_2d_assign[0:self.batch_num, :, 1:]
        G_Antenna_assign = G_Antenna_assign[0:self.batch_num, :, 1:]
        LSL_other6beams_assign = LSL_other6beams_assign[0:self.batch_num, 0:6, 1:]
        G_Antenna_other6beams_assign = G_Antenna_other6beams_assign[0:self.batch_num, 0:6, 1:]

        if self.indoor_calib:
            self.LSL_calib_assign = LSL_assign
            LSL_assign = LSL_org_assign

        AP_assign_user = AP_assign_user[0:self.batch_num, 1:, :]
        assigned_batch_index = assigned_batch_index[0:self.batch_num]

        AP_assign_user_index=AP_assign_user* tf.tile(tf.expand_dims(mask[0:self.batch_num,:],axis=1), [1, 58, 1])
        AP_assign_user_index = tf.expand_dims(tf.reduce_sum(tf.abs(AP_assign_user_index), 2),axis=1)
        AP_assign_user_index=tf.tile(AP_assign_user_index, [1, 58, 1])

        LSL_assign_UAVs= LSL_assign*tf.cast(AP_assign_user_index >= self.GUE_ratio*self.Nuser_drop, 'float32')
        LSL_assign_GUEs = LSL_assign * tf.cast(AP_assign_user_index < self.GUE_ratio * self.Nuser_drop, 'float32')

        return LSL_assign, AP_assign_user, assigned_batch_index,D_2d_assign,G_Antenna_assign,LSL_other6beams_assign,G_Antenna_other6beams_assign

    # This is to assign UEs to BSs. If a batch has at least 1 BS with no associated UE then we totaly ignore this batch and find another one.
    def Assign_AP(self, LSL,D,D_2d,pl,shadowing_LOS,shadowing_NLOS,G_Antenna,p_LOS,LSL_NTN):

        if self.open_access:
            LSL=LSL[:, 0:57, :] + self.P_Tx_dB
            if self.RSS_offloading:
                LSL = LSL + self.toes
        if self.sat_user:
            self.Nuser_drop = self.Nuser_drop-self.Nuser_drop_sat

        if self.indoor_calib:
            LSL_org = LSL
            LSL = self.LSGclass.LSL_calib
        else:
            LSL_org =LSL

        # Finding the load on each BS
        LSG_all_TN_GUEs = -LSL
        BSs_assigment = tf.cast(LSG_all_TN_GUEs == tf.tile(tf.expand_dims(tf.reduce_max(LSG_all_TN_GUEs, axis=1), axis=1), [1, 57, 1]),"float32")
        BSs_load_all = tf.reduce_sum(BSs_assigment, axis=2) #/ 3.0

        LSL_assign = tf.zeros([LSL.shape[0], LSL.shape[1], 1], dtype='float32')
        
        #
        pl_assign = tf.zeros([pl.shape[0], pl.shape[1], 1], dtype='float32')
        shadowing_LOS_assign = tf.zeros([shadowing_LOS.shape[0], shadowing_LOS.shape[1], 1], dtype='float32')
        shadowing_NLOS_assign = tf.zeros([shadowing_NLOS.shape[0], shadowing_NLOS.shape[1], 1], dtype='float32')
        G_Antenna_assign=tf.zeros([G_Antenna.shape[0],G_Antenna.shape[1], 1], dtype='float32')
        D_assign=tf.zeros([D.shape[0], D.shape[1], 1], dtype='float32')
        D_2d_assign=tf.zeros([D_2d.shape[0], D_2d.shape[1], 1], dtype='float32')
        p_LOS_assign=tf.zeros([p_LOS.shape[0], p_LOS.shape[1], 1], dtype='float32')
        LSL_org_assign = tf.zeros([LSL_org.shape[0], LSL_org.shape[1], 1], dtype='float32')
        LSL_NTN_assign = tf.zeros([LSL_NTN.shape[0], LSL_NTN.shape[1], 1], dtype='float32')
        BSs_load_all_assign = tf.zeros([BSs_load_all.shape[0], BSs_load_all.shape[1]], dtype='float32')

        # Xuser_assigned = tf.zeros(Xuser.shape[0],1,3)
        #
        self.user_index=np.array([])
        AP_assign_user = tf.zeros([LSL.shape[0], 1,LSL.shape[2]], dtype='float32')

        LSL_sort = tf.math.argmin(LSL, axis=1)  #
        LSL_min = tf.math.reduce_min(LSL,axis=1,keepdims=True)
        d_sort = tf.reduce_sum(tf.cast(LSL==LSL_min,'float32')*D_2d,axis=1)
        # d_sort =tf.squeeze(d_sort)
        # Status=1
        # Make sure mask does not have zero value!!!!!!
        mask = tf.expand_dims(tf.range(1.0,LSL.shape[2]+1), axis=0)  # Q
        mask = tf.tile(mask, [LSL.shape[0], 1])  # Q
        assigned_batch_index = tf.range(0, LSL.shape[0])
        # Q
        if self.open_access:
            self.Nap=self.Nap #+1  #+7
        elif self.open_access==False:
            self.Nap=self.Nap

        for i in range(self.Nap):
            # print(i)
            # ind_i=np.argwhere(d_sort==i)
            # idnof user assigned to AP+i

            # ----how many users assigned to AP_i
            ind_ap_i = tf.cast(LSL_sort == i, 'float32')*tf.cast(d_sort >self.Dist2D_exclud,'float32')
            # compute valid batch (AP_i has atleast one user assigned)
            valid_batch = tf.reduce_sum(ind_ap_i, axis=1)
            valid_batch = tf.squeeze(tf.where(valid_batch > 0))
            # -----------Keep valid batch
            # Note: In the future, if we want to keep the un valid batches, we can perform operations in this part of the code. Instead of just delteting them.
            # ind_i_val = tf.gather(ind_i_val,valid_batch)
            ind_ap_i = tf.gather(ind_ap_i, valid_batch, axis=0)
            LSL_sort = tf.gather(LSL_sort, valid_batch, axis=0)
            d_sort = tf.gather(d_sort, valid_batch, axis=0)
            LSL = tf.gather(LSL, valid_batch, axis=0)
            
            #
            pl = tf.gather(pl, valid_batch, axis=0)
            shadowing_LOS = tf.gather(shadowing_LOS, valid_batch, axis=0)
            shadowing_NLOS = tf.gather(shadowing_NLOS, valid_batch, axis=0)
            G_Antenna = tf.gather(G_Antenna, valid_batch, axis=0)
            D = tf.gather(D, valid_batch, axis=0)
            D_2d = tf.gather(D_2d, valid_batch, axis=0)
            p_LOS = tf.gather(p_LOS, valid_batch, axis=0)
            LSL_org = tf.gather(LSL_org, valid_batch, axis=0)
            LSL_NTN = tf.gather(LSL_NTN, valid_batch, axis=0)
            BSs_load_all = tf.gather(BSs_load_all, valid_batch, axis=0)

            #
            
            assigned_batch_index = tf.gather(assigned_batch_index, valid_batch, axis=0)
            LSL_assign = tf.gather(LSL_assign, valid_batch, axis=0)
            
            #
            pl_assign = tf.gather(pl_assign, valid_batch, axis=0)
            shadowing_LOS_assign = tf.gather(shadowing_LOS_assign, valid_batch, axis=0)
            shadowing_NLOS_assign = tf.gather(shadowing_NLOS_assign, valid_batch, axis=0)
            G_Antenna_assign = tf.gather(G_Antenna_assign, valid_batch, axis=0)
            D_assign = tf.gather(D_assign, valid_batch, axis=0)
            D_2d_assign = tf.gather(D_2d_assign, valid_batch, axis=0)
            p_LOS_assign = tf.gather(p_LOS_assign, valid_batch, axis=0)
            LSL_org_assign = tf.gather(LSL_org_assign, valid_batch, axis=0)
            LSL_NTN_assign = tf.gather(LSL_NTN_assign, valid_batch, axis=0)
            #
            
            mask = tf.gather(mask, valid_batch, axis=0)

            AP_assign_user = tf.gather(AP_assign_user, valid_batch, axis=0)
            # --------------------------------------------
            # -------
              
            mask_i = mask * ind_ap_i
            mask_i=tf.transpose(tf.random.shuffle(tf.transpose(mask_i,[1,0])),[1,0])
            
            ap_assign_user=tf.reduce_max(mask*tf.cast(mask_i>0.0,"float32"),axis=1,keepdims=True)-1
            ap_assign_user=tf.gather_nd(mask_i,tf.concat([tf.expand_dims(tf.constant(range( mask_i.shape[0])),axis=1),tf.cast(ap_assign_user,"int32")],axis=1))
            self.user_index=np.append(self.user_index,ap_assign_user.numpy()/LSL.shape[2])

            mask_ap = tf.expand_dims(tf.scatter_nd(tf.concat([tf.expand_dims(tf.constant(range( mask.shape[0])),axis=1),
                                    tf.cast(tf.expand_dims(ap_assign_user,axis=1),"int32")-1],axis=1),
                                    tf.ones(mask.shape[0]),[mask.shape[0],LSL.shape[2]]),axis=1)

            LSL_selected_user = tf.reduce_sum(LSL *mask_ap , axis=2, keepdims=True)


            #LSG+Tx Power for TN and NTN seperated and before assigment
            LSGwithTxPower_TN_before_assigment=-LSL[:,0:57,:]-self.P_Tx_dB + self.P_over_noise_db - self.noise_figure_user
            LSGwithTxPower_TN_before_assigment=tf.expand_dims(tf.reduce_max(LSGwithTxPower_TN_before_assigment,axis=1),axis=1)
            LSGwithTxPower_LEOcenterBeam_before_assigment = tf.expand_dims(tf.reduce_max(-LSL[:,57:64,:]-self.P_Tx_sat_db + self.P_over_noise_db_sat - self.noise_figure_user_sat,axis=1), axis=1)
            # LSGwithTxPower_LEOcenterBeam_before_assigment = tf.expand_dims(-LSL[:, 57, :]-self.P_Tx_sat_db + self.P_over_noise_db_sat - self.noise_figure_user_sat, axis=1)

            #
            pl_selected_user = tf.reduce_sum(mask_ap * pl, axis=2, keepdims=True)
            shadowing_LOS_selected_user = tf.reduce_sum(mask_ap * shadowing_LOS, axis=2, keepdims=True)
            shadowing_NLOS_selected_user = tf.reduce_sum(mask_ap * shadowing_NLOS, axis=2, keepdims=True)
            G_Antenna_selected_user = tf.reduce_sum(mask_ap * G_Antenna, axis=2, keepdims=True)
            D_selected_user = tf.reduce_sum(mask_ap * D, axis=2, keepdims=True)
            D_2d_selected_user = tf.reduce_sum(mask_ap * D_2d, axis=2, keepdims=True)
            p_LOS_selected_user = tf.reduce_sum(mask_ap * p_LOS, axis=2, keepdims=True)
            LSL_org_selected_user = tf.reduce_sum(mask_ap * LSL_org, axis=2, keepdims=True)
            LSL_NTN_selected_user = tf.reduce_sum(mask_ap * LSL_NTN, axis=2, keepdims=True)


            #
            LSL_assign = tf.concat([LSL_assign, LSL_selected_user], axis=2)
            
            #
            pl_assign = tf.concat([pl_assign, pl_selected_user], axis=2)
            shadowing_LOS_assign = tf.concat([shadowing_LOS_assign, shadowing_LOS_selected_user], axis=2)
            shadowing_NLOS_assign = tf.concat([shadowing_NLOS_assign, shadowing_NLOS_selected_user], axis=2)
            G_Antenna_assign = tf.concat([G_Antenna_assign, G_Antenna_selected_user], axis=2)
            D_assign = tf.concat([D_assign, D_selected_user], axis=2)
            D_2d_assign = tf.concat([D_2d_assign, D_2d_selected_user], axis=2)
            p_LOS_assign = tf.concat([p_LOS_assign, p_LOS_selected_user], axis=2)
            LSL_org_assign = tf.concat([LSL_org_assign, LSL_org_selected_user], axis=2)
            LSL_NTN_assign = tf.concat([LSL_NTN_assign, LSL_NTN_selected_user], axis=2)
            
            AP_assign_user = tf.concat([AP_assign_user, mask_ap], axis=1)

        LSL_assign = LSL_assign[0:self.batch_num, :,1:]  # the assigned user index to AP_i. The entries have value between [0,self.Nuser_drop]
        #
        pl_assign = pl_assign[0:self.batch_num, :, 1:]
        shadowing_LOS_assign = shadowing_LOS_assign[0:self.batch_num, :, 1:]
        shadowing_NLOS_assign = shadowing_NLOS_assign[0:self.batch_num, :, 1:]
        G_Antenna_assign = G_Antenna_assign[0:self.batch_num, :, 1:]
        D_assign = D_assign[0:self.batch_num, :, 1:]
        D_2d_assign = D_2d_assign[0:self.batch_num, :, 1:]
        p_LOS_assign = p_LOS_assign[0:self.batch_num, :, 1:]
        LSL_org_assign = LSL_org_assign[0:self.batch_num, :, 1:]
        LSL_NTN_assign = LSL_NTN_assign[0:self.batch_num, :, 1:]+self.P_Tx_sat_db+ self.sat_bias
        assigned_batch_index = assigned_batch_index[0:self.batch_num]
        BSs_load_all = BSs_load_all[0:self.batch_num, :]

        if self.indoor_calib:
            self.LSL_calib_assign = LSL_assign
            LSL_assign = LSL_org_assign

        #
        if self.one_tier == False:
            AP_assign_user = AP_assign_user[0:self.batch_num,1:,:]
            AP_assign_user_index=AP_assign_user* tf.tile(tf.expand_dims(mask[0:self.batch_num,:],axis=1), [1, 57, 1])
            AP_assign_user_index = tf.expand_dims(tf.reduce_sum(tf.abs(AP_assign_user_index), 2),axis=1)
            AP_assign_user_index=tf.tile(AP_assign_user_index, [1, 57, 1])

        if self.one_tier == True:
            AP_assign_user = AP_assign_user[0:self.batch_num,1:,:]
            AP_assign_user_index=AP_assign_user* tf.tile(tf.expand_dims(mask[0:self.batch_num,:],axis=1), [1, 21, 1])
            AP_assign_user_index = tf.expand_dims(tf.reduce_sum(tf.abs(AP_assign_user_index), 2),axis=1)
            AP_assign_user_index=tf.tile(AP_assign_user_index, [1, 21, 1])

        LSL_assign_UAVs= LSL_assign*tf.cast(AP_assign_user_index >= self.GUE_ratio*LSL.shape[2], 'float32')
        LSL_assign_GUEs = LSL_assign * tf.cast(AP_assign_user_index < self.GUE_ratio * LSL.shape[2], 'float32')

        return LSL_assign, AP_assign_user, assigned_batch_index,pl_assign,shadowing_LOS_assign,shadowing_NLOS_assign,G_Antenna_assign,D_assign,p_LOS_assign,D_2d_assign,LSGwithTxPower_TN_before_assigment,LSGwithTxPower_LEOcenterBeam_before_assigment,LSL_assign_UAVs,LSL_assign_GUEs,LSL_NTN_assign, BSs_load_all


    def dist_serving_BS(self,Xap,Xuser,AP_assign_user,assigned_batch_index):
            
        Xap = tf.gather(Xap,assigned_batch_index,axis=0)
        Xuser = tf.gather(Xuser,assigned_batch_index,axis=0)
        Xuser_assigned = tf.expand_dims(Xuser,axis=1)*tf.expand_dims(AP_assign_user,axis=3)
        Xuser_assigned = tf.reduce_sum(Xuser_assigned,axis=2)
        if self.DeploymentType =="PPP":
            D_assign, D_2d_assign, BS_wrapped_Cord_assign, Xuser_assigned = self.Deployment.Deploy.Dist(Xap,Xuser_assigned, self.EX, self.EY)
        elif self.DeploymentType == "Hex":
            D_assign, D_2d_assign, BS_wrapped_Cord_assign, Xuser_assigned = self.Deployment.Deploy.Dist(Xap, Xuser_assigned)

        return D_assign, D_2d_assign, BS_wrapped_Cord_assign

    def Assign_AP_OpenAccess_OnlyLEO(self, LSL, D_2d,G_Antenna,G_Antenna_other,Xuser):
        #Important note 2022/10/06: BO TN thresholds 57 only will work in case3 where there is 1 UAV per TN cell. Otherwise, huge modification is needed in the code

        self.P_Tx_TN = tf.math.pow(10.0, (self.P_Tx_dB - 30.0) / 10.0)
        self.P_Tx_NTN = tf.math.pow(10.0, (self.P_Tx_sat_db - 30.0) / 10.0)
        self.PSD = tf.math.pow(10.0, (-174.0 - 30.0) / 10.0)
        self.BW_TN = self.bandwidth
        self.BW_NTN = self.bandwidth_sat
        self.NF_TN = tf.math.pow(10.0, self.noise_figure_user / 10.0)
        self.NF_NTN = tf.math.pow(10.0, self.noise_figure_user_sat / 10.0)

        if self.one_tier == False:
            LSL_TN = LSL[:, 0:57, :] + self.P_Tx_dB
            LSL_NTN =LSL[:, 57:64, :] + self.P_Tx_sat_db+ self.sat_bias
        if self.one_tier == True:
            LSL_TN = LSL[:, 0:21, :] + self.P_Tx_dB
            LSL_NTN =LSL[:, 21:28, :] + self.P_Tx_sat_db+ self.sat_bias
        LSG_TN = tf.math.pow(10, (-LSL_TN) / 10)
        LSG_NTN = tf.math.pow(10, (-LSL_NTN) / 10)
        LSG = tf.concat([LSG_TN, LSG_NTN], axis=1)
        LSG = 10.0 * (tf.math.log(LSG) / tf.math.log(10.0))
        LSL = -LSG

        LSL_min_TN = tf.math.reduce_min(LSL_TN, axis=1, keepdims=True)
        LSG_min_TN = tf.math.pow(10, (-LSL_min_TN) / 10)

        LSL_TN_sorted=tf.sort(LSL_TN,axis=1)
        LSG_TN_sorted=-LSL_TN_sorted
        LSG_TN_sorted = tf.math.pow(10, (LSG_TN_sorted) / 10)
        snr_link = LSG_TN_sorted[:,0:self.k_dominant_BS,:] * self.P_Tx_TN / (self.BW_TN * self.NF_TN)
        num_TN = LSG_min_TN * self.P_Tx_TN / (self.BW_TN * self.NF_TN)
        denom_TN = tf.expand_dims(tf.reduce_sum(snr_link, axis=1), axis=1) - num_TN + self.PSD
        sinr_onlyground = num_TN / denom_TN
        sinr_onlyground = 10.0 * (tf.math.log(sinr_onlyground) / tf.math.log(10.0))

        LSL_min_NTN = tf.math.reduce_min(LSL_NTN, axis=1, keepdims=True)
        LSG_min_NTN = tf.math.pow(10, (-LSL_min_NTN) / 10)
        snr_link_NTN = LSG_NTN * self.P_Tx_NTN / (self.BW_NTN * self.NF_NTN)
        num_NTN = LSG_min_NTN * self.P_Tx_NTN / (self.BW_NTN * self.NF_NTN)
        denom_NTN = tf.expand_dims(tf.reduce_sum(snr_link_NTN, axis=1), axis=1) - num_NTN + self.PSD
        sinr_onlyLEO = num_NTN / denom_NTN
        sinr_onlyLEO = 10.0 * (tf.math.log(sinr_onlyLEO) / tf.math.log(10.0))

        if self.FRF_3:
            sinr_onlyLEO = (LSG_min_NTN * self.P_Tx_NTN) / (self.BW_NTN * self.NF_NTN * self.PSD)
            sinr_onlyLEO = 10.0 * (tf.math.log(sinr_onlyLEO) / tf.math.log(10.0))

        # TN_Assign = tf.cast(sinr_onlyground >= -5.0, "float32") #The 2.0 is just a small offset for insurance
        # TN_Assign = tf.tile(TN_Assign, [1, 57, 1])
        # NTN_Assign = tf.cast(sinr_onlyground < -5.0, "float32")
        # NTN_Assign = tf.tile(NTN_Assign, [1, 7, 1])
        # self.T1_TN=tf.tile(tf.expand_dims(self.T1_TN,0), [sinr_onlyground.shape[0], 1, 1]) #if you are not using BO loop uncomment this
        T2_NTN=tf.tile(tf.expand_dims(self.T1_TN[:,:,-1], axis=2), [1, 1, sinr_onlyground.shape[2]])
        self.T1_TN=self.T1_TN[:,:,:-1]
        NTN_Assign = tf.cast(sinr_onlyground < self.T1_TN, "float32") * tf.cast(sinr_onlyLEO > T2_NTN, "float32")
        TN_Assign = tf.cast(NTN_Assign == 0.0, "float32")
        NTN_Assign = tf.tile(NTN_Assign, [1, 7, 1])
        if self.one_tier == False:
            TN_Assign = tf.tile(TN_Assign, [1, 57, 1])
        if self.one_tier == True:
            TN_Assign = tf.tile(TN_Assign, [1, 21, 1])
        self.Offloaded_UEs=tf.reduce_sum(NTN_Assign, axis=2)
        self.Min_Offloaded_UEs=tf.reduce_min(self.Offloaded_UEs)
        self.Max_Offloaded_UEs = tf.reduce_max(self.Offloaded_UEs)
        self.Mean_Offloaded_UEs = tf.reduce_mean(self.Offloaded_UEs)/(self.UAV_ratio * self.Nuser_drop)
        self.Offloaded_UEs_perElv = tf.reduce_mean(self.Offloaded_UEs)

        LSL_assign_UAVs_Offloaded = NTN_Assign * LSL_NTN
        LSL_assign_UAVs_Not_Offloaded = TN_Assign * LSL_TN

        UAVs_Offloaded_height = NTN_Assign[:,0,:] * Xuser[:,:,2]

        ##======== This is for choosing some of the UAVs based on cap reasons not just under -5dB thershold
        if self.Cap_cond:
            LSL_assign_UAVs_Offloaded_ref = LSL_assign_UAVs_Offloaded

            LSL_assign_UAVs_Offloaded_sorted = tf.sort(LSL_assign_UAVs_Offloaded_ref + tf.cast(LSL_assign_UAVs_Offloaded_ref == 0.0, "float32") * 10000, axis=2)
            LSL_assign_UAVs_Offloaded_sorted = LSL_assign_UAVs_Offloaded_sorted * tf.cast(LSL_assign_UAVs_Offloaded_sorted != 10000.0, "float32")
            LSL_assign_UAVs_Offloaded_1 = LSL_assign_UAVs_Offloaded_sorted[:, :, 0:self.desired_UAVs_offload]
            LSL_assign_UAVs_Offloaded_1 = tf.ones(LSL_assign_UAVs_Offloaded_1.shape, 'float32') * LSL_assign_UAVs_Offloaded_1
            LSL_assign_UAVs_Offloaded_2 = LSL_assign_UAVs_Offloaded_sorted[:, :, self.desired_UAVs_offload:]
            LSL_assign_UAVs_Offloaded_2 = tf.zeros(LSL_assign_UAVs_Offloaded_2.shape, 'float32') * LSL_assign_UAVs_Offloaded_2
            LSL_assign_UAVs_Offloaded = tf.concat([LSL_assign_UAVs_Offloaded_1, LSL_assign_UAVs_Offloaded_2], axis=2)

            UnOffloaded_UAVs_cond = tf.expand_dims(tf.reduce_max(tf.expand_dims(LSL_assign_UAVs_Offloaded[:,0,:], axis=1), axis=2),axis=2)
            UnOffloaded_UAVs_cond = tf.tile(UnOffloaded_UAVs_cond, [1, 1, 58])

            UnOffloaded_UAVs_cast = tf.cast(tf.expand_dims(LSL_assign_UAVs_Offloaded_ref[:,0,:],axis=1) > UnOffloaded_UAVs_cond, "float32")
            UnOffloaded_UAVs_cast = tf.tile(UnOffloaded_UAVs_cast, [1, 57, 1])

            TN_Assign = TN_Assign + UnOffloaded_UAVs_cast
            LSL_assign_UAVs_Not_Offloaded = TN_Assign * LSL_TN

        if self.LSG_all==True:
            LSG_assign_UAVs_Offloaded=-LSL_assign_UAVs_Offloaded
            LSG_assign_UAVs_Not_Offloaded = -LSL_assign_UAVs_Not_Offloaded

        else:
            LSG_assign_UAVs_Offloaded=-LSL_assign_UAVs_Offloaded[0:self.batch_num,:,:]
            LSG_assign_UAVs_Not_Offloaded = -LSL_assign_UAVs_Not_Offloaded[0:self.batch_num, :, :]

        return LSG_assign_UAVs_Offloaded,LSG_assign_UAVs_Not_Offloaded

    def Assign_AP_IndoorIncluded(self, LSL, D, D_2d, pl, shadowing_LOS, shadowing_NLOS, G_Antenna, p_LOS):
        if self.sat_user:
            self.Nuser_drop = self.Nuser_drop - self.Nuser_drop_sat

        if self.indoor_calib:
            LSL_org = LSL
            LSL = self.LSGclass.LSL_calib
        else:
            LSL_org = LSL
        LSL_assign = tf.zeros([LSL.shape[0], LSL.shape[1], 1], dtype='float32')

        #
        pl_assign = tf.zeros([pl.shape[0], pl.shape[1], 1], dtype='float32')
        shadowing_LOS_assign = tf.zeros([shadowing_LOS.shape[0], shadowing_LOS.shape[1], 1], dtype='float32')
        shadowing_NLOS_assign = tf.zeros([shadowing_NLOS.shape[0], shadowing_NLOS.shape[1], 1], dtype='float32')
        G_Antenna_assign = tf.zeros([G_Antenna.shape[0], G_Antenna.shape[1], 1], dtype='float32')
        D_assign = tf.zeros([D.shape[0], D.shape[1], 1], dtype='float32')
        D_2d_assign = tf.zeros([D_2d.shape[0], D_2d.shape[1], 1], dtype='float32')
        p_LOS_assign = tf.zeros([p_LOS.shape[0], p_LOS.shape[1], 1], dtype='float32')
        LSL_org_assign = tf.zeros([LSL_org.shape[0], LSL_org.shape[1], 1], dtype='float32')

        # Xuser_assigned = tf.zeros(Xuser.shape[0],1,3)
        #
        self.user_index = np.array([])
        AP_assign_user = tf.zeros([LSL.shape[0], 1, self.Nuser_drop], dtype='float32')

        LSL_sort = tf.math.argmin(LSL, axis=1)  #
        LSL_min = tf.math.reduce_min(LSL, axis=1, keepdims=True)
        d_sort = tf.reduce_sum(tf.cast(LSL == LSL_min, 'float32') * D_2d, axis=1)
        # d_sort =tf.squeeze(d_sort)
        # Status=1
        # Make sure mask does not have zero value!!!!!!
        mask = tf.expand_dims(tf.range(1.0, self.Nuser_drop + 1), axis=0)  # Q
        mask = tf.tile(mask, [LSL.shape[0], 1])  # Q
        assigned_batch_index = tf.range(0, LSL.shape[0])
        # Q
        if self.open_access:
            self.Nap = self.Nap + 7
        elif self.open_access == False:
            self.Nap = self.Nap

        for i in range(self.Nap):
            # print(i)
            # ind_i=np.argwhere(d_sort==i)
            # idnof user assigned to AP+i

            # ----how many users assigned to AP_i
            ind_ap_i = tf.cast(LSL_sort == i, 'float32') * tf.cast(d_sort > self.Dist2D_exclud, 'float32')
            # compute valid batch (AP_i has atleast one user assigned)
            valid_batch = tf.reduce_sum(ind_ap_i, axis=1)
            valid_batch = tf.squeeze(tf.where(valid_batch > 0))
            # -----------Keep valid batch
            # Note: In the future, if we want to keep the un valid batches, we can perform operations in this part of the code. Instead of just delteting them.
            # ind_i_val = tf.gather(ind_i_val,valid_batch)
            ind_ap_i = tf.gather(ind_ap_i, valid_batch, axis=0)
            LSL_sort = tf.gather(LSL_sort, valid_batch, axis=0)
            d_sort = tf.gather(d_sort, valid_batch, axis=0)
            LSL = tf.gather(LSL, valid_batch, axis=0)

            #
            pl = tf.gather(pl, valid_batch, axis=0)
            shadowing_LOS = tf.gather(shadowing_LOS, valid_batch, axis=0)
            shadowing_NLOS = tf.gather(shadowing_NLOS, valid_batch, axis=0)
            G_Antenna = tf.gather(G_Antenna, valid_batch, axis=0)
            D = tf.gather(D, valid_batch, axis=0)
            D_2d = tf.gather(D_2d, valid_batch, axis=0)
            p_LOS = tf.gather(p_LOS, valid_batch, axis=0)
            LSL_org = tf.gather(LSL_org, valid_batch, axis=0)
            #

            assigned_batch_index = tf.gather(assigned_batch_index, valid_batch, axis=0)
            LSL_assign = tf.gather(LSL_assign, valid_batch, axis=0)

            #
            pl_assign = tf.gather(pl_assign, valid_batch, axis=0)
            shadowing_LOS_assign = tf.gather(shadowing_LOS_assign, valid_batch, axis=0)
            shadowing_NLOS_assign = tf.gather(shadowing_NLOS_assign, valid_batch, axis=0)
            G_Antenna_assign = tf.gather(G_Antenna_assign, valid_batch, axis=0)
            D_assign = tf.gather(D_assign, valid_batch, axis=0)
            D_2d_assign = tf.gather(D_2d_assign, valid_batch, axis=0)
            p_LOS_assign = tf.gather(p_LOS_assign, valid_batch, axis=0)
            LSL_org_assign = tf.gather(LSL_org_assign, valid_batch, axis=0)
            #

            mask = tf.gather(mask, valid_batch, axis=0)

            AP_assign_user = tf.gather(AP_assign_user, valid_batch, axis=0)
            # --------------------------------------------
            # -------

            mask_i = mask * ind_ap_i
            mask_i = tf.transpose(tf.random.shuffle(tf.transpose(mask_i, [1, 0])), [1, 0])

            ap_assign_user = tf.reduce_max(mask * tf.cast(mask_i > 0.0, "float32"), axis=1, keepdims=True) - 1
            ap_assign_user = tf.gather_nd(mask_i, tf.concat(
                [tf.expand_dims(tf.constant(range(mask_i.shape[0])), axis=1), tf.cast(ap_assign_user, "int32")],
                axis=1))
            self.user_index = np.append(self.user_index, ap_assign_user.numpy() / self.Nuser_drop)

            mask_ap = tf.expand_dims(tf.scatter_nd(tf.concat([tf.expand_dims(tf.constant(range(mask.shape[0])), axis=1),
                                                              tf.cast(tf.expand_dims(ap_assign_user, axis=1),
                                                                      "int32") - 1], axis=1),
                                                   tf.ones(mask.shape[0]), [mask.shape[0], self.Nuser_drop]), axis=1)

            LSL_selected_user = tf.reduce_sum(LSL * mask_ap, axis=2, keepdims=True)

            # LSG+Tx Power for TN and NTN seperated and before assigment
            LSGwithTxPower_TN_before_assigment = -LSL[:, 0:57,
                                                  :] - self.P_Tx_dB + self.P_over_noise_db - self.noise_figure_user
            LSGwithTxPower_TN_before_assigment = tf.expand_dims(
                tf.reduce_max(LSGwithTxPower_TN_before_assigment, axis=1), axis=1)
            LSGwithTxPower_LEOcenterBeam_before_assigment = tf.expand_dims(tf.reduce_max(
                -LSL[:, 57:64, :] - self.P_Tx_sat_db + self.P_over_noise_db_sat - self.noise_figure_user_sat, axis=1),
                                                                           axis=1)
            # LSGwithTxPower_LEOcenterBeam_before_assigment = tf.expand_dims(-LSL[:, 57, :]-self.P_Tx_sat_db + self.P_over_noise_db_sat - self.noise_figure_user_sat, axis=1)

            #
            pl_selected_user = tf.reduce_sum(mask_ap * pl, axis=2, keepdims=True)
            shadowing_LOS_selected_user = tf.reduce_sum(mask_ap * shadowing_LOS, axis=2, keepdims=True)
            shadowing_NLOS_selected_user = tf.reduce_sum(mask_ap * shadowing_NLOS, axis=2, keepdims=True)
            G_Antenna_selected_user = tf.reduce_sum(mask_ap * G_Antenna, axis=2, keepdims=True)
            D_selected_user = tf.reduce_sum(mask_ap * D, axis=2, keepdims=True)
            D_2d_selected_user = tf.reduce_sum(mask_ap * D_2d, axis=2, keepdims=True)
            p_LOS_selected_user = tf.reduce_sum(mask_ap * p_LOS, axis=2, keepdims=True)
            LSL_org_selected_user = tf.reduce_sum(mask_ap * LSL_org, axis=2, keepdims=True)
            #

            LSL_assign = tf.concat([LSL_assign, LSL_selected_user], axis=2)

            #
            pl_assign = tf.concat([pl_assign, pl_selected_user], axis=2)
            shadowing_LOS_assign = tf.concat([shadowing_LOS_assign, shadowing_LOS_selected_user], axis=2)
            shadowing_NLOS_assign = tf.concat([shadowing_NLOS_assign, shadowing_NLOS_selected_user], axis=2)
            G_Antenna_assign = tf.concat([G_Antenna_assign, G_Antenna_selected_user], axis=2)
            D_assign = tf.concat([D_assign, D_selected_user], axis=2)
            D_2d_assign = tf.concat([D_2d_assign, D_2d_selected_user], axis=2)
            p_LOS_assign = tf.concat([p_LOS_assign, p_LOS_selected_user], axis=2)
            LSL_org_assign = tf.concat([LSL_org_assign, LSL_org_selected_user], axis=2)

            AP_assign_user = tf.concat([AP_assign_user, mask_ap], axis=1)

        LSL_assign = LSL_assign[0:self.batch_num, :,
                     1:]  # the assigned user index to AP_i. The entries have value between [0,self.Nuser_drop]

        #
        pl_assign = pl_assign[0:self.batch_num, :, 1:]
        shadowing_LOS_assign = shadowing_LOS_assign[0:self.batch_num, :, 1:]
        shadowing_NLOS_assign = shadowing_NLOS_assign[0:self.batch_num, :, 1:]
        G_Antenna_assign = G_Antenna_assign[0:self.batch_num, :, 1:]
        D_assign = D_assign[0:self.batch_num, :, 1:]
        D_2d_assign = D_2d_assign[0:self.batch_num, :, 1:]
        p_LOS_assign = p_LOS_assign[0:self.batch_num, :, 1:]
        LSL_org_assign = LSL_org_assign[0:self.batch_num, :, 1:]
        if self.indoor_calib:
            self.LSL_calib_assign = LSL_assign
            LSL_assign = LSL_org_assign

        #

        AP_assign_user = AP_assign_user[0:self.batch_num, 1:, :]
        assigned_batch_index = assigned_batch_index[0:self.batch_num]
        return LSL_assign, AP_assign_user, assigned_batch_index, pl_assign, shadowing_LOS_assign, shadowing_NLOS_assign, G_Antenna_assign, D_assign, p_LOS_assign, D_2d_assign, LSGwithTxPower_TN_before_assigment, LSGwithTxPower_LEOcenterBeam_before_assigment

    def BO_offloading_Assigned_UAVs(self, LSL):

        self.P_Tx_TN = tf.math.pow(10.0, (self.P_Tx_dB - 30.0) / 10.0)
        self.P_Tx_NTN = tf.math.pow(10.0, (self.P_Tx_sat_db - 30.0) / 10.0)
        self.PSD = tf.math.pow(10.0, (-174.0 - 30.0) / 10.0)
        self.BW_TN = self.bandwidth
        self.BW_NTN = self.bandwidth_sat
        self.NF_TN = tf.math.pow(10.0, self.noise_figure_user / 10.0)
        self.NF_NTN = tf.math.pow(10.0, self.noise_figure_user_sat / 10.0)

        if self.one_tier == False:
            LSL_TN = LSL[:, 0:57, :]
            LSL_NTN = LSL[:, 57:64, :]
        if self.one_tier == True:
            LSL_TN = LSL[:, 0:21, :]
            LSL_NTN = LSL[:, 21:28, :]
        LSG_TN = tf.math.pow(10, (-LSL_TN) / 10)
        LSG_NTN = tf.math.pow(10, (-LSL_NTN) / 10)
        LSG = tf.concat([LSG_TN, LSG_NTN], axis=1)

        SNR_TN = LSG_TN * self.P_Tx_TN / (self.BW_TN * self.NF_TN)
        num_TN = tf.linalg.diag_part(SNR_TN)
        denom_TN = tf.reduce_sum(SNR_TN, axis=1) - num_TN + self.PSD
        sinr_onlyground = num_TN / denom_TN
        sinr_onlyground = 10.0 * (tf.math.log(sinr_onlyground) / tf.math.log(10.0))
        sinr_onlyground = tf.expand_dims(sinr_onlyground,axis=1)

        LSL_min_NTN = tf.math.reduce_min(LSL_NTN, axis=1, keepdims=True)
        LSG_min_NTN = tf.math.pow(10, (-LSL_min_NTN) / 10)
        snr_link_NTN = LSG_NTN * self.P_Tx_NTN / (self.BW_NTN * self.NF_NTN)
        num_NTN = LSG_min_NTN * self.P_Tx_NTN / (self.BW_NTN * self.NF_NTN)
        denom_NTN = tf.expand_dims(tf.reduce_sum(snr_link_NTN, axis=1), axis=1) - num_NTN + self.PSD
        sinr_onlyLEO = num_NTN / denom_NTN
        sinr_onlyLEO = 10.0 * (tf.math.log(sinr_onlyLEO) / tf.math.log(10.0))
        if self.FRF_3:
            sinr_onlyLEO = (LSG_min_NTN * self.P_Tx_NTN) / (self.BW_NTN * self.NF_NTN* self.PSD)

        sinr_onlyLEO = 10.0 * (tf.math.log(sinr_onlyLEO) / tf.math.log(10.0))



        T2_NTN = tf.tile(tf.expand_dims(self.T1_TN[:, :, -1], axis=2), [1, 1, sinr_onlyground.shape[2]])
        self.T1_TN = self.T1_TN[:, :, :-1]
        NTN_Assign = tf.cast(sinr_onlyground <= self.T1_TN, "float32") * tf.cast(sinr_onlyLEO >= T2_NTN, "float32")
        TN_Assign = tf.cast(NTN_Assign == 0.0, "float32")
        NTN_Assign = tf.tile(NTN_Assign, [1, 7, 1])
        if self.one_tier == False:
            TN_Assign = tf.tile(TN_Assign, [1, 57, 1])
        if self.one_tier == True:
            TN_Assign = tf.tile(TN_Assign, [1, 21, 1])
        self.Offloaded_UEs = tf.reduce_sum(NTN_Assign, axis=2)
        self.Min_Offloaded_UEs = tf.reduce_min(self.Offloaded_UEs)
        self.Max_Offloaded_UEs = tf.reduce_max(self.Offloaded_UEs)
        self.Mean_Offloaded_UEs = tf.reduce_mean(self.Offloaded_UEs) / (self.Nap)
        self.Offloaded_UEs_perElv = tf.reduce_mean(self.Offloaded_UEs)
        Offloaded_UEs_perc = self.Mean_Offloaded_UEs

        LSL_assign_UAVs_Offloaded = NTN_Assign * LSL_NTN
        LSL_assign_UAVs_Not_Offloaded = TN_Assign * LSL_TN

        # UAVs_Offloaded_height = NTN_Assign[:, 0, :] * Xuser[:, :, 2]

        LSG_assign_UAVs_Offloaded = -LSL_assign_UAVs_Offloaded
        LSG_assign_UAVs_Not_Offloaded = -LSL_assign_UAVs_Not_Offloaded
        return LSG_assign_UAVs_Offloaded, LSG_assign_UAVs_Not_Offloaded, Offloaded_UEs_perc