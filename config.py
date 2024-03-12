"""
This is the Configuration script of the simulator

@authors: Mohamed Benzaghta and Rasoul Nikbakht
"""

import tensorflow as tf
import math
import numpy as np

class Config():

    def __init__(self):

        # ------------ Define the general paramters that will be used in the upcoming functions.
        self.batch_num = 200   #Number of iterations to run
        self.beta_open_loop = 1
        self.Zuser = 1.5            #Hight of user in m
        self.Zap = 25.0             #Hight of BSs in m
        self.N = 2                  #Number of cell in a tire  N=1 for 7 cells and N=2 for 19 cells
        self.EX = 1000              #X-axis grid length m, for PPP deployment type
        self.EY = 1000              #Y-axis grid length m, for PPP deployment type
        self.radius = 250           #ISD/2 in meters
        self.fc= 2.0                #TN freq in GHz
        self.fc_Hz = 2.0e9          #TN freq in Hz
        self.c = 3.0e8              #Speed of light m/s
        self.DeploymentType= "Hex"  #Deployment choice, Hex or PPP

        if self.DeploymentType=="Hex":
            self.Nap = 19                          #Number of BSs without sectoring
            if self.N==1:
                self.Nap = 7
            self.Nuser_drop = 15*3*self.Nap        #Number of UEs dropped #use 57 if debugging UE is on
        elif self.DeploymentType =="PPP":
            self.Nap = 19                           #Number of BSs without sectoring
            self.Nuser_drop = 15*3*self.Nap         #Number of UEs dropped

        self.bandwidth = 1e7 #TN Bandwidth in Hz
        self.noise_figure_user= 9 #TN Noise Figure in dB

        # ------------ Indoor Settings
        self.indoor = False
        self.in_ratio=0.0 #0.8                   #Indoor UEs ratio
        self.out_ratio=1.0 #0.2                  #Outdoor UEs ratio
        self.indoor_calib = False           #Used to check with 3GPP calibration
        if self.indoor_calib:
            self.Dist2D_exclud = 0          #Excludtion distance if UE is so close to the BSs,can be used as 35m if wanted
        else:
            self.Dist2D_exclud =35.0

        self.sectoring_status = True        #To sectorize each physical BS position to 3 sectors, 120 degree

        # ------------ Sat parameters
        self.calibration_sat = False        #Used to check with 3GPP calibration
        self.sat_user= False                #If only interested in NTN. No integration with TN.
        self.open_access= True              #Integration of TN-NTN
        self.Nleo= 7                        #Number of LEO Beams
        self.hex_N_sat = 4                  #Similar to TN, number of tiers of the Hex grid
        self.Zleo=600.0e3                   #Hight of LEO in m
        self.RE = 6371.0e3                  #Radius of Earth in m
        # self.LEO_x_cord=0.0               #This is replaced by alpha_factor and fed into the call function of Deployment and NTN_LSG #Determines the Elevation angle. For 87 Elv, make it 29.5e3. For 84 Elv, 55.0e3.
        # for z in [29.5e3, 55.0e3]: self.LEO_x_cord=z
        self.LEO_y_cord=0.0                 #Determines the Elevation angle. No need to be changed.
        self.fc_sat= 2.0                    #NTN freq in GHz
        self.fc_Hz_sat = 2.0e9              #NTN freq in Hz
        self.radius_sat = math.sqrt(3.0)/2.0*25e3       #LEO beam radius. sqrt(3)/2 comes from our hex refrence code because radius of the Sat is equivalent to the parameter size.
        self.Nuser_drop_sat = 7             #If needed to assosciate UEs to the other beams of the LEO.
        self.antenna_gain_max_db_sat = 30   #LEO max antenna gain

        self.FRF_3 = True  # Offloading based on FRF3 or FRF1
        if self.FRF_3:
            self.bandwidth_sat = 1e7            #NTN Bandwidth in Hz. FRF3=10MHz and FRF1=30MHz
        else :
            self.bandwidth_sat = 3e7

        if self.calibration_sat:
            self.bandwidth_sat = 3e7

        # ------------ TN BSs power/noise parameters
        # self.P_Tx_dB = 46.0   #BS Tx power in dBm
        self.noise_db = -174+10*tf.math.log(self.bandwidth)/tf.math.log(10.0)   #TN Noise in dBm
        self.noise_figure_user = 9  #UE noise figure in dB
        # self.P_over_noise_db = self.P_Tx_dB - self.noise_db #power over noise

        #------------ NTN LEO power/noise parameters
        self.P_Tx_sat_db = 34+30-30+(10*tf.math.log(self.bandwidth_sat/1.0e6)/tf.math.log(10.0))   #LEO Tx power in dBm. This is equivalent to 34dBW/MHz EIRP in the 3GPP document, so we found Tx power from EIRP 34+30 is convertion to dBm +BW in MHz in log -30 Max antenna gain
        self.noise_db_sat = -174+10*tf.math.log(self.bandwidth_sat)/tf.math.log(10.0) #-104.0   #-174+10*tf.math.log(self.bandwidth_sat)/tf.math.log(10.0)  #in dBm
        self.noise_figure_user_sat = 9  #in dB
        self.sat_bias=0.0 #worst5=24.5, worst10=26.5, worst15=28
        self.InterBeamInterference_factor = 0.0 #13.3025 is the max amount of inter-beam interference introduce in our closed-access experiment
        self.P_over_noise_db_sat = self.P_Tx_sat_db - self.noise_db_sat

        # ------------ UE power/noise parameters
        self.UE_P_Tx_dbm = 23.0         #TN UE Tx power in dBm
        self.UE_P_Tx_sat_dbm = 23.0     #NTN UE Tx power in dBm
        self.UE_bandwidth= 360.0e3      #UE Bandwidth
        self.UE_noise_db=-174+10*tf.math.log(self.UE_bandwidth)/tf.math.log(10.0)
        self.UE_P_over_noise_db = self.UE_P_Tx_dbm - self.UE_noise_db
        self.UE_P_over_noise_db_sat = self.UE_P_Tx_sat_dbm - self.UE_noise_db

        # ------------ Fractional Power Control (FPC) for TN UEs
        self.FPC = True
        if self.FPC:
            self.Po=-85.0
            self.alpha=0.8

        # ------------ UAVs deployment
        self.UAVs = True
        # self.Zuav=150.0        #UAV Height
        self.GUE_ratio = 0.6667 #Case5: 0.6667, Case4: 0.8, Case3: 0.93334, Case2:0.993334 #Use 1 and 0 in case of UE debugging
        self.UAV_ratio =  0.3333 #Case5: 0.3333, Case4: 0.2, Case3: 0.06666, Case2:0.006666
        self.Zuav = tf.random.uniform([2 * self.batch_num, int(self.UAV_ratio * self.Nuser_drop)+1, 1], 150.0, 150.0) #+1 for case5,3,2

        self.k_dominant_BS=57    #Used for SINR offloading. This is dominant BSs interfernce instead of looking at entire 57 BSs. This value should be desired k with +1

        # --------- Capacity restrictions
        self.Cap_cond = False
        self.desired_UAVs_offload=58

        # --------- Hotspot parameters (the code part is in hex class)
        self.hotspot=False
        self.hotspot_radius = 40   #250 this is for no hotspots
        self.Num_hotspot=19
        self.UEs_in_hotspot= 30#*3 #15 this is for no hotspots

        # --------- To support 1-tier only evaluations (21-cells [index: 4,5,8,9,10,13,14])
        self.one_tier = False #Do not use this, just change N from 2 to 1
        self.LSG_all = False
        self.UAVs_highway = True

        # --------- Corridor hights
        self.h_corr1 = 300.0
        self.h_corr2 = 300.0
        self.h_corr3 = 300.0
        self.h_corr4 = 300.0

        # --------- Assigment based on RSS+toe (where toe is going to be optimized by BO for load balancing)
        self.RSS_offloading = False

        # --------- Tilt-optimization
        # self.BS_tilt=-40.0 #minus sign refer to being down-tilted

        # --------- Specialized BO
        self.Specialized_BO = False
        self.IterativeBO_1Threshold = False

        # --------- UEs debugging
        self.GUEs_debug = False


        return


