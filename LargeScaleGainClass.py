"""
This is the TN Large Scale Gain Class for GUEs of the simulator:
    In this class, the LSGs between the TN BSs and GUEs are calculated.
    Path loss, shadowing and Antenna Gain are computed here.
    Note: All of our terrestrial network ground UEs or UAVs UEs are assumed to be deployed for Urban Macro Scenarios (UMa)

@authors: Mohamed Benzaghta and Rasoul Nikbakht
"""

import math
import tensorflow as tf
from config import  Config
class Large_Scale_Gain(Config):

    #Define the general paramters that will be used in the upcoming functions.
    def __init__(self):
        Config.__init__(self)
        self.exponent=3.8               #famous n in PL formula
        self.shadowing_sigma_LOS= 4.0    #shadowing SD for LOS
        self.shadowing_sigma_NLOS = 6.0   #shadowing SD for NLOS
        # self.shadowing_sigma_NLOS_in=0.0
        self.shadowing_sigma_NLOS_in_low=0.0 #4.4
        self.shadowing_sigma_NLOS_in_high=0.0 #6.4
        self.hE=1.0                     #effective enviroment height taken as 1m. 3GPP document 38.901, p.29.
        self.P_over_noise =100         #Noise power in dB BW*-174(power spectral) Add noise dependent BW
        self.SLA_cut= 30                #in dB
        self.Amax_cut=30                #in dB
        self.phi_3dB=65              #3dB HPBW of antenna patter in degree for Azimuth
        self.thetha_3dB=10             #3dB HPBW of antenna patter in degree for Elevation
        self.Antenna_elements = 1
        self.Lambda = self.fc*1e9/self.c
        self.Antenna_spacing = self.Lambda/2
        if self.sat_user:
            self.Nuser_drop = self.Nuser_drop-self.Nuser_drop_sat

        
    def call(self,D,D_2d,D_in,D_2d_in,D_2d_building,Azi_phi_deg,Elv_thetha_deg,Azi_phi_deg_in, Elv_thetha_deg_in,Zuser_indoor,BS_tilt):
        self.BS_tilt = BS_tilt
       
        if self.indoor==True:
            p_LOS_out= self.Prob_LOS_3GPP(D_2d)
            p_LOS_in= self.Prob_LOS_3GPP_in(D_2d_building)
            p_LOS= tf.concat([p_LOS_out,p_LOS_in],axis=2)
        
            G_dB_out,G_linear_out,pl_out,shadowing_LOS_out,shadowing_NLOS_out =self.PL_3GPP_UMa(D, D_2d, self.Zap, self.Zuser, self.fc,p_LOS_out)   #This is the Gain (negative of PL) in dB and scaler form calculated in PL_3GPP_UMa function below.
            G_dB_in,G_linear_in,pl_in,shadowing_in =self.PL_3GPP_UMa_in(D_in, D_2d_in,D_2d_building, self.Zap, Zuser_indoor, self.fc,p_LOS_in)

            G_dB=tf.concat([G_dB_out,G_dB_in],axis=2)
            if self.indoor_calib:
                G_dB_out_calib, G_linear_out_calib, pl_out_calib, shadowing_LOS_out_calib, shadowing_NLOS_out_calib = self.PL_3GPP_UMa(D, D_2d,self.Zap,self.Zuser, self.fc, tf.ones(p_LOS_out.shape))  # This is the Gain (negative of PL) in dB and scaler form calculated in PL_3GPP_UMa function below.
                G_dB_in_calib, G_linear_in_calib, pl_in_calib, shadowing_in_calib = self.PL_3GPP_UMa_in(D_in, D_2d_in, D_2d_building, self.Zap,Zuser_indoor, self.fc, tf.ones(p_LOS_in.shape))
                G_dB_calib = tf.concat([G_dB_out_calib, G_dB_in_calib], axis=2)
                self.Pl_calib = tf.concat([self.Pl_calib,self.Pl_in_calib],axis=2)
            G_linear=tf.concat([G_linear_out,G_linear_in],axis=2)
            pl=tf.concat([pl_out,pl_in],axis=2)

            shadowing_NLOS=tf.concat([shadowing_NLOS_out,shadowing_in],axis=2)
            shadowing_LOS=tf.concat([shadowing_LOS_out,shadowing_in],axis=2)

            G_sector_out = self.sectoring(Azi_phi_deg,Elv_thetha_deg) # this line includes both antenna gain and sectoring effect
            G_sector_in = self.sectoring(Azi_phi_deg_in, Elv_thetha_deg_in)

            G_sector=tf.concat([G_sector_out,G_sector_in],axis=2)
        
        elif self.indoor==False:
            p_LOS= self.Prob_LOS_3GPP(D_2d)
            G_dB,G_linear,pl,shadowing_LOS,shadowing_NLOS =self.PL_3GPP_UMa(D, D_2d, self.Zap, self.Zuser, self.fc,p_LOS)
            G_sector = self.sectoring(Azi_phi_deg,Elv_thetha_deg)
        #---------------------------------------------------------
        if self.sectoring_status:
            shadowing_LOS= tf.tile(shadowing_LOS, [1, 3, 1])
            shadowing_NLOS= tf.tile(shadowing_NLOS, [1, 3, 1])
            LSG= tf.tile(G_dB, [1, 3, 1])+G_sector
            LSL=-LSG
            p_LOS = tf.tile(p_LOS, [1, 3, 1])
            pl= tf.tile(pl, [1, 3, 1])
        else:
            LSG = G_dB
            LSL = -LSG
            G_sector = G_sector[:,0:self.Nap,:]
        #----------------------------------------------------
        if self.indoor_calib:
            if self.sectoring_status:
                # LSG_calib = tf.tile(G_dB_calib, [1, 3, 1])+G_sector
                LSG_calib = -tf.tile(self.Pl_calib, [1, 3, 1]) + G_sector

            else:
                # LSG_calib = G_dB_calib
                LSG_calib = -self.Pl_calib
            LSL_calib = -LSG_calib
            self.LSL_calib = LSL_calib

        return LSL,LSG,G_sector,p_LOS,pl,shadowing_LOS,shadowing_NLOS
        # Incorporate sectoring using different aneta gain
    def sectoring(self,Azi_phi_deg, Elv_thetha_deg):
        if Elv_thetha_deg.shape[2] == 0:
            Elv_thetha_deg = tf.tile(Elv_thetha_deg, [1, 3, 1]) + self.BS_tilt[:, :, 0:Elv_thetha_deg.shape[2]]
        else:
            Elv_thetha_deg=tf.tile(Elv_thetha_deg, [1, 3, 1])+self.BS_tilt[:,:,0:Elv_thetha_deg.shape[2]] #This is to consider the tilt for every sector seperately
        G_secor0 = self.Antenna_gain_3GPP(Azi_phi_deg, Elv_thetha_deg[:,0:19,:])

        # Azi_phi_deg1= (tf.cast((Azi_phi_deg+120)<180,"float32")*tf.cast((Azi_phi_deg+120)>-180,"float32")*(Azi_phi_deg+120) +
        #              tf.cast((Azi_phi_deg + 120) < -180, "float32") * (Azi_phi_deg + 120 + 360) +
        #              tf.cast((Azi_phi_deg + 120) > 180, "float32") * (Azi_phi_deg + 120 - 360))
        Azi_phi_deg1 = (tf.cast((Azi_phi_deg+120)>180,"float32")*(Azi_phi_deg+120-360) +
                     tf.cast((Azi_phi_deg + 120) < 180.001, "float32") * (Azi_phi_deg + 120 ))
        G_secor1 = self.Antenna_gain_3GPP(Azi_phi_deg1, Elv_thetha_deg[:,19:38,:])
        
        # Azi_phi_deg2 = (tf.cast((Azi_phi_deg+240)<180,"float32")*tf.cast((Azi_phi_deg+240)>-180,"float32")*(Azi_phi_deg+240) +
        #              tf.cast((Azi_phi_deg + 240) < -180, "float32") * (Azi_phi_deg + 240 + 360) +
        #              tf.cast((Azi_phi_deg + 240) > 180, "float32") * (Azi_phi_deg + 240 - 360))
        Azi_phi_deg2 = (tf.cast((Azi_phi_deg+240)>180,"float32")*(Azi_phi_deg+240.0-360.0) +
                     tf.cast((Azi_phi_deg + 240.0) < 180.001, "float32") * (Azi_phi_deg + 240))
        G_secor2 = self.Antenna_gain_3GPP(Azi_phi_deg2,Elv_thetha_deg[:,38:,:])
        
        G_sector = tf.concat([G_secor0,G_secor1,G_secor2],axis=1)
        self.Azi_phi_sector = tf.concat([Azi_phi_deg,Azi_phi_deg1,Azi_phi_deg2],axis=1)
        # G_sector = tf.zeros(G_sector.shape,'float32')
        return G_sector
    #Calculating antenna gain based on its patter and formulas provided in 3GPP Document 38.901 P.23
    def Antenna_gain_3GPP(self,Azi_phi_deg,Elv_thetha_deg):
        # -------------------------- Array gain
        ind = tf.constant([[[[i for i in range(self.Antenna_elements)]]]])
        phase = tf.expand_dims(Elv_thetha_deg, axis=3)
        phase = tf.complex(0.0, 2 * math.pi * self.Antenna_spacing * tf.math.cos(
            phase * math.pi / 180.0) / self.Lambda * tf.cast(ind, 'float32'))

        w = tf.complex(tf.ones(phase.shape, 'float32'), 0.0)  # replace the w with beamforming vector
        array_gain = tf.math.abs(tf.reduce_sum(tf.math.exp(-phase) * w, axis=3))
        self.array_gain = 10.0 * tf.math.log(array_gain) / tf.math.log(10.0)
        self.array_gain = self.cut_by_30(self.array_gain)
        #-----------------------------------------------------------------

        Azi_cut= 12*(tf.pow(Azi_phi_deg/self.phi_3dB,2))
        Elv_cut=12*(tf.pow((Elv_thetha_deg-90)/self.thetha_3dB,2))

        G_azi = self.cut_by_30(-Azi_cut)
        self.G_azi = 8+ G_azi

        G_elv = self.cut_by_30(-Elv_cut)
        self.G_elv1 =  G_elv
        self.G_elv = 8 + G_elv

        #Total loss
        # G_total=-(G_azi+G_elv)
        if self.sectoring_status:
            G_total = G_azi + G_elv
        else:
            G_total = G_elv

        G_Antenna=8+self.cut_by_30(G_total)+self.array_gain

        self.G_Antenna = G_Antenna
        return G_Antenna
    def cut_by_30(self,G_Antenna):
        con= tf.cast(G_Antenna>-30.0,"float32")
        G_Antenna= G_Antenna*con-30*(1-con)
        return G_Antenna
    
    #Calculating probability of LOS based on 3GPP Document 38.901 P.30, UMa scenario    
    def Prob_LOS_3GPP(self,D_2d):
        batch_num = D_2d.shape[0]
        Nuser_drop_out = int(self.out_ratio * self.GUE_ratio * self.Nuser_drop)
        con_0a= tf.cast(D_2d<=18,"float32")
        P_LOS1=1*con_0a
        
        con_0b= tf.cast(D_2d>18,"float32")
        P_LOS2=((18/D_2d)+(tf.math.exp(-(D_2d/63))*(1-(18/D_2d))))*con_0b
        
        P_LOS=P_LOS1+P_LOS2   #This is the probability of LOS before the experiment.
        
        P_random = tf.random.uniform([batch_num,self.Nap,D_2d.shape[2]],0,1)  #This is to perform the experiment.
        if self.sat_user:
            if self.indoor:
                P_random = tf.random.uniform([batch_num, self.Nap, D_2d.shape[2]], 0, 1)

        elif self.sat_user==False:
            if self.indoor:
                P_random = tf.random.uniform([batch_num,self.Nap,Nuser_drop_out],0,1)

            
        LOS = tf.cast(P_LOS>P_random,"float32") #This gives a 0/1 matrix. If 1 UE is in LOS else NLOS
        # if self.indoor_calib:
        #     LOS = tf.ones(LOS.shape,'float32')
        #
        # LOS = tf.ones(LOS.shape, 'float32')
        return LOS
    
    def Prob_LOS_3GPP_in(self,D_2d_building):
        Nuser_drop_in = int(self.in_ratio * self.GUE_ratio * self.Nuser_drop)
        # batch_num = D_2d_in.shape[0]

        # LOS_in = tf.zeros(D_2d_in.shape,'float32')
        D_2d = D_2d_building
        batch_num = D_2d.shape[0]
        Nuser_drop_out = int(self.out_ratio * self.GUE_ratio * self.Nuser_drop)
        con_0a = tf.cast(D_2d <= 18, "float32")
        P_LOS1 = 1 * con_0a

        con_0b = tf.cast(D_2d > 18, "float32")
        P_LOS2 = ((18 / D_2d) + (tf.math.exp(-(D_2d / 63)) * (1 - (18 / D_2d)))) * con_0b

        P_LOS = P_LOS1 + P_LOS2  # This is the probability of LOS before the experiment.

        P_random = tf.random.uniform([batch_num, self.Nap, self.Nuser_drop], 0,
                                     1)  # This is to perform the experiment.
        if self.sat_user:
            if self.indoor:
                P_random = tf.random.uniform([batch_num, self.Nap, D_2d_building.shape[2]], 0, 1)
                
        elif self.sat_user==False:
            if self.indoor:
                P_random = tf.random.uniform([batch_num, self.Nap, Nuser_drop_in], 0, 1)

        LOS_in = tf.cast(P_LOS > P_random, "float32")  # This gives a 0/1 matrix. If 1 UE is in LOS else NLOS
        # if self.indoor_calib:
        #     LOS_in = tf.ones(LOS_in.shape,'float32')
        # LOS_in = tf.ones(LOS_in.shape, 'float32')
        return LOS_in
    
    #This function calculates the Path Loss based on 3GPP Document 38.901 Urban Macro scenaario is consedired (p.27), LOS condition. Pathloss [dB], fc is in GHz and d is in meters
    def PL_3GPP_UMa (self,D,D_2d,Zap,Zuser,fc,LOS):
        if self.fc<6:
            pl1_coff=[28.0,22.0]
            pl2_coff=[28.0,9.0]
        else:
            pl1_coff=[32.4,20.0]
            pl2_coff=[32.4,10]
        fc_Hz=self.fc*tf.pow(10.0,9)                #in calculaating Break Point distance fc is in Hz
        Zap_e=Zap-self.hE                           #effective antenna height at BS
        Zuser_e=Zuser-self.hE                       #effective antenna height at UE
        d_BP=4*Zap_e*Zuser_e*(fc_Hz/self.c)         # Breaking Point. Note: In calculating the Break Point, we ignored the note about effective enviroment height in P.29 ((Note 1) of the 38.901 document and it is taken as 1m.
        
        con_LOS= tf.cast(LOS==1.0,"float32")  #If LOS condition, perform the following PL calculations
        
        con_1= tf.cast(D_2d>=0.0,"float32")* tf.cast(D_2d<=d_BP,"float32")           #This is the first condition in the PL formula if the 2D is less than the breakpoint. Note: 0.0 should be changed to 10.0 when we add the condition of min. distance of UE to BS of 35m.
        con_2= tf.cast(D_2d>=d_BP,"float32")* tf.cast(D_2d<=50000,"float32")         #This is the second condition in the PL formula if the 2D is bigger than the breakpoint. This gives a 0/1 matrix. 1 if true and 0 if false.
        pl_1=pl1_coff[0]+(pl1_coff[1]*(tf.math.log(D)/tf.math.log(10.0)))+(20.0*(tf.math.log(fc)/tf.math.log(10.0)))          #PL part 1 formula for 2D distance less than the breakpoint.
        pl_1=pl_1*con_1             #Multiply the PL part 1 to the condition 1 so we get the tensor size of desired PL values.
        
        pl_2=pl2_coff[0]+(40.0*(tf.math.log(D)/tf.math.log(10.0)))+(20.0*(tf.math.log(fc)/tf.math.log(10.0)))-(pl2_coff[1]*(tf.math.log(tf.pow(d_BP,2.0)+tf.pow(self.Zap-self.Zuser,2.0))/tf.math.log(10.0))) #PL part 2 formula for 2D distance higher than the breakpoint.
        pl_2=pl_2*con_2             #Multiply the PL part 2 to the condition 2 so we get the tensor size of desired PL values.
        
        pla=pl_1+pl_2
        self.Pla_calib =pla
        shadowing_LOS =tf.random.normal(pla.shape, 0, self.shadowing_sigma_LOS, tf.float32)
        # self.shadowing_LOS = tf.tile(shadowing_LOS,[1,3,1])
        pla1=pla+shadowing_LOS
        pla1=pla1*con_LOS  
        
        con_NLOS= tf.cast(LOS==0.0,"float32")  #If NLOS condition, perform the following PL calculations
        
        plba=13.54+(39.08*(tf.math.log(D)/tf.math.log(10.0)))+(20.0*(tf.math.log(fc)/tf.math.log(10.0)))-(0.6*(Zuser-1.5))
        
        con_3= tf.cast(plba>pla,"float32")
        plb1=plba*con_3
        
        con_4= tf.cast(plba<=pla,"float32")
        plb2=pla*con_4
        
        plb=plb1+plb2
        self.Plb_calib = plb
        shadowing_NLOS =tf.random.normal(plb.shape, 0, self.shadowing_sigma_NLOS, tf.float32)
        plb=plb+shadowing_NLOS
        # self.shadowing_NLOS = tf.tile(shadowing_NLOS,[1,3,1])
        plb=plb*con_NLOS
        
        pl=pla1+plb
        # self.pl = tf.tile(pl,[1,3,1])
        G_dB=-pl                         
        G_linear=tf.pow(10.0,G_dB/10) 
        self.Pl_calib = self.Pla_calib+self.Plb_calib
        return G_dB,G_linear,pl,shadowing_LOS,shadowing_NLOS
    
    def PL_3GPP_UMa_in (self,D_in,D_2d_in,D_2d_building,Zap,Zuser_indoor,fc,LOS_in):
        Zuser= Zuser_indoor
        D =D_in
        D_2d = D_2d_building

        if self.fc < 6:
            pl1_coff = [28.0, 22.0]
            pl2_coff = [28.0, 9.0]
        else:
            pl1_coff = [32.4, 20.0]
            pl2_coff = [32.4, 10]
        fc_Hz = self.fc * tf.pow(10.0, 9)  # in calculaating Break Point distance fc is in Hz
        
        #hE calculations
        # con_Pc_1=tf.cast(Zuser<13.0,"float32")
        # C_1=0.0*con_Pc_1
        
        # con_gD2d_1=tf.cast(D_2d<=18.0,"float32")
        # g_D2d_1=0.0*con_gD2d_1
        con_gD2d_2=tf.cast(D_2d>18.0,"float32")
        g_D2d= ((5.0/4.0)*tf.pow(D_2d/100.0,3.0)*tf.exp(-D_2d/150.0))*con_gD2d_2
        # g_D2d=g_D2d_1+g_D2d_2
        
        con_Pc = tf.cast(Zuser>=13.0,"float32")*tf.cast(Zuser<=23.0,"float32")
        Pc = tf.pow((Zuser*con_Pc+(13+Zuser)*(1-con_Pc)-13.0)/10.0,1.5)
        C=Pc*g_D2d*con_Pc


        # C=C_1+C_2
        Prob_C=1/(1+C)

        P_random = tf.random.uniform(Prob_C.shape, 0, 1)

        Ind_1_m = tf.cast(Prob_C > P_random, "float32")  # This gives a 0/1 matrix. If 1 UE is in LOS else NLOS
        Ind_rand_m = (1-Ind_1_m)

        hE=self.hE*Ind_1_m + Ind_rand_m*(tf.random.uniform(Prob_C.shape,0,1)*(Zuser-1.5-12)+12)
        #
        Zap_e = Zap-hE  # effective antenna height at BS
        Zuser_e = Zuser-hE  # effective antenna height at UE
        d_BP = 4 * Zap_e * Zuser_e * (
                    fc_Hz / self.c)  # Breaking Point. Note: In calculating the Break Point, we ignored the note about effective enviroment height in P.29 ((Note 1) of the 38.901 document and it is taken as 1m.
        # d_BP =0.0
        con_LOS = tf.cast(LOS_in == 1.0, "float32")  # If LOS condition, perform the following PL calculations

        con_1 = tf.cast(D_2d >= 0.0, "float32") * tf.cast(D_2d <= d_BP,
                                                          "float32")  # This is the first condition in the PL formula if the 2D is less than the breakpoint. Note: 0.0 should be changed to 10.0 when we add the condition of min. distance of UE to BS of 35m.
        con_2 = tf.cast(D_2d >= d_BP, "float32") * tf.cast(D_2d <= 50000,
                                                           "float32")  # This is the second condition in the PL formula if the 2D is bigger than the breakpoint. This gives a 0/1 matrix. 1 if true and 0 if false.
        pl_1 = pl1_coff[0] + (pl1_coff[1] * (tf.math.log(D) / tf.math.log(10.0))) + (20.0 * (
                    tf.math.log(fc) / tf.math.log(10.0)))  # PL part 1 formula for 2D distance less than the breakpoint.
        pl_1 = pl_1 * con_1  # Multiply the PL part 1 to the condition 1 so we get the tensor size of desired PL values.

        pl_2 = pl2_coff[0] + (40.0 * (tf.math.log(D) / tf.math.log(10.0))) + (
                    20.0 * (tf.math.log(fc) / tf.math.log(10.0))) - (pl2_coff[1] * (
                    tf.math.log(tf.pow(d_BP, 2.0) + tf.pow(self.Zap - self.Zuser, 2.0)) / tf.math.log(
                10.0)))  # PL part 2 formula for 2D distance higher than the breakpoint.
        pl_2 = pl_2 * con_2  # Multiply the PL part 2 to the condition 2 so we get the tensor size of desired PL values.

        pla = pl_1 + pl_2

        # shadowing_LOS = tf.random.normal(pla.shape, 0, self.shadowing_sigma_LOS, tf.float32)
        # # self.shadowing_LOS = tf.tile(shadowing_LOS,[1,3,1])
        pla = pla #+ shadowing_LOS

        pla1 = pla * con_LOS

        con_NLOS = tf.cast(LOS_in == 0.0, "float32")  # If NLOS condition, perform the following PL calculations

        plba = 13.54 + (39.08 * (tf.math.log(D) / tf.math.log(10.0))) + (
                    20.0 * (tf.math.log(fc) / tf.math.log(10.0))) - (0.6 * (Zuser - 1.5))

        con_3 = tf.cast(plba > pla, "float32")
        plb1 = plba * con_3

        con_4 = tf.cast(plba <= pla, "float32")
        plb2 = pla * con_4

        plb = plb1 + plb2
        shadowing_NLOS = tf.random.normal(plb.shape, 0, self.shadowing_sigma_NLOS, tf.float32)
        plb = plb #+ shadowing_NLOS
        # self.shadowing_NLOS = tf.tile(shadowing_NLOS,[1,3,1])
        plb = plb * con_NLOS

        pl = pla1 + plb
        self.Pl_in_calib = pl

        #Building penetration loss low
        L_glass=2.0+0.2*fc
        L_concrete=5.0+4.0*fc
        L_IRR_glass=23.0+0.3*fc
        
        PL_tw_low=5.0-(10.0*(tf.math.log( (0.3*tf.pow(10.0,-L_glass/10.0))+(0.7*tf.pow(10.0,-L_concrete/10.0)))/tf.math.log(10.0)))
        PL_tw_High=5.0-(10.0*(tf.math.log( (0.7*tf.pow(10.0,-L_IRR_glass/10.0))+(0.3*tf.pow(10.0,-L_concrete/10.0)))/tf.math.log(10.0)))
        PL_tw_low=20.0
        PL_tw_High=20.0
        #Building inside loss low and shadowing (50% high and 50% low as mentioned in calibration scenario)
        Nuser_drop_in = int(self.in_ratio * self.GUE_ratio * self.Nuser_drop)
        Nuser_drop_in_low=int(Nuser_drop_in/2)
        
        D_2d_in_low=D_2d_in[:,:,0:Nuser_drop_in_low]
        D_2d_in_high=D_2d_in[:,:,Nuser_drop_in_low:Nuser_drop_in]

        D_2d_building_low=D_2d_building[:,:,0:Nuser_drop_in_low]
        D_2d_building_high=D_2d_building[:,:,Nuser_drop_in_low:Nuser_drop_in]
        
        shadowing_NLOS_in_low =tf.random.normal(D_2d_in_low.shape, 0, self.shadowing_sigma_NLOS_in_low, tf.float32)
        shadowing_NLOS_in_high =tf.random.normal(D_2d_in_high.shape, 0, self.shadowing_sigma_NLOS_in_high, tf.float32)
        
        PL_inside_low=0.5*(D_2d_in_low-D_2d_building_low)+shadowing_NLOS_in_low+PL_tw_low
        PL_inside_high=0.5*(D_2d_in_high-D_2d_building_high)+shadowing_NLOS_in_high+PL_tw_High

        PL_inside=tf.concat([PL_inside_low,PL_inside_high],axis=2)
        
        shadowing_in=tf.concat([shadowing_NLOS_in_low,shadowing_NLOS_in_high],axis=2)
        # shadowing_LOS_in = tf.zeros(shadowing_NLOS_in.shape,'float32')

        #
        
        pl_in=pl+PL_inside
        G_dB_in=-pl_in                         
        G_linear_in=tf.pow(10.0,G_dB_in/10) 
        
        #
        return G_dB_in,G_linear_in,pl_in,shadowing_in
