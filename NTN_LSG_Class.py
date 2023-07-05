"""
This is the NTN Large Scale Gain Class of the simulator:
    In this class, the LSGs between the TN BSs and GUEs/UAVs are calculated.
    Path loss, shadowing and Antenna Gain are computed here.
    Note: All of our terrestrial network ground UEs or UAVs UEs are assumed to be deployed for Urban Macro Scenarios (UMa)

@authors: Mohamed Benzaghta and Rasoul Nikbakht
"""

import tensorflow as tf
import math
from scipy.special import j1
import numpy as np
from config import  Config
from DeploymentClass import  Deployment

class NTN_Large_Scale_Gain(Config):
    def __init__(self):
        Config.__init__(self)
        self.Deployment = Deployment()
        self.shadowing_sigma_LOS = 1.2 #4
        if self.calibration_sat:
            self.shadowing_sigma_LOS= .74
        self.shadowing_sigma_NLOS= 9.2 #6
        self.k=2*math.pi*self.fc_Hz_sat/self.c
        self.a=1
        self.ka=self.k*self.a

        return

    def call(self,NTN_D2d,Azi_phi_deg_Sat,Elv_thetha_deg_Sat,Xuser,alpha_factor):
        self.LEO_x_cord = alpha_factor
        NTN_D,NTN_Elevation, NTN_Boresight=self.Elevation_angle (NTN_D2d,Azi_phi_deg_Sat,Elv_thetha_deg_Sat,Xuser)
        NTN_AntennaGain_dB=self.NTN_AntennaGain(NTN_Boresight)
        NTN_LOS=self.NTN_Prob_LOS_3GPP(NTN_Elevation)
        PLg=self.NTN_PLg()
        PLs=self.NTN_PLs()
        PLe=self.NTN_PLe()
        NTN_G_dB, NTN_G_linear, NTN_PL, NTN_shadowing_LOS, NTN_shadowing_NLOS=self.NTN_PL_3GPP_UMa(NTN_D,NTN_Elevation,NTN_LOS,PLg,PLs,PLe)
        NTN_LSG=NTN_G_dB+NTN_AntennaGain_dB
        NTN_LSL=-NTN_LSG
        return NTN_LSL,NTN_LSG,NTN_AntennaGain_dB,NTN_LOS,NTN_PL,NTN_shadowing_LOS,NTN_shadowing_NLOS

    def call_in_beforeAssigment_centerBeam(self,NTN_D2d):
        _,_, _,_,_,_,_,_,D_2d_in,D_2d_building,_,_,_,_,_, _ = self.Deployment.Call()
        D_2d_in = tf.expand_dims(D_2d_in[:, 9, :], axis=1)
        D_2d_building = tf.expand_dims(D_2d_building[:, 9, :], axis=1)

        NTN_D,NTN_Elevation, NTN_Boresight=self.Elevation_angle (NTN_D2d)
        NTN_AntennaGain_dB=self.NTN_AntennaGain(NTN_Boresight)
        NTN_LOS=self.NTN_Prob_LOS_3GPP(NTN_Elevation)
        PLg=self.NTN_PLg()
        PLs=self.NTN_PLs()
        PLe=self.NTN_PLe()
        NTN_G_dB, NTN_G_linear, NTN_PL, NTN_shadowing_LOS, NTN_shadowing_NLOS=self.NTN_PL_3GPP_UMa_in_beforeAssigment(NTN_D, NTN_Elevation, NTN_LOS, PLg, PLs, PLe,D_2d_in,D_2d_building)


        NTN_LSG=NTN_G_dB+NTN_AntennaGain_dB
        NTN_LSL=-NTN_LSG
        return NTN_LSL,NTN_LSG,NTN_AntennaGain_dB,NTN_LOS,NTN_PL,NTN_shadowing_LOS,NTN_shadowing_NLOS

    def call_in_beforeAssigment_otherBeam(self, NTN_D2d,D_2d_in,D_2d_building):
        D_2d_in = tf.expand_dims(D_2d_in[:, 9, :], axis=1)
        D_2d_in = tf.tile(D_2d_in, [1, 6, 1])
        D_2d_building = tf.expand_dims(D_2d_building[:, 9, :], axis=1)
        D_2d_building = tf.tile(D_2d_building, [1, 6, 1])

        NTN_D, NTN_Elevation, NTN_Boresight = self.Elevation_angle(NTN_D2d)
        NTN_AntennaGain_dB = self.NTN_AntennaGain(NTN_Boresight)
        NTN_LOS = self.NTN_Prob_LOS_3GPP(NTN_Elevation)
        PLg = self.NTN_PLg()
        PLs = self.NTN_PLs()
        PLe = self.NTN_PLe()
        NTN_G_dB, NTN_G_linear, NTN_PL, NTN_shadowing_LOS, NTN_shadowing_NLOS = self.NTN_PL_3GPP_UMa_in_beforeAssigment(NTN_D, NTN_Elevation, NTN_LOS, PLg, PLs, PLe, D_2d_in, D_2d_building)

        NTN_LSG = NTN_G_dB + NTN_AntennaGain_dB
        NTN_LSL = -NTN_LSG
        return NTN_LSL, NTN_LSG, NTN_AntennaGain_dB, NTN_LOS, NTN_PL, NTN_shadowing_LOS, NTN_shadowing_NLOS

    def call_in_afterAssigment(self, NTN_D2d):
        _ = self.Deployment.Call()
        D_2d_building_sat = NTN_D2d
        D_2d_inside_building_sat = tf.math.minimum(
                tf.random.uniform([D_2d_building_sat.shape[0], D_2d_building_sat.shape[1], D_2d_building_sat.shape[2]], 0.0, 25.0),
                tf.random.uniform([D_2d_building_sat.shape[0], D_2d_building_sat.shape[1], D_2d_building_sat.shape[2]], 0.0, 25.0))
        D_2d_in_sat = D_2d_building_sat + D_2d_inside_building_sat

        NTN_D, NTN_Elevation, NTN_Boresight = self.Elevation_angle(NTN_D2d)
        NTN_AntennaGain_dB = self.NTN_AntennaGain(NTN_Boresight)
        NTN_LOS = self.NTN_Prob_LOS_3GPP(NTN_Elevation)
        PLg = self.NTN_PLg()
        PLs = self.NTN_PLs()
        PLe = self.NTN_PLe()
        NTN_G_dB, NTN_G_linear, NTN_PL, NTN_shadowing_LOS, NTN_shadowing_NLOS =self. NTN_PL_3GPP_UMa_in_afterAssigmentSatUEs(NTN_D,NTN_Elevation,NTN_LOS,PLg,PLs,PLe,D_2d_in_sat,D_2d_building_sat)

        NTN_LSG = NTN_G_dB + NTN_AntennaGain_dB
        NTN_LSL = -NTN_LSG
        return NTN_LSL, NTN_LSG, NTN_AntennaGain_dB, NTN_LOS, NTN_PL, NTN_shadowing_LOS, NTN_shadowing_NLOS
    
    def Elevation_angle (self,NTN_D2d,Azi_phi_deg_Sat,Elv_thetha_deg_Sat,Xuser):

        beta=(NTN_D2d)/40030.0e3*360.0
        NTN_D=tf.math.sqrt(tf.math.pow(self.RE, 2)+tf.math.pow(self.RE+self.Zleo, 2)-2*self.RE*(self.RE+self.Zleo)*tf.math.cos(np.deg2rad(beta)))
        NTN_Elevation=tf.math.acos(((self.RE+self.Zleo)*tf.math.sin(np.deg2rad(beta))/NTN_D))*180.0/math.pi

        #These are the global coordinates
        NTN_D = tf.tile(NTN_D, [1, 7, 1])
        NTN_Elevation = tf.tile(NTN_Elevation, [1, 7, 1])
        Azi_phi_deg_Sat = tf.tile(Azi_phi_deg_Sat, [1, 7, 1])
        Elv_thetha_deg_Sat = tf.tile(Elv_thetha_deg_Sat, [1, 7, 1])
        NTN_D2d = tf.tile(NTN_D2d, [1, 7, 1])

        # Angles between LEO and UE in local coordinates[degrees]
        # Each beam of the same LEO will have different ones
        Azi_phi_deg_Sat_0=tf.expand_dims(Azi_phi_deg_Sat[:,0,:],axis=1)+0.0
        Azi_phi_deg_Sat_1 = tf.expand_dims(Azi_phi_deg_Sat[:, 1, :], axis=1)+30.0
        Azi_phi_deg_Sat_2 = tf.expand_dims(Azi_phi_deg_Sat[:, 2, :], axis=1)+90.0
        Azi_phi_deg_Sat_3 = tf.expand_dims(Azi_phi_deg_Sat[:, 3, :], axis=1)+150.0
        Azi_phi_deg_Sat_4 = tf.expand_dims(Azi_phi_deg_Sat[:, 4, :], axis=1)+210.0
        Azi_phi_deg_Sat_5 = tf.expand_dims(Azi_phi_deg_Sat[:, 5, :], axis=1)+270.0
        Azi_phi_deg_Sat_6=tf.expand_dims(Azi_phi_deg_Sat[:,6,:],axis=1)+330.0
        Azi_phi_deg_Sat_local = tf.concat([Azi_phi_deg_Sat_0, Azi_phi_deg_Sat_1, Azi_phi_deg_Sat_2, Azi_phi_deg_Sat_3, Azi_phi_deg_Sat_4, Azi_phi_deg_Sat_5,Azi_phi_deg_Sat_6],axis=1)

        Elv_thetha_deg_Sat_0 = tf.expand_dims(Elv_thetha_deg_Sat[:, 0, :], axis=1) + 0.0
        Elv_thetha_deg_Satt_1to6 = tf.expand_dims(Elv_thetha_deg_Sat[:, 1, :], axis=1) - 4.41 #This is the seperation between beams in the elveation
        Elv_thetha_deg_Satt_1to6 = tf.tile(Elv_thetha_deg_Satt_1to6, [1, 6, 1])
        Elv_thetha_deg_Sat_local = tf.concat([Elv_thetha_deg_Sat_0, Elv_thetha_deg_Satt_1to6], axis=1)

        #We take the LEO as a reference. We define two 3d vectors to find the angle between them which is thetha the boresight angle seen from the beam
        #The first vector goes from the beam to the ground, we use formulas from TR38.821 page 36
        U0=tf.ones([NTN_D2d.shape[0],1, 1])*(tf.math.sin(180.0*math.pi/180.0) * tf.math.cos(0.0*math.pi/180.0))
        U1 = tf.ones([NTN_D2d.shape[0], 1, 1]) * (tf.math.sin(175.59 * math.pi / 180.0) * tf.math.cos(30.0 * math.pi / 180.0))
        U2 = tf.ones([NTN_D2d.shape[0], 1, 1]) * (tf.math.sin(175.59 * math.pi / 180.0) * tf.math.cos(90.0 * math.pi / 180.0))
        U3 = tf.ones([NTN_D2d.shape[0], 1, 1]) * (tf.math.sin(175.59 * math.pi / 180.0) * tf.math.cos(150.0 * math.pi / 180.0))
        U4 = tf.ones([NTN_D2d.shape[0], 1, 1]) * (tf.math.sin(175.59 * math.pi / 180.0) * tf.math.cos(210.0 * math.pi / 180.0))
        U5 = tf.ones([NTN_D2d.shape[0], 1, 1]) * (tf.math.sin(175.59 * math.pi / 180.0) * tf.math.cos(270.0 * math.pi / 180.0))
        U6 = tf.ones([NTN_D2d.shape[0], 1, 1]) * (tf.math.sin(175.59 * math.pi / 180.0) * tf.math.cos(330.0 * math.pi / 180.0))
        U = tf.concat([U0, U1, U2, U3, U4,U5,U6], axis=1)

        V0=tf.ones([NTN_D2d.shape[0],1, 1])*(tf.math.sin(180.0*math.pi/180.0) * tf.math.sin(0.0*math.pi/180.0))
        V1 = tf.ones([NTN_D2d.shape[0], 1, 1]) * (tf.math.sin(175.59 * math.pi / 180.0) * tf.math.sin(30.0 * math.pi / 180.0))
        V2 = tf.ones([NTN_D2d.shape[0], 1, 1]) * (tf.math.sin(175.59 * math.pi / 180.0) * tf.math.sin(90.0 * math.pi / 180.0))
        V3 = tf.ones([NTN_D2d.shape[0], 1, 1]) * (tf.math.sin(175.59 * math.pi / 180.0) * tf.math.sin(150.0 * math.pi / 180.0))
        V4 = tf.ones([NTN_D2d.shape[0], 1, 1]) * (tf.math.sin(175.59 * math.pi / 180.0) * tf.math.sin(210.0 * math.pi / 180.0))
        V5 = tf.ones([NTN_D2d.shape[0], 1, 1]) * (tf.math.sin(175.59 * math.pi / 180.0) * tf.math.sin(270.0 * math.pi / 180.0))
        V6 = tf.ones([NTN_D2d.shape[0], 1, 1]) * (tf.math.sin(175.59 * math.pi / 180.0) * tf.math.sin(330.0 * math.pi / 180.0))
        V = tf.concat([V0, V1, V2, V3, V4,V5,V6], axis=1)
        x_leo=U*self.Zleo
        y_leo=V*self.Zleo
        z_leo=tf.ones([NTN_D2d.shape[0], 7, 1]) *self.Zleo
        Xleo=tf.concat([x_leo, y_leo, z_leo], axis=2)

        # The second vector is the position of the UE seen by the LEO
        Xuser_LEOprospective_x=tf.expand_dims(Xuser[:,:,0],axis=2)-self.LEO_x_cord
        Xuser_LEOprospective_y = tf.expand_dims(Xuser[:,:,1],axis=2)-self.LEO_y_cord
        Xuser_LEOprospective_z = self.Zleo - tf.expand_dims(Xuser[:,:,2],axis=2)
        Xuser_LEOprospective = tf.concat([Xuser_LEOprospective_x, Xuser_LEOprospective_y, Xuser_LEOprospective_z], axis=2)

        # Xleo=tf.expand_dims(Xleo, axis=2)
        # Xuser_LEOprospective = tf.expand_dims(Xuser_LEOprospective, axis=1)
        Xleo_Norm=tf.expand_dims(tf.norm(Xleo,axis=2),axis=2)
        Yleo_Norm = tf.expand_dims(tf.norm(Xuser_LEOprospective, axis=2), axis=1)
        Norm_tensor=Xleo_Norm*Yleo_Norm
        dotProduct_tensor=tf.keras.backend.batch_dot(Xleo,Xuser_LEOprospective,axes=2)
        cos_input=dotProduct_tensor/Norm_tensor
        cos_input_1=tf.cast(cos_input >= 1.0,"float32") #Gives you 1 even if it is like 1.000001 because otherwise boresight will be nan not zero
        cos_input_2 = tf.cast(cos_input < 1.0, "float32")*cos_input
        cos_input=cos_input_1+cos_input_2
        NTN_Boresight=(tf.math.acos(cos_input)*180/math.pi)+0.0001
        
        return NTN_D,NTN_Elevation, NTN_Boresight
        
        
    def NTN_AntennaGain(self, NTN_Boresight):
        
        NTN_Boresight=tf.abs(NTN_Boresight)
        con_1= tf.cast(NTN_Boresight==0.0,"float32")
        con_2= tf.cast(NTN_Boresight>0,"float32")  * tf.cast(NTN_Boresight<=90,"float32")
    
        AG1=1.0*con_1
            
        j=j1((self.ka*tf.math.sin(np.deg2rad(NTN_Boresight))))
        AG2=4*tf.math.pow(j/(self.ka*tf.math.sin(np.deg2rad(NTN_Boresight))),2)*con_2
        
        NTN_AntennaGain=AG1+AG2
        NTN_AntennaGain_dB=10.0*(tf.math.log(NTN_AntennaGain)/tf.math.log(10.0))+self.antenna_gain_max_db_sat

        return NTN_AntennaGain_dB

    
    def NTN_Prob_LOS_3GPP(self, NTN_Elevation):
        batch_num = NTN_Elevation.shape[0]

        Elevation = NTN_Elevation
        con_10= tf.cast(Elevation>=0,"float32")  * tf.cast(Elevation<=14,"float32")
        con_20= tf.cast(Elevation>=15,"float32") * tf.cast(Elevation<=24,"float32")
        con_30= tf.cast(Elevation>=25,"float32") * tf.cast(Elevation<=34,"float32")
        con_40= tf.cast(Elevation>=35,"float32") * tf.cast(Elevation<=44,"float32")
        con_50= tf.cast(Elevation>=45,"float32") * tf.cast(Elevation<=54,"float32")
        con_60= tf.cast(Elevation>=55,"float32") * tf.cast(Elevation<=64,"float32")
        con_70= tf.cast(Elevation>=65,"float32") * tf.cast(Elevation<=74,"float32") 
        con_80= tf.cast(Elevation>=75,"float32") * tf.cast(Elevation<85,"float32")
        con_90= tf.cast(Elevation>=85,"float32") * tf.cast(Elevation<=90,"float32")
        
        P_LOS10=0.282*con_10
        P_LOS20=0.331*con_20
        P_LOS30=0.398*con_30  
        P_LOS40=0.468*con_40
        P_LOS50=0.537*con_50
        P_LOS60=0.612*con_60
        P_LOS70=0.738*con_70
        P_LOS80=0.82*con_80
        P_LOS90=0.981*con_90
            
        P_LOS=P_LOS10+P_LOS20+P_LOS30+P_LOS40+P_LOS50+P_LOS60+P_LOS70+P_LOS80+P_LOS90
            
        P_random = tf.random.uniform([batch_num,1,NTN_Elevation.shape[2]],0,1)
            
        NTN_LOS = tf.cast(P_LOS>P_random,"float32") 
        # Debug
        # LOS = tf.ones(LOS.shape,'float32')
        #P_LOS,P_random,
        # NTN_LOS = tf.expand_dims(NTN_LOS,axis=2)
        if self.calibration_sat:
            NTN_LOS = tf.ones(NTN_LOS.shape,'float32')

        if self.UAVs:
            NTN_LOS = tf.ones(NTN_LOS.shape, 'float32')
        return NTN_LOS
    
    #Total path loss (PL) 
    def NTN_PL_3GPP_UMa(self,NTN_D,NTN_Elevation,NTN_LOS,PLg,PLs,PLe):
        
        Elevation=NTN_Elevation
        #PL=PLb+PLg+PLs+PLe
        
        #LOS conditions
        
        con_LOS= tf.cast(NTN_LOS==1.0,"float32")
        con_NLOS= tf.cast(NTN_LOS==0.0,"float32")
        
        #free space propgation loss
        FSPL=32.45+(20.0*(tf.math.log(self.fc_sat)/tf.math.log(10.0)))+(20.0*(tf.math.log(NTN_D)/tf.math.log(10.0)))
        
        #clutter loss (Note: This clutter for S band only, for Ka look back at 3GPP NTN document P.50)
        CL_10= tf.cast(Elevation>=0,"float32")  * tf.cast(Elevation<=14,"float32")*34.3
        CL_20= tf.cast(Elevation>=15,"float32") * tf.cast(Elevation<=24,"float32")*30.9
        CL_30= tf.cast(Elevation>=25,"float32") * tf.cast(Elevation<=34,"float32")*29.0
        CL_40= tf.cast(Elevation>=35,"float32") * tf.cast(Elevation<=44,"float32")*27.7
        CL_50= tf.cast(Elevation>=45,"float32") * tf.cast(Elevation<=54,"float32")*26.8
        CL_60= tf.cast(Elevation>=55,"float32") * tf.cast(Elevation<=64,"float32")*26.2
        CL_70= tf.cast(Elevation>=65,"float32") * tf.cast(Elevation<=74,"float32")*25.8
        CL_80= tf.cast(Elevation>=75,"float32") * tf.cast(Elevation<=84,"float32")*25.5
        CL_90= tf.cast(Elevation>=85,"float32") * tf.cast(Elevation<=90,"float32")*25.5
        
        CL=(CL_10+CL_20+CL_30+CL_40+CL_50+CL_60+CL_70+CL_80+CL_90)
        # CL=0.0
        
        #Shadowing
        NTN_shadowing_LOS =tf.random.normal(NTN_LOS.shape, 0, self.shadowing_sigma_LOS, tf.float32)
        NTN_shadowing_NLOS =tf.random.normal(NTN_LOS.shape, 0, self.shadowing_sigma_NLOS, tf.float32)
        
        #basic path loss (PLb)
        plb_LOS=(FSPL+NTN_shadowing_LOS)*con_LOS
        # plb_LOS = FSPL  * con_LOS
        plb_NLOS=(FSPL+NTN_shadowing_NLOS+CL)*con_NLOS
        
        PLb=plb_LOS+plb_NLOS

        NTN_PL=PLb+PLg+PLs+PLe


        NTN_G_dB=-NTN_PL
        NTN_G_linear=tf.pow(10.0,NTN_G_dB/10)
        
        
        return NTN_G_dB, NTN_G_linear, NTN_PL, NTN_shadowing_LOS, NTN_shadowing_NLOS

    def NTN_PL_3GPP_UMa_in_beforeAssigment(self, NTN_D, NTN_Elevation, NTN_LOS, PLg, PLs, PLe,D_2d_in,D_2d_building):
        Elevation = NTN_Elevation
        # PL=PLb+PLg+PLs+PLe

        # LOS conditions

        con_LOS = tf.cast(NTN_LOS == 1.0, "float32")
        con_NLOS = tf.cast(NTN_LOS == 0.0, "float32")

        # free space propgation loss
        FSPL = 32.45 + (20.0 * (tf.math.log(self.fc_sat) / tf.math.log(10.0))) + (
                    20.0 * (tf.math.log(NTN_D) / tf.math.log(10.0)))

        # clutter loss (Note: This clutter for S band only, for Ka look back at 3GPP NTN document P.50)
        CL_10 = tf.cast(Elevation >= 0, "float32") * tf.cast(Elevation <= 14, "float32") * 34.3
        CL_20 = tf.cast(Elevation >= 15, "float32") * tf.cast(Elevation <= 24, "float32") * 30.9
        CL_30 = tf.cast(Elevation >= 25, "float32") * tf.cast(Elevation <= 34, "float32") * 29.0
        CL_40 = tf.cast(Elevation >= 35, "float32") * tf.cast(Elevation <= 44, "float32") * 27.7
        CL_50 = tf.cast(Elevation >= 45, "float32") * tf.cast(Elevation <= 54, "float32") * 26.8
        CL_60 = tf.cast(Elevation >= 55, "float32") * tf.cast(Elevation <= 64, "float32") * 26.2
        CL_70 = tf.cast(Elevation >= 65, "float32") * tf.cast(Elevation <= 74, "float32") * 25.8
        CL_80 = tf.cast(Elevation >= 75, "float32") * tf.cast(Elevation <= 84, "float32") * 25.5
        CL_90 = tf.cast(Elevation >= 85, "float32") * tf.cast(Elevation <= 90, "float32") * 25.5

        CL = (CL_10 + CL_20 + CL_30 + CL_40 + CL_50 + CL_60 + CL_70 + CL_80 + CL_90)
        # CL=0.0

        # Shadowing
        NTN_shadowing_LOS = tf.random.normal(NTN_LOS.shape, 0, self.shadowing_sigma_LOS, tf.float32)
        NTN_shadowing_NLOS = tf.random.normal(NTN_LOS.shape, 0, self.shadowing_sigma_NLOS, tf.float32)

        # basic path loss (PLb)
        plb_LOS = (FSPL + NTN_shadowing_LOS) * con_LOS
        # plb_LOS = FSPL  * con_LOS
        plb_NLOS = (FSPL + NTN_shadowing_NLOS + CL) * con_NLOS

        PLb = plb_LOS + plb_NLOS

        NTN_PL = PLb + PLg + PLs + PLe


        PL_tw = 20.0
        # D_2d_in_sat=self.Deployment.D_2d_in_sat[:,NTN_D.shape[1],:]
        # D_building_sat=self.Deployment.D_building_sat[:,NTN_D.shape[1],:]
        PL_inside = 0.5 * (D_2d_in-D_2d_building) + PL_tw
        NTN_PL = NTN_PL + PL_inside

        NTN_G_dB = -NTN_PL
        NTN_G_linear = tf.pow(10.0, NTN_G_dB / 10)

        return NTN_G_dB, NTN_G_linear, NTN_PL, NTN_shadowing_LOS, NTN_shadowing_NLOS

    def NTN_PL_3GPP_UMa_in_afterAssigmentSatUEs(self, NTN_D, NTN_Elevation, NTN_LOS, PLg, PLs, PLe,D_2d_in,D_2d_building):

        Elevation = NTN_Elevation
        # PL=PLb+PLg+PLs+PLe

        # LOS conditions

        con_LOS = tf.cast(NTN_LOS == 1.0, "float32")
        con_NLOS = tf.cast(NTN_LOS == 0.0, "float32")

        # free space propgation loss
        FSPL = 32.45 + (20.0 * (tf.math.log(self.fc_sat) / tf.math.log(10.0))) + (
                    20.0 * (tf.math.log(NTN_D) / tf.math.log(10.0)))

        # clutter loss (Note: This clutter for S band only, for Ka look back at 3GPP NTN document P.50)
        CL_10 = tf.cast(Elevation >= 0, "float32") * tf.cast(Elevation <= 14, "float32") * 34.3
        CL_20 = tf.cast(Elevation >= 15, "float32") * tf.cast(Elevation <= 24, "float32") * 30.9
        CL_30 = tf.cast(Elevation >= 25, "float32") * tf.cast(Elevation <= 34, "float32") * 29.0
        CL_40 = tf.cast(Elevation >= 35, "float32") * tf.cast(Elevation <= 44, "float32") * 27.7
        CL_50 = tf.cast(Elevation >= 45, "float32") * tf.cast(Elevation <= 54, "float32") * 26.8
        CL_60 = tf.cast(Elevation >= 55, "float32") * tf.cast(Elevation <= 64, "float32") * 26.2
        CL_70 = tf.cast(Elevation >= 65, "float32") * tf.cast(Elevation <= 74, "float32") * 25.8
        CL_80 = tf.cast(Elevation >= 75, "float32") * tf.cast(Elevation <= 84, "float32") * 25.5
        CL_90 = tf.cast(Elevation >= 85, "float32") * tf.cast(Elevation <= 90, "float32") * 25.5

        CL = (CL_10 + CL_20 + CL_30 + CL_40 + CL_50 + CL_60 + CL_70 + CL_80 + CL_90)
        # CL=0.0

        # Shadowing
        NTN_shadowing_LOS = tf.random.normal(NTN_LOS.shape, 0, self.shadowing_sigma_LOS, tf.float32)
        NTN_shadowing_NLOS = tf.random.normal(NTN_LOS.shape, 0, self.shadowing_sigma_NLOS, tf.float32)

        # basic path loss (PLb)
        plb_LOS = (FSPL + NTN_shadowing_LOS) * con_LOS
        # plb_LOS = FSPL  * con_LOS
        plb_NLOS = (FSPL + NTN_shadowing_NLOS + CL) * con_NLOS

        PLb = plb_LOS + plb_NLOS

        NTN_PL = PLb + PLg + PLs + PLe


        PL_tw = 20.0
        # D_2d_in_sat=self.Deployment.D_2d_in_sat[:,NTN_D.shape[1],:]
        # D_building_sat=self.Deployment.D_building_sat[:,NTN_D.shape[1],:]
        PL_inside = 0.5 * (D_2d_in-D_2d_building) + PL_tw
        NTN_PL = NTN_PL + PL_inside

        NTN_G_dB = -NTN_PL
        NTN_G_linear = tf.pow(10.0, NTN_G_dB / 10)

        return NTN_G_dB, NTN_G_linear, NTN_PL, NTN_shadowing_LOS, NTN_shadowing_NLOS
    #Atmospheric absorption (PLg)
    def NTN_PLg (self):
        
        PLg=0
        return PLg
        
    #Scintillation (PLs)
    def NTN_PLs (self):
        
        PLs=0
        return PLs
        
        
    #Building entry loss (PLe)
    def NTN_PLe (self):
        
        PLe=0.0
        return PLe
    
    
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    