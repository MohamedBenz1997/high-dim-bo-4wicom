"""
This is the TN Large Scale Gain Class for UAVs of the simulator:
    In this class, the LSGs between the TN BSs and UAVs are calculated.
    Path loss, shadowing and Antenna Gain are computed here.
    Note: All of our terrestrial network ground UEs or UAVs UEs are assumed to be deployed for Urban Macro Scenarios (UMa)

@authors: Mohamed Benzaghta and Rasoul Nikbakht
"""

import math
import tensorflow as tf
from config import Config
import numpy as np
from scipy.integrate import quad

class Large_Scale_Gain_drone(Config):

    def __init__(self):
        Config.__init__(self)
        self.exponent = 3.8
        self.shadowing_sigma_LOS1 = 4.0
        self.shadowing_sigma_NLOS = 6.0
        self.hE = 1.0  # effective enviroment height taken as 1m. 3GPP document 38.901, p.29.
        self.SLA_cut = 30  # in dB
        self.Amax_cut = 30  # in dB
        self.phi_3dB = 70  # 3dB HPBW of antenna patter in degree for Azimuth
        # self.thetha_3dB = 10 # 3dB HPBW of antenna patter in degree for Elevation (vertical)
        self.Antenna_elements = 1
        self.Lambda = self.fc * 1e9 / self.c
        self.Antenna_spacing = self.Lambda * 0.5
        # self.Element_gain = 14.0

    def call(self, D_drone, D_2d_drone, Azi_phi_deg_drone, Elv_thetha_deg_drone, Zdrone, BS_tilt, BS_HPBW_v):
        #Getting the BS tilts and horizontal HPBW from the run file
        self.BS_tilt=BS_tilt
        thetha_3dB = BS_HPBW_v

        p_LOS = self.Prob_LOS_3GPP_UAVs(D_2d_drone, Zdrone)
        G_dB, G_linear, pl, shadowing_LOS, shadowing_NLOS = self.PL_3GPP_UMa_UAVs(D_drone, D_2d_drone, p_LOS, Zdrone)
        G_sector = self.sectoring(Azi_phi_deg_drone,Elv_thetha_deg_drone, thetha_3dB)  # this line includes both antenna gain and sectoring effect

        if self.sectoring_status:
            shadowing_LOS = tf.tile(shadowing_LOS, [1, 3, 1])
            shadowing_NLOS = tf.tile(shadowing_NLOS, [1, 3, 1])
            LSG = tf.tile(G_dB, [1, 3, 1]) + G_sector
            LSL = -LSG
            p_LOS = tf.tile(p_LOS, [1, 3, 1])
            pl = tf.tile(pl, [1, 3, 1])
        else:
            LSG = G_dB
            LSL = -LSG
            G_sector = G_sector[:, 0:self.Nap, :]

        return LSL, LSG, G_sector, p_LOS, pl, shadowing_LOS, shadowing_NLOS


    # Incorporate sectoring using different aneta gain
    def sectoring(self, Azi_phi_deg, Elv_thetha_deg, thetha_3dB):
        Azi_phi_deg = Azi_phi_deg + 30.0 #This is to make the sectors similar to 3GPP deployment, 30,150,270 deg
        thetha_3dB = thetha_3dB[:, :, 0:Azi_phi_deg.shape[2]] #This is to take just the GUEs number or UAV numbers when calculating the AG
        if Elv_thetha_deg.shape[2] == 0:
            Elv_thetha_deg = tf.tile(Elv_thetha_deg, [1, 3, 1]) + self.BS_tilt[:, :, 0:Elv_thetha_deg.shape[2]]
        else:
            Elv_thetha_deg=tf.tile(Elv_thetha_deg, [1, 3, 1])+self.BS_tilt[:,:,0:Elv_thetha_deg.shape[2]] #This is to consider the tilt for every sector seperately

        if self.N==1:
            G_secor0 = self.Antenna_gain_3GPP(Azi_phi_deg, Elv_thetha_deg[:, 0:7, :],thetha_3dB[:, 0:7, :])
        elif self.N==2:
            G_secor0 = self.Antenna_gain_3GPP(Azi_phi_deg, Elv_thetha_deg[:,0:19,:],thetha_3dB[:,0:19,:])

        # Azi_phi_deg1= (tf.cast((Azi_phi_deg+120)<180,"float32")*tf.cast((Azi_phi_deg+120)>-180,"float32")*(Azi_phi_deg+120) +
        #              tf.cast((Azi_phi_deg + 120) < -180, "float32") * (Azi_phi_deg + 120 + 360) +
        #              tf.cast((Azi_phi_deg + 120) > 180, "float32") * (Azi_phi_deg + 120 - 360))
        Azi_phi_deg1 = (tf.cast((Azi_phi_deg + 120) > 180, "float32") * (Azi_phi_deg + 120 - 360) +
                        tf.cast((Azi_phi_deg + 120) < 180.001, "float32") * (Azi_phi_deg + 120))

        if self.N==1:
            G_secor1 = self.Antenna_gain_3GPP(Azi_phi_deg1, Elv_thetha_deg[:, 7:14, :],thetha_3dB[:, 7:14, :])
        elif self.N==2:
            G_secor1 = self.Antenna_gain_3GPP(Azi_phi_deg1, Elv_thetha_deg[:,19:38,:],thetha_3dB[:,19:38,:])
        # Azi_phi_deg2 = (tf.cast((Azi_phi_deg+240)<180,"float32")*tf.cast((Azi_phi_deg+240)>-180,"float32")*(Azi_phi_deg+240) +
        #              tf.cast((Azi_phi_deg + 240) < -180, "float32") * (Azi_phi_deg + 240 + 360) +
        #              tf.cast((Azi_phi_deg + 240) > 180, "float32") * (Azi_phi_deg + 240 - 360))
        Azi_phi_deg2 = (tf.cast((Azi_phi_deg + 240) > 180, "float32") * (Azi_phi_deg + 240.0 - 360.0) +
                        tf.cast((Azi_phi_deg + 240.0) < 180.001, "float32") * (Azi_phi_deg + 240))

        if self.N==1:
            G_secor2 = self.Antenna_gain_3GPP(Azi_phi_deg2, Elv_thetha_deg[:, 14:, :],thetha_3dB[:, 14:, :])
        elif self.N==2:
            G_secor2 = self.Antenna_gain_3GPP(Azi_phi_deg2, Elv_thetha_deg[:,38:,:],thetha_3dB[:,38:,:])

        G_sector = tf.concat([G_secor0, G_secor1, G_secor2], axis=1)
        self.Azi_phi_sector = tf.concat([Azi_phi_deg, Azi_phi_deg1, Azi_phi_deg2], axis=1)
        # G_sector = tf.zeros(G_sector.shape,'float32')
        return G_sector

    # Calculating antenna gain based on its patter and formulas provided in 3GPP Document 38.901 P.23
    def Antenna_gain_3GPP(self, Azi_phi_deg, Elv_thetha_deg, thetha_3dB):
        # -------------------------- Array gain
        ind = tf.constant([[[[i for i in range(self.Antenna_elements)]]]])
        phase = tf.expand_dims(Elv_thetha_deg, axis=3)
        phase = tf.complex(0.0, 2 * math.pi * self.Antenna_spacing * tf.math.cos(
            phase * math.pi / 180.0) / self.Lambda * tf.cast(ind, 'float32'))

        w = tf.complex(tf.ones(phase.shape, 'float32'), 0.0)  # replace the w with beamforming vector
        array_gain = tf.math.abs(tf.reduce_sum(tf.math.exp(-phase) * w, axis=3))
        self.array_gain = 10.0 * tf.math.log(array_gain) / tf.math.log(10.0)
        self.array_gain = self.cut_by_30(self.array_gain)
        # -----------------------------------------------------------------
        Azi_cut = 12 * (tf.pow(Azi_phi_deg / self.phi_3dB, 2))
        Elv_cut = 12 * (tf.pow((Elv_thetha_deg-90) / thetha_3dB, 2))
        G_azi = self.cut_by_30(-Azi_cut)
        # self.G_azi = self.Element_gain + G_azi

        Element_gain_thetha_3dB = self.AG_based_thetha_3dB(thetha_3dB)
        self.G_azi = Element_gain_thetha_3dB + G_azi

        G_elv = self.cut_by_30(-Elv_cut)
        self.G_elv1 = G_elv
        # self.G_elv = self.Element_gain + G_elv
        self.G_elv = Element_gain_thetha_3dB + G_elv

        if self.sectoring_status:
            G_total = G_azi + G_elv
        else:
            G_total = G_elv
        # G_Antenna = self.Element_gain + self.cut_by_30(G_total) + self.array_gain
        G_Antenna = Element_gain_thetha_3dB + self.cut_by_30(G_total) + self.array_gain

        self.G_Antenna = G_Antenna
        return G_Antenna

    def cut_by_30(self, G_Antenna):
        con = tf.cast(G_Antenna > -30.0, "float32")
        G_Antenna = G_Antenna * con - 30 * (1 - con)
        return G_Antenna

    # Calculating probability of LOS based on 3GPP Document 38.901 P.30, UMa scenario
    def Prob_LOS_3GPP_UAVs(self, D_2d, Zdrone):
        Zdrone=tf.ones(D_2d.shape)*Zdrone
        # if self.one_tier == False:
        #     Zdrone = tf.tile(tf.transpose(Zdrone, [0, 2, 1]), [1, 19, 1])
        #
        # if self.one_tier == True:
        #     Zdrone = tf.tile(tf.transpose(Zdrone, [0, 2, 1]), [1, 7, 1])

        # Zdrone = tf.expand_dims(Zdrone, axis=1)
        # UAVs Height Conditions
        h_LOS_a = tf.cast(Zdrone <= 22.5, "float32")  # If UAV is less than 22.5m
        h_LOS_b = tf.cast(Zdrone > 22.5, "float32") * tf.cast(Zdrone <= 100.0, "float32")  # If UAV is between 22.5-100m
        h_LOS_c = tf.cast(Zdrone > 100.0, "float32")  # If UAV is above 100m
        # -----------------------------------------------------------------------
        # For UAVs less than 22.5m
        P_LOS_a1 = tf.cast(D_2d <= 18, "float32")
        C_h = tf.cast(Zdrone >= 13, "float32") * tf.math.pow(((tf.math.abs(Zdrone - 13.0)) / 10), 1.5) * tf.cast(
            Zdrone <= 23.0, "float32")
        P_LOS_a2 = (18 / D_2d + tf.math.exp(-(D_2d / 63)) * (1 - 18 / D_2d)) * (
                    1 + C_h * 5 / 4 * tf.math.pow(D_2d / 100, 3) * tf.math.exp(-D_2d / 150.0)) * tf.cast(D_2d >= 18,
                                                                                                         "float32")
        P_LOS_a = (P_LOS_a1 + P_LOS_a2) * h_LOS_a
        # ------------------------------------------------------------------------
        # For UAV user between 22.5 and 100 m
        d1 = tf.math.maximum(460.0 * (tf.math.log(Zdrone) / tf.math.log(10.0)) - 700.0, 18.0)
        p1 = 4300.0 * (tf.math.log(Zdrone) / tf.math.log(10.0)) - 3800.0
        P_LOS_b1 = tf.cast(D_2d <= d1, "float32")
        P_LOS_b2 = (d1 / D_2d + tf.math.exp(-D_2d / p1) * (1 - d1 / D_2d)) * tf.cast(D_2d > d1, "float32")
        P_LOS_b = (P_LOS_b1 + P_LOS_b2) * h_LOS_b
        # ----------------------------------------------------------------------------
        # For UAVs above 100m
        P_LOS_c = 1.0 * h_LOS_c
        # ---------------------------------------------------------------------------
        # Final LOS Matrix
        P_LOS = P_LOS_a + P_LOS_b + P_LOS_c
        P_random = tf.random.uniform(P_LOS.shape, 0, 1)
        LOS = tf.cast(P_random < P_LOS, "float32")
        # if self.drone_calib:
        #     LOS = tf.ones(LOS.shape, 'float32')
        # LOS = tf.ones(LOS.shape, 'float32')
        return LOS

    # This function calculates the Path Loss based on 3GPP Document 38.901 Urban Macro scenaario is consedired (p.27), LOS condition. Pathloss [dB], fc is in GHz and d is in meters
    def PL_3GPP_UMa_UAVs(self, D, D_2d, LOS, Zdrone):
        if self.fc < 6:
            pl1_coff = [28.0, 22.0]
            pl2_coff = [28.0, 9.0]
        else:
            pl1_coff = [32.4, 20.0]
            pl2_coff = [32.4, 10]

        Zap_e = self.Zap - self.hE
        Zuser_e = self.Zuser - self.hE
        d_BP = 4 * Zap_e * Zuser_e * (
                    self.fc_Hz / self.c)  # Breaking Point. Note: In calculating the Break Point, we ignored the note about effective enviroment height in P.29 ((Note 1) of the 38.901 document and it is taken as 1m.

        # UAVs Height Conditions
        # Zdrone = tf.ones(D_2d.shape) * Zdrone
        # if self.one_tier == False:
        #     Zdrone = tf.tile(tf.transpose(Zdrone, [0, 2, 1]), [1, 19, 1])
        #
        # if self.one_tier == True:
        #     Zdrone = tf.tile(tf.transpose(Zdrone, [0, 2, 1]), [1, 7, 1])
        # Zdrone = tf.expand_dims(Zdrone, axis=1)
        h_LOS_a = tf.cast(Zdrone <= 22.5, "float32")  # If UAV is less than 22.5m
        h_LOS_b = tf.cast(Zdrone > 22.5, "float32") * tf.cast(Zdrone <= 300.0, "float32")
        h_NLOS_a = tf.cast(Zdrone <= 22.5, "float32")  # If UAV is between 22.5-100m
        h_NLOS_b = tf.cast(Zdrone > 22.5, "float32") * tf.cast(Zdrone <= 100, "float32")
        # ------------------------------------------------------
        # If LOS condition, perform the following PL calculations
        con_LOS = tf.cast(LOS == 1.0, "float32")

        con_1 = tf.cast(D_2d >= 0.0, "float32") * tf.cast(D_2d <= d_BP,
                                                          "float32")  # This is the first condition in the PL formula if the 2D is less than the breakpoint. Note: 0.0 should be changed to 10.0 when we add the condition of min. distance of UE to BS of 35m.
        con_2 = tf.cast(D_2d >= d_BP, "float32") * tf.cast(D_2d <= 50000,
                                                           "float32")  # This is the second condition in the PL formula if the 2D is bigger than the breakpoint. This gives a 0/1 matrix. 1 if true and 0 if false.
        pl_1 = pl1_coff[0] + (pl1_coff[1] * (tf.math.log(D) / tf.math.log(10.0))) + (20.0 * (
                    tf.math.log(self.fc) / tf.math.log(
                10.0)))  # PL part 1 formula for 2D distance less than the breakpoint.
        pl_1 = pl_1 * con_1  # Multiply the PL part 1 to the condition 1 so we get the tensor size of desired PL values.

        pl_2 = pl2_coff[0] + (40.0 * (tf.math.log(D) / tf.math.log(10.0))) + (
                    20.0 * (tf.math.log(self.fc) / tf.math.log(10.0))) - \
               (pl2_coff[1] * (tf.math.log(tf.pow(d_BP, 2.0) + tf.pow(self.Zap - Zdrone, 2.0)) / tf.math.log(
                   10.0)))  # PL part 2 formula for 2D distance higher than the breakpoint.
        pl_2 = pl_2 * con_2  # Multiply the PL part 2 to the condition 2 so we get the tensor size of desired PL values.

        pla = pl_1 + pl_2

        plb = 28.0 + 22.0 * tf.math.log(D) / tf.math.log(10.0) + 20.0 * tf.math.log(self.fc) / tf.math.log(10.0)
        self.Pl_calib = pla * h_LOS_a + plb * h_LOS_b
        pl_los = pla * h_LOS_a + plb * h_LOS_b
        # --------------shadowing
        shadowing_LOS_a = tf.random.normal(pl_los.shape, 0, self.shadowing_sigma_LOS1, tf.float32)
        shadowing_LOS_a = shadowing_LOS_a * h_LOS_a
        # self.shadowing_LOS = tf.tile(shadowing_LOS,[1,3,1])
        shadowing_LOS_b = tf.random.normal(pl_los.shape, 0, 1, tf.float32) * 4.64 * tf.math.exp(-.0066 * Zdrone)
        shadowing_LOS_b = shadowing_LOS_b * h_LOS_b
        shadowing_LOS = shadowing_LOS_a + shadowing_LOS_b
        PL_LOS = pl_los + shadowing_LOS

        # ----------------------------------------

        # If NLOS condition, perform the following PL calculations
        # -- distance dependent loss
        con_NLOS = tf.cast(LOS == 0.0, "float32")

        pla_NLOS = 13.54 + (39.08 * (tf.math.log(D) / tf.math.log(10.0))) + (
                    20.0 * (tf.math.log(self.fc) / tf.math.log(10.0))) - (0.6 * (Zdrone - 1.5))

        pla_NLOS = tf.cast(pla_NLOS > pla, "float32") * pla_NLOS + tf.cast(pla_NLOS <= pla, "float32") * pla
        pla_NLOS = pla_NLOS * h_NLOS_a
        plb_NLOS = -17.5 + (46.0 - 7.0 * tf.math.log(Zdrone) / tf.math.log(10.0)) * tf.math.log(D) / tf.math.log(
            10.0) + 20.0 * tf.math.log(40 * math.pi * self.fc / 3) / tf.math.log(10.0)
        plb_NLOS = plb_NLOS * h_NLOS_b

        pl_NLOS = pla_NLOS + plb_NLOS

        # --------------------------
        # shadowing
        shadowing_NLOS = tf.random.normal(pla_NLOS.shape, 0, self.shadowing_sigma_NLOS, tf.float32)
        # ----------------------------------------------------------
        # Complete PL for NLOS condition
        PL_NLOS = pl_NLOS + shadowing_NLOS

        # Complete PL for NLOS condition
        pl = PL_LOS * con_LOS + PL_NLOS * con_NLOS
        # self.pl = tf.tile(pl,[1,3,1])
        G_dB = -pl
        G_linear = tf.pow(10.0, G_dB / 10)

        return G_dB, G_linear, pl, shadowing_LOS, shadowing_NLOS

    def AG_based_thetha_3dB(self,thetha_3dB):
        thetha_3dB_ref = thetha_3dB
        thetha_3dB = thetha_3dB[0,:,0].numpy()

        def f_65(x, thetha):
            return np.sin(x) * 10.0 ** (-1.2 * ((x - np.deg2rad(90.0)) / np.deg2rad(65.0)) ** 2)

        def f_10(x, thetha):
            return np.sin(x) * 10.0 ** (-1.2 * ((x - np.deg2rad(90.0)) / np.deg2rad(thetha)) ** 2)

        a1 = 0
        b1 = np.pi

        AG_values = []  # To store the AG values for each thetha_3dB

        for thetha in thetha_3dB:
            result_65, _ = quad(f_65, a1, b1, args=(thetha,))
            result_10, _ = quad(f_10, a1, b1, args=(thetha,))
            AG = 10 * np.log10((10 ** 0.8 * result_65) / result_10)
            AG_values.append(AG)

        Element_gain_thetha_3dB = tf.convert_to_tensor(AG_values, dtype=tf.float32)
        Element_gain_thetha_3dB = tf.expand_dims(tf.expand_dims(Element_gain_thetha_3dB,axis=0),axis=2)
        Element_gain_thetha_3dB = tf.tile(Element_gain_thetha_3dB, [thetha_3dB_ref.shape[0], 1, thetha_3dB_ref.shape[2]])

        return Element_gain_thetha_3dB
