# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 14:33:45 2021

@author: Benzo

Note1: All of our terrestrial network ground UEs or UAVs UEs are assumed to be deployed for Urban Macro Scenarios (UMa)

"""
import math
import tensorflow as tf
from config import  Config
class Large_Scale_Gain(Config):

    #Define the general paramters that will be used in the upcoming functions.
    def __init__(self):
        Config.__init__(self)
        self.exponent=3.8               
        self.shadowing_sigma_LOS1= 4.0
        self.shadowing_sigma_LOS2= 4.64*math.exp(-0.0066*self.Zuser)
        self.shadowing_sigma_NLOS=6.0   
        self.hE=1.0                     #effective enviroment height taken as 1m. 3GPP document 38.901, p.29.
        self.SLA_cut= 30                #in dB
        self.Amax_cut=30                #in dB
        self.phi_3dB=65                 #3dB HPBW of antenna patter in degree for Azimuth
        self.thetha_3dB=10              #3dB HPBW of antenna patter in degree for Elevation
        self.Antenna_elements = 1
        self.Lambda = self.fc*1e9/self.c
        self.Antenna_spacing = self.Lambda/2
        
    def call(self,D,D_2d,Azi_phi_deg,Elv_thetha_deg):
        p_LOS = self.Prob_LOS_3GPP_UAVs(D_2d)
        G_dB,G_linear,pl,shadowing_LOS,shadowing_NLOS =self.PL_3GPP_UMa_UAVs(D, D_2d,p_LOS)   
        # G_Antenna = self.Antenna_gain_3GPP(Azi_phi_deg,Elv_thetha_deg)
        # LSG = G_dB + G_Antenna
        G_sector = self.sectoring(Azi_phi_deg,Elv_thetha_deg) # this line includes both antenna gain and sectoring effect
        LSG= tf.tile(G_dB, [1, 3, 1])+G_sector
        LSL=-LSG
        p_LOS = tf.tile(p_LOS, [1, 3, 1])
        
        #
        pl= tf.tile(pl, [1, 3, 1])
        shadowing_LOS= tf.tile(shadowing_LOS, [1, 3, 1])
        shadowing_NLOS= tf.tile(shadowing_NLOS, [1, 3, 1])
        #
        
        return LSL,LSG,G_sector,p_LOS,pl,shadowing_LOS,shadowing_NLOS
    
        # Incorporate sectoring using different aneta gain
    def sectoring(self,Azi_phi_deg, Elv_thetha_deg):
        G_secor0 = self.Antenna_gain_3GPP(Azi_phi_deg, Elv_thetha_deg)
        
        Azi_phi_deg1= (tf.cast((Azi_phi_deg+120)<180,"float32")*tf.cast((Azi_phi_deg+120)>-180,"float32")*(Azi_phi_deg+120) +
                     tf.cast((Azi_phi_deg + 120) < -180, "float32") * (Azi_phi_deg + 120 + 360) +
                     tf.cast((Azi_phi_deg + 120) > 180, "float32") * (Azi_phi_deg + 120 - 360))
        G_secor1 = self.Antenna_gain_3GPP(Azi_phi_deg1, Elv_thetha_deg)
        
        Azi_phi_deg2 = (tf.cast((Azi_phi_deg+240)<180,"float32")*tf.cast((Azi_phi_deg+240)>-180,"float32")*(Azi_phi_deg+240) +
                     tf.cast((Azi_phi_deg + 240) < -180, "float32") * (Azi_phi_deg + 240 + 360) +
                     tf.cast((Azi_phi_deg + 240) > 180, "float32") * (Azi_phi_deg + 240 - 360))
        G_secor2 = self.Antenna_gain_3GPP(Azi_phi_deg2,Elv_thetha_deg)
        
        G_sector = tf.concat([G_secor0,G_secor1,G_secor2],axis=1)
        # G_sector = tf.zeros(G_sector.shape,'float32')
        return G_sector
    
    #Calculating antenna gain based on its patter and formulas provided in 3GPP Document 38.901 P.23
    def Antenna_gain_3GPP(self,Azi_phi_deg,Elv_thetha_deg):
    
        Azi_cut= 12*(tf.pow(Azi_phi_deg/self.phi_3dB,2))
        Elv_cut=12*(tf.pow((Elv_thetha_deg-90)/self.thetha_3dB,2))
        
        con_1a= tf.cast(Azi_cut<self.Amax_cut,"float32")
        G_azi1=-Azi_cut*con_1a
        con_1b= tf.cast(Azi_cut>=self.Amax_cut,"float32")
        G_azi2=-self.Amax_cut*con_1b
        
        G_azi=G_azi1+G_azi2
        
        con_2a= tf.cast(Elv_cut<self.SLA_cut,"float32")
        G_elv1=-Elv_cut*con_2a
        con_2b= tf.cast(Elv_cut>=self.SLA_cut,"float32")
        G_elv2=-self.SLA_cut*con_2b
        
        G_elv=G_elv1+G_elv2
        
        #Total loss
        G_total=-(G_azi+G_elv)
        
        # G_total=-(G_elv)
        
        con_3a= tf.cast(G_total<self.Amax_cut,"float32")
        Atten_total1= -G_total*con_3a
        con_3b= tf.cast(G_total>=self.Amax_cut,"float32")
        Atten_total2= -self.Amax_cut*con_3b
        Atten_total=Atten_total1+Atten_total2
        G_Antenna=8+Atten_total
        # G_Antenna=Atten_total-Atten_total
        #-------------------------- Array gain
        # ind = tf.constant([[[[i for i in range(self.Antenna_elements)]]]])
        # phase = tf.expand_dims(Elv_thetha_deg,axis=3)*tf.cast(ind,'float32')
        # phase = tf.complex(0.0,-2*math.pi*self.Antenna_spacing/self.Lambda*tf.math.cos(phase))
        # w = tf.complex(tf.ones(phase.shape,'float32'),0.0)  # replace the w with beamforming vector
        # array_gain = tf.math.abs(tf.reduce_sum(tf.math.exp(phase)*w,axis=3))
        # #---------------------------------------
        # G_Antenna = G_Antenna+10.0*tf.math.log(array_gain)/tf.math.log(10.0)
        #debug
        # G_Antenna = tf.ones(G_Antenna.shape,'float32')
        # G_Antenna = tf.random.normal(G_Antenna.shape, -22, 10, tf.float32)
        return G_Antenna  

    #Calculating probability of LOS based on 3GPP Document 38.901 P.30, UMa scenario    
    def Prob_LOS_3GPP_UAVs(self,D_2d):
        
        batch_num = D_2d.shape[0]
        
        #UAVs Height Conditions
        con_a= tf.cast(self.Zuser<=22.5,"float32") #If UAV is less than 22.5m
        con_b= tf.cast(self.Zuser>22.5,"float32")*tf.cast(self.Zuser<=100.0,"float32") #If UAV is between 22.5-100m
        con_c= tf.cast(self.Zuser>100.0,"float32") #If UAV is above 100m
        
        # For UAVs less than 22.5m
        con_0a= tf.cast(D_2d<=18,"float32")
        P_LOS1=1*con_0a
        
        con_0b= tf.cast(D_2d>18,"float32")
        P_LOS2=((18/D_2d)+(tf.math.exp(-(D_2d/63))*(1-(18/D_2d))))*con_0b
        
        P_LOS=P_LOS1+P_LOS2   
        
        P_random = tf.random.uniform([batch_num,self.Nap,self.Nuser_drop],0,1)  
        
        LOS_a = tf.cast(P_LOS>P_random,"float32")*con_a
        
        # For UAVs between 22.5-100m
        d1_a=460.0*(tf.math.log(self.Zuser)/tf.math.log(10.0))-700.0
        con_d1a= tf.cast(d1_a>18.0,"float32")
        d1a=d1_a*con_d1a
        con_d1b= tf.cast(d1_a<=18.0,"float32")
        d1b=18.0*con_d1b
        
        d1=d1a+d1b
        p1=4300.0*(tf.math.log(self.Zuser)/tf.math.log(10.0))-3800.0

        con_1a= tf.cast(D_2d<=d1,"float32")
        P_LOS1b=1*con_1a         
        
        con_1b= tf.cast(D_2d>d1,"float32")
        P_LOS2b=((d1/D_2d)+(tf.math.exp(-(D_2d/p1))*(1-(d1/D_2d))))*con_1b
        
        P_LOSb=P_LOS1b+P_LOS2b
        LOS_b = tf.cast(P_LOSb>P_random,"float32")*con_b
        
        # For UAVs above 100m
        LOS_c = 1.0*con_c
        
        #Final LOS Matrix
        LOS=LOS_a+LOS_b+LOS_c
        
        # Debug
        # LOS = tf.ones(LOS.shape,'float32')
        return LOS
    
    #This function calculates the Path Loss based on 3GPP Document 38.901 Urban Macro scenaario is consedired (p.27), LOS condition. Pathloss [dB], fc is in GHz and d is in meters
    def PL_3GPP_UMa_UAVs(self,D,D_2d,LOS):
        
        Zap_e=self.Zap-self.hE                           
        Zuser_e=self.Zuser-self.hE                      
        d_BP=4*Zap_e*Zuser_e*(self.fc_Hz/self.c)      # Breaking Point. Note: In calculating the Break Point, we ignored the note about effective enviroment height in P.29 ((Note 1) of the 38.901 document and it is taken as 1m.
        
        #UAVs Height Conditions
        con_a= tf.cast(self.Zuser<=22.5,"float32") #If UAV is less than 22.5m
        con_b= tf.cast(self.Zuser>22.5,"float32")*tf.cast(self.Zuser<=100.0,"float32") #If UAV is between 22.5-100m
        con_c= tf.cast(self.Zuser>22.5,"float32") 
        
        #If LOS condition, perform the following PL calculations
        con_LOS= tf.cast(LOS==1.0,"float32")  
        
        con_1= tf.cast(D_2d>=0.0,"float32")* tf.cast(D_2d<=d_BP,"float32")           #This is the first condition in the PL formula if the 2D is less than the breakpoint. Note: 0.0 should be changed to 10.0 when we add the condition of min. distance of UE to BS of 35m.
        con_2= tf.cast(D_2d>=d_BP,"float32")* tf.cast(D_2d<=50000,"float32")         #This is the second condition in the PL formula if the 2D is bigger than the breakpoint. This gives a 0/1 matrix. 1 if true and 0 if false.
        pl_1=28.0+(22.0*(tf.math.log(D)/tf.math.log(10.0)))+(20.0*(tf.math.log(self.fc)/tf.math.log(10.0)))          #PL part 1 formula for 2D distance less than the breakpoint.
        pl_1=pl_1*con_1             #Multiply the PL part 1 to the condition 1 so we get the tensor size of desired PL values.
        
        pl_2=28.0+(40.0*(tf.math.log(D)/tf.math.log(10.0)))+(20.0*(tf.math.log(self.fc)/tf.math.log(10.0)))-(9.0*(tf.math.log(tf.pow(d_BP,2.0)+tf.pow(self.Zap-self.Zuser,2.0))/tf.math.log(10.0))) #PL part 2 formula for 2D distance higher than the breakpoint.
        pl_2=pl_2*con_2             #Multiply the PL part 2 to the condition 2 so we get the tensor size of desired PL values.
        
        pla=pl_1+pl_2
        shadowing_LOS1 =tf.random.normal(pla.shape, 0, self.shadowing_sigma_LOS1, tf.float32)
        # self.shadowing_LOS = tf.tile(shadowing_LOS,[1,3,1])
        pla1=pla+shadowing_LOS1
        pla1=pla1*con_LOS*con_a
        
        shadowing_LOS2 =tf.random.normal(pla.shape, 0, self.shadowing_sigma_LOS2, tf.float32)
        pla1b=pl_1+shadowing_LOS2
        pla1b=pla1b*con_LOS*con_c
        
        #Complete shadowing LOS for debugging
        shadowing_LOS=(shadowing_LOS1*con_a)+(shadowing_LOS2*con_c)
        
        
        #Complete PL for LOS condition
        PL_LOS=(pla1+pla1b)
        
        
        #If NLOS condition, perform the following PL calculations
        con_NLOS= tf.cast(LOS==0.0,"float32") 
        
        plba=13.54+(39.08*(tf.math.log(D)/tf.math.log(10.0)))+(20.0*(tf.math.log(self.fc)/tf.math.log(10.0)))-(0.6*(self.Zuser-1.5))
        
        con_3= tf.cast(plba>pla,"float32")
        plb1=plba*con_3
        
        con_4= tf.cast(plba<=pla,"float32")
        plb2=pla*con_4
        
        plb=plb1+plb2
        shadowing_NLOS =tf.random.normal(plb.shape, 0, self.shadowing_sigma_NLOS, tf.float32)
        # self.shadowing_NLOS = tf.tile(shadowing_NLOS,[1,3,1])
        plb1=plb*con_NLOS*con_a
        plb1b=-17.5+(46-7*(tf.math.log(self.Zuser)/tf.math.log(10.0)))*(tf.math.log(D)/tf.math.log(10.0))+(20.0*(tf.math.log(40*math.pi*self.fc/3)/tf.math.log(10.0)))*con_NLOS*con_b

        #Complete PL for NLOS condition
        PL_NLOS=(plb1+plb1b)+shadowing_NLOS
        
        #Complete PL for NLOS condition
        pl=PL_LOS+PL_NLOS
        # self.pl = tf.tile(pl,[1,3,1])
        G_dB=-pl                         
        G_linear=tf.pow(10.0,G_dB/10) 
        
        return G_dB,G_linear,pl,shadowing_LOS,shadowing_NLOS
