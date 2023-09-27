"""
This is the Deployment Class of the simulator:
    In this class, the cells and UEs (GUEs/UAVs, Indoor/Outdoor) are deployed.
    Distances 2D/3D and Azimuth/Elevation angles are computed in here.

@authors: Mohamed Benzaghta and Rasoul Nikbakht
"""

import tensorflow as tf
from config import Config
import math
from pppclass import DeployPPP
from hexcalss import DeployHex


class Deployment(Config):
    def __init__(self):
        Config.__init__(self)
        if self.DeploymentType=="Hex":
            self.Deploy= DeployHex()
        elif self.DeploymentType=="PPP":
            self.Deploy = DeployPPP()


    def Call(self, alpha_factor): #,beta_open_loop=1
        self.LEO_x_cord = alpha_factor
        Xap, Xuser = self.Deploy.call()

        if self.sat_user:
            self.Xuser_sat = Xuser[:,self.Nuser_drop-self.Nuser_drop_sat:self.Nuser_drop,:]
            self.Xap_with_sat = tf.concat([ Xap,tf.tile(tf.constant([[[self.LEO_x_cord, self.LEO_y_cord, self.Zleo],[0.0, -43301.27018922193, self.Zleo],[0.0, 43301.27018922193, self.Zleo],[-37500.0, -21650.635094610963, self.Zleo],[-37500.0, 21650.635094610963, self.Zleo],[37500.0, -21650.635094610963, self.Zleo],[37500.0, 21650.635094610963, self.Zleo]]]), [Xap.shape[0], 1, 1]) ], axis=1)
            self.Nuser_drop = self.Nuser_drop-self.Nuser_drop_sat
            Xuser = Xuser[:,0:self.Nuser_drop,:]
        
        elif self.indoor==False:
            Xap=Xap
            Xuser=Xuser
            Zuser_indoor=0.0
            Xuser_in=0.0
            Zuser_in=0.0
            Xuser_out=0.0
            D_in=0.0
            D_2d_in=0.0
            D_2d_building=0.0
            BS_wrapped_Cord_in=0.0
            Azi_phi_deg_in =0.0
            Elv_thetha_deg_in=0.0
            Zuser_in=0.0
        
        if self.DeploymentType =="Hex":
            if self.sat_user:
                D_sat, D_2d_sat, BS_wrapped_Cord_sat, Xuser = self.Deploy.Dist(Xap, self.Xuser_sat) #Distance from Sat UEs to all BSs
                Azi_phi_deg_sat, Elv_thetha_deg_sat = self.Azi_Elv_Angles(BS_wrapped_Cord_sat, self.Xuser_sat, D_2d_sat)
                _, D_2d_UE_to_sat, _, Xuser = self.Deploy.Dist(self.Xap_with_sat , self.Xuser_sat) #Distance from Sat UES to all BSs+LEO
                self.D_2d_sat = D_2d_sat
                self.D_sat = D_sat
                self.BS_wrapped_Cord_sat = BS_wrapped_Cord_sat
                self.Azi_phi_deg_sat=Azi_phi_deg_sat
                self.Elv_thetha_deg_sat=Elv_thetha_deg_sat
                self.D_2d_UE_to_sat=D_2d_UE_to_sat
                self.D_2d_building_sat=0.0
                self.D_2d_in_sat=0.0
                self.D_building_sat=0.0
                self.Zuser_indoor_sat=0.0
                self.D_in_sat=0.0
                self.Azi_phi_deg_in_sat=0.0
                self.Elv_thetha_deg_in_sat=0.0
                if self.indoor == True:
                    D_2d_building_sat = D_2d_sat
                    self.D_2d_building_sat=D_2d_building_sat
                    D_building_sat = D_sat
                    self.D_building_sat=D_building_sat
                    BS_wrapped_Cord_in_sat = BS_wrapped_Cord_sat
                    D_2d_inside_building_sat = tf.math.minimum(
                        tf.random.uniform([D_2d_building_sat.shape[0], 1, D_2d_building_sat.shape[2]], 0.0, 25.0),
                        tf.random.uniform([D_2d_building_sat.shape[0], 1, D_2d_building_sat.shape[2]], 0.0, 25.0))
                    D_2d_in_sat = D_2d_building_sat + D_2d_inside_building_sat
                    self.D_2d_in_sat=D_2d_in_sat

                    # finding total 3D distance
                    Zuser_indoor_sat = tf.expand_dims(tf.squeeze(Zuser_in_sat , axis=2), axis=1)
                    self.Zuser_indoor_sat=Zuser_indoor_sat
                    D_in_sat = tf.sqrt(tf.pow(D_2d_in_sat, 2.0) + tf.pow(self.Zap - Zuser_indoor_sat, 2.0))
                    self.D_in_sat=D_in_sat
                    Azi_phi_deg_in_sat, Elv_thetha_deg_in_sat = self.Azi_Elv_Angles( BS_wrapped_Cord_in_sat,self.Xuser_sat, D_2d_in_sat)
                    self.Azi_phi_deg_in_sat=Azi_phi_deg_in_sat
                    self.Elv_thetha_deg_in_sat =Elv_thetha_deg_in_sat

                    D_sat, D_2d_sat, BS_wrapped_Cord_sat, Xuser = self.Deploy.Dist(Xap, Xuser_out)
                    Azi_phi_deg_sat, Elv_thetha_deg_sat= self.Azi_Elv_Angles(BS_wrapped_Cord_sat, Xuser_out, D_2d_sat)
                    self.D_2d_sat = D_2d_sat
                    self.D_sat = D_sat
                    self.BS_wrapped_Cord_sat = BS_wrapped_Cord_sat
                    self.Azi_phi_deg_sat = Azi_phi_deg_sat
                    self.Elv_thetha_deg_sat = Elv_thetha_deg_sat

                elif self.indoor==False:
                    D, D_2d, BS_wrapped_Cord, Xuser = self.Deploy.Dist(Xap,Xuser)
                    Azi_phi_deg, Elv_thetha_deg = self.Azi_Elv_Angles(BS_wrapped_Cord,Xuser,D_2d)

            if self.open_access:
                if self.N==1:
                    Xap = Xap
                elif self.N == 2:
                    Xap = tf.concat([Xap, tf.tile(tf.constant([[[self.LEO_x_cord, self.LEO_y_cord, self.Zleo],[0.0, -43301.27018922193, self.Zleo],[0.0, 43301.27018922193, self.Zleo],[-37500.0, -21650.635094610963, self.Zleo],[-37500.0, 21650.635094610963, self.Zleo],[37500.0, -21650.635094610963, self.Zleo],[37500.0, 21650.635094610963, self.Zleo]]]),[Xap.shape[0], 1, 1])], axis=1)

                # Nuser_drop_GUE = int(self.GUE_ratio * self.Nuser_drop)
                # Nuser_drop_UAV = int(self.UAV_ratio * self.Nuser_drop)
                # Xuser_GUE = Xuser[:, 0:Nuser_drop_GUE, :] #uncomment this if you are dealing with not 9UAVs
                # Xuser_UAV_ref = tf.expand_dims(Xuser[:, Nuser_drop_GUE:self.Nuser_drop, 0], axis=2)
                # Yuser_UAV_ref = tf.expand_dims(Xuser[:, Nuser_drop_GUE:self.Nuser_drop, 1], axis=2)
                # Zuser_UAV_ref = tf.expand_dims(Xuser[:, Nuser_drop_GUE:self.Nuser_drop, 2],axis=2) # I will use this for size ref
                #
                # # I will use this for size ref
                # Zuser_UAV = tf.ones(Zuser_UAV_ref.shape)*self.Zuav
                # Xuser_UAV = tf.concat([Xuser_UAV_ref, Yuser_UAV_ref, Zuser_UAV], axis=2)
                # Xuser = tf.concat([Xuser_GUE, Xuser_UAV], axis=1)

                D, D_2d, BS_wrapped_Cord, Xuser = self.Deploy.Dist(Xap, Xuser)
                Azi_phi_deg, Elv_thetha_deg = self.Azi_Elv_Angles(BS_wrapped_Cord, Xuser, D_2d)
                D_UAV=0.0
                D_2d_UAV=0.0
                BS_wrapped_Cord_UAV=0.0
                Azi_phi_deg_UAV=0.0
                Elv_thetha_deg_UAV=0.0

                if self.indoor == True:
                    Nuser_drop_in = int(self.in_ratio * self.GUE_ratio * self.Nuser_drop)
                    Nuser_drop_out = int(self.out_ratio * self.GUE_ratio * self.Nuser_drop)
                    self.Nuser_drop_out = Nuser_drop_out
                    Xuser_out = Xuser_GUE[:, 0:Nuser_drop_out, :]

                    Xuser_in_ref = tf.expand_dims(Xuser_GUE[:, Nuser_drop_out:self.Nuser_drop, 0], axis=2)
                    Yuser_in_ref = tf.expand_dims(Xuser_GUE[:, Nuser_drop_out:self.Nuser_drop, 1], axis=2)
                    Zuser_in_ref = tf.expand_dims(Xuser_GUE[:, Nuser_drop_out:self.Nuser_drop, 2],axis=2)  # I will use this for size ref

                    Nf = tf.random.uniform(Zuser_in_ref.shape, 4.0, 8.0)
                    nf = tf.random.uniform(Nf.shape, 0, 1) * (Nf - 1) + 1.0
                    Zuser_in = 3.0 * (nf - 1) + 1.5

                    Xuser_in = tf.concat([Xuser_in_ref, Yuser_in_ref, Zuser_in], axis=2)
                    Xuser_GUE = tf.concat([Xuser_out, Xuser_in], axis=1)

                    D, D_2d, BS_wrapped_Cord, Xuser = self.Deploy.Dist(Xap,Xuser_out)  # Note D, D2d is for Outside users referreing to D3d_out and D2d_out in 3GPP document
                    D_building, D_2d_building, BS_wrapped_Cord_in, Xuser_in = self.Deploy.Dist(Xap, Xuser_in)
                    D_2d_inside_building = tf.math.minimum(tf.random.uniform([D_2d_building.shape[0], 1, D_2d_building.shape[2]], 0.0, 25.0),tf.random.uniform([D_2d_building.shape[0], 1, D_2d_building.shape[2]], 0.0, 25.0))
                    D_2d_in = D_2d_building + D_2d_inside_building
                    # D_2d_in = D_2d_building + tf.random.uniform(D_2d_building.shape,0.0,25.0)

                    # finding total 3D distance
                    Zuser_indoor = tf.expand_dims(tf.squeeze(Zuser_in, axis=2), axis=1)

                    D_in = tf.sqrt(tf.pow(D_2d_in, 2.0) + tf.pow(self.Zap - Zuser_indoor, 2.0))

                    Azi_phi_deg, Elv_thetha_deg = self.Azi_Elv_Angles(BS_wrapped_Cord, Xuser_out, D_2d)
                    Azi_phi_deg_in, Elv_thetha_deg_in = self.Azi_Elv_Angles(BS_wrapped_Cord_in, Xuser_in, D_2d_in)

                    D_UAV, D_2d_UAV, BS_wrapped_Cord_UAV, Xuser_UAV = self.Deploy.Dist(Xap, Xuser_UAV)
                    Azi_phi_deg_UAV, Elv_thetha_deg_UAV = self.Azi_Elv_Angles(BS_wrapped_Cord_UAV, Xuser_UAV, D_2d_UAV)


        elif self.DeploymentType =="PPP":
            
            if self.indoor==True:
                D, D_2d, BS_wrapped_Cord, Xuser_out = self.Deploy.Dist(Xap, Xuser_out)
                D_in, D_2d_in, BS_wrapped_Cord_in, Xuser_in = self.Deploy.Dist(Xap, Xuser_in)

                D_2d_inside_building = tf.math.minimum(
                    tf.random.uniform([D_2d_building.shape[0], 1, D_2d_building.shape[2]], 0.0, 25.0),
                    tf.random.uniform([D_2d_building.shape[0], 1, D_2d_building.shape[2]], 0.0, 25.0))
                D_2d_in = D_2d_building + D_2d_inside_building
                # D_2d_in=D_2d_in+tf.random.uniform(D_2d_in.shape,0.0,25.0)
                
                Azi_phi_deg, Elv_thetha_deg = self.Azi_Elv_Angles(BS_wrapped_Cord, Xuser_out, D_2d)
                Azi_phi_deg_in, Elv_thetha_deg_in = self.Azi_Elv_Angles(BS_wrapped_Cord_in, Xuser_in, D_2d_in)
                
            elif self.indoor==False:
                D, D_2d, BS_wrapped_Cord, Xuser = self.Deploy.Dist(Xap, Xuser)
                Azi_phi_deg, Elv_thetha_deg = self.Azi_Elv_Angles(BS_wrapped_Cord, Xuser, D_2d)

        return Xap,Xuser, Xuser_in,Zuser_indoor,Xuser_out,D,D_2d,D_in,D_2d_in,D_2d_building,BS_wrapped_Cord,BS_wrapped_Cord_in,Azi_phi_deg,Elv_thetha_deg,Azi_phi_deg_in, Elv_thetha_deg_in,D_UAV, D_2d_UAV, BS_wrapped_Cord_UAV,Azi_phi_deg_UAV, Elv_thetha_deg_UAV

    def Azi_Elv_Angles(self,BS_wrapped_Cord,Xuser,D_2d):
        
        Xuser=tf.expand_dims(Xuser,axis=1)
        
        x_diff = BS_wrapped_Cord[:,:,:,0]-Xuser[:,:,:,0]
        y_diff = BS_wrapped_Cord[:,:,:,1]-Xuser[:,:,:,1]
        z_diff = BS_wrapped_Cord[:,:,:,2]-Xuser[:,:,:,2]
        
        Azi_phi_deg=tf.math.atan2(y_diff,x_diff)*180/math.pi 
        Elv_thetha_deg=(tf.math.atan2(z_diff,D_2d)*180.0/math.pi)+90.0
        
        return Azi_phi_deg,Elv_thetha_deg