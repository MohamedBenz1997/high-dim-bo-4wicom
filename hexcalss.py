import numpy as np
import tensorflow as tf
from config import Config
import math
# from __future__ import division
# from __future__ import print_function
import collections
import math
from libs.hexfunctions import *
from config import Config
from plot_class import Plot
import random
import matplotlib.pyplot as plt
import os
class DeployHex(Config):
    def __init__(self):
        Config.__init__(self)
        # Deploy cells using https://www.redblobgames.com/grids/hexagons/#map-storage lib
        # Define the hex objects for the first tire


        self.side = 2*self.radius/np.sqrt(3.0)
        T0 = []
        for i in range(2 * self.N + 1):
            for j in range(2 * self.N + 1):
                for k in range(2 * self.N + 1):
                    if (i - self.N) + (j - self.N) + (k - self.N) == 0:
                        h = Hex(i - self.N, j - self.N, k - self.N)
                        T0.append(h)
        self.T0 =T0
        #################################################################
        # Find the hex center for one of the hex-group in the second tire
        T0_center_pos = [0] * len(T0)
        orgin = Point(0.0, 0.0)
        # radius = 250
        size = Point(2 / math.sqrt(3.0) * self.radius, 2 / math.sqrt(3.0) * self.radius)
        layout = Layout(layout_flat, size, orgin)
        self.layout =layout
        # fig, ax = plt.subplots(1, 1)
        for i in range(len(T0)):
            T0_center_pos[i] = hex_to_pixel(layout, T0[i])
            # ax.plot(T0_center_pos[i].x, T0_center_pos[i].y, "ko")
            # ax.text(T0_center_pos[i].x, T0_center_pos[i].y, str(i))
            # corners = polygon_corners(layout, T0[i])
            # x, y = get_corners(corners)
            # ax.plot(x, y, 'cyan')
        self.T0_center_pos = T0_center_pos
        ################################################
        # Find the corners of the center cell in tire 0
        center_cell_id = int((len(T0)-1)/2)
        corners_T0_center = polygon_corners(layout, T0[center_cell_id])
        corners = np.zeros([len(corners_T0_center),2],'float32')
        corners[:,0] = [corners_T0_center[i].x for i in range(len(corners_T0_center))]
        corners[:,1] = [corners_T0_center[i].y for i in range(len(corners_T0_center))]
        self.corners_T0_center = corners
        # construct the equations for the sides of the hex in the form of y = ax+b
        a = (np.roll(self.corners_T0_center[:,1],-1)+1e-2-self.corners_T0_center[:,1])/(1e-15+np.roll(self.corners_T0_center[:,0],-1)-self.corners_T0_center[:,0])
        b = self.corners_T0_center[:,1]-a*self.corners_T0_center[:,0]
        self.corner_lines = [np.expand_dims(np.expand_dims(a,axis=0),axis=0),np.expand_dims(np.expand_dims(b,axis=0),axis=0)]
        self.lines_sign =np.array([[[1,1,1,-1,-1,-1]]])
        ###############################################################
        # Find of the images of the hex located in the center of tire 0
        c = Hex(0.0, 0.0, 0.0)
        c_miror0 = Hex(2 * self.N + 1, -self.N - 1, -self.N)
        # c_miror0_pos = hex_to_pixel(layout,c_miror0)
        rotate = hex_subtract(c_miror0, c)
        c_miror = [0] * 6
        pos_miror = [0] * 6
        for i in range(6):
            rotate = hex_rotate_right(rotate)
            c_miror[i] = rotate
            pos_miror[i] = hex_to_pixel(layout, rotate)
        wrap_shift_x = [pos_miror[i].x for i in range(len(pos_miror))]
        wrap_shift_y = [pos_miror[i].y for i in range(len(pos_miror))]
        wrap_shift_z = [0]*len(pos_miror)
        self.Wrap_shift = tf.constant([[[wrap_shift_x,wrap_shift_y,wrap_shift_z]]])
        self.Wrap_shift = tf.concat([self.Wrap_shift,tf.constant([[[[0.0],[0.0],[0.0]]]])],axis=3)
        self.c_miror = c_miror
        self.c_miror = c_miror
            # ax.plot(pos_miror[i].x,pos_miror[i].y,'o')
            # ax.text(pos_miror[i].x,pos_miror[i].y,str(i))
        # colors = ["red", "blue", "green", "orange", "purple", 'brown']
        # for i in range(len(c_miror)):
        #     center_i = c_miror[i]
        #     for j in range(len(T0)):
        #         Ti_center_j = hex_add(center_i, T0[j])
        #         Ti_center_j_pos = hex_to_pixel(layout, Ti_center_j)
        #         ax.plot(Ti_center_j_pos.x, Ti_center_j_pos.y, "ko")
        #         ax.text(Ti_center_j_pos.x, Ti_center_j_pos.y, str(j))
        #         corners = polygon_corners(layout, Ti_center_j)
        #         x, y = get_corners(corners)
        #         ax.plot(x, y, colors[i])
        # plt.show()
    def call(self):
        batch_num = self.batch_num*3 # Assigning the number of iterations to power 2 to ensure convergence. Becaause there are some iterations will be ignored if at least 1 BS does not have a UE associated with it.
        x0 = tf.random.uniform([batch_num, self.Nuser_drop, 1], -self.side,
                               self.side)  # Assigning a random values tensor for UEs x-axis which have the size of number of batches X number of UE. Having a limit of the x-axis grid defined previously as EX
        y0 = tf.random.uniform([batch_num, self.Nuser_drop, 1], -self.radius,
                               self.radius)  # Same logic as x-axis coordinate, this time for y.
        z0 = self.Zuser + tf.zeros([batch_num, self.Nuser_drop, 1],
                                   dtype='float32')  # For z-axis it is not random coordinates because we are defining the height of UEs at the beggining. We will sum that value to a zero value tensor with size desired, batches X number of UE
        # Xuser = tf.concat([x0, y0, z0], axis=2)  # The 3D coordinat

        check_in_hex = (y0-self.corner_lines[0]*x0-self.corner_lines[1])*self.lines_sign>=0
        check_in_hex = tf.cast(tf.reduce_sum(tf.cast(check_in_hex,'float32'),axis=2)== self.corners_T0_center.shape[0],"float32")
        x0_vec = tf.reshape(x0,[-1,1])
        y0_vec = tf.reshape(y0,[-1,1])
        z0_vec = tf.reshape(z0,[-1,1])
        check_in_hex = tf.reshape(check_in_hex,[-1])
        valid_user_id = tf.squeeze(tf.where(check_in_hex ==1))

        valid_user_id = valid_user_id[0:2*self.batch_num*self.Nuser_drop]
        # get valid users
        x0_vec = tf.gather(x0_vec,valid_user_id,axis=0)
        y0_vec = tf.gather(y0_vec,valid_user_id,axis=0)
        z0_vec = tf.gather(z0_vec, valid_user_id, axis=0)

        x0 = tf.reshape(x0_vec,[2*self.batch_num, self.Nuser_drop,1])
        y0 = tf.reshape(y0_vec, [2*self.batch_num, self.Nuser_drop,1])
        z0 = tf.reshape(z0_vec, [2*self.batch_num, self.Nuser_drop,1])
        Xuser = tf.concat([x0, y0, z0], axis=2)  # The 3D coordinat

        
        # move user_drop_per_cell users to each cell randomly
        Xap = self.T0_center_pos
        Zap= tf.tile(tf.constant([[self.Zap]]),[self.Nap,1])
        Xap = tf.concat([Xap,Zap],axis=1)
        ind_ap_rand = tf.random.uniform([2*self.batch_num*self.Nuser_drop],0,self.Nap,dtype='int32')
        ind_ap_rand = ind_ap_rand+self.Nap*tf.constant([i for i in range(ind_ap_rand.shape[0])],dtype="int32")
        Xap_vec = tf.gather(tf.tile(Xap,[ind_ap_rand.shape[0],1]),ind_ap_rand,axis=0)
        # Xuser = Xuser+tf.expand_dims(Xap,axis=2)
        # Xuser = tf.reshape(Xuser,[2*self.batch_num,self.Nap,self.user_drop_per_cell,3])
        Xap_vec_fixedZ=tf.concat([Xap_vec[:,0:2],tf.zeros([Xap_vec.shape[0],1])],axis=1)
        if self.hotspot==False:
            Xuser = tf.reshape(Xuser,[-1,3])+Xap_vec_fixedZ
            Xuser = tf.reshape(Xuser,[-1,self.Nuser_drop,3])
        Xap = tf.tile(tf.expand_dims(Xap,axis=0),[2*self.batch_num,1,1])

        # --------- Move users to created hotspots
        if self.hotspot:
            if self.one_tier == True:
                #Creat hotspot areas in the first cell, 100m in y-axis above the deployed BSs, x-axis fixed
                hotspot_center_cord = [(-433.0127, -150.0),
                                       (-433.0127, 350.0),
                                       (0.0, -400.0),
                                       (0.0, 100.0),
                                       (0.0, 600.0),
                                       (433.0127, -150.0),
                                       (433.0127, 350.0)]
                hotspot_center_cord = tf.concat([hotspot_center_cord], axis=1)
                hotspot_center_fixedHight = tf.expand_dims(tf.zeros(hotspot_center_cord.shape[0]), axis=1)
                hotspot_center_cord = tf.concat([hotspot_center_cord, hotspot_center_fixedHight], axis=1)

                ## Move UEs from the center hex cell to the hotspot areas
                #UEs at cell center to be moved to the hotpots
                Xuser_hotspot = tf.zeros(Xuser[:,0:self.UEs_in_hotspot*self.Num_hotspot,:].shape, 'float32')
                ind_hotspot = tf.tile(tf.constant([0,1,2,3,4,5,6], 'int32'),[self.UEs_in_hotspot*2 * self.batch_num]) #creat index numbers based on total UEs in hotspotsin all simulations considering all bateches
                hotspot_vec = tf.gather(tf.tile(hotspot_center_cord, [ind_hotspot.shape[0], 1]), ind_hotspot, axis=0)
                #extend hotspot location tensor to be same size as all UEs tensor size
                Xuser_hotspot = tf.reshape(Xuser_hotspot, [-1, 3]) + hotspot_vec #Shift Users to the hotspot area
                Xuser_hotspot = tf.reshape(Xuser_hotspot, [-1,self.UEs_in_hotspot*self.Num_hotspot, 3])
                #Remove UEs randomly around their hotspot center
                Xuser_hotspot_Xaxis = tf.expand_dims(Xuser_hotspot[:, :,0],axis=2)
                Xuser_hotspot_Yaxis = tf.expand_dims(Xuser_hotspot[:,:, 1], axis=2)
                Xuser_hotspot_Zaxis = tf.expand_dims(Xuser_hotspot[:, :, 2], axis=2)+self.Zuser
                Xuser_hotspot_Xaxis = Xuser_hotspot_Xaxis + tf.random.uniform(Xuser_hotspot_Xaxis.shape, -self.hotspot_radius, self.hotspot_radius)
                Xuser_hotspot_Yaxis = Xuser_hotspot_Yaxis + tf.random.uniform(Xuser_hotspot_Yaxis.shape, -self.hotspot_radius, self.hotspot_radius)
                Xuser_hotspot = tf.concat([Xuser_hotspot_Xaxis, Xuser_hotspot_Yaxis, Xuser_hotspot_Zaxis], axis=2)
                #Other Users to be uniformly distibuted among all cells
                Xuser_not_hotspot_Xaxis = tf.expand_dims(Xuser[:, self.UEs_in_hotspot*self.Num_hotspot:,0],axis=2)
                Xuser_not_hotspot_Yaxis = tf.expand_dims(Xuser[:, self.UEs_in_hotspot*self.Num_hotspot:, 1], axis=2)
                Xuser_not_hotspot_Zaxis = tf.expand_dims(Xuser[:, self.UEs_in_hotspot*self.Num_hotspot:, 2], axis=2)
                Xuser_not_hotspot_Xaxis = tf.random.uniform(Xuser_not_hotspot_Xaxis.shape, -550.0, 550.0)
                Xuser_not_hotspot_Yaxis = tf.random.uniform(Xuser_not_hotspot_Yaxis.shape, -550.0, 550.0)
                Xuser_not_hotspot = tf.concat([Xuser_not_hotspot_Xaxis, Xuser_not_hotspot_Yaxis, Xuser_not_hotspot_Zaxis], axis=2)
                Xuser = tf.concat([Xuser_hotspot, Xuser_not_hotspot], axis=1)

            if self.one_tier==False:
                Xuser = tf.reshape(Xuser, [-1, 3]) + Xap_vec_fixedZ
                Xuser = tf.reshape(Xuser, [-1, self.Nuser_drop, 3])
                #Creat hotspot areas in the first cell, 100m in y-axis above the deployed BSs, x-axis fixed
                hotspot_center_cord = [(-866.0254037844387-75.0, -300.0),
                                       (-866.0254037844387-75.0, 200.0),
                                       (866.0254037844387-75.0, 700.0),
                                       (433.01270189221935-75.0, -550.0),
                                       (-433.01270189221935-75.0, -50.0),
                                       (-433.01270189221935-75.0, 450.0),
                                       (-433.01270189221935-75.0, 950.0),
                                       (0.0-75.0, -800.0),
                                       (0.0-75.0, -300.0),
                                       (0.0-75.0, 200.0),
                                       (0.0-75.0, 700.0),
                                       (0.0-75.0, 1200.0),
                                       (-433.01270189221935-75.0, -550.0),
                                       (433.01270189221935-75.0, -50.0),
                                       (433.01270189221935-75.0, 450.0),
                                       (433.01270189221935-75.0, 950.0),
                                       (866.0254037844387-75.0, -300.0),
                                       (866.0254037844387-75.0, 200.0),
                                       (-866.0254037844387-75.0, 700.0)]
                hotspot_center_cord = tf.concat([hotspot_center_cord], axis=1)
                hotspot_center_fixedHight = tf.expand_dims(tf.zeros(hotspot_center_cord.shape[0]), axis=1)
                hotspot_center_cord = tf.concat([hotspot_center_cord, hotspot_center_fixedHight], axis=1)

                ## Move UEs from the center hex cell to the hotspot areas
                #UEs at cell center to be moved to the hotpots
                Xuser_hotspot = tf.zeros(Xuser[:,0:self.UEs_in_hotspot*self.Num_hotspot,:].shape, 'float32')
                ind_hotspot = tf.tile(tf.constant([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18], 'int32'),[self.UEs_in_hotspot*2 * self.batch_num]) #creat index numbers based on total UEs in hotspotsin all simulations considering all bateches
                hotspot_vec = tf.gather(tf.tile(hotspot_center_cord, [ind_hotspot.shape[0], 1]), ind_hotspot, axis=0)
                #extend hotspot location tensor to be same size as all UEs tensor size
                Xuser_hotspot = tf.reshape(Xuser_hotspot, [-1, 3]) + hotspot_vec #Shift Users to the hotspot area
                Xuser_hotspot = tf.reshape(Xuser_hotspot, [-1,self.UEs_in_hotspot*self.Num_hotspot, 3])
                #Remove UEs randomly around their hotspot center
                Xuser_hotspot_Xaxis = tf.expand_dims(Xuser_hotspot[:, :,0],axis=2)
                Xuser_hotspot_Yaxis = tf.expand_dims(Xuser_hotspot[:,:, 1], axis=2)
                Xuser_hotspot_Zaxis = tf.expand_dims(Xuser_hotspot[:, :, 2], axis=2)+self.Zuser
                Xuser_hotspot_Xaxis = Xuser_hotspot_Xaxis + tf.random.uniform(Xuser_hotspot_Xaxis.shape, -self.hotspot_radius, self.hotspot_radius)
                Xuser_hotspot_Yaxis = Xuser_hotspot_Yaxis + tf.random.uniform(Xuser_hotspot_Yaxis.shape, -self.hotspot_radius, self.hotspot_radius)
                Xuser_hotspot = tf.concat([Xuser_hotspot_Xaxis, Xuser_hotspot_Yaxis, Xuser_hotspot_Zaxis], axis=2)
                #Other Users to be uniformly distibuted among all cells
                Xuser_not_hotspot = Xuser[:, self.UEs_in_hotspot * self.Num_hotspot:, :]
                Xuser = tf.concat([Xuser_hotspot, Xuser_not_hotspot], axis=1)

        if self.UAVs_highway:
            Xuser_ground=Xuser[:, 0:int(self.GUE_ratio * self.Nuser_drop),:]

            #UAVs in Corridor
            Xuser_UAVs_Xaxis = tf.expand_dims(Xuser[:, int(self.GUE_ratio * self.Nuser_drop):,0],axis=2)
            Xuser_UAVs_Yaxis = tf.expand_dims(Xuser[:, int(self.GUE_ratio * self.Nuser_drop):, 1], axis=2)
            Xuser_UAVs_Zaxis = tf.expand_dims(Xuser[:, int(self.GUE_ratio * self.Nuser_drop):, 2], axis=2)

            Xuser_UAVs_Xaxis=tf.random.uniform(Xuser_UAVs_Xaxis.shape, -615.0, 615.0)
            Xuser_UAVs_Yaxis=tf.random.uniform(Xuser_UAVs_Yaxis.shape, -660.0, 660.0)


            Xuser_UAVs_Xaxis_Highway1 = tf.random.uniform(Xuser_UAVs_Xaxis[:,0:int((self.UAV_ratio * self.Nuser_drop)/4),:].shape, -650.0, -610.0)
            Xuser_UAVs_Xaxis_Highway2 = tf.random.uniform(Xuser_UAVs_Xaxis[:,int((self.UAV_ratio * self.Nuser_drop)/4):int((self.UAV_ratio * self.Nuser_drop)/2),:].shape, -780, 780)
            Xuser_UAVs_Xaxis_Highway3 = tf.random.uniform(Xuser_UAVs_Xaxis[:,int((self.UAV_ratio * self.Nuser_drop)/2):int((self.UAV_ratio * self.Nuser_drop)*3/4),:].shape, -780, 780)
            Xuser_UAVs_Xaxis_Highway4 = tf.random.uniform(Xuser_UAVs_Xaxis[:,int((self.UAV_ratio * self.Nuser_drop)*3/4):,:].shape, 610.0, 650.0)
            Xuser_UAVs_Xaxis = tf.concat([Xuser_UAVs_Xaxis_Highway1, Xuser_UAVs_Xaxis_Highway2, Xuser_UAVs_Xaxis_Highway3, Xuser_UAVs_Xaxis_Highway4], axis=1)

            Xuser_UAVs_Yaxis1 = tf.random.uniform(Xuser_UAVs_Xaxis[:,0:int((self.UAV_ratio * self.Nuser_drop)/4),:].shape,  -780.0, 780.0)
            Xuser_UAVs_Yaxis2 = tf.random.uniform(Xuser_UAVs_Xaxis[:,int((self.UAV_ratio * self.Nuser_drop)/4):int((self.UAV_ratio * self.Nuser_drop)/2),:].shape, -650.0, -610.0)
            Xuser_UAVs_Yaxis3 = tf.random.uniform(Xuser_UAVs_Xaxis[:,int((self.UAV_ratio * self.Nuser_drop)/2):int((self.UAV_ratio * self.Nuser_drop)*3/4),:].shape, 610.0, 650.0)
            Xuser_UAVs_Yaxis4 = tf.random.uniform(Xuser_UAVs_Xaxis[:,int((self.UAV_ratio * self.Nuser_drop)*3/4):,:].shape, -780, 780)
            Xuser_UAVs_Yaxis = tf.concat([Xuser_UAVs_Yaxis1, Xuser_UAVs_Yaxis2, Xuser_UAVs_Yaxis3, Xuser_UAVs_Yaxis4], axis=1)

            Xuser_UAVs_Zaxis1 = tf.random.uniform(Xuser_UAVs_Xaxis[:,0:int((self.UAV_ratio * self.Nuser_drop)/4),:].shape, 150.0, 150.0)
            Xuser_UAVs_Zaxis2 = tf.random.uniform(Xuser_UAVs_Xaxis[:,int((self.UAV_ratio * self.Nuser_drop)/4):int((self.UAV_ratio * self.Nuser_drop)/2),:].shape, 120.0, 120.0)
            Xuser_UAVs_Zaxis3 = tf.random.uniform(Xuser_UAVs_Xaxis[:,int((self.UAV_ratio * self.Nuser_drop)/2):int((self.UAV_ratio * self.Nuser_drop)*3/4),:].shape, 120.0, 120.0)
            Xuser_UAVs_Zaxis4 = tf.random.uniform(Xuser_UAVs_Xaxis[:,int((self.UAV_ratio * self.Nuser_drop)*3/4):,:].shape, 150.0, 150.0)
            Xuser_UAVs_Zaxis = tf.concat([Xuser_UAVs_Zaxis1, Xuser_UAVs_Zaxis2, Xuser_UAVs_Zaxis3, Xuser_UAVs_Zaxis4], axis=1)

            Xuser_UAVs = tf.concat([Xuser_UAVs_Xaxis, Xuser_UAVs_Yaxis, Xuser_UAVs_Zaxis], axis=2)


            Xuser = tf.concat([Xuser_ground, Xuser_UAVs], axis=1)

        if self.UAVs_highway==False:
            Xuser_ground = Xuser[:, 0:int(self.GUE_ratio * self.Nuser_drop):, :]
            Xuser_UAVs_x_y = Xuser[:, int(self.GUE_ratio * self.Nuser_drop):, 0:2]
            Xuser_UAVs = tf.concat([Xuser_UAVs_x_y, self.Zuav], axis=2)
            Xuser = tf.concat([Xuser_ground, Xuser_UAVs], axis=1)


        self.Xuser = Xuser
        self.Xap = Xap

        return Xap,Xuser
    
    
    def Dist(self,Xap,Xuser):
        Xap_wrap = tf.expand_dims(Xap,axis=3)+self.Wrap_shift
        # Compute the distance between each userBS and their images
        xdiff = tf.expand_dims(Xap_wrap,axis=2)-tf.expand_dims(tf.expand_dims(Xuser,axis=1),axis=4)
        D = tf.sqrt(tf.reduce_sum(tf.math.square(xdiff),axis=3))
        D_2d = tf.sqrt(tf.reduce_sum(tf.math.square(xdiff[:,:,:,0:2,:]),axis=3))
        # D_ind = tf.argmin(D,axis=3)
        D_3d = tf.reduce_min(D,axis=3)
        ind_min = tf.cast(D==tf.expand_dims(D_3d,axis=3),'float32')
        # D_2d = tf.reduce_sum(D_2d*ind_min,axis=3)
        D_2d = tf.reduce_min(D_2d, axis=3)
        BS_wrapped_Cord = tf.expand_dims(ind_min,axis=3)
        BS_wrapped_Cord= BS_wrapped_Cord *tf.expand_dims(Xap_wrap,axis=2)
        BS_wrapped_Cord = tf.reduce_sum(BS_wrapped_Cord,axis=4)

        #This is to move the user close to the BS based on D2d exclusion creteria
        D_2d_exclusion=tf.expand_dims(tf.cast(tf.reduce_sum(tf.cast(D_2d < self.Dist2D_exclud, "float32"),axis=1)>0.0, "float32"),axis=2)
        users_in_exlusion=D_2d_exclusion * Xuser
        moved_users = users_in_exlusion+ (tf.cast(users_in_exlusion != 0.0, "float32")*tf.constant([[[self.Dist2D_exclud*1.5, self.Dist2D_exclud*1.5, 0.0]]], dtype=tf.float32))
        stayed_users = tf.cast(D_2d_exclusion == 0.0, "float32")* Xuser
        Xuser = stayed_users + moved_users
        #Do the calculations again
        # Compute the distance between each userBS and their images
        xdiff = tf.expand_dims(Xap_wrap,axis=2)-tf.expand_dims(tf.expand_dims(Xuser,axis=1),axis=4)
        D = tf.sqrt(tf.reduce_sum(tf.math.square(xdiff),axis=3))
        D_2d = tf.sqrt(tf.reduce_sum(tf.math.square(xdiff[:,:,:,0:2,:]),axis=3))
        # D_ind = tf.argmin(D,axis=3)
        D_3d = tf.reduce_min(D,axis=3)
        ind_min = tf.cast(D==tf.expand_dims(D_3d,axis=3),'float32')
        # D_2d = tf.reduce_sum(D_2d*ind_min,axis=3)
        D_2d = tf.reduce_min(D_2d, axis=3)
        BS_wrapped_Cord = tf.expand_dims(ind_min,axis=3)
        BS_wrapped_Cord= BS_wrapped_Cord *tf.expand_dims(Xap_wrap,axis=2)
        BS_wrapped_Cord = tf.reduce_sum(BS_wrapped_Cord,axis=4)
        return D_3d,D_2d,BS_wrapped_Cord,Xuser

        # Xap_wrap = tf.expand_dims(tf.Xap_wrap,axis=2)+
        # return D,D_2d,
    def plot_hex(self):
        Xap = self.Xap
        Xuser_assigned = self.Xuser
        # if self.one_tier:
        #     self.T0=[self.T0[i] for i in [4,5,8,9,10,13,14]]
        T0_center_pos = [0] * len(self.T0)
        fig, ax = plt.subplots(1, 1)
        if self.UAVs_highway:
            plt.vlines(-630, -780, 780, 'red', linewidth=8)
            plt.hlines(-630, -780, 780, 'green', linewidth=8)
            plt.hlines(630, -780, 780, 'brown', linewidth=8)
            plt.vlines(630, -780, 780, 'gray', linewidth=8)
        for i in range(len(self.T0)):
            T0_center_pos[i] = hex_to_pixel(self.layout, self.T0[i])
            ax.plot(T0_center_pos[i].x, T0_center_pos[i].y, "ko")
            if i==0:
                ax.plot(T0_center_pos[i].x, T0_center_pos[i].y, "ko", label = 'BSs')
            ax.text(T0_center_pos[i].x, T0_center_pos[i].y, str(i)) #This shows the index on the center of the cell
            corners = polygon_corners(self.layout, self.T0[i])
            x, y = get_corners(corners)
            ax.plot(x, y, 'black',  zorder=1)
        for i in range(int(self.GUE_ratio * self.Nuser_drop), self.Nuser_drop):
        # for i in range(0,int(self.GUE_ratio * self.Nuser_drop)):
            if i == 0:
                # ax.plot(Xap[0, i, 0], Xap[0, i, 1], 'x', color='brown', label="BS")
                # ax.text(Xap[0, i, 0], Xap[0, i, 1], str(i), color='brown', label="BS")
                ax.plot(Xuser_assigned[0, i, 0], Xuser_assigned[0, i, 1], 'o', color='blue', label="User")
                # ax.text(Xuser_assigned[0, i, 0], Xuser_assigned[0, i, 1], str(i), color='blue', label="User")
            else:
                # ax.plot(Xap[0, i, 0], Xap[0, i, 1], 'x', color='brown')
                # ax.text(Xap[0, i, 0], Xap[0, i, 1], str(i), color='brown')
                ax.plot(Xuser_assigned[0, i, 0], Xuser_assigned[0, i, 1], 'o', color='blue')
                # ax.text(Xuser_assigned[0, i, 0], Xuser_assigned[0, i, 1], str(i), color='blue')
            # ax.plot([Xuser_assigned[0, i, 0], Xap[0, i, 0]], [Xuser_assigned[0, 3*i, 1], Xap[0, i, 1]], color='red')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.legend()
        # ax.set_facecolor('teal')
        # plt.legend("BS")
        # plt.savefig('./2DNetworkLayout.jpg', format='jpg')
        plt.show()
        pass
    def show_wrap_around(self,Xuser,BS_wrapped_Cord,savefig=False):
        colors = ["red", "blue", "green", "orange", "purple", 'brown',"black"]
        c_miror = self.c_miror
        c = Hex(0.0, 0.0, 0.0)
        c_miror.append(c)
        T0 = self.T0
        layout = self.layout
        fig, ax = plt.subplots(1, 1)
        for i in range(len(c_miror)):
            center_i = c_miror[i]
            for j in range(len(T0)):
                Ti_center_j = hex_add(center_i, T0[j])
                Ti_center_j_pos = hex_to_pixel(layout, Ti_center_j)
                ax.plot(Ti_center_j_pos.x, Ti_center_j_pos.y,  c="gray", alpha=0.05, marker="o")
                ax.text(Ti_center_j_pos.x, Ti_center_j_pos.y, str(j))
                corners = polygon_corners(layout, Ti_center_j)
                x, y = get_corners(corners)
                if i==6:
                    ax.plot(x, y, "blue",alpha=0.5)
                else:
                    ax.plot(x, y, "gray",alpha=0.3)
        user_id = 250
        BS_wrapped_Cord_test_user = BS_wrapped_Cord[0,:,user_id,:]
        for i in range(self.Nap):
            if i==0:
                plt.plot(BS_wrapped_Cord_test_user[i, 0], BS_wrapped_Cord_test_user[i, 1], '*', color='red',
                         label="Wrap AP")
            else:
                plt.plot(BS_wrapped_Cord_test_user[i, 0],BS_wrapped_Cord_test_user[i, 1], '*', color='red')
        plt.plot(Xuser[0,user_id , 0], Xuser[0, user_id, 1], 'o', color="green", label="User")
        ax.legend()
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        plt.title("Wraparound model from a user viewpoint")
        dir = os.getcwd()
        if savefig:
            plt.savefig(dir + '/results/wrap_around.pdf', bbox_inches='tight')
        plt.show()
        pass



#
# #test the class
# config = Config()
# ----------- This is where you see the deployed users on the hex grid
# deployment = DeployHex()
# deployment.call()
# deployment.plot_hex()

#Wraparound showing
# Xap,Xuser=deployment.call()
# D_3d,D_2d,BS_wrapped_Cord=deployment.Dist(Xap,Xuser)
# deployment.show_wrap_around(Xuser,BS_wrapped_Cord)
# ------------
# D_3d,D_2d,BS_wrapped_Cord=deployment.Dist(deployment.Xap,deployment.Xuser)
#
# deployment.show_wrap_around(deployment.Xuser,BS_wrapped_Cord,savefig=True)
# # deployment.dist(deployment.Xap,deployment.Xuser,deployment.Wrap_shift)
