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

            Xuser_ground = Xuser[:, 0:int(self.GUE_ratio * self.Nuser_drop), :]

            #Mohamed UAVs Corr
            #################################################################
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

            Xuser_UAVs_Zaxis1 = tf.random.uniform(Xuser_UAVs_Xaxis[:,0:int((self.UAV_ratio * self.Nuser_drop)/4),:].shape, self.h_corr1, self.h_corr1)
            Xuser_UAVs_Zaxis2 = tf.random.uniform(Xuser_UAVs_Xaxis[:,int((self.UAV_ratio * self.Nuser_drop)/4):int((self.UAV_ratio * self.Nuser_drop)/2),:].shape, self.h_corr2, self.h_corr2)
            Xuser_UAVs_Zaxis3 = tf.random.uniform(Xuser_UAVs_Xaxis[:,int((self.UAV_ratio * self.Nuser_drop)/2):int((self.UAV_ratio * self.Nuser_drop)*3/4),:].shape, self.h_corr3, self.h_corr3)
            Xuser_UAVs_Zaxis4 = tf.random.uniform(Xuser_UAVs_Xaxis[:,int((self.UAV_ratio * self.Nuser_drop)*3/4):,:].shape, self.h_corr4, self.h_corr4)
            Xuser_UAVs_Zaxis = tf.concat([Xuser_UAVs_Zaxis1, Xuser_UAVs_Zaxis2, Xuser_UAVs_Zaxis3, Xuser_UAVs_Zaxis4], axis=1)
            #################################################################

            #Matteo UAVs Corr
            #################################################################
            # Xuser_UAVs_Xaxis = tf.expand_dims(Xuser[:, int(self.GUE_ratio * self.Nuser_drop):,0],axis=2)
            # Xuser_UAVs_Yaxis = tf.expand_dims(Xuser[:, int(self.GUE_ratio * self.Nuser_drop):, 1], axis=2)
            # Xuser_UAVs_Zaxis = tf.expand_dims(Xuser[:, int(self.GUE_ratio * self.Nuser_drop):, 2], axis=2)
            #
            # Xuser_UAVs_Xaxis=tf.random.uniform(Xuser_UAVs_Xaxis.shape, -200.0, 200.0)
            # Xuser_UAVs_Yaxis=tf.random.uniform(Xuser_UAVs_Yaxis.shape, -400.0, 400.0)
            #
            #
            # Xuser_UAVs_Xaxis_Highway1 = tf.random.uniform(Xuser_UAVs_Xaxis[:,0:int((self.UAV_ratio * self.Nuser_drop)/4),:].shape, -220.0, -180.0)
            # Xuser_UAVs_Xaxis_Highway2 = tf.random.uniform(Xuser_UAVs_Xaxis[:,int((self.UAV_ratio * self.Nuser_drop)/4):int((self.UAV_ratio * self.Nuser_drop)/2),:].shape, -120, -80)
            # Xuser_UAVs_Xaxis_Highway3 = tf.random.uniform(Xuser_UAVs_Xaxis[:,int((self.UAV_ratio * self.Nuser_drop)/2):int((self.UAV_ratio * self.Nuser_drop)*3/4),:].shape, 80, 120)
            # Xuser_UAVs_Xaxis_Highway4 = tf.random.uniform(Xuser_UAVs_Xaxis[:,int((self.UAV_ratio * self.Nuser_drop)*3/4):,:].shape, 180.0, 220.0)
            # Xuser_UAVs_Xaxis = tf.concat([Xuser_UAVs_Xaxis_Highway1, Xuser_UAVs_Xaxis_Highway2, Xuser_UAVs_Xaxis_Highway3, Xuser_UAVs_Xaxis_Highway4], axis=1)
            #
            # Xuser_UAVs_Yaxis1 = tf.random.uniform(Xuser_UAVs_Xaxis[:,0:int((self.UAV_ratio * self.Nuser_drop)/4),:].shape,  -400.0, 400.0)
            # Xuser_UAVs_Yaxis2 = tf.random.uniform(Xuser_UAVs_Xaxis[:,int((self.UAV_ratio * self.Nuser_drop)/4):int((self.UAV_ratio * self.Nuser_drop)/2),:].shape, -400.0, 400.0)
            # Xuser_UAVs_Yaxis3 = tf.random.uniform(Xuser_UAVs_Xaxis[:,int((self.UAV_ratio * self.Nuser_drop)/2):int((self.UAV_ratio * self.Nuser_drop)*3/4),:].shape, -400.0, 400.0)
            # Xuser_UAVs_Yaxis4 = tf.random.uniform(Xuser_UAVs_Xaxis[:,int((self.UAV_ratio * self.Nuser_drop)*3/4):,:].shape, -400.0, 400.0)
            # Xuser_UAVs_Yaxis = tf.concat([Xuser_UAVs_Yaxis1, Xuser_UAVs_Yaxis2, Xuser_UAVs_Yaxis3, Xuser_UAVs_Yaxis4], axis=1)
            #
            # Xuser_UAVs_Zaxis1 = tf.random.uniform(Xuser_UAVs_Xaxis[:,0:int((self.UAV_ratio * self.Nuser_drop)/4),:].shape, self.h_corr1, self.h_corr1)
            # Xuser_UAVs_Zaxis2 = tf.random.uniform(Xuser_UAVs_Xaxis[:,int((self.UAV_ratio * self.Nuser_drop)/4):int((self.UAV_ratio * self.Nuser_drop)/2),:].shape, self.h_corr2, self.h_corr2)
            # Xuser_UAVs_Zaxis3 = tf.random.uniform(Xuser_UAVs_Xaxis[:,int((self.UAV_ratio * self.Nuser_drop)/2):int((self.UAV_ratio * self.Nuser_drop)*3/4),:].shape, self.h_corr3, self.h_corr3)
            # Xuser_UAVs_Zaxis4 = tf.random.uniform(Xuser_UAVs_Xaxis[:,int((self.UAV_ratio * self.Nuser_drop)*3/4):,:].shape, self.h_corr4, self.h_corr4)
            # Xuser_UAVs_Zaxis = tf.concat([Xuser_UAVs_Zaxis1, Xuser_UAVs_Zaxis2, Xuser_UAVs_Zaxis3, Xuser_UAVs_Zaxis4], axis=1)
            #################################################################

            Xuser_UAVs = tf.concat([Xuser_UAVs_Xaxis, Xuser_UAVs_Yaxis, Xuser_UAVs_Zaxis], axis=2)
            Xuser = tf.concat([Xuser_ground, Xuser_UAVs], axis=1)

            if self.GUEs_debug:
                #Scenario: 1 GUE, All BS, all sectors

                fixed_x01 = tf.random.uniform([2*self.batch_num, 1, 1], -866.0254-95.26, -866.0254-95.26)
                fixed_y01 = tf.random.uniform([2 * self.batch_num, 1, 1], -500.0+55.0, -500.0+55.0)
                fixed_z01 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser01 = tf.concat([fixed_x01, fixed_y01, fixed_z01], axis=2)
                fixed_x02 = tf.random.uniform([2*self.batch_num, 1, 1], -866.0254+95.26, -866.0254+95.26)
                fixed_y02 = tf.random.uniform([2 * self.batch_num, 1, 1], -500.0+55.0, -500.0+55.0)
                fixed_z02 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser02 = tf.concat([fixed_x02, fixed_y02, fixed_z02], axis=2)
                fixed_x03 = tf.random.uniform([2*self.batch_num, 1, 1], -866.0254, -866.0254)
                fixed_y03 = tf.random.uniform([2 * self.batch_num, 1, 1], -500.0-110.0, -500.0-110.0)
                fixed_z03 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser03 = tf.concat([fixed_x03, fixed_y03, fixed_z03], axis=2)
                Xuser0 = tf.concat([Xuser01, Xuser02, Xuser03], axis=1)

                fixed_x11 = tf.random.uniform([2*self.batch_num, 1, 1], -866.0254-95.26, -866.0254-95.26)
                fixed_y11 = tf.random.uniform([2 * self.batch_num, 1, 1], 0.0+55.0, 0.0+55.0)
                fixed_z11 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser11 = tf.concat([fixed_x11, fixed_y11, fixed_z11], axis=2)
                fixed_x12 = tf.random.uniform([2*self.batch_num, 1, 1], -866.0254+95.26, -866.0254+95.26)
                fixed_y12 = tf.random.uniform([2 * self.batch_num, 1, 1], 0.0+55.0, 0.0+55.0)
                fixed_z12 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser12 = tf.concat([fixed_x12, fixed_y12, fixed_z12], axis=2)
                fixed_x13 = tf.random.uniform([2*self.batch_num, 1, 1], -866.0254, -866.0254)
                fixed_y13 = tf.random.uniform([2 * self.batch_num, 1, 1], 0.0-110.0, 0.0-110.0)
                fixed_z13 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser13 = tf.concat([fixed_x13, fixed_y13, fixed_z13], axis=2)
                Xuser1 = tf.concat([Xuser11, Xuser12, Xuser13], axis=1)


                fixed_x21 = tf.random.uniform([2*self.batch_num, 1, 1], -866.0254-95.26, -866.0254-95.26)
                fixed_y21 = tf.random.uniform([2 * self.batch_num, 1, 1], 500.0+55.0, 500.0+55.0)
                fixed_z21 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser21 = tf.concat([fixed_x21, fixed_y21, fixed_z21], axis=2)
                fixed_x22 = tf.random.uniform([2*self.batch_num, 1, 1], -866.0254+95.26, -866.0254+95.26)
                fixed_y22 = tf.random.uniform([2 * self.batch_num, 1, 1], 500.0+55.0, 500.0+55.0)
                fixed_z22 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser22 = tf.concat([fixed_x22, fixed_y22, fixed_z22], axis=2)
                fixed_x23 = tf.random.uniform([2*self.batch_num, 1, 1], -866.0254, -866.0254)
                fixed_y23 = tf.random.uniform([2 * self.batch_num, 1, 1], 500.0-110.0, 500.0-110.0)
                fixed_z23 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser23 = tf.concat([fixed_x23, fixed_y23, fixed_z23], axis=2)
                Xuser2 = tf.concat([Xuser21, Xuser22, Xuser23], axis=1)

                fixed_x31 = tf.random.uniform([2*self.batch_num, 1, 1], -433.0127-95.26, -433.0127-95.26)
                fixed_y31 = tf.random.uniform([2 * self.batch_num, 1, 1], -750.0+55.0, -750.0+55.0)
                fixed_z31 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser31 = tf.concat([fixed_x31, fixed_y31, fixed_z31], axis=2)
                fixed_x32 = tf.random.uniform([2*self.batch_num, 1, 1], -433.0127+95.26, -433.0127+95.26)
                fixed_y32 = tf.random.uniform([2 * self.batch_num, 1, 1], -750.0+55.0, -750.0+55.0)
                fixed_z32 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser32 = tf.concat([fixed_x32, fixed_y32, fixed_z32], axis=2)
                fixed_x33 = tf.random.uniform([2*self.batch_num, 1, 1], -433.0127, -433.0127)
                fixed_y33 = tf.random.uniform([2 * self.batch_num, 1, 1], -750.0-110.0, -750.0-110.0)
                fixed_z33 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser33 = tf.concat([fixed_x33, fixed_y33, fixed_z33], axis=2)
                Xuser3 = tf.concat([Xuser31, Xuser32, Xuser33], axis=1)

                fixed_x41 = tf.random.uniform([2*self.batch_num, 1, 1], -433.0127-95.26, -433.0127-95.26)
                fixed_y41 = tf.random.uniform([2 * self.batch_num, 1, 1], -250.0+55.0, -250.0+55.0)
                fixed_z41 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser41 = tf.concat([fixed_x41, fixed_y41, fixed_z41], axis=2)
                fixed_x42 = tf.random.uniform([2*self.batch_num, 1, 1], -433.0127+95.26, -433.0127+95.26)
                fixed_y42 = tf.random.uniform([2 * self.batch_num, 1, 1], -250.0+55.0, -250.0+55.0)
                fixed_z42 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser42 = tf.concat([fixed_x42, fixed_y42, fixed_z42], axis=2)
                fixed_x43 = tf.random.uniform([2*self.batch_num, 1, 1], -433.0127, -433.0127)
                fixed_y43 = tf.random.uniform([2 * self.batch_num, 1, 1], -250.0-110.0, -250.0-110.0)
                fixed_z43 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser43 = tf.concat([fixed_x43, fixed_y43, fixed_z43], axis=2)
                Xuser4 = tf.concat([Xuser41, Xuser42, Xuser43], axis=1)

                fixed_x51 = tf.random.uniform([2*self.batch_num, 1, 1], -433.0127-95.26, -433.0127-95.26)
                fixed_y51 = tf.random.uniform([2 * self.batch_num, 1, 1], 250.0+55.0, 250.0+55.0)
                fixed_z51 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser51 = tf.concat([fixed_x51, fixed_y51, fixed_z51], axis=2)
                fixed_x52 = tf.random.uniform([2*self.batch_num, 1, 1], -433.0127+95.26, -433.0127+95.26)
                fixed_y52 = tf.random.uniform([2 * self.batch_num, 1, 1], 250.0+55.0, 250.0+55.0)
                fixed_z52 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser52 = tf.concat([fixed_x52, fixed_y52, fixed_z52], axis=2)
                fixed_x53 = tf.random.uniform([2*self.batch_num, 1, 1], -433.0127, -433.0127)
                fixed_y53 = tf.random.uniform([2 * self.batch_num, 1, 1], 250.0-110.0, 250.0-110.0)
                fixed_z53 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser53 = tf.concat([fixed_x53, fixed_y53, fixed_z53], axis=2)
                Xuser5 = tf.concat([Xuser51, Xuser52, Xuser53], axis=1)

                fixed_x61 = tf.random.uniform([2*self.batch_num, 1, 1], -433.0127-95.26, -433.0127-95.26)
                fixed_y61 = tf.random.uniform([2 * self.batch_num, 1, 1], 750.0+55.0, 750.0+55.0)
                fixed_z61 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser61 = tf.concat([fixed_x61, fixed_y61, fixed_z61], axis=2)
                fixed_x62 = tf.random.uniform([2*self.batch_num, 1, 1], -433.0127+95.26, -433.0127+95.26)
                fixed_y62 = tf.random.uniform([2 * self.batch_num, 1, 1], 750.0+55.0, 750.0+55.0)
                fixed_z62 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser62 = tf.concat([fixed_x62, fixed_y62, fixed_z62], axis=2)
                fixed_x63 = tf.random.uniform([2*self.batch_num, 1, 1], -433.0127, -433.0127)
                fixed_y63 = tf.random.uniform([2 * self.batch_num, 1, 1], 750.0-110.0, 750.0-110.0)
                fixed_z63 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser63 = tf.concat([fixed_x63, fixed_y63, fixed_z63], axis=2)
                Xuser6 = tf.concat([Xuser61, Xuser62, Xuser63], axis=1)

                fixed_x71 = tf.random.uniform([2*self.batch_num, 1, 1],  0.0-95.26,  0.0-95.26)
                fixed_y71 = tf.random.uniform([2 * self.batch_num, 1, 1], -1000.0+55.0, -1000.0+55.0)
                fixed_z71 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser71 = tf.concat([fixed_x71, fixed_y71, fixed_z71], axis=2)
                fixed_x72 = tf.random.uniform([2*self.batch_num, 1, 1],  0.0+95.26,  0.0+95.26)
                fixed_y72 = tf.random.uniform([2 * self.batch_num, 1, 1], -1000.0+55.0, -1000.0+55.0)
                fixed_z72 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser72 = tf.concat([fixed_x72, fixed_y72, fixed_z72], axis=2)
                fixed_x73 = tf.random.uniform([2*self.batch_num, 1, 1],  0.0,  0.0)
                fixed_y73 = tf.random.uniform([2 * self.batch_num, 1, 1], -1000.0-110.0, -1000.0-110.0)
                fixed_z73 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser73 = tf.concat([fixed_x73, fixed_y73, fixed_z73], axis=2)
                Xuser7 = tf.concat([Xuser71, Xuser72, Xuser73], axis=1)

                fixed_x81 = tf.random.uniform([2*self.batch_num, 1, 1],  0.0-95.26,  0.0-95.26)
                fixed_y81 = tf.random.uniform([2 * self.batch_num, 1, 1], -500.0+55.0, -500.0+55.0)
                fixed_z81 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser81 = tf.concat([fixed_x81, fixed_y81, fixed_z81], axis=2)
                fixed_x82 = tf.random.uniform([2*self.batch_num, 1, 1],  0.0+95.26,  0.0+95.26)
                fixed_y82 = tf.random.uniform([2 * self.batch_num, 1, 1], -500.0+55.0, -500.0+55.0)
                fixed_z82 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser82 = tf.concat([fixed_x82, fixed_y82, fixed_z82], axis=2)
                fixed_x83 = tf.random.uniform([2*self.batch_num, 1, 1],  0.0,  0.0)
                fixed_y83 = tf.random.uniform([2 * self.batch_num, 1, 1], -500.0-110.0, -500.0-110.0)
                fixed_z83 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser83 = tf.concat([fixed_x83, fixed_y83, fixed_z83], axis=2)
                Xuser8 = tf.concat([Xuser81, Xuser82, Xuser83], axis=1)

                fixed_x91 = tf.random.uniform([2*self.batch_num, 1, 1],  0.0-95.26,  0.0-95.26)
                fixed_y91 = tf.random.uniform([2 * self.batch_num, 1, 1], 0.0+55.0, 0.0+55.0)
                fixed_z91 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser91 = tf.concat([fixed_x91, fixed_y91, fixed_z91], axis=2)
                fixed_x92 = tf.random.uniform([2*self.batch_num, 1, 1],  0.0+95.26,  0.0+95.26)
                fixed_y92 = tf.random.uniform([2 * self.batch_num, 1, 1], 0.0+55.0, 0.0+55.0)
                fixed_z92 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser92 = tf.concat([fixed_x92, fixed_y92, fixed_z92], axis=2)
                fixed_x93 = tf.random.uniform([2*self.batch_num, 1, 1],  0.0,  0.0)
                fixed_y93 = tf.random.uniform([2 * self.batch_num, 1, 1], 0.0-110.0, 0.0-110.0)
                fixed_z93 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser93 = tf.concat([fixed_x93, fixed_y93, fixed_z93], axis=2)
                Xuser9 = tf.concat([Xuser91, Xuser92, Xuser93], axis=1)

                fixed_x101 = tf.random.uniform([2*self.batch_num, 1, 1],  0.0-95.26,  0.0-95.26)
                fixed_y101 = tf.random.uniform([2 * self.batch_num, 1, 1], 500.0+55.0, 500.0+55.0)
                fixed_z101 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser101 = tf.concat([fixed_x101, fixed_y101, fixed_z101], axis=2)
                fixed_x102 = tf.random.uniform([2*self.batch_num, 1, 1],  0.0+95.26,  0.0+95.26)
                fixed_y102 = tf.random.uniform([2 * self.batch_num, 1, 1], 500.0+55.0, 500.0+55.0)
                fixed_z102 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser102 = tf.concat([fixed_x102, fixed_y102, fixed_z102], axis=2)
                fixed_x103 = tf.random.uniform([2*self.batch_num, 1, 1],  0.0,  0.0)
                fixed_y103 = tf.random.uniform([2 * self.batch_num, 1, 1], 500.0-110.0, 500.0-110.0)
                fixed_z103 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser103 = tf.concat([fixed_x103, fixed_y103, fixed_z103], axis=2)
                Xuser10 = tf.concat([Xuser101, Xuser102, Xuser103], axis=1)

                fixed_x111 = tf.random.uniform([2*self.batch_num, 1, 1],  0.0-95.26,  0.0-95.26)
                fixed_y111 = tf.random.uniform([2 * self.batch_num, 1, 1], 1000.0+55.0, 1000.0+55.0)
                fixed_z111 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser111 = tf.concat([fixed_x111, fixed_y111, fixed_z111], axis=2)
                fixed_x112 = tf.random.uniform([2*self.batch_num, 1, 1],  0.0+95.26,  0.0+95.26)
                fixed_y112 = tf.random.uniform([2 * self.batch_num, 1, 1], 1000.0+55.0, 1000.0+55.0)
                fixed_z112 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser112 = tf.concat([fixed_x112, fixed_y112, fixed_z112], axis=2)
                fixed_x113 = tf.random.uniform([2*self.batch_num, 1, 1],  0.0,  0.0)
                fixed_y113 = tf.random.uniform([2 * self.batch_num, 1, 1], 1000.0-110.0, 1000.0-110.0)
                fixed_z113 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser113 = tf.concat([fixed_x113, fixed_y113, fixed_z113], axis=2)
                Xuser11 = tf.concat([Xuser111, Xuser112, Xuser113], axis=1)

                fixed_x121 = tf.random.uniform([2*self.batch_num, 1, 1],  433.0127-95.26,  433.0127-95.26)
                fixed_y121 = tf.random.uniform([2 * self.batch_num, 1, 1], -750.0+55.0, -750.0+55.0)
                fixed_z121 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser121 = tf.concat([fixed_x121, fixed_y121, fixed_z121], axis=2)
                fixed_x122 = tf.random.uniform([2*self.batch_num, 1, 1],  433.0127+95.26,  433.0127+95.26)
                fixed_y122 = tf.random.uniform([2 * self.batch_num, 1, 1], -750.0+55.0, -750.0+55.0)
                fixed_z122 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser122 = tf.concat([fixed_x122, fixed_y122, fixed_z122], axis=2)
                fixed_x123 = tf.random.uniform([2*self.batch_num, 1, 1],  433.0127,  433.0127)
                fixed_y123 = tf.random.uniform([2 * self.batch_num, 1, 1], -750.0-110.0, -750.0-110.0)
                fixed_z123 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser123 = tf.concat([fixed_x123, fixed_y123, fixed_z123], axis=2)
                Xuser12 = tf.concat([Xuser121, Xuser122, Xuser123], axis=1)

                fixed_x131 = tf.random.uniform([2*self.batch_num, 1, 1],  433.0127-95.26,  433.0127-95.26)
                fixed_y131 = tf.random.uniform([2 * self.batch_num, 1, 1], -250.0+55.0, -250.0+55.0)
                fixed_z131 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser131 = tf.concat([fixed_x131, fixed_y131, fixed_z131], axis=2)
                fixed_x132 = tf.random.uniform([2*self.batch_num, 1, 1],  433.0127+95.26,  433.0127+95.26)
                fixed_y132 = tf.random.uniform([2 * self.batch_num, 1, 1], -250.0+55.0, -250.0+55.0)
                fixed_z132 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser132 = tf.concat([fixed_x132, fixed_y132, fixed_z132], axis=2)
                fixed_x133 = tf.random.uniform([2*self.batch_num, 1, 1],  433.0127,  433.0127)
                fixed_y133 = tf.random.uniform([2 * self.batch_num, 1, 1], -250.0-110.0, -250.0-110.0)
                fixed_z133 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser133 = tf.concat([fixed_x133, fixed_y133, fixed_z133], axis=2)
                Xuser13 = tf.concat([Xuser131, Xuser132, Xuser133], axis=1)

                fixed_x141 = tf.random.uniform([2*self.batch_num, 1, 1],  433.0127-95.26,  433.0127-95.26)
                fixed_y141 = tf.random.uniform([2 * self.batch_num, 1, 1], 250.0+55.0, 250.0+55.0)
                fixed_z141 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser141 = tf.concat([fixed_x141, fixed_y141, fixed_z141], axis=2)
                fixed_x142 = tf.random.uniform([2*self.batch_num, 1, 1],  433.0127+95.26,  433.0127+95.26)
                fixed_y142 = tf.random.uniform([2 * self.batch_num, 1, 1], 250.0+55.0, 250.0+55.0)
                fixed_z142 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser142 = tf.concat([fixed_x142, fixed_y142, fixed_z142], axis=2)
                fixed_x143 = tf.random.uniform([2*self.batch_num, 1, 1],  433.0127,  433.0127)
                fixed_y143 = tf.random.uniform([2 * self.batch_num, 1, 1], 250.0-110.0, 250.0-110.0)
                fixed_z143 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser143 = tf.concat([fixed_x143, fixed_y143, fixed_z143], axis=2)
                Xuser14 = tf.concat([Xuser141, Xuser142, Xuser143], axis=1)

                fixed_x151 = tf.random.uniform([2*self.batch_num, 1, 1],  433.0127-95.26,  433.0127-95.26)
                fixed_y151 = tf.random.uniform([2 * self.batch_num, 1, 1], 750.0+55.0, 750.0+55.0)
                fixed_z151 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser151 = tf.concat([fixed_x151, fixed_y151, fixed_z151], axis=2)
                fixed_x152 = tf.random.uniform([2*self.batch_num, 1, 1],  433.0127+95.26,  433.0127+95.26)
                fixed_y152 = tf.random.uniform([2 * self.batch_num, 1, 1], 750.0+55.0, 750.0+55.0)
                fixed_z152 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser152 = tf.concat([fixed_x152, fixed_y152, fixed_z152], axis=2)
                fixed_x153 = tf.random.uniform([2*self.batch_num, 1, 1],  433.0127,  433.0127)
                fixed_y153 = tf.random.uniform([2 * self.batch_num, 1, 1], 750.0-110.0, 750.0-110.0)
                fixed_z153 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser153 = tf.concat([fixed_x153, fixed_y153, fixed_z153], axis=2)
                Xuser15 = tf.concat([Xuser151, Xuser152, Xuser153], axis=1)

                fixed_x161 = tf.random.uniform([2*self.batch_num, 1, 1],  866.0254-95.26,  866.0254-95.26)
                fixed_y161 = tf.random.uniform([2 * self.batch_num, 1, 1], -500.0+55.0, -500.0+55.0)
                fixed_z161 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser161 = tf.concat([fixed_x161, fixed_y161, fixed_z161], axis=2)
                fixed_x162 = tf.random.uniform([2*self.batch_num, 1, 1],  866.0254+95.26,  866.0254+95.26)
                fixed_y162 = tf.random.uniform([2 * self.batch_num, 1, 1], -500.0+55.0, -500.0+55.0)
                fixed_z162 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser162 = tf.concat([fixed_x162, fixed_y162, fixed_z162], axis=2)
                fixed_x163 = tf.random.uniform([2*self.batch_num, 1, 1],  866.0254,  866.0254)
                fixed_y163 = tf.random.uniform([2 * self.batch_num, 1, 1], -500.0-110.0, -500.0-110.0)
                fixed_z163 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser163 = tf.concat([fixed_x163, fixed_y163, fixed_z163], axis=2)
                Xuser16 = tf.concat([Xuser161, Xuser162, Xuser163], axis=1)

                fixed_x171 = tf.random.uniform([2*self.batch_num, 1, 1],  866.0254-95.26,  866.0254-95.26)
                fixed_y171 = tf.random.uniform([2 * self.batch_num, 1, 1], 0.0+55.0, 0.0+55.0)
                fixed_z171 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser171 = tf.concat([fixed_x171, fixed_y171, fixed_z171], axis=2)
                fixed_x172 = tf.random.uniform([2*self.batch_num, 1, 1],  866.0254+95.26,  866.0254+95.26)
                fixed_y172 = tf.random.uniform([2 * self.batch_num, 1, 1], 0.0+55.0, 0.0+55.0)
                fixed_z172 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser172 = tf.concat([fixed_x172, fixed_y172, fixed_z172], axis=2)
                fixed_x173 = tf.random.uniform([2*self.batch_num, 1, 1],  866.0254,  866.0254)
                fixed_y173 = tf.random.uniform([2 * self.batch_num, 1, 1], 0.0-110.0, 0.0-110.0)
                fixed_z173 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser173 = tf.concat([fixed_x173, fixed_y173, fixed_z173], axis=2)
                Xuser17 = tf.concat([Xuser171, Xuser172, Xuser173], axis=1)

                fixed_x181 = tf.random.uniform([2*self.batch_num, 1, 1],  866.0254-95.26,  866.0254-95.26)
                fixed_y181 = tf.random.uniform([2 * self.batch_num, 1, 1], 500.0+55.0, 500.0+55.0)
                fixed_z181 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser181 = tf.concat([fixed_x181, fixed_y181, fixed_z181], axis=2)
                fixed_x182 = tf.random.uniform([2*self.batch_num, 1, 1],  866.0254+95.26,  866.0254++95.26)
                fixed_y182 = tf.random.uniform([2 * self.batch_num, 1, 1], 500.0+55.0, 500.0+55.0)
                fixed_z182 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser182 = tf.concat([fixed_x182, fixed_y182, fixed_z182], axis=2)
                fixed_x183 = tf.random.uniform([2*self.batch_num, 1, 1],  866.0254,  866.0254)
                fixed_y183 = tf.random.uniform([2 * self.batch_num, 1, 1], 500.0-110.0, 500.0-110.0)
                fixed_z183 = tf.random.uniform([2 * self.batch_num, 1, 1], 1.5, 1.5)
                Xuser183 = tf.concat([fixed_x183, fixed_y183, fixed_z183], axis=2)
                Xuser18 = tf.concat([Xuser181, Xuser182, Xuser183], axis=1)


                Xuser = tf.concat([Xuser0, Xuser1, Xuser2, Xuser3, Xuser4, Xuser5, Xuser6, Xuser7, Xuser8, Xuser9, Xuser10, Xuser11, Xuser12, Xuser13, Xuser14, Xuser15, Xuser16, Xuser17, Xuser18], axis=1)

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
            #Mohamed Corr
            plt.vlines(-630, -780, 780, 'red', linewidth=8)
            plt.hlines(-630, -780, 780, 'green', linewidth=8)
            plt.hlines(630, -780, 780, 'brown', linewidth=8)
            plt.vlines(630, -780, 780, 'gray', linewidth=8)

            # #Matteo Corr
            # plt.vlines(-200, -400, 400, 'red', linewidth=8)
            # plt.vlines(-100, -400, 400, 'green', linewidth=8)
            # plt.vlines(100, -400, 400, 'brown', linewidth=8)
            # plt.vlines(200, -400, 400, 'gray', linewidth=8)

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
        # for i in range(self.Nuser_drop):
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
#D_3d,D_2d,BS_wrapped_Cord=deployment.Dist(deployment.Xap,deployment.Xuser)
#
# deployment.show_wrap_around(deployment.Xuser,BS_wrapped_Cord,savefig=True)
# # deployment.dist(deployment.Xap,deployment.Xuser,deployment.Wrap_shift)
