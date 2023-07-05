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
from NTN_LSG_Class import NTN_Large_Scale_Gain
from plot_class import Plot
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

class DeployHex(Config):
    def __init__(self):
        Config.__init__(self)
        # Deploy cells using https://www.redblobgames.com/grids/hexagons/#map-storage lib
        # Define the hex objects for the first tire

        self.N= self.hex_N_sat
        # self.size = self.radius_sat
        self.radius = self.radius_sat
        self.side = 2*self.radius/np.sqrt(3.0)
        orgin_hex = Hex(0,0,0)
        # T0 = []
        t0=[[]]*(self.N+1)
        for i in range(2 * self.N+1):
            for j in range(2 * self.N+1):
                for k in range(2 * self.N+1):
                    if (i - self.N) + (j - self.N) + (k - self.N) == 0:
                        h = Hex(i - self.N, j - self.N, k - self.N)
                        # T0.append(h)
                        dist = hex_distance(h, orgin_hex)
                        # print(dist)
                        t_ring =t0[dist].copy()
                        t_ring.append(h)
                        t0[dist]=t_ring
        T0 =[]
        ring_ind =[[]]*(self.N+1)
        for i in range(self.N+1):
            ring_ind[i] =np.array(range(len(t0[i])))+len(T0)
            T0 = T0 + t0[i]
        # t00 = [[]] * (self.N + 1)
        # t00[0] = orgin_hex
        # for i in range(1,self.N+1):
        #     repeat =True
        #     hex_up =hex_subtract(Hex(0, i, -i),orgin_hex)
        #     t_ring = list(t00[i])
        #     t_ring.append(hex_up)
        #     hex_rotate = hex_up
        #     while repeat:
        #         hex_rotate = hex_rotate_left(hex_rotate)
        #         dist = hex_distance(hex_rotate, hex_up)
        #         if dist==0:
        #             repeat=False
        #         else:
        #             t_ring.append(hex_rotate)
        #     t00[i] =t_ring
        self.ring_ind = ring_ind
        self.T0 =T0
        self.t0 =t0 # hex in each tire
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
        # center_cell_id = int((len(T0)-1)/2)
        corners_T0_center = polygon_corners(layout, T0[0])
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
        batch_num = int(self.batch_num*1.5) # Assigning the number of iterations to power 2 to ensure convergence. Becaause there are some iterations will be ignored if at least 1 BS does not have a UE associated with it.
        x0 = tf.random.uniform([batch_num, self.Nuser_drop_sat, 1], -self.side,
                               self.side)  # Assigning a random values tensor for UEs x-axis which have the size of number of batches X number of UE. Having a limit of the x-axis grid defined previously as EX
        y0 = tf.random.uniform([batch_num, self.Nuser_drop_sat, 1], -self.radius,
                               self.radius)  # Same logic as x-axis coordinate, this time for y.
        z0 = self.Zuser + tf.zeros([batch_num, self.Nuser_drop_sat, 1],
                                   dtype='float32')  # For z-axis it is not random coordinates because we are defining the height of UEs at the beggining. We will sum that value to a zero value tensor with size desired, batches X number of UE
        # Xuser = tf.concat([x0, y0, z0], axis=2)  # The 3D coordinat

        check_in_hex = (y0-self.corner_lines[0]*x0-self.corner_lines[1])*self.lines_sign>=0
        check_in_hex = tf.cast(tf.reduce_sum(tf.cast(check_in_hex,'float32'),axis=2)== self.corners_T0_center.shape[0],"float32")
        x0_vec = tf.reshape(x0,[-1,1])
        y0_vec = tf.reshape(y0,[-1,1])
        z0_vec = tf.reshape(z0,[-1,1])
        check_in_hex = tf.reshape(check_in_hex,[-1])
        valid_user_id = tf.squeeze(tf.where(check_in_hex ==1))

        valid_user_id = valid_user_id[0:self.batch_num*self.Nuser_drop_sat]
        # get valid users
        x0_vec = tf.gather(x0_vec,valid_user_id,axis=0)
        y0_vec = tf.gather(y0_vec,valid_user_id,axis=0)
        z0_vec = tf.gather(z0_vec, valid_user_id, axis=0)

        x0 = tf.reshape(x0_vec,[self.batch_num, self.Nuser_drop_sat,1])
        y0 = tf.reshape(y0_vec, [self.batch_num, self.Nuser_drop_sat,1])
        z0 = tf.reshape(z0_vec, [self.batch_num, self.Nuser_drop_sat,1])
        Xuser = tf.concat([x0, y0, z0], axis=2)  # The 3D coordinat


        # # move user_drop_per_cell users to each cell randomly
        Xap = self.T0_center_pos
        Zap= tf.tile(tf.constant([[self.Zap]]),[len(Xap),1])
        Xap = tf.concat([Xap,Zap],axis=1)
        ind_ap_rand = tf.random.uniform([self.batch_num*self.Nuser_drop_sat],0,self.Nap,dtype='int32')
        ind_ap_rand = ind_ap_rand+self.Nleo *tf.constant([i for i in range(ind_ap_rand.shape[0])],dtype="int32")
        Xap_vec = tf.gather(tf.tile(Xap,[ind_ap_rand.shape[0],1]),ind_ap_rand,axis=0)
        # # Xuser = Xuser+tf.expand_dims(Xap,axis=2)
        # # Xuser = tf.reshape(Xuser,[2*self.batch_num,self.Nap,self.user_drop_per_cell,3])
        Xap_vec_fix_z = tf.concat([Xap_vec[:,0:2],tf.zeros([Xap_vec.shape[0],1])],axis=1)
        Xuser = tf.reshape(Xuser,[-1,3])+Xap_vec_fix_z
        Xuser = tf.reshape(Xuser,[-1,self.Nuser_drop_sat,3])

        Xap = tf.tile(tf.expand_dims(Xap,axis=0),[self.batch_num,1,1])
        self.Xuser = Xuser
        self.Xap = Xap
        return Xap,Xuser
    def Dist(self,Xap,Xuser):
        # Xap = tf.expand_dims(Xap,axis=3)
        # Compute the distance between each userBS and their images
        xdiff = tf.expand_dims(Xap,axis=2)-tf.expand_dims(Xuser,axis=1)
        D = tf.sqrt(tf.reduce_sum(tf.math.square(xdiff),axis=3))
        D_2d = tf.sqrt(tf.reduce_sum(tf.math.square(xdiff[:,:,:,0:2]),axis=3))
        # D_ind = tf.argmin(D,axis=3)
        # D_3d = tf.reduce_min(D,axis=3)
        # D_2d = tf.reduce_min(D_2d,axis=3)
        # BS_wrapped_Cord = tf.expand_dims(tf.cast(D==tf.expand_dims(D_3d,axis=3),'float32'),axis=3)
        # BS_wrapped_Cord= BS_wrapped_Cord *tf.expand_dims(Xap_wrap,axis=2)
        # BS_wrapped_Cord = tf.reduce_sum(BS_wrapped_Cord,axis=4)
        return D_2d

    def assign_beam(self,power_received_linear):
        ring_ind = np.append(self.ring_ind[0],self.ring_ind[1])
        # ring_ind = np.append(ring_ind,self.ring_ind[2])
        beam_assigned  = tf.cast(tf.expand_dims(tf.math.argmax(power_received_linear, axis=1),axis=1),"float32")
        #keep the useres in the first two rings
        beam_assigned = tf.cast(beam_assigned,"float32")*tf.cast(beam_assigned<ring_ind.shape[0] ,"float32")+\
                       -tf.cast(beam_assigned>ring_ind.shape[0]-1 ,"float32")

        beam_mask = tf.expand_dims(tf.constant([range(len(self.T0))],"float32"),axis=2)
        beam_mask = tf.cast(beam_mask==beam_assigned,"float32")
        signal_mask = beam_mask
        signal = tf.gather(power_received_linear,ring_ind,axis=1)*tf.gather(signal_mask,ring_ind,axis=1)

        signal = tf.reduce_sum(signal,axis=1)
        I_mask = tf.expand_dims(tf.cast(signal>0,"float32"),axis=1)



        # SIR_mask = 1-beam_mask
        I= tf.reduce_sum(power_received_linear*I_mask,axis=1)-signal

        signal = signal.numpy()
        I = I.numpy()

        signal = signal[I!=0]
        I = I[I!=0]
        SIR = signal / I
        SIR_db =  10*tf.math.log(SIR)/tf.math.log(10.0)
        # ring_ind =self.ring_ind[0]
        SNR = signal/self.P_over_noise

        SNR_db = 10*tf.math.log(SNR)/tf.math.log(10.0)
        signal_db = 10*tf.math.log(signal)/tf.math.log(10.0)
        I_db = 10*tf.math.log(I)/tf.math.log(10.0)
        SINR = tf.pow(10,(signal_db + self.P_over_noise_db_sat-self.noise_figure_user)/10)/\
               (tf.pow(10.0,(I_db+self.P_over_noise_db_sat-self.noise_figure_user)/10.0)+1)
        SINR_db = 10*tf.math.log(SINR)/tf.math.log(10.0)
        # for beam in ring_ind:
        #     is_beam_selected =  beam_assigned == beam
        #     SNR = (snr_downlink_full[:,beam,:]* tf.cast(is_beam_selected,"float32")).numpy()
        #     SNR = SNR[SNR!= 0]
        #     SNR_vec = np.append(SNR_vec,SNR)
        # # #SIR
        # # SIR_vec = np.array([])
        # # SIR_vec = (snr_downlink_full[:, beam, :] * (1 - tf.cast(is_beam_selected, "float32"))).numpy()
        return SNR_db.numpy(),SIR_db.numpy(),SINR_db.numpy(),I_db.numpy(),signal_db.numpy()



        # Xap_wrap = tf.expand_dims(tf.Xap_wrap,axis=2)+
        # return D,D_2d,
    def plot_hex(self):
        Xap = self.Nleo
        Xuser_assigned = self.Xuser
        T0_center_pos = [0] * len(self.T0)
        fig, ax = plt.subplots(1, 1)
        for i in range(len(self.T0)):
            T0_center_pos[i] = hex_to_pixel(self.layout, self.T0[i])
            ax.plot(T0_center_pos[i].x, T0_center_pos[i].y, "ko")
            ax.text(T0_center_pos[i].x, T0_center_pos[i].y, str(i))
            corners = polygon_corners(self.layout, self.T0[i])
            x, y = get_corners(corners)
            ax.plot(x, y, 'cyan')
        for i in range(self.Nuser_drop_sat):
            if i == 0:
                # ax.plot(Xap[0, i, 0], Xap[0, i, 1], 'x', color='brown', label="BS")
                # ax.text(Xap[0, i, 0], Xap[0, i, 1], str(i), color='brown', label="BS")
                ax.plot(Xuser_assigned[0, i, 0], Xuser_assigned[0, i, 1], 'o', color='blue', label="User")
                ax.text(Xuser_assigned[0, i, 0], Xuser_assigned[0, i, 1], str(i), color='blue', label="User")
            else:
                # ax.plot(Xap[0, i, 0], Xap[0, i, 1], 'x', color='brown')
                # ax.text(Xap[0, i, 0], Xap[0, i, 1], str(i), color='brown')
                ax.plot(Xuser_assigned[0, i, 0], Xuser_assigned[0, i, 1], 'o', color='blue')
                ax.text(Xuser_assigned[0, i, 0], Xuser_assigned[0, i, 1], str(i), color='blue')
            # ax.plot([Xuser_assigned[0, i, 0], Xap[0, i, 0]], [Xuser_assigned[0, 3*i, 1], Xap[0, i, 1]], color='red')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend()
        plt.show()
        pass
    # def plot_wrap(self,Xuser,BS_wrapped_Cord):


#
# #test the class
# config = Config()
deployment = DeployHex()
Xap,Xuser=deployment.call()
D_2d = deployment.Dist(Xap,Xuser)
deployment.plot_hex()
sat = NTN_Large_Scale_Gain()
NTN_LSL,NTN_LSG,NTN_AntennaGain_dB,NTN_LOS,NTN_PL,NTN_shadowing_LOS,NTN_shadowing_NLOS=sat.call(D_2d)
power_received_linear = tf.math.pow(10.0,NTN_LSG/10+deployment.P_Tx_sat_db/10)
snr_downlink_full=(power_received_linear/deployment.P_over_noise)

################
SNR_assigned,SIR_assigned,SINR_assigned,I_assigned,signal_assigned= deployment.assign_beam(power_received_linear)

# sinr =10*tf.math.log(snr_downlink_full[:,3,:]/( tf.reduce_sum(snr_downlink_full,axis=1)-snr_downlink_full[:,3,:]+1))/tf.math.log(10.0)
# interference = 10*tf.math.log( tf.reduce_sum(snr_downlink_full,axis=1)-snr_downlink_full[:,3,:])/tf.math.log(10.0)
# snr = 10*tf.math.log(snr_downlink_full[:,3,:])/tf.math.log(10.0)
interference =  tf.reduce_sum(snr_downlink_full,axis=1)-snr_downlink_full[:,0,:]
interference_db = 10*tf.math.log(interference)/tf.math.log(10.0)
sir =10*tf.math.log(snr_downlink_full[:,0,:]/(interference))/tf.math.log(10.0)
snr = 10*tf.math.log(snr_downlink_full[:,0,:])/tf.math.log(10.0)

# interference =  tf.reduce_sum(tf.gather(snr_downlink_full,[1,7,11,17],axis=1),axis=1)
# interference_db = 10*tf.math.log(interference)/tf.math.log(10.0)
# sir =10*tf.math.log(snr_downlink_full[:,9,:]/(interference))/tf.math.log(10.0)
#
# snr = 10*tf.math.log(snr_downlink_full[:,9,:])/tf.math.log(10.0)


plot = Plot()
plot.cdfplot([signal_assigned,I_assigned],"power (dB)",["signal","I"])
plot.cdfplot([SINR_assigned,SIR_assigned],"dB",["SINR","SIR"])
# plot.cdfplot([-NTN_LSG.numpy()[:,[0],:]],"dB",["Loss"])
# deployment.dist(deployment.Xap,deployment.Xuser,deployment.Wrap_shift)
_,_,_,_,LSG_assigned= deployment.assign_beam(tf.math.pow(10.0,NTN_LSG/10))
plot.cdfplot([-LSG_assigned],"dB",["Loss"])
