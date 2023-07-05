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

import matplotlib.pyplot as plt
import os


class DeployHex(Config):
    def __init__(self):
        Config.__init__(self)
        # Deploy cells using https://www.redblobgames.com/grids/hexagons/#map-storage lib
        # Define the hex objects for the first tire

        self.side = 2 * self.radius / np.sqrt(3.0)
        T0 = []
        for i in range(2 * self.N + 1):
            for j in range(2 * self.N + 1):
                for k in range(2 * self.N + 1):
                    if (i - self.N) + (j - self.N) + (k - self.N) == 0:
                        h = Hex(i - self.N, j - self.N, k - self.N)
                        T0.append(h)
        self.T0 = T0
        #################################################################
        # Find the hex center for one of the hex-group in the second tire
        T0_center_pos = [0] * len(T0)
        orgin = Point(0.0, 0.0)
        # radius = 250
        size = Point(2 / math.sqrt(3.0) * self.radius, 2 / math.sqrt(3.0) * self.radius)
        layout = Layout(layout_flat, size, orgin)
        self.layout = layout
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
        center_cell_id = int((len(T0) - 1) / 2)
        corners_T0_center = polygon_corners(layout, T0[center_cell_id])
        corners = np.zeros([len(corners_T0_center), 2], 'float32')
        corners[:, 0] = [corners_T0_center[i].x for i in range(len(corners_T0_center))]
        corners[:, 1] = [corners_T0_center[i].y for i in range(len(corners_T0_center))]
        self.corners_T0_center = corners
        # construct the equations for the sides of the hex in the form of y = ax+b
        a = (np.roll(self.corners_T0_center[:, 1], -1) + 1e-2 - self.corners_T0_center[:, 1]) / (
                    1e-15 + np.roll(self.corners_T0_center[:, 0], -1) - self.corners_T0_center[:, 0])
        b = self.corners_T0_center[:, 1] - a * self.corners_T0_center[:, 0]
        self.corner_lines = [np.expand_dims(np.expand_dims(a, axis=0), axis=0),
                             np.expand_dims(np.expand_dims(b, axis=0), axis=0)]
        self.lines_sign = np.array([[[1, 1, 1, -1, -1, -1]]])
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
        wrap_shift_z = [0] * len(pos_miror)
        self.Wrap_shift = tf.constant([[[wrap_shift_x, wrap_shift_y, wrap_shift_z]]])
        self.Wrap_shift = tf.concat([self.Wrap_shift, tf.constant([[[[0.0], [0.0], [0.0]]]])], axis=3)
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
        batch_num = self.batch_num * 3  # Assigning the number of iterations to power 2 to ensure convergence. Becaause there are some iterations will be ignored if at least 1 BS does not have a UE associated with it.
        x0 = tf.random.uniform([batch_num, self.Ndrone_drop, 1], -self.side,
                               self.side)  # Assigning a random values tensor for UEs x-axis which have the size of number of batches X number of UE. Having a limit of the x-axis grid defined previously as EX
        y0 = tf.random.uniform([batch_num, self.Ndrone_drop, 1], -self.radius,
                               self.radius)  # Same logic as x-axis coordinate, this time for y.
        z0 = tf.random.uniform([batch_num, self.Ndrone_drop, 1], self.z_drone_min,
                               self.z_drone_max)   # For z-axis it is not random coordinates because we are defining the height of UEs at the beggining. We will sum that value to a zero value tensor with size desired, batches X number of UE
        # Xuser = tf.concat([x0, y0, z0], axis=2)  # The 3D coordinat

        check_in_hex = (y0 - self.corner_lines[0] * x0 - self.corner_lines[1]) * self.lines_sign >= 0
        check_in_hex = tf.cast(
            tf.reduce_sum(tf.cast(check_in_hex, 'float32'), axis=2) == self.corners_T0_center.shape[0], "float32")
        x0_vec = tf.reshape(x0, [-1, 1])
        y0_vec = tf.reshape(y0, [-1, 1])
        z0_vec = tf.reshape(z0, [-1, 1])
        check_in_hex = tf.reshape(check_in_hex, [-1])
        valid_user_id = tf.squeeze(tf.where(check_in_hex == 1))

        valid_user_id = valid_user_id[0:2 * self.batch_num * self.Ndrone_drop]
        # get valid users
        x0_vec = tf.gather(x0_vec, valid_user_id, axis=0)
        y0_vec = tf.gather(y0_vec, valid_user_id, axis=0)
        z0_vec = tf.gather(z0_vec, valid_user_id, axis=0)

        x0 = tf.reshape(x0_vec, [2 * self.batch_num, self.Nuser_drop, 1])
        y0 = tf.reshape(y0_vec, [2 * self.batch_num, self.Nuser_drop, 1])
        z0 = tf.reshape(z0_vec, [2 * self.batch_num, self.Nuser_drop, 1])
        Xuser = tf.concat([x0, y0, z0], axis=2)  # The 3D coordinat

        # move user_drop_per_cell users to each cell randomly
        Xap = self.T0_center_pos
        Zap = tf.tile(tf.constant([[self.Zap]]), [self.Nap, 1])
        Xap = tf.concat([Xap, Zap], axis=1)
        ind_ap_rand = tf.random.uniform([2 * self.batch_num * self.Ndrone_drop], 0, self.Nap, dtype='int32')
        ind_ap_rand = ind_ap_rand + self.Nap * tf.constant([i for i in range(ind_ap_rand.shape[0])], dtype="int32")
        Xap_vec = tf.gather(tf.tile(Xap, [ind_ap_rand.shape[0], 1]), ind_ap_rand, axis=0)
        # Xuser = Xuser+tf.expand_dims(Xap,axis=2)
        # Xuser = tf.reshape(Xuser,[2*self.batch_num,self.Nap,self.user_drop_per_cell,3])
        Xap_vec_fixedZ = tf.concat([Xap_vec[:, 0:2], tf.zeros([Xap_vec.shape[0], 1])], axis=1)
        Xuser = tf.reshape(Xuser, [-1, 3]) + Xap_vec_fixedZ
        Xuser = tf.reshape(Xuser, [-1, self.Ndrone_drop, 3])
        Xap = tf.tile(tf.expand_dims(Xap, axis=0), [2 * self.batch_num, 1, 1])
        self.Xuser = Xuser
        self.Xap = Xap
        return Xap, Xuser

    def Dist(self, Xap, Xuser):
        Xap_wrap = tf.expand_dims(Xap, axis=3) + self.Wrap_shift
        # Compute the distance between each userBS and their images
        xdiff = tf.expand_dims(Xap_wrap, axis=2) - tf.expand_dims(tf.expand_dims(Xuser, axis=1), axis=4)
        D = tf.sqrt(tf.reduce_sum(tf.math.square(xdiff), axis=3))
        D_2d = tf.sqrt(tf.reduce_sum(tf.math.square(xdiff[:, :, :, 0:2, :]), axis=3))
        # D_ind = tf.argmin(D,axis=3)
        D_3d = tf.reduce_min(D, axis=3)
        ind_min = tf.cast(D == tf.expand_dims(D_3d, axis=3), 'float32')
        # D_2d = tf.reduce_sum(D_2d*ind_min,axis=3)
        D_2d = tf.reduce_min(D_2d, axis=3)
        BS_wrapped_Cord = tf.expand_dims(ind_min, axis=3)
        BS_wrapped_Cord = BS_wrapped_Cord * tf.expand_dims(Xap_wrap, axis=2)
        BS_wrapped_Cord = tf.reduce_sum(BS_wrapped_Cord, axis=4)
        return D_3d, D_2d, BS_wrapped_Cord

        # Xap_wrap = tf.expand_dims(tf.Xap_wrap,axis=2)+
        # return D,D_2d,

    def plot_hex(self):
        Xap = self.Xap
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
        for i in range(self.Nuser_drop):
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

    def show_wrap_around(self, Xuser, BS_wrapped_Cord, savefig=False):
        colors = ["red", "blue", "green", "orange", "purple", 'brown', "black"]
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
                ax.plot(Ti_center_j_pos.x, Ti_center_j_pos.y, c="gray", alpha=0.05, marker="o")
                ax.text(Ti_center_j_pos.x, Ti_center_j_pos.y, str(j))
                corners = polygon_corners(layout, Ti_center_j)
                x, y = get_corners(corners)
                if i == 6:
                    ax.plot(x, y, "blue", alpha=0.5)
                else:
                    ax.plot(x, y, "gray", alpha=0.3)
        user_id = 3
        BS_wrapped_Cord_test_user = BS_wrapped_Cord[0, :, user_id, :]
        for i in range(self.Nap):
            if i == 0:
                plt.plot(BS_wrapped_Cord_test_user[i, 0], BS_wrapped_Cord_test_user[i, 1], '*', color='red',
                         label="Wrap AP")
            else:
                plt.plot(BS_wrapped_Cord_test_user[i, 0], BS_wrapped_Cord_test_user[i, 1], '*', color='red')
        plt.plot(Xuser[0, user_id, 0], Xuser[0, user_id, 1], 'o', color="green", label="User")
        ax.legend()
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        plt.title("Wraparound model from a user viewpoint")
        dir = os.getcwd()
        if savefig:
            plt.savefig(dir + '/results/wrap_around.pdf', bbox_inches='tight')
        plt.show()
        pass
    # def plot_wrap(self,Xuser,BS_wrapped_Cord):

#
# #test the class
# config = Config()
# deployment = DeployHex()
# deployment.call()
# deployment.plot_hex()
# D_3d,D_2d,BS_wrapped_Cord=deployment.Dist(deployment.Xap,deployment.Xuser)
#
# deployment.show_wrap_around(deployment.Xuser,BS_wrapped_Cord,savefig=True)
# # deployment.dist(deployment.Xap,deployment.Xuser,deployment.Wrap_shift)
