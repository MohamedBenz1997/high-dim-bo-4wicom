# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 14:18:36 2021

@author: Benzo
"""
import tensorflow as tf
from config import Config
import math


class DeployPPP(Config):
    def __init__(self):
        Config.__init__(self)

    def call(self, beta_open_loop=1):
        # self.batch_num= batch_num       #Assigning number of batches based on the number of iterations that we want. This will be entered to the function from the runner script.
        batch_num = self.batch_num * 2  # Assigning the number of iterations to power 2 to ensure convergence. Becaause there are some iterations will be ignored if at least 1 BS does not have a UE associated with it.
        x0 = tf.random.uniform([batch_num, self.Nuser_drop, 1], 0,
                               self.EX)  # Assigning a random values tensor for UEs x-axis which have the size of number of batches X number of UE. Having a limit of the x-axis grid defined previously as EX
        y0 = tf.random.uniform([batch_num, self.Nuser_drop, 1], 0,
                               self.EY)  # Same logic as x-axis coordinate, this time for y.
        z0 = self.Zuser + tf.zeros([batch_num, self.Nuser_drop, 1],
                                   dtype='float32')  # For z-axis it is not random coordinates because we are defining the height of UEs at the beggining. We will sum that value to a zero value tensor with size desired, batches X number of UE
        Xuser = tf.concat([x0, y0, z0],
                          axis=2)  # The 3D coordinates of each user will be the combination of x,y,z coordinates defined in the previous 3 steps.
        # The same logic of assigning coordinates to UEs is done here for BSs
        x = tf.random.uniform([batch_num, self.Nap, 1], 0, self.EX)
        y = tf.random.uniform([batch_num, self.Nap, 1], 0, self.EY)
        z = self.Zap + tf.zeros([batch_num, self.Nap, 1], dtype='float32')
        Xap = tf.concat([x, y, z], axis=2)

        # D, D_2d, BS_wrapped_Cord = self.Dist(Xap, Xuser, self.EX,
        #                                      self.EY)  # This is the 3D and 2D distances between UEs and BSs calculated in the Dist function below. (all UEs from all patches are considered)
        return Xap, Xuser

    # Calculating the 2D and 3D distances between UEs and BSs
    # Note: X1 is BSs coordinates, X2 is UEs coordinates
    # Q
    def Dist(self, X1, X2):
        EX = self.EX
        EY = self.EY
        N1 = X1.shape[1]
        N2 = X2.shape[1]
        # ----------The pair distances
        xvec1 = tf.expand_dims(X1[:, :, 0], axis=2)
        xvec2 = tf.expand_dims(X2[:, :, 0], axis=2)
        xmat1 = tf.tile(xvec1, [1, 1, N2])
        xmat2 = tf.tile(tf.transpose(xvec2, perm=[0, 2, 1]), [1, N1, 1])
        xdiff = xmat1 - xmat2
        xdist2 = tf.pow(tf.math.minimum(tf.math.abs(xdiff), EX - tf.math.abs(xdiff)), 2)

        cond_x1 = tf.cast(tf.math.abs(xdiff) <= EX - tf.math.abs(xdiff), "float32")
        x_coord_new1 = xmat1 * cond_x1

        cond_x2a = tf.cast(tf.math.abs(xdiff) > EX - tf.math.abs(xdiff), "float32") * tf.cast(xmat2 >= xmat1, "float32")
        x_coord_new2 = (xmat2 + (EX - tf.math.abs(xdiff))) * cond_x2a

        cond_x2b = tf.cast(tf.math.abs(xdiff) > EX - tf.math.abs(xdiff), "float32") * tf.cast(xmat2 < xmat1, "float32")
        x_coord_new3 = (xmat2 - (EX - tf.math.abs(xdiff))) * cond_x2b

        x_coord_new = x_coord_new1 + x_coord_new2 + x_coord_new3
        x_coord_new = tf.expand_dims(x_coord_new, axis=3)

        yvec1 = tf.expand_dims(X1[:, :, 1], axis=2)
        yvec2 = tf.expand_dims(X2[:, :, 1], axis=2)
        ymat1 = tf.tile(yvec1, [1, 1, N2])
        ymat2 = tf.tile(tf.transpose(yvec2, perm=[0, 2, 1]), [1, N1, 1])
        ydiff = ymat1 - ymat2
        ydist2 = tf.pow(tf.minimum(tf.math.abs(ydiff), EY - tf.math.abs(ydiff)), 2)

        cond_y1 = tf.cast(tf.math.abs(ydiff) <= EY - tf.math.abs(ydiff), "float32")
        y_coord_new1 = ymat1 * cond_y1

        cond_y2a = tf.cast(tf.math.abs(ydiff) > EY - tf.math.abs(ydiff), "float32") * tf.cast(ymat2 >= ymat1, "float32")
        y_coord_new2 = (ymat2 + (EY - tf.math.abs(ydiff))) * cond_y2a

        cond_y2b = tf.cast(tf.math.abs(ydiff) > EX - tf.math.abs(ydiff), "float32") * tf.cast(ymat2 < ymat1, "float32")
        y_coord_new3 = (ymat2 - (EY - tf.math.abs(ydiff))) * cond_y2b

        y_coord_new = y_coord_new1 + y_coord_new2 + y_coord_new3

        y_coord_new = tf.expand_dims(y_coord_new, axis=3)

        zvec1 = tf.expand_dims(X1[:, :, 2], axis=2)
        zvec2 = tf.expand_dims(X2[:, :, 2], axis=2)
        zmat1 = tf.tile(zvec1, [1, 1, N2]);
        zmat2 = tf.tile(tf.transpose(zvec2, perm=[0, 2, 1]), [1, N1, 1]);
        zdiff = zmat1 - zmat2
        zdist2 = tf.pow(zdiff, 2)

        z_coord_new = tf.expand_dims(zmat1, axis=3)

        BS_wrapped_Cord = tf.concat([x_coord_new, y_coord_new, z_coord_new], axis=3)

        D = tf.math.sqrt(xdist2 + ydist2 + zdist2)  # 3D distance
        D_2d = tf.math.sqrt(xdist2 + ydist2)  # 2D distance

        return D, D_2d, BS_wrapped_Cord