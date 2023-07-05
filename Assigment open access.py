def Assign_AP_OpenAccess(self, LSL):

    self.P_Tx_TN = tf.math.pow(10.0, (self.P_Tx_dB - 30.0) / 10.0)
    self.PSD = tf.math.pow(10.0, (-174.0 - 30.0) / 10.0)
    self.BW_TN = self.bandwidth
    self.NF_TN = tf.math.pow(10.0, self.noise_figure_user / 10.0)

    LSL_assign = tf.zeros([LSL.shape[0], LSL.shape[1], 1], dtype='float32')
    D_2d_assign = tf.zeros([D_2d.shape[0], D_2d.shape[1], 1], dtype='float32')
    self.user_index = np.array([])
    AP_assign_user = tf.zeros([LSL.shape[0], 1, self.Nuser_drop], dtype='float32')

    LSL_TN = LSL[:, 0:57, :] + self.P_Tx_dB
    LSL_NTN = tf.expand_dims(LSL[:, 57, :], axis=1)+ self.P_Tx_sat_db+self.sat_bias
    LSG_TN = tf.math.pow(10, (-LSL_TN) / 10)
    LSG_NTN = tf.math.pow(10, (-LSL_NTN) / 10)
    LSG=tf.concat([LSG_TN,LSG_NTN ],axis=1)
    LSG=10.0 * (tf.math.log(LSG) / tf.math.log(10.0))
    LSL=-LSG

    LSL_min_TN = tf.math.reduce_min(LSL_TN, axis=1, keepdims=True)
    LSG_min_TN = tf.math.pow(10, (-LSL_min_TN) / 10)


    snr_link = LSG_TN * self.P_Tx_TN / (self.BW_TN * self.NF_TN)
    num_TN = LSG_min_TN * self.P_Tx_TN / (self.BW_TN * self.NF_TN)
    denom_TN = tf.expand_dims(tf.reduce_sum(snr_link, axis=1),axis=1) - num_TN + self.PSD
    sinr_onlyground = num_TN / denom_TN
    sinr_onlyground = 10.0 * (tf.math.log(sinr_onlyground) / tf.math.log(10.0))

    sinr_onlyLEO = (-LSL_NTN) + self.P_over_noise_db_sat - self.noise_figure_user_sat

    # now I need to compare SINR_TN with SINR_NTN which should have a size of TensorShape([20, 1, 5700])

    TN_Assign = tf.cast(sinr_onlyground > sinr_onlyLEO, "float32")
    TN_Assign = tf.tile(TN_Assign, [1, 57, 1])
    NTN_Assign = tf.cast(sinr_onlyground <= sinr_onlyLEO, "float32")
    LSL_Assign_ref=tf.concat([TN_Assign,NTN_Assign ],axis=1)*LSL

    LSL_TN_sort = tf.math.argmin(LSL_Assign_ref[:, 0:57, :], axis=1)
    LSL_NTN_sort = tf.math.argmax(LSL_Assign_ref, axis=1)
    LSL_NTN_sort_cond =tf.cast(LSL_NTN_sort == 57 , "int64")
    LSL_NTN_sort=LSL_NTN_sort_cond*LSL_NTN_sort
    LSL_sort=LSL_TN_sort+LSL_NTN_sort

    LSL_TN_min = tf.math.reduce_min(LSL_Assign_ref[:, 0:57, :], axis=1, keepdims=True)
    LSL_NTN_min =tf.math.reduce_max(tf.expand_dims(LSL_Assign_ref[:,57,:],axis=1), axis=1, keepdims=True)
    LSL_min = LSL_TN_min+LSL_NTN_min

    d_sort = tf.reduce_sum(tf.cast(LSL == LSL_min, 'float32') * D_2d, axis=1)

    mask = tf.expand_dims(tf.range(1.0, self.Nuser_drop + 1), axis=0)
    mask = tf.tile(mask, [LSL.shape[0], 1])  # Q
    assigned_batch_index = tf.range(0, LSL.shape[0])

    if self.open_access:
        self.Nap = self.Nap + 1  # +7
    elif self.open_access == False:
        self.Nap = self.Nap

    for i in range(self.Nap):
        print(i)
        ind_ap_i = tf.cast(LSL_sort == i, 'float32') * tf.cast(d_sort > self.Dist2D_exclud, 'float32')
        valid_batch = tf.reduce_sum(ind_ap_i, axis=1)
        valid_batch = tf.squeeze(tf.where(valid_batch > 0))
        ind_ap_i = tf.gather(ind_ap_i, valid_batch, axis=0)
        LSL_sort = tf.gather(LSL_sort, valid_batch, axis=0)
        d_sort = tf.gather(d_sort, valid_batch, axis=0)
        LSL = tf.gather(LSL, valid_batch, axis=0)
        D_2d = tf.gather(D_2d, valid_batch, axis=0)

        assigned_batch_index = tf.gather(assigned_batch_index, valid_batch, axis=0)
        LSL_assign = tf.gather(LSL_assign, valid_batch, axis=0)
        D_2d_assign = tf.gather(D_2d_assign, valid_batch, axis=0)

        mask = tf.gather(mask, valid_batch, axis=0)

        AP_assign_user = tf.gather(AP_assign_user, valid_batch, axis=0)

        mask_i = mask * ind_ap_i
        mask_i = tf.transpose(tf.random.shuffle(tf.transpose(mask_i, [1, 0])), [1, 0])

        ap_assign_user = tf.reduce_max(mask * tf.cast(mask_i > 0.0, "float32"), axis=1, keepdims=True) - 1
        ap_assign_user = tf.gather_nd(mask_i, tf.concat([tf.expand_dims(tf.constant(range(mask_i.shape[0])), axis=1), tf.cast(ap_assign_user, "int32")], axis=1))
        self.user_index = np.append(self.user_index, ap_assign_user.numpy() / self.Nuser_drop)

        mask_ap = tf.expand_dims(tf.scatter_nd(tf.concat([tf.expand_dims(tf.constant(range(mask.shape[0])), axis=1),tf.cast(tf.expand_dims(ap_assign_user, axis=1), "int32") - 1],axis=1),tf.ones(mask.shape[0]), [mask.shape[0], self.Nuser_drop]), axis=1)

        LSL_selected_user = tf.reduce_sum(LSL * mask_ap, axis=2, keepdims=True)
        D_2d_selected_user = tf.reduce_sum(mask_ap * D_2d, axis=2, keepdims=True)

        LSL_assign = tf.concat([LSL_assign, LSL_selected_user], axis=2)
        D_2d_assign = tf.concat([D_2d_assign, D_2d_selected_user], axis=2)

        AP_assign_user = tf.concat([AP_assign_user, mask_ap], axis=1)

    LSL_assign = LSL_assign[0:self.batch_num, :,1:]
    D_2d_assign = D_2d_assign[0:self.batch_num, :, 1:]

    if self.indoor_calib:
        self.LSL_calib_assign = LSL_assign
        LSL_assign = LSL_org_assign

    AP_assign_user = AP_assign_user[0:self.batch_num, 1:, :]
    assigned_batch_index = assigned_batch_index[0:self.batch_num]

    return LSL_assign,D_2d_assign


    # from plot_class import Plot
    # import numpy as np
    # import matplotlib.pyplot as plt
    # plot = Plot()
    # plot.cdfplot([sinr_onlyground.numpy()], xlabel="Received Signal Strength [dBm]", lines_label=["TN"])
    #
    # plot = Plot()
    # plot.cdfplot([sinr_onlyLEO.numpy()], xlabel="Received Signal Strength [dBm]", lines_label=["TN"])