from VSBO_utils import *

from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

#My simulator import
import tensorflow as tf
from TerrestrialClass import Terrestrial
from SinrClass import SINR
from config import Config
from plot_class import Plot
SINR = SINR()
config = Config()
plot = Plot()

with localconverter(ro.default_converter + pandas2ri.converter):
    tmvtnorm = importr('tmvtnorm')

rpy2.robjects.numpy2ri.activate()

sf_stop = 10.0


class BOtorch(object):
    def __init__(self, input_dim, bounds, data_size, tilts_vector, HPBW_v_vector, Ptx_thresholds_vector, obj_vector, *args, **kwargs):
        # self.X = X
        self.X_dim = input_dim
        # self.Y = Y
        self.bound = bounds
        self.obj_info = {}

        self.data_size =data_size
        self.tilts_vector = tilts_vector
        self.HPBW_v_vector = HPBW_v_vector
        self.Ptx_thresholds_vector = Ptx_thresholds_vector
        self.obj_vector = obj_vector

    def data_initialize(self,):

        train_x = self.tilts_vector[:, :, 0]
        train_x = torch.from_numpy(train_x.numpy()).double()
        train_obj = self.obj_vector

        for j in range(self.data_size):
            ##Setting Random tilts for all BSs creating a data set
            BS_tilt = tf.random.uniform(self.tilts_vector.shape, -20, 0)
            ########################################################
            ##Different sets of samples in the initial data-set
            ########################################################
            # if j >= 0 and j <= 24:
            #     BS_tilt = tf.random.uniform(tilts_vector.shape, -10.0, -5.0)
            # elif j >= 25 and j <= 49:
            #     BS_tilt = tf.random.uniform(tilts_vector.shape, 10.0, 10.0)
            # elif j >= 50 and j <= 74:
            #     BS_tilt = tf.random.uniform(tilts_vector.shape, -10.0, 10.0)
            # elif j >= 75 and j <= 100:
            #     BS_tilt = tf.random.uniform(tilts_vector.shape, -10.0, 10.0)
            #     # Define the excluded range
            #     excluded_range = tf.constant([-5.0, 5.0])  # -10,10
            #     # Mask out values within the excluded range
            #     condition = tf.logical_and(BS_tilt >= excluded_range[0], BS_tilt <= excluded_range[1])
            #     replacement_values = tf.random.uniform(tf.shape(BS_tilt), -10.0, -5.0)
            #     BS_tilt = tf.where(condition, replacement_values, BS_tilt)
            ########################################################

            new_train_x = torch.from_numpy(BS_tilt[:, :, 0].numpy()).double()

            # BS_tilt = self.tilts_vector  # This is for getting the SINR for the opt thresholds after finishing
            BS_tilt = tf.tile(BS_tilt, [2 * config.batch_num, 1, config.Nuser_drop])

            BS_HPBW_v = self.HPBW_v_vector  # This is for getting the SINR for the opt thresholds after finishing
            BS_HPBW_v = tf.tile(BS_HPBW_v, [2 * config.batch_num, 1, config.Nuser_drop])

            P_Tx_TN = self.Ptx_thresholds_vector  # This is for getting the SINR for the opt thresholds after finishing
            P_Tx_TN = tf.tile(P_Tx_TN, [2 * config.batch_num, 1, config.Nuser_drop])

            # Run the simulator
            data = Terrestrial()
            data.alpha_factor = 0.0  # LEO at 90deg
            data.BS_tilt = BS_tilt
            data.BS_HPBW_v = BS_HPBW_v
            data.call()
            # Import of the UAVs and GUEs LSG and SINR data
            Xuser_GUEs = data.Xuser_GUEs
            Xuser_UAVs = data.Xuser_UAVs
            LSG_UAVs_Corridors = data.LSG_UAVs_Corridors
            LSG_GUEs = data.LSG_GUEs
            sinr_TN_UAVs_Corridors = SINR.sinr_TN(LSG_UAVs_Corridors, P_Tx_TN)
            sinr_TN_GUEs = SINR.sinr_TN(LSG_GUEs, P_Tx_TN)

            # BO objective: Sum of log of the RSS
            SINR_sumOftheLog_Obj, Rate_sumOftheLog_Obj, sinr_total_UAVs, sinr_total_GUEs = SINR.BO_Multi_Obj_Cooridor(
                sinr_TN_UAVs_Corridors, sinr_TN_GUEs, alpha=0.0)
            Rate_sumOftheLog_Obj, Rate_GUEs, Rate_UAVs = SINR.BO_Obj_Rates_and_Outage(LSG_GUEs, LSG_UAVs_Corridors,
                                                                                      P_Tx_TN,
                                                                                      alpha=0.0)
            Rate_obj = Rate_sumOftheLog_Obj[0].__float__()
            new_obj = torch.tensor([[Rate_obj]], dtype=torch.double)

            ## Serving BSs indexes and UAVs locations
            # BSs_id_UAVs, Xuser_UAVs_x, Xuser_UAVs_y = SINR.Cell_id(LSG_UAVs_Corridors, Xuser_UAVs)
            # BSs_id_GUEs, Xuser_GUEs_x, Xuser_GUEs_y = SINR.Cell_id(LSG_GUEs, Xuser_GUEs)

            # Append the thresholds and objectives
            train_x = torch.cat((train_x, new_train_x), dim=0)
            train_obj = torch.cat((train_obj, new_obj), dim=0)

        # Save the torch tensors to a file with .pt extension to be loaded using python later
        # file_name = "2023_09_25_Corr_SAforSB_SAonly_Down.pt"
        # torch.save({"train_x": train_x, "train_obj": train_obj}, file_name)

        self.X = train_x
        self.Y = train_obj

    @catch_error
    def fit_model(self, X, bounds, model_class, rand_init=0, **kwargs):
        X_normalize = normalize(X, bounds=bounds)
        self.Y_standard = standardize(self.Y)
        GP_model = model_class(X_normalize, self.Y_standard, bounds=bounds, ard_num_dims=X_normalize.shape[-1],
                               **kwargs)
        if (rand_init > 0 and rand_init < 10):
            GP_model.covar_module.base_kernel.lengthscale = torch.rand(
                GP_model.covar_module.base_kernel.lengthscale.size(), device=device, dtype=dtype)
            GP_model.covar_module.outputscale = torch.rand(GP_model.covar_module.outputscale.size(), device=device,
                                                           dtype=dtype)
            GP_model.likelihood.noise = rand_init + 1
        elif rand_init >= 10:
            raise ValueError('Too many cov mat singular!')
        mll = ExactMarginalLogLikelihood(likelihood=GP_model.likelihood, model=GP_model)
        mll = mll.to(X_normalize)
        init_loss = self.get_loss(GP_model, X_normalize, mll)
        fit_gpytorch_model(mll)
        final_loss = self.get_loss(GP_model, X_normalize, mll)
        return X_normalize, GP_model, mll, init_loss, final_loss

    def GP_fitting(self, model_class, **kwargs):
        self.model_class = model_class
        self.X_normalize, self.model, self.mll, self.init_loss, self.final_loss = self.fit_model(self.X, self.bound,
                                                                                                 model_class, **kwargs)

    def get_loss(self, model, X_normalize, mll):
        model.train()
        output = model(X_normalize)
        return - mll(output, model.train_targets)

    def acq_optimize(self, model, dim, acq_func, optim_method='LBFGS', **kwargs):
        # pdb.set_trace()
        acq = acq_func(model, **kwargs)
        '''
        if(acq_func=='EI'):
            acq = ExpectedImprovement(model,best_f=self.Y_standard.max().item())
        else:
            print('No implementation on this acquisition function!')
        '''
        if (optim_method == 'LBFGS'):
            candidates, _ = optimize_acqf(
                acq_function=acq,
                bounds=torch.stack([
                    torch.zeros(dim, dtype=dtype, device=device),
                    torch.ones(dim, dtype=dtype, device=device)
                ]),
                q=1,
                num_restarts=10,
                raw_samples=20,
            )
            new_x_normalize = candidates.detach()
        elif (optim_method == 'CMAES'):
            es = cma.CMAEvolutionStrategy(
                x0=np.random.rand(dim),
                sigma0=0.2,
                inopts={'bounds': [0, 1], "popsize": 50},
            )
            with torch.no_grad():
                while not es.stop():
                    xs = es.ask()
                    XS = torch.tensor(xs, device=device, dtype=dtype)
                    YS = -acq(XS.unsqueeze(-2))
                    ys = YS.view(-1).double().numpy()
                    es.tell(xs, ys)
            new_x_normalize = torch.from_numpy(es.best.x).to(XS).reshape((1, dim))
            # pdb.set_trace()
        else:
            print('No implementation on this optimization!')
        return acq, new_x_normalize

    def BO_acq_optim(self, optim_method='LBFGS'):
        self.optim_method = optim_method
        self.acq, self.new_x_normalize = self.acq_optimize(self.model, self.X_dim, ExpectedImprovement,
                                                           optim_method=optim_method,
                                                           best_f=self.Y_standard.max().item())

    def data_update(self):
        new_x = unnormalize(self.new_x_normalize, bounds=self.bound)

        BS_tilt = tf.constant(new_x.numpy())
        BS_tilt = tf.expand_dims(BS_tilt, axis=2)
        BS_tilt = tf.tile(BS_tilt, [2 * config.batch_num, 1, config.Nuser_drop])

        # Run simulator based on new candidates
        ########################################################

        # If power and tilts are not being optimized
        P_Tx_TN = tf.tile(Ptx_thresholds_vector, [2 * config.batch_num, 1, config.Nuser_drop])
        BS_HPBW_v = tf.tile(HPBW_v_vector, [2 * config.batch_num, 1, config.Nuser_drop])

        data = Terrestrial()
        data.alpha_factor = 0.0  # LEO at 90deg
        data.BS_tilt = BS_tilt
        data.BS_HPBW_v = BS_HPBW_v
        data.call()

        # Import of the UAVs and GUEs LSG and SINR data
        LSG_UAVs_Corridors = data.LSG_UAVs_Corridors
        LSG_GUEs = data.LSG_GUEs
        sinr_TN_UAVs_Corridors = SINR.sinr_TN(LSG_UAVs_Corridors, P_Tx_TN)
        sinr_TN_GUEs = SINR.sinr_TN(LSG_GUEs, P_Tx_TN)

        # BO objective
        Rate_sumOftheLog_Obj, Rate_GUEs, Rate_UAVs = SINR.BO_Obj_Rates_and_Outage(LSG_GUEs, LSG_UAVs_Corridors, P_Tx_TN,
                                                                                  alpha=0.5)
        Rate_obj = Rate_sumOftheLog_Obj[0].__float__()
        new_y = torch.tensor([[Rate_obj]], dtype=torch.double)

        self.X, self.Y = torch.cat((self.X, new_x)), torch.cat((self.Y, new_y))

    def erase_last_instance(self):
        self.X = self.X[:-1]
        self.Y = self.Y[:-1]


class VSBO(BOtorch):
    def __init__(self, N_FS, *args, **kwargs):
        super(VSBO, self).__init__(*args, **kwargs)
        self.N_FS = N_FS
        self.active_f_dims = self.X_dim
        self.active_f_list = torch.tensor([1 for i in range(self.X_dim)], dtype=torch.bool, device=device)

    def CMAES_initialize(self):
        self.es = cma.CMAEvolutionStrategy(
            x0=np.random.rand(self.X_dim),
            sigma0=0.2,
            inopts={'bounds': [0, 1], "popsize": self.N_FS},
        )

    def CMAES_update(self):
        # pdb.set_trace()
        _ = self.es.ask()
        X_normalize = normalize(self.X, bounds=self.bound)
        X_normalize_np = X_normalize.numpy()
        Y_np = self.Y.numpy()
        self.es.tell([X_normalize_np[j] for j in range(-self.N_FS, 0, 1)], [-Y_np[j] for j in range(-self.N_FS, 0, 1)])
        # [mu1,mu2,cov11,cov12,cov21,cov22,cov22_inv,cond_cov]
        self.conditiona_normal_list = get_conditional_normal(self.es.mean, (self.es.sigma ** 2) * self.es.C,
                                                             ~self.active_f_list)
        self.cond_cov_cholesky = np.linalg.cholesky(self.conditiona_normal_list[-1])

    def GP_fitting_active(self, model_class, **kwargs):
        self.model_class = model_class
        self.X_normalize_active, self.model_active, self.mll_active, self.init_loss_active, self.final_loss_active = self.fit_model(
            self.X[:, self.active_f_list], self.bound[:, self.active_f_list], model_class, **kwargs)

    def BO_acq_optim_active(self, optim_method='LBFGS'):
        self.optim_method = optim_method
        self.acq_active, self.new_x_normalize_active = self.acq_optimize(self.model_active, self.active_f_dims,
                                                                         ExpectedImprovement, optim_method=optim_method,
                                                                         best_f=self.Y_standard.max().item())

    def calc_important_score(self, model, method='KLrel', *args, **kwargs):
        if (method == 'ard'):
            return FS_ARD(model)
        elif (method == 'KLrel'):
            return FS_KLrel(model, *args, **kwargs)
        elif (method == 'fANOVA'):
            return FS_fANOVA(*args, **kwargs)
            # pdb.set_trace()
            # f = fANOVA(kwargs['X'],kwargs['Y'])
        else:
            print("This immportant score calculation method has not been implemented!")

    def variable_selection_2(self, FS_score_method, *args, **kwargs):
        # pdb.set_trace()
        self.FS_score_method = FS_score_method
        # self.GP_fitting(self.model_class,**kwargs)
        self.FS_important_scores = self.calc_important_score(self.model, self.FS_score_method, dim=self.X_dim,
                                                             active_f_list=torch.tensor([1 for i in range(self.X_dim)],
                                                                                        dtype=torch.bool,
                                                                                        device=device), *args, **kwargs)
        kwargs_old = kwargs.copy()
        _, self.indices = torch.sort(self.FS_important_scores, descending=True)
        print(self.indices)
        if (self.X_dim == self.active_f_dims):
            self.stepwise_forward_2(0, torch.tensor([0 for k in range(self.X_dim)], dtype=torch.bool, device=device),
                                    **kwargs)
        else:
            delta_Y = torch.max(self.Y) - torch.max(self.Y[:-self.N_FS])
            if (delta_Y <= 0):
                self.stepwise_forward_2(0, self.active_f_list, **kwargs)
            else:
                mark = 0
                prev_active_dim_remain = self.active_f_dims
                prev_active_index = torch.where(self.active_f_list == 1)[0]
                start_point = 0
                if ('variable_type' in kwargs.keys()):
                    kwargs = kwargs_old.copy()
                    kwargs['variable_type'] = kwargs_old['variable_type'][self.active_f_list]
                FS_important_scores_active = self.calc_important_score(self.model_active, self.FS_score_method,
                                                                       dim=self.active_f_dims,
                                                                       active_f_list=self.active_f_list, *args,
                                                                       **kwargs)
                _, indices_active = torch.sort(FS_important_scores_active, descending=True)
                # RFE
                prev_loss = self.final_loss_active
                get_loss_interval = 0
                for k in range(self.active_f_dims - 1, 0, -1):
                    try:
                        if ('variable_type' in kwargs.keys()):
                            kwargs = kwargs_old.copy()
                            kwargs['variable_type'] = kwargs_old['variable_type'][prev_active_index[indices_active[:k]]]
                        _, _, _, _, sub_final_loss = self.fit_model(self.X[:, prev_active_index[indices_active[:k]]],
                                                                    self.bound[:,
                                                                    prev_active_index[indices_active[:k]]],
                                                                    self.model_class, **kwargs)
                    except ValueError as e:
                        if (e.args[0] == 'Too many cov mat singular!'):
                            break
                        else:
                            raise ValueError(e.args[0])
                    if (sub_final_loss <= prev_loss):
                        prev_loss = sub_final_loss
                        prev_active_dim_remain -= 1
                    else:
                        loss_interv = sub_final_loss - prev_loss
                        get_loss_interval += 1
                        break
                new_indices = prev_active_index[indices_active[:prev_active_dim_remain]]
                for j in range(self.X_dim):
                    if (self.indices[j] in new_indices):
                        continue
                    try:
                        if ('variable_type' in kwargs.keys()):
                            kwargs = kwargs_old.copy()
                            kwargs['variable_type'] = kwargs_old['variable_type'][
                                torch.cat([new_indices, torch.tensor([self.indices[j]])])]
                        _, _, _, _, sub_final_loss = self.fit_model(
                            self.X[:, torch.cat([new_indices, torch.tensor([self.indices[j]])])],
                            self.bound[:, torch.cat([new_indices, torch.tensor([self.indices[j]])])], self.model_class,
                            **kwargs)
                    except ValueError as e:
                        if (e.args[0] == 'Too many cov mat singular!'):
                            new_indices = torch.cat([new_indices, torch.tensor([self.indices[j]])])
                            continue
                        else:
                            raise ValueError(e.args[0])
                    if (get_loss_interval == 0):
                        loss_interv = prev_loss - sub_final_loss
                        prev_loss = sub_final_loss
                        get_loss_interval += 1
                        new_indices = torch.cat([new_indices, torch.tensor([self.indices[j]])])
                        continue
                    if (prev_loss - sub_final_loss < loss_interv / sf_stop):
                        break
                    else:
                        loss_interv = prev_loss - sub_final_loss
                        prev_loss = sub_final_loss
                        new_indices = torch.cat([new_indices, torch.tensor([self.indices[j]])])
                self.active_f_dims = len(new_indices)
                self.active_f_list = torch.tensor([0 for k in range(self.X_dim)], dtype=torch.bool, device=device)
                self.active_f_list[new_indices] = 1

    def variable_selection_nomom(self, FS_score_method, *args, **kwargs):
        self.FS_score_method = FS_score_method
        self.GP_fitting(self.model_class, **kwargs)
        self.FS_important_scores = self.calc_important_score(self.model, self.FS_score_method, dim=self.X_dim,
                                                             active_f_list=torch.tensor([1 for i in range(self.X_dim)],
                                                                                        dtype=torch.bool,
                                                                                        device=device), *args, **kwargs)
        kwargs_old = kwargs.copy()
        _, self.indices = torch.sort(self.FS_important_scores, descending=True)
        print(self.indices)
        self.stepwise_forward_2(0, torch.tensor([0 for k in range(self.X_dim)], dtype=torch.bool, device=device),
                                **kwargs)

    def stepwise_forward_2(self, start_point, important_variables, **kwargs):
        # pdb.set_trace()
        get_loss_interval = -1
        if_fs = 0
        kwargs_new = kwargs.copy()
        for j in range(start_point, self.X_dim):
            if (important_variables[self.indices[j]] == 1 and get_loss_interval == -1):
                continue
            try:
                if ('variable_type' in kwargs.keys()):
                    kwargs_new['variable_type'] = kwargs['variable_type'][self.indices[:j + 1]]
                _, _, _, _, sub_final_loss = self.fit_model(self.X[:, self.indices[:j + 1]],
                                                            self.bound[:, self.indices[:j + 1]], self.model_class,
                                                            **kwargs_new)
            except ValueError as e:
                if (e.args[0] == 'Too many cov mat singular!'):
                    continue
                else:
                    raise ValueError(e.args[0])
            if (get_loss_interval == -1):
                prev_loss = sub_final_loss
                get_loss_interval += 1
            elif (get_loss_interval == 0):
                loss_interv = prev_loss - sub_final_loss
                prev_loss = sub_final_loss
                get_loss_interval += 1
            else:
                if (loss_interv <= 0 or prev_loss - sub_final_loss < loss_interv / sf_stop):
                    if_fs = 1
                    self.active_f_list = torch.tensor([0 for k in range(self.X_dim)], dtype=torch.bool, device=device)
                    self.active_f_list[self.indices[:j]] = 1
                    self.active_f_dims = j
                    break
                else:
                    loss_interv = prev_loss - sub_final_loss
                    prev_loss = sub_final_loss
        if not if_fs:
            self.active_f_dims = self.X_dim
            self.active_f_list = torch.tensor([1 for k in range(self.X_dim)], dtype=torch.bool, device=device)

    # use rtmvnorm in R to sample truncated multivariate normal distribution, use rpy2 to embed R code in python
    def truncated_multivariate_normal_sampling(self, mu, cov_mat, n_samp):
        # pdb.set_trace()
        mu = FloatVector(mu)
        x_dim, _ = cov_mat.shape
        cov = ro.r.matrix(cov_mat, nrow=x_dim, ncol=x_dim)
        lb = FloatVector(np.zeros(x_dim))
        ub = FloatVector(np.ones(x_dim))
        return np.array(tmvtnorm.rtmvnorm(n=n_samp, mean=mu, sigma=cov, lower=lb, upper=ub, algorithm='gibbs', burn=100,
                                          thinning=5))

    def data_update(self, method='CMAES_posterior', n_sampling=1):
        # self.lessiv_n_sampling = n_sampling
        # pdb.set_trace()
        new_x = torch.tensor([0 for i in range(self.X_dim)], dtype=dtype, device=device).reshape((1, self.X_dim))
        new_x[:, self.active_f_list] = self.new_x_normalize_active
        if (self.X_dim > self.active_f_dims):
            # pdb.set_trace()
            if (method == 'rand'):
                new_x[:, ~self.active_f_list] = torch.rand(1, self.X_dim - self.active_f_dims, device=device,
                                                           dtype=dtype)
            if (method == 'mix'):
                # pdb.set_trace()
                rand_s = np.random.uniform(0, 1)
                if (rand_s <= 0.5):
                    new_x[:, ~self.active_f_list] = torch.rand(1, self.X_dim - self.active_f_dims, device=device,
                                                               dtype=dtype)
                else:
                    new_x[:, ~self.active_f_list] = normalize(self.X, bounds=self.bound)[
                        self.Y.argmax(), ~self.active_f_list].reshape((1, self.X_dim - self.active_f_dims))
            elif (method == 'CMAES_posterior'):
                CMA_cond_mean = self.conditiona_normal_list[0] + np.dot(
                    np.dot(self.conditiona_normal_list[3], self.conditiona_normal_list[6]), (
                                self.new_x_normalize_active.numpy().reshape((self.active_f_dims,)) -
                                self.conditiona_normal_list[1]))
                CMA_cond_cov = self.conditiona_normal_list[-1]
                self.new_x_normalize_inactive = self.truncated_multivariate_normal_sampling(CMA_cond_mean, CMA_cond_cov,
                                                                                            n_sampling)
                # self.new_x_normalize_inactive = np.random.multivariate_normal(CMA_cond_mean,CMA_cond_cov,n_sampling)
                # x_arr,y_arr = np.where(self.new_x_normalize_inactive>1)
                # self.new_x_normalize_inactive[x_arr,y_arr] = 1
                # x_arr,y_arr = np.where(self.new_x_normalize_inactive<0)
                # self.new_x_normalize_inactive[x_arr,y_arr] = 0
                new_x_multi = new_x.repeat(n_sampling, 1)
                new_x_multi[:, ~self.active_f_list] = torch.tensor(self.new_x_normalize_inactive, device=device,
                                                                   dtype=dtype).reshape(
                    (n_sampling, self.X_dim - self.active_f_dims))
                if (n_sampling > 1):
                    post = self.model.posterior(new_x_multi)
                    new_x = new_x_multi[post.mean.argmax()].reshape((1, self.X_dim))
                else:
                    new_x = new_x_multi
            else:
                print("The method to get the value of less important variables has not been implemented!")
        new_x = unnormalize(new_x, bounds=self.bound)

        BS_tilt = tf.constant(new_x.numpy())
        BS_tilt = tf.expand_dims(BS_tilt, axis=2)
        BS_tilt = tf.tile(BS_tilt, [2 * config.batch_num, 1, config.Nuser_drop])

        # Run simulator based on new candidates
        ########################################################

        # If power and tilts are not being optimized
        P_Tx_TN = tf.tile(self.Ptx_thresholds_vector, [2 * config.batch_num, 1, config.Nuser_drop])
        BS_HPBW_v = tf.tile(self.HPBW_v_vector, [2 * config.batch_num, 1, config.Nuser_drop])

        data = Terrestrial()
        data.alpha_factor = 0.0  # LEO at 90deg
        data.BS_tilt = tf.constant(BS_tilt.numpy(), dtype=tf.float32)
        data.BS_HPBW_v = BS_HPBW_v
        data.call()

        # Import of the UAVs and GUEs LSG and SINR data
        LSG_UAVs_Corridors = data.LSG_UAVs_Corridors
        LSG_GUEs = data.LSG_GUEs
        sinr_TN_UAVs_Corridors = SINR.sinr_TN(LSG_UAVs_Corridors, P_Tx_TN)
        sinr_TN_GUEs = SINR.sinr_TN(LSG_GUEs, P_Tx_TN)

        # BO objective
        Rate_sumOftheLog_Obj, Rate_GUEs, Rate_UAVs = SINR.BO_Obj_Rates_and_Outage(LSG_GUEs, LSG_UAVs_Corridors, P_Tx_TN,
                                                                                  alpha=0.0)
        Rate_obj = Rate_sumOftheLog_Obj[0].__float__()
        new_y = torch.tensor([[Rate_obj]], dtype=torch.double)
        self.X, self.Y = torch.cat((self.X, new_x)), torch.cat((self.Y, new_y))