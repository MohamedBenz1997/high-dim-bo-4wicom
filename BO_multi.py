"""
This is the BO Class of the simulator:
    This is the TN-NTN thresholds are computed
@authors: Mohamed Benzaghta
"""

import torch
from botorch.models.gp_regression import FixedNoiseGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.monte_carlo import qExpectedImprovement
from config import Config
import numpy as np



class BO_multi_class():

    def __init__(self):
        # Initial Data needed for BOtorch

        # Case study 00a ---- 1 TN and 1 NTN threshold, FRF=3 (UAV uniform location, 150m height), SeveralElv
        # TN_thresholds_TNonly = torch.tensor([[-2.2308874], [7.257044], [2.53331], [5.868122], [0.634923]])
        # TN_thresholds_NTNonly = torch.tensor([[2.4263191], [-2.614038], [7.674938], [-9.603079], [9.644947]])
        # TN_thresholds_TNonly = torch.tile(TN_thresholds_TNonly, (1, 57))
        # self.TN_thresholds = torch.cat((TN_thresholds_TNonly, TN_thresholds_NTNonly), 1)

        # # Case study ---- Load Balancing Experiment
        # self.TN_thresholds = torch.tensor([
        #                         [],
        #
        # [],
        #
        # []])
        #
        #
        # # Case study ---- Load Balancing Experiment
        # self.min_Rates = torch.tensor([[],
        #                                [],
        #                                []])


        # Case study  ---- Uptilt experiment (1 tilt for all)
        self.TN_thresholds = torch.tensor([[-30.0], [-20.0], [0.0], [30.0]])


        Case study ---- Load Balancing Experiment
        self.min_Rates = torch.tensor([[1409.420654296875],
                                       [1560.7459716796875],
                                       [1459.7220458984375],
                                       [1251.6058349609375]])

        # # Case study  ---- Uptilt experiment (19 tilts)
        self.TN_thresholds = torch.tensor([
                                [-14.836289, -14.536041, -12.097961, -14.917168, -11.168421,
       -14.974743, -13.407147, -13.346384, -14.327436, -13.425077,
       -14.976399, -12.951084, -11.38052 , -12.994371, -11.346596,
       -13.723012, -11.841258, -14.054642, -13.687011],

        [-11.658585, -12.468614, -13.257273, -13.897732, -14.479568,
       -14.697688, -12.194918, -14.037255, -12.533116, -13.359023,
       -12.005245, -12.638441, -14.130384, -11.45158 , -11.087755,
       -13.71873 , -14.297129, -11.118753, -11.263265],

        [-14.011549 , -13.448909 , -12.605523 , -12.861307 , -12.324459 ,
       -11.185198 , -12.627681 , -12.527599 , -12.641043 , -14.716791 ,
       -14.121136 , -11.645224 , -12.033152 , -12.112118 , -11.7338505,
       -13.827951 , -14.9779625, -14.543519 , -12.135164],

        [-11.202621 , -13.045666 , -12.342211 , -13.933815 , -11.222408 ,
       -11.754723 , -13.204211 , -11.559556 , -12.891558 , -13.673327 ,
       -12.857828 , -14.596149 , -14.099388 , -11.187959 , -12.774414 ,
       -13.043673 , -11.085531 , -14.106184 , -11.0170765],

        [-12.536031 , -13.685306 , -12.83218  , -12.242033 , -12.546473 ,
       -14.214848 , -12.796724 , -11.348222 , -11.870888 , -11.996638 ,
       -12.9209175, -12.427996 , -13.078766 , -13.43652  , -14.905382 ,
       -12.930549 , -13.784942 , -13.159779 , -14.797204]])
        #
        # Case study ---- Load Balancing Experiment
        self.min_Rates = torch.tensor([[2621.710205078125],
                                       [2488.554443359375],
                                       [2611.023193359375],
                                       [2428.089111328125],
                                       [2561.807861328125]])

        #Take the updated iterations
        # TN_thresholds_before = np.load('/tmp/pycharm_project_502/TN_thresholds.npy')
        # Rates_before = np.load('/tmp/pycharm_project_502/Rates.npy')
        # self.TN_thresholds = torch.tensor(TN_thresholds_before)
        # self.min_Rates = torch.tensor(Rates_before)

    def call_forConfig(self):
        config = Config()
        BO_output=self.BO_qEI()
        # BO_output = self.BO_testing()
        #If you want to fix the TN threshold to all cells use this part
        #############################################################
        # BO_output_zeros = torch.zeros(BO_output.shape)
        # BO_output_1st = BO_output[0, 0]
        # BO_output_last = BO_output[0, -1]
        # BO_output_TN=BO_output_zeros[:,:-1]+BO_output_1st
        # BO_output_NTN = BO_output_zeros[:,-1] + BO_output_last
        # BO_output_NTN = BO_output_NTN[:, None]
        # BO_output = torch.cat((BO_output_TN, BO_output_NTN), 1)
        #############################################################

        self.TN_thresholds = torch.cat((self.TN_thresholds, BO_output), 0)

        #For x-cells
        BO_output = BO_output.expand(config.batch_num, 1, 19) #Change here

        return BO_output

    def append_minRate(self, new_minRate):
        # Create tensor to append
        new_minRate_tensor = torch.tensor(new_minRate, dtype=float).reshape(1, 1)
        # Append new tensor value
        self.min_Rates = torch.cat((self.min_Rates, new_minRate_tensor), 0)


    # ------------ BO using aquestion function qEI
    def BO_qEI(self):
        # NOISE_SE = torch.tensor([0.001, 0.001]) #The one I use for exp 00a and 01a
        # NOISE_SE = torch.tensor([0.00001, 0.00001]) #good for uptilt
        NOISE_SE = torch.tensor([0.000001, 0.000001])  # perfect for uptilt ISD 500m


        train_x = self.TN_thresholds
        #If you want to fix the TN threshold to all BSs use this part (19 BSs cells)
        #############################################################
        T_NTN = train_x[0, -1]
        T_NTN = T_NTN[None, None]
        train_x = train_x[:, 0:19]
        #############################################################

        #If you want to fix the TN threshold to all cells use this part
        #############################################################
        # T_NTN = train_x[:, -1]
        # # T_NTN = T_NTN[None, None]
        # T_NTN = T_NTN[:, None]
        # train_x = train_x[:, 0:1]
        # train_x = torch.cat((train_x, T_NTN), 1)
        #############################################################

        #If you want to have 3 TN thresholds
        #############################################################
        # train_x1 = train_x[:, 0:1]
        # train_x2 = train_x[:, 19:20]
        # train_x3 = train_x[:, 38:39]
        # train_x = torch.cat((train_x1, train_x2, train_x3), 1)
        #############################################################

        train_obj = self.min_Rates
        best_init_y = train_obj.max().item()

        # Bounds for 1 cell
        # bounds = torch.tensor([[-90.0],
        #                        [90.0]])

        # Bounds for 1 cell
        # bounds = torch.tensor([[-10.0, -10.0],
        #                        [10.0, 10.0]])

        # Bounds for 3 cell
        # bounds = torch.tensor([[-20.0, -20.0, -20.0 ],
        #                        [0.0, 0.0, 0.0 ]])

        # Bounds for TN & NTN thresholds
        bounds_lower = torch.tensor([[-45.0]])
        bounds_lower = bounds_lower.tile((19,))
        bounds_higher = torch.tensor([[10.0]])
        bounds_higher = bounds_higher.tile((19,))
        bounds = torch.cat((bounds_lower, bounds_higher), 0)

        models = []
        for i in range(train_obj.shape[-1]):
            train_y = train_obj[..., i:i + 1]
            train_yvar = torch.full_like(train_y, NOISE_SE[i] ** 2)
            models.append(
                FixedNoiseGP(train_x, train_y, train_yvar, outcome_transform=Standardize(m=1))
            )
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)

        EI = qExpectedImprovement(model=model, best_f=best_init_y)
        candidate, _ = optimize_acqf(acq_function=EI, bounds=bounds, q=1, num_restarts=200, raw_samples=512)

        #If you want to fix the TN threshold to all cells use this part
        #############################################################
        # candidate_TN = torch.tile(candidate[0][0], (1, 57)) #Change here
        # candidate_NTN = torch.tile(candidate[0][1], (1, 1))
        # candidate = torch.cat((candidate_TN, candidate_NTN), 1)
        #############################################################
        # BO_output_TN = candidate[0][0].__float__()
        # BO_output_NTN = candidate[0][1].__float__()

        #If you want to fix the TN threshold to every cell in the deployment (3 TN thresholds experiment)
        #############################################################
        # candidate_TN1 = torch.tile(candidate[0][0], (1, 19))
        # candidate_TN2 = torch.tile(candidate[0][1], (1, 19))
        # candidate_TN3 = torch.tile(candidate[0][2], (1, 19))
        # candidate = torch.cat((candidate_TN1, candidate_TN2, candidate_TN3), 1)
        #############################################################

        # return BO_output_TN,BO_output_NTN
        BO_output = candidate

        return BO_output

    # def BO_testing(self):
    #
    #     NOISE_SE = 0.5
    #     train_yvar = torch.tensor(NOISE_SE ** 2)
    #
    #     train_x = self.TN_thresholds
    #     #If you want to fix the TN threshold to all BSs use this part (19 BSs cells)
    #     #############################################################
    #     T_NTN = train_x[0, -1]
    #     T_NTN = T_NTN[None, None]
    #     train_x = train_x[:, 0:19]
    #     #############################################################
    #
    #     train_obj = self.min_Rates
    #     best_init_y = train_obj.max().item()
    #
    #     model_obj = FixedNoiseGP(train_x, train_obj, train_yvar.expand_as(train_obj)).to(train_x)
    #     model = ModelListGP(model_obj)
    #     mll = SumMarginalLogLikelihood(model.likelihood, model)
    #
    #     EI = qExpectedImprovement(model=model, best_f=best_init_y)
    #     candidate, _ = optimize_acqf(acq_function=EI, bounds=bounds, q=1, num_restarts=200, raw_samples=512)
    #
    #     # return BO_output_TN,BO_output_NTN
    #     BO_output = candidate
    #
    #     return BO_output