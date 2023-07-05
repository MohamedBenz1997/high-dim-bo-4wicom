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
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler
from config import Config
import numpy as np



class BO_multi_qNEI_class():

    def __init__(self):

        # Initial Data needed for BOtorch
        # Case study  ---- Uptilt experiment (19 tilts)
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
        BO_output=self.BO_qNEI()

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
    def BO_qNEI(self):
        NOISE_SE = torch.tensor([0.000001, 0.000001])  # perfect for uptilt ISD 500m

        train_x = self.TN_thresholds
        train_obj = self.min_Rates
        best_init_y = train_obj.max().item()

        # Bounds for TN & NTN thresholds
        bounds_lower = torch.tensor([[-45.0]])
        bounds_lower = bounds_lower.tile((19,))
        bounds_higher = torch.tensor([[10.0]])
        bounds_higher = bounds_higher.tile((19,))
        bounds = torch.cat((bounds_lower, bounds_higher), 0)


        ############################################
        models = []
        for i in range(train_obj.shape[-1]):
            train_y = train_obj[..., i:i + 1]
            train_yvar = torch.full_like(train_y, NOISE_SE[i] ** 2)
            models.append(
                FixedNoiseGP(train_x, train_y, train_yvar, outcome_transform=Standardize(m=1))
            )
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        ############################################

        MC_SAMPLES = 256
        BATCH_SIZE = 1
        NUM_RESTARTS = 10
        RAW_SAMPLES = 512


        # define the qEI and qNEI acquisition modules using a QMC sampler
        qmc_sampler = SobolQMCNormalSampler(MC_SAMPLES)

        # for best_f, we use the best observed noisy values as an approximation
        qNEI = qNoisyExpectedImprovement(
                                         model=model,
                                         X_baseline=train_x,
                                         sampler=qmc_sampler,)

        candidate, _ = optimize_acqf(
                acq_function=qNEI,
                bounds=bounds,
                q=BATCH_SIZE,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,  # used for intialization heuristic
                options={"batch_limit": 5, "maxiter": 200},)

        BO_output = candidate

        return BO_output

