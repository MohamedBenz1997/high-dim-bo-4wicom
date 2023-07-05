"""
This is the BO Class of the simulator:
    This is the TN-NTN thresholds are computed
@authors: Mohamed Benzaghta
"""
import torch
from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import  fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.optim import  optimize_acqf

class BO_class():
    def __init__(self):
        # Initial Data needed for BOtorch
        self.TN_thresholds = torch.tensor([[-3.5], [-4.0],[-5.5],[-6.5],[-8.0]], dtype=float)
        self.min_Rates = torch.tensor([[959.26514], [967.94],[716.4624],[583.17334],[424.69684]], dtype=float)

    def call_forConfig(self):
        BO_output_scaler=self.BO_qEI() #I need this to update the config file

        # Transfor it into a tensor with dimension 1,1
        BO_output = torch.tensor(BO_output_scaler, dtype=float).reshape(1, 1)
        # Append this value to.self TN_thresholds
        self.TN_thresholds = torch.cat((self.TN_thresholds, BO_output), 0)

        return BO_output_scaler

    def append_minRate(self, new_minRate):
        # Create tensor to append
        new_minRate_tensor = torch.tensor(new_minRate, dtype=float).reshape(1, 1)
        # Append new tensor value
        self.min_Rates = torch.cat((self.min_Rates, new_minRate_tensor), 0)


    # ------------ BO using aquestion function qEI
    def BO_qEI(self):
        min_Rate = self.min_Rates
        TN_threshold = self.TN_thresholds

        best_observed_value = min_Rate.max().item()

        # Getting the next candidate sampling point
        init_x = TN_threshold
        init_y = min_Rate
        best_init_y = best_observed_value
        bounds = torch.tensor([[-8.5], [-0.0]])

        # Because we have 1 KPI, single opt function
        single_model = SingleTaskGP(init_x, init_y)
        mll = ExactMarginalLogLikelihood(single_model.likelihood, single_model)
        fit_gpytorch_model(mll)

        # at this step we have the GP regression model, now we use the aqueasatipn function
        EI = qExpectedImprovement(model=single_model, best_f=best_init_y)

        # optimize the acquestion function
        candidate, _ = optimize_acqf(acq_function=EI, bounds=bounds, q=1, num_restarts=200, raw_samples=512)
        BO_output = candidate[0][0].__float__()

        return BO_output