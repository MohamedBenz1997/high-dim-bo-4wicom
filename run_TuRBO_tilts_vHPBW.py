import math
from dataclasses import dataclass

import torch
from botorch.acquisition import qExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize
from torch.quasirandom import SobolEngine

import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import HorseshoePrior

dtype = torch.double

#My simulator import
import tensorflow as tf
from TerrestrialClass import Terrestrial
from SinrClass import SINR
from config import Config
from plot_class import Plot
SINR = SINR()
config = Config()
plot = Plot()
from scipy.io import savemat

def WiSe(x):

    # Obtaining tilts
    x_tilts = x[0:57]
    dim_tilts = 57
    lower_bound_tilts = -15.0
    upper_bound_tilts = 45.0
    bounds_tilts = torch.cat((torch.zeros(1, dim_tilts) + lower_bound_tilts, torch.zeros(1, dim_tilts) + upper_bound_tilts))
    new_x_tilts = unnormalize(x_tilts, bounds_tilts)
    BS_tilt = tf.constant(new_x_tilts.numpy())
    BS_tilt = tf.expand_dims(tf.expand_dims(BS_tilt, axis=0), axis=2)

    # #Specifiying tilts
    # BS_tilt = tf.expand_dims(tf.constant([[
    #     -14.4560, - 14.1319, - 13.9468,   32.3200,   29.4375,   31.5648, - 12.2652, - 11.5641, - 13.2996, - 14.7087,
    #     - 14.8936, - 10.4940,   41.5800, - 13.9275, - 12.6330, - 13.6806,   29.9685, - 13.5248,   31.2413, - 13.8916,
    #     - 13.5129,   24.2786,   39.4246, - 13.5935,   18.0583, - 13.9555, - 11.6119, - 12.6597, - 14.3336,   28.7924,
    #     - 13.3019,   34.6571,   28.2556,   32.8013,   35.0363, - 13.8791, - 11.7589, - 13.9402,   42.6654, - 13.2376,
    #     20.8807, - 11.0360, - 13.9017, - 11.0590, - 10.0500,   44.3375,   40.0510, - 14.8062, - 14.8100, - 14.1966,
    #     - 10.2223, - 11.5097, - 14.0140, - 12.6885, - 12.2853, - 14.6965, - 14.6113]]), axis=2)
    BS_tilt = tf.expand_dims(tf.expand_dims(tf.random.uniform((57,), -12.0, -12.0, tf.float32), axis=0), axis=2)

    BS_tilt = tf.tile(BS_tilt, [2 * config.batch_num, 1, config.Nuser_drop])

    # Obtaining vHPBW
    x_vHPBW = x[57:]
    dim_vHPBW = 57
    lower_bound_vHPBW = 5.0
    upper_bound_vHPBW = 40.0
    bounds_vHPBW = torch.cat((torch.zeros(1, dim_vHPBW) + lower_bound_vHPBW, torch.zeros(1, dim_vHPBW) + upper_bound_vHPBW))
    new_x_vHPBW = unnormalize(x_vHPBW, bounds_vHPBW)
    BS_HPBW_v = tf.constant(new_x_vHPBW.numpy())
    BS_HPBW_v = tf.expand_dims(tf.expand_dims(BS_HPBW_v, axis=0), axis=2)

    # #Specifiying tilts
    # BS_HPBW_v = tf.expand_dims(tf.constant([[
    #     13.6397,   14.9121,   12.9279,   13.2973,    7.2157,    5.6928,   11.5787,   10.2861,   13.1135,   13.0961,
    #     15.0843,    7.3380,   20.7834,   11.0466,   10.8063,   13.3503,    5.5691,   11.4909,    7.5647,   13.5551,
    #     12.3120,    6.7062,   17.5115,   10.1048,    6.9021,   10.8138,   11.6734,   10.9175,   12.3684,   17.5539,
    #     14.0744,    8.5939,   12.4248,    7.5120,    8.2983,   13.5497,    9.6919,   12.7216,   16.0811,   13.1879,
    #     39.3443,    8.0549,   14.9334,   11.2906,   12.7351,   33.1241,   11.5078,   11.6127,   11.8795,   16.0908,
    #     6.0201,   14.0386,   14.3254,   16.5708,   10.6480,   14.4024,   13.0798]]), axis=2)
    BS_HPBW_v = tf.expand_dims(tf.expand_dims(tf.random.uniform((57,), 10.0, 10.0, tf.float32), axis=0), axis=2)

    BS_HPBW_v = tf.tile(BS_HPBW_v, [2 * config.batch_num, 1, config.Nuser_drop])

    # Run simulator based on new candidates
    ########################################################
    Ptx_thresholds_vector = tf.expand_dims(tf.expand_dims(tf.random.uniform((57,), 46.0, 46.0, tf.float32), axis=0),axis=2)

    # If power and tilts are not being optimized
    P_Tx_TN = tf.tile(Ptx_thresholds_vector, [2 * config.batch_num, 1, config.Nuser_drop])

    data = Terrestrial()
    data.alpha_factor = 0.0  # LEO at 90deg
    data.BS_tilt = tf.constant(BS_tilt.numpy(), dtype=tf.float32)
    data.BS_HPBW_v = tf.constant(BS_HPBW_v.numpy(), dtype=tf.float32)
    data.call()

    # Import of the UAVs and GUEs LSG and SINR data
    LSG_UAVs_Corridors = data.LSG_UAVs_Corridors
    LSG_GUEs = data.LSG_GUEs
    sinr_TN_UAVs_Corridors = SINR.sinr_TN(LSG_UAVs_Corridors, P_Tx_TN)
    sinr_TN_GUEs = SINR.sinr_TN(LSG_GUEs, P_Tx_TN)

    # BO objective
    SINR_sumOftheLog_Obj, Rate_sumOftheLog_Obj, sinr_total_UAVs, sinr_total_GUEs = SINR.BO_Multi_Obj_Cooridor(sinr_TN_UAVs_Corridors, sinr_TN_GUEs, alpha=0.0)
    Rate_sumOftheLog_Obj, Rate_GUEs, Rate_UAVs = SINR.BO_Obj_Rates_and_Outage(LSG_GUEs, LSG_UAVs_Corridors, P_Tx_TN,alpha=0.0)
    Rate_obj = Rate_sumOftheLog_Obj[0].__float__()
    new_y = torch.tensor([[Rate_obj]], dtype=torch.double)

    KPI = new_y
    return KPI

# TurBO initial paramaters
dim = 114
batch_size = 4
n_init = 2 * dim
max_cholesky_size = float("inf")  # Always use Cholesky


@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )


def update_state(state, Y_next):
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state

def get_initial_points(dim, n_pts, seed=0):
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    X_init = sobol.draw(n=n_pts).to(dtype=dtype)
    return X_init

def generate_batch(
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    batch_size,
    n_candidates=None,  # Number of candidates for Thompson sampling
    num_restarts=10,
    raw_samples=512,
    acqf="ts",  # "ei" or "ts"
):
    assert acqf in ("ts", "ei")
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    # Scale the TR to be proportional to the lengthscales
    x_center = X[Y.argmax(), :].clone()
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    if acqf == "ts":
        dim = X.shape[-1]
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=dtype)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_candidates, dim, dtype=dtype) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim - 1, size=(len(ind),))] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        with torch.no_grad():  # We don't need gradients when using TS
            X_next = thompson_sampling(X_cand, num_samples=batch_size)

    elif acqf == "ei":
        ei = qExpectedImprovement(model, train_Y.max())
        X_next, acq_value = optimize_acqf(
            ei,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

    return X_next

X_turbo = get_initial_points(dim, n_init)
Y_turbo = torch.tensor(
    [WiSe(x) for x in X_turbo], dtype=dtype
).unsqueeze(-1)

## Save initial data-set
#file_name = "2023_12_28_Corr_50m_30deg_2bydim_57for10deg_initial_dataset.pt"
#torch.save({"X_turbo": X_turbo, "Y_turbo": Y_turbo}, file_name)

## Load initial data-set
#file_name = "2023_12_27_Corr_50m_30deg_2bydim_114for10deg_initial_dataset.pt"
#loaded_data = torch.load(file_name)
#X_turbo = loaded_data["X_turbo"]
#Y_turbo = loaded_data["Y_turbo"]

## Load initial data-set mix
# file_name = "2023_12_28_Corr_50m_30deg_2bydim_57for10deg_initial_dataset.pt"
# loaded_data = torch.load(file_name)
# X_turbo_SA = loaded_data["X_turbo"]
# Y_turbo_SA = loaded_data["Y_turbo"]
# X_turbo = torch.cat((X_turbo, X_turbo_SA), dim=0)
# Y_turbo = torch.cat((Y_turbo, Y_turbo_SA), dim=0)

state = TurboState(dim, batch_size=batch_size)

NUM_RESTARTS = 10
RAW_SAMPLES = 512
N_CANDIDATES = min(5000, max(2000, 200 * dim))

torch.manual_seed(0)
i = 0
while not state.restart_triggered:  # Run until TuRBO converges
    # Fit a GP model
    train_Y = (Y_turbo - Y_turbo.mean()) / Y_turbo.std()
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
    covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
        MaternKernel(
            nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0)
        )
    )
    model = SingleTaskGP(
        X_turbo, train_Y, covar_module=covar_module, likelihood=likelihood
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    # Do the fitting and acquisition function optimization inside the Cholesky context
    with gpytorch.settings.max_cholesky_size(max_cholesky_size):
        # Fit the model
        fit_gpytorch_mll(mll)

        # Create a batch
        X_next = generate_batch(
            state=state,
            model=model,
            X=X_turbo,
            Y=train_Y,
            batch_size=batch_size,
            n_candidates=N_CANDIDATES,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,
            acqf="ts",
        )

    Y_next = torch.tensor(
        [WiSe(x) for x in X_next], dtype=dtype
    ).unsqueeze(-1)

    # Update state
    state = update_state(state=state, Y_next=Y_next)

    # Append data
    X_turbo = torch.cat((X_turbo, X_next), dim=0)
    Y_turbo = torch.cat((Y_turbo, Y_next), dim=0)

    #Un-normalize the candidates
    X_turbo_tilts = X_turbo[:,0:57]
    dim_tilts = 57
    lower_bound_tilts = -15.0
    upper_bound_tilts = 45.0
    bounds_tilts = torch.cat((torch.zeros(1, dim_tilts) + lower_bound_tilts, torch.zeros(1, dim_tilts) + upper_bound_tilts))
    Thresholds_tilts = unnormalize(X_turbo_tilts, bounds_tilts)
    X_turbo_vHPBW = X_turbo[:,57:]
    dim_vHPBW = 57
    lower_bound_vHPBW = 5.0
    upper_bound_vHPBW = 40.0
    bounds_vHPBW = torch.cat((torch.zeros(1, dim_vHPBW) + lower_bound_vHPBW, torch.zeros(1, dim_vHPBW) + upper_bound_vHPBW))
    Thresholds_vHPBW = unnormalize(X_turbo_vHPBW, bounds_vHPBW)

    Thresholds = torch.cat((Thresholds_tilts, Thresholds_vHPBW), dim=1)

    #BO outputs
    Obj = Y_turbo
    Thresholds = Thresholds.numpy()
    Thresholds = tf.convert_to_tensor(Thresholds)
    Obj = Obj.numpy()
    Obj = tf.convert_to_tensor(Obj)
    best_observed_objective_value = tf.reduce_max(Obj, axis=0)
    optimum_thresholds = tf.tile(tf.cast(Obj == best_observed_objective_value, "float64"), [1, 1]) * Thresholds
    optimum_thresholds = tf.reduce_sum(optimum_thresholds, axis=0)
    Full_tilts = optimum_thresholds[0:57]
    Full_vHPBW = optimum_thresholds[57:]

    if i == 0:
        best_rate_so_far = tf.zeros(best_observed_objective_value.shape, dtype='float64')
    best_rate_so_far = tf.concat([best_rate_so_far, best_observed_objective_value], axis=0)

    # Saving BO data for matlab
    data_BO = {"Thresholds": Thresholds.numpy(),
               "Obj": Obj.numpy(),
               "best_observed_objective_value": best_observed_objective_value.numpy(),
               "best_rate_so_far": best_rate_so_far.numpy(),
               "Full_tilts": Full_tilts.numpy(),
               "Full_vHPBW": Full_vHPBW.numpy(),}
    file_name = "2023_12_30_TuRBO_Corr_150m_tilts_vHPBW_iteration{}.mat".format(i)
    savemat(file_name, data_BO)

    #Increment the counter
    i += 1

    # Print current status
    print(
        f"{len(X_turbo)}) Best value: {state.best_value:.2e}, TR length: {state.length:.2e}"
    )

