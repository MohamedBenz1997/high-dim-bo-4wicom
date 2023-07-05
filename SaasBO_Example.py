import os

import torch
from torch.quasirandom import SobolEngine

from botorch import fit_fully_bayesian_model_nuts
from botorch.acquisition import qExpectedImprovement
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from botorch.test_functions import Branin

# ################################################
WARMUP_STEPS = 256
NUM_SAMPLES = 128
THINNING = 16
#
# train_X = torch.rand(10, 4)
# test_X = torch.rand(5, 4)
# train_Y = torch.sin(train_X[:, :1])
# test_Y = torch.sin(test_X[:, :1])
#
# #By default, we infer the unknown noise variance in the data. You can also pass in a known noise variance (train_Yvar) for each observation, which may be useful in cases where you for example know that the problem is noise-free and can then set the noise variance to a small value such as 1e-6.
# #gp = SaasFullyBayesianSingleTaskGP(train_X=train_X, train_Y=train_Y, train_Yvar=torch.full_like(train_Y, 1e-6))
#
# gp = SaasFullyBayesianSingleTaskGP(
#     train_X=train_X,
#     train_Y=train_Y,
#     outcome_transform=Standardize(m=1)
# )
# fit_fully_bayesian_model_nuts(
#     gp,
#     warmup_steps=WARMUP_STEPS,
#     num_samples=NUM_SAMPLES,
#     thinning=THINNING,
#     disable_progbar=True,
# )
# with torch.no_grad():
#     posterior = gp.posterior(test_X)
###################################################################################


# Optimize Branin embedded in a 30D space
###################################################
branin = Branin()
def branin_emb(x):
    lb, ub = branin.bounds
    return branin(lb + (ub - lb) * x[..., :2])

DIM = 57
# Evaluation budget
N_INIT = 10
N_ITERATIONS = 100
BATCH_SIZE = 1
print(f"Using a total of {N_INIT + BATCH_SIZE * N_ITERATIONS} function evaluations")

# Run the optimization
X = SobolEngine(dimension=DIM, scramble=True, seed=0).draw(N_INIT)
Y = branin_emb(X).unsqueeze(-1)
print(f"Best initial point: {Y.min().item():.3f}")

for i in range(N_ITERATIONS):
    train_Y = -1 * Y  # Flip the sign since we want to minimize f(x)
    gp = SaasFullyBayesianSingleTaskGP(
        train_X=X,
        train_Y=train_Y,
        train_Yvar=torch.full_like(train_Y, 1e-6),
        outcome_transform=Standardize(m=1),
    )
    fit_fully_bayesian_model_nuts(
        gp,
        warmup_steps=WARMUP_STEPS,
        num_samples=NUM_SAMPLES,
        thinning=THINNING,
        disable_progbar=True,
    )

    EI = qExpectedImprovement(model=gp, best_f=train_Y.max())
    candidates, acq_values = optimize_acqf(
        EI,
        bounds=torch.cat((torch.zeros(1, DIM), torch.ones(1, DIM))),
        q=BATCH_SIZE,
        num_restarts=10,
        raw_samples=1024,
    )

    Y_next = torch.cat([branin_emb(x).unsqueeze(-1) for x in candidates]).unsqueeze(-1)
    if Y_next.min() < Y.min():
        ind_best = Y_next.argmin()
        x0, x1 = candidates[ind_best, :2].tolist()
        print(
            f"{i + 1}) New best: {Y_next[ind_best].item():.3f} @ "
            f"[{x0:.3f}, {x1:.3f}]"
        )
    X = torch.cat((X, candidates))
    Y = torch.cat((Y, Y_next))

