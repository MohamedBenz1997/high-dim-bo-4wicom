#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Run one replication.
"""
from typing import Callable, Dict, List, Optional, Union
import time
import torch
import numpy as np
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.sampling import draw_sobol_samples
from MORBO_gen import (
    TS_select_batch_MORBO,
)
from MORBO_state import TRBOState
from MORBO_trust_region import TurboHParams
from torch import Tensor

from torch.quasirandom import SobolEngine
from botorch.utils.transforms import unnormalize

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
from tqdm import tqdm

supported_labels = ["morbo"]

BASE_SEED = 12346
def WiSe(x):

    dim = 57
    lower_bound_WiSe = -15.0
    upper_bound_WiSe = 45.0
    bounds_WiSe = torch.cat((torch.zeros(1, dim) + lower_bound_WiSe, torch.zeros(1, dim) + upper_bound_WiSe))
    new_x = unnormalize(x, bounds_WiSe)
    BS_tilt = tf.constant(new_x.numpy())
    BS_tilt = tf.expand_dims(tf.expand_dims(BS_tilt, axis=0), axis=2)

    # #Specifiying tilts
    # BS_tilt = tf.expand_dims(tf.constant([[
    #     -14.8470, - 13.4243,    7.4896,    8.3783,   12.4242,   12.1912, - 10.1185, - 14.4888, - 14.9273, - 14.9706,
    #     13.2545, - 14.9025,   15.1222, - 13.1998, - 14.8966, - 14.7850,   15.8751,   15.0329,    5.3124,   12.9737,
    #     10.1356,   13.6932,   16.2333, - 14.8296,    0.9250, - 13.7361, - 4.1275, - 14.1719, - 14.6700,   14.2649,
    #     - 13.0908,   13.9583,   13.1613,   14.4604, - 14.6686, - 14.9878, - 14.6371, - 11.9384, - 14.8977, - 13.3471,
    #     - 13.0281, - 14.6113, - 14.9524, - 14.9475,   14.9813, - 13.8463,   15.2721, - 14.8567, - 14.7609, - 14.9244,
    #     - 14.8238, - 14.9055, - 14.7492,   17.7670, - 13.0705, - 14.9241, - 14.9249]]), axis=2)

    BS_tilt = tf.tile(BS_tilt, [2 * config.batch_num, 1, config.Nuser_drop])

    # Run simulator based on new candidates
    ########################################################
    HPBW_v_vector = tf.expand_dims(tf.expand_dims(tf.random.uniform((57,), 10.0, 10.0, tf.float32), axis=0), axis=2)
    Ptx_thresholds_vector = tf.expand_dims(tf.expand_dims(tf.random.uniform((57,), 46.0, 46.0, tf.float32), axis=0),axis=2)

    # If power and tilts are not being optimized
    P_Tx_TN = tf.tile(Ptx_thresholds_vector, [2 * config.batch_num, 1, config.Nuser_drop])
    BS_HPBW_v = tf.tile(HPBW_v_vector, [2 * config.batch_num, 1, config.Nuser_drop])

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
    # SINR_sumOftheLog_Obj, Rate_sumOftheLog_Obj, sinr_total_UAVs, sinr_total_GUEs = SINR.BO_Multi_Obj_Cooridor(sinr_TN_UAVs_Corridors, sinr_TN_GUEs, alpha=0.5)
    Rate_sumOftheLog_Obj_GUEs, _, _ = SINR.BO_Obj_Rates_and_Outage(LSG_GUEs, LSG_UAVs_Corridors, P_Tx_TN,alpha=0.0)
    Rate_sumOftheLog_Obj_UAVs, _, _ = SINR.BO_Obj_Rates_and_Outage(LSG_GUEs, LSG_UAVs_Corridors, P_Tx_TN,alpha=1.0)
    Rate_obj_GUEs = Rate_sumOftheLog_Obj_GUEs[0].__float__()
    Rate_obj_UAVs = Rate_sumOftheLog_Obj_UAVs[0].__float__()
    new_y = torch.tensor([[Rate_obj_GUEs,Rate_obj_UAVs]], dtype=torch.double)

    KPI = new_y
    return KPI

def run_one_replication(
    seed: int,
    label: str,
    max_evals: int,
    evalfn: str,
    batch_size: int,
    dim: int,
    n_initial_points: int,
    n_trust_regions: int = TurboHParams.n_trust_regions,
    max_tr_size: int = TurboHParams.max_tr_size,
    min_tr_size: int = TurboHParams.min_tr_size,
    max_reference_point: Optional[List[float]] = None,
    failure_streak: Optional[int] = None,  # This is better to set automatically
    success_streak: int = TurboHParams.success_streak,
    raw_samples: int = TurboHParams.raw_samples,
    n_restart_points: int = TurboHParams.n_restart_points,
    length_init: float = TurboHParams.length_init,
    length_min: float = TurboHParams.length_min,
    length_max: float = TurboHParams.length_max,
    trim_trace: bool = TurboHParams.trim_trace,
    hypervolume: bool = TurboHParams.hypervolume,
    max_cholesky_size: int = TurboHParams.max_cholesky_size,
    use_ard: bool = TurboHParams.use_ard,
    verbose: bool = TurboHParams.verbose,
    qmc: bool = TurboHParams.qmc,
    track_history: bool = TurboHParams.track_history,
    sample_subset_d: bool = TurboHParams.sample_subset_d,
    fixed_scalarization: bool = TurboHParams.fixed_scalarization,
    winsor_pct: float = TurboHParams.winsor_pct,
    trunc_normal_perturb: bool = TurboHParams.trunc_normal_perturb,
    switch_strategy_freq: Optional[int] = TurboHParams.switch_strategy_freq,
    tabu_tenure: int = TurboHParams.tabu_tenure,
    decay_restart_length_alpha: float = TurboHParams.decay_restart_length_alpha,
    use_noisy_trbo: bool = TurboHParams.use_noisy_trbo,
    observation_noise_std: Optional[List[float]] = None,
    observation_noise_bias: Optional[List[float]] = None,
    use_simple_rff: bool = TurboHParams.use_simple_rff,
    use_approximate_hv_computations: bool = TurboHParams.use_approximate_hv_computations,
    approximate_hv_alpha: Optional[float] = TurboHParams.approximate_hv_alpha,
    recompute_all_hvs: bool = True,
    restart_hv_scalarizations: bool = True,
    dtype: torch.device = torch.double,
    device: Optional[torch.device] = None,
    save_callback: Optional[Callable[[Tensor], None]] = None,
    save_during_opt: bool = True,
) -> None:
    r"""Run the BO loop for given number of iterations. Supports restarting of
    prematurely killed experiments.

    Args:
        seed: The random seed.
        label: The algorith ("morbo")
        max_evals: evaluation budget
        evalfn: The test problem name
        batch_size: The size of each batch in BO
        dim: The input dimension (this is a parameter for some problems)
        n_initial_points: The number of initial sobol points

    The remaining parameters and default values are defined in trust_region.py.
    """
    assert label in supported_labels, "Label not supported!"
    start_time = time.time()
    seed = BASE_SEED + seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    tkwargs = {"dtype": dtype}
    bounds = torch.empty((2, dim), dtype=dtype)
    constraints = None
    objective = None


    num_objectives = len(max_reference_point)

    lower_bound = 0.0
    upper_bound = 1.0
    lower_tensor = torch.zeros(1, dim, dtype=dtype) + lower_bound
    upper_tensor = torch.zeros(1, dim, dtype=dtype) + upper_bound
    bounds = torch.cat((lower_tensor, upper_tensor), 0)
    num_outputs = 2

    # Automatically set the failure streak if it isn't specified
    failure_streak = max(dim // 3, 10) if failure_streak is None else failure_streak

    tr_hparams = TurboHParams(
        length_init=length_init,
        length_min=length_min,
        length_max=length_max,
        batch_size=batch_size,
        success_streak=success_streak,
        failure_streak=failure_streak,
        max_tr_size=max_tr_size,
        min_tr_size=min_tr_size,
        trim_trace=trim_trace,
        n_trust_regions=n_trust_regions,
        verbose=verbose,
        qmc=qmc,
        use_ard=use_ard,
        sample_subset_d=sample_subset_d,
        track_history=track_history,
        fixed_scalarization=fixed_scalarization,
        n_initial_points=n_initial_points,
        n_restart_points=n_restart_points,
        raw_samples=raw_samples,
        max_reference_point=max_reference_point,
        hypervolume=hypervolume,
        winsor_pct=winsor_pct,
        trunc_normal_perturb=trunc_normal_perturb,
        decay_restart_length_alpha=decay_restart_length_alpha,
        switch_strategy_freq=switch_strategy_freq,
        tabu_tenure=tabu_tenure,
        use_noisy_trbo=use_noisy_trbo,
        use_simple_rff=use_simple_rff,
        use_approximate_hv_computations=use_approximate_hv_computations,
        approximate_hv_alpha=approximate_hv_alpha,
        restart_hv_scalarizations=restart_hv_scalarizations,
    )

    trbo_state = TRBOState(
        dim=dim,
        max_evals=max_evals,
        num_outputs=num_outputs,
        num_objectives=num_objectives,
        bounds=bounds,
        tr_hparams=tr_hparams,
        constraints=constraints,
        objective=objective,
    )

    # For saving outputs
    n_evals = []
    true_hv = []
    pareto_X = []
    pareto_Y = []
    n_points_in_tr = [[] for _ in range(n_trust_regions)]
    n_points_in_tr_collected_by_other = [[] for _ in range(n_trust_regions)]
    n_points_in_tr_collected_by_sobol = [[] for _ in range(n_trust_regions)]
    tr_sizes = [[] for _ in range(n_trust_regions)]
    tr_centers = [[] for _ in range(n_trust_regions)]
    tr_restarts = [[] for _ in range(n_trust_regions)]
    fit_times = []
    gen_times = []
    true_ref_point = torch.tensor(max_reference_point, dtype=dtype)

    # Create initial points
    n_points = min(n_initial_points, max_evals - trbo_state.n_evals)
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    X_init = sobol.draw(n=n_points).to(dtype=dtype)

    Y_init = torch.stack(
        [torch.tensor(WiSe(x), dtype=dtype) for x in X_init]
    ).squeeze(1)

    trbo_state.update(
        X=X_init,
        Y=Y_init,
        new_ind=torch.full(
            (X_init.shape[0],), 0, dtype=torch.long
        ),
    )
    trbo_state.log_restart_points(X=X_init, Y=Y_init)

    # Initializing the trust regions. This also initializes the models.
    for i in range(n_trust_regions):
        trbo_state.initialize_standard(
            tr_idx=i,
            restart=False,
            switch_strategy=False,
            X_init=X_init,
            Y_init=Y_init,
        )

    # Update TRs data across trust regions, if necessary
    trbo_state.update_data_across_trs()

    # Set the initial TR indices to -2
    trbo_state.TR_index_history.fill_(-2)

    # Getting next suggestions
    all_tr_indices = [-1] * n_points

    counter_WiSE = 0
    while trbo_state.n_evals < max_evals:
        start_gen = time.time()
        selection_output = TS_select_batch_MORBO(trbo_state=trbo_state)
        gen_times.append(time.time() - start_gen)
        if trbo_state.tr_hparams.verbose:
            print(f"Time spent on generating candidates: {gen_times[-1]:.1f} seconds")

        X_cand = selection_output.X_cand
        tr_indices = selection_output.tr_indices
        all_tr_indices.extend(tr_indices.tolist())
        trbo_state.tabu_set.log_iteration()
        Y_cand = torch.stack(
            [torch.tensor(WiSe(x), dtype=dtype) for x in X_cand]
        ).squeeze(1)

        # Log TR info
        for i, tr in enumerate(trbo_state.trust_regions):
            inds = torch.cat(
                [torch.where((x == trbo_state.X_history).all(dim=-1))[0] for x in tr.X]
            )
            tr_inds = trbo_state.TR_index_history[inds]
            assert len(tr_inds) == len(tr.X)
            n_points_in_tr[i].append(len(tr_inds))
            n_points_in_tr_collected_by_sobol[i].append(sum(tr_inds == -2).cpu().item())
            n_points_in_tr_collected_by_other[i].append(
                sum((tr_inds != i) & (tr_inds != -2)).cpu().item()
            )
            tr_sizes[i].append(tr.length.item())
            tr_centers[i].append(tr.X_center.cpu().squeeze().tolist())

        # Append data to the global history and fit new models
        start_fit = time.time()
        trbo_state.update(X=X_cand, Y=Y_cand, new_ind=tr_indices)
        should_restart_trs = trbo_state.update_trust_regions_and_log(
            X_cand=X_cand,
            Y_cand=Y_cand,
            tr_indices=tr_indices,
            batch_size=batch_size,
            verbose=verbose,
        )
        fit_times.append(time.time() - start_fit)
        if trbo_state.tr_hparams.verbose:
            print(f"Time spent on model fitting: {fit_times[-1]:.1f} seconds")

        switch_strategy = trbo_state.check_switch_strategy()
        if switch_strategy:
            should_restart_trs = [True for _ in should_restart_trs]
        if any(should_restart_trs):
            for i in range(trbo_state.tr_hparams.n_trust_regions):
                if should_restart_trs[i]:
                    n_points = min(n_restart_points, max_evals - trbo_state.n_evals)
                    if n_points <= 0:
                        break  # out of budget
                    if trbo_state.tr_hparams.verbose:
                        print(f"{trbo_state.n_evals}) Restarting trust region {i}")
                    trbo_state.TR_index_history[trbo_state.TR_index_history == i] = -1
                    init_kwargs = {}
                    if trbo_state.tr_hparams.restart_hv_scalarizations:
                        # generate new point
                        X_center = trbo_state.gen_new_restart_design()
                        Y_center = f(X_center)
                        init_kwargs["X_init"] = X_center
                        init_kwargs["Y_init"] = Y_center
                        init_kwargs["X_center"] = X_center
                        trbo_state.update(
                            X=X_center,
                            Y=Y_center,
                            new_ind=torch.tensor(
                                [i], dtype=torch.long
                            ),
                        )
                        trbo_state.log_restart_points(X=X_center, Y=Y_center)

                    trbo_state.initialize_standard(
                        tr_idx=i,
                        restart=True,
                        switch_strategy=switch_strategy,
                        **init_kwargs,
                    )
                    if trbo_state.tr_hparams.restart_hv_scalarizations:
                        # we initialized the TR with one data point.
                        # this passes historical information to that new TR
                        trbo_state.update_data_across_trs()
                    tr_restarts[i].append(
                        trbo_state.n_evals.item()
                    )  # Where it restarted

        if trbo_state.tr_hparams.verbose:
            print(f"Total refill points: {trbo_state.total_refill_points}")

        # Save state at this evaluation and move to cpu
        n_evals.append(trbo_state.n_evals.item())
        if trbo_state.hv is not None:
            # The objective is None if there are no constraints
            obj = objective if objective else lambda x: x
            partitioning = DominatedPartitioning(
                ref_point=true_ref_point, Y=obj(trbo_state.pareto_Y)
            )
            hv = partitioning.compute_hypervolume().item()
            if trbo_state.tr_hparams.verbose:
                print(f"{trbo_state.n_evals}) Current hypervolume: {hv:.3f}")

            pareto_X.append(trbo_state.pareto_X.tolist())
            pareto_Y.append(trbo_state.pareto_Y.tolist())
            true_hv.append(hv)

            if observation_noise_std is not None:
                f.record_current_pf_and_hv(obj=obj, constraints=trbo_state.constraints)
        else:
            if trbo_state.tr_hparams.verbose:
                print(f"{trbo_state.n_evals}) Current hypervolume is zero!")
            pareto_X.append([])
            pareto_Y.append([])
            true_hv.append(0.0)
        trbo_state.update_data_across_trs()

        output = {
            "n_evals": n_evals,
            "X_history": trbo_state.X_history.cpu(),
            "metric_history": trbo_state.Y_history.cpu(),
            "true_pareto_X": pareto_X,
            "true_pareto_Y": pareto_Y,
            "true_hv": true_hv,
            "n_points_in_tr": n_points_in_tr,
            "n_points_in_tr_collected_by_other": n_points_in_tr_collected_by_other,
            "n_points_in_tr_collected_by_sobol": n_points_in_tr_collected_by_sobol,
            "tr_sizes": tr_sizes,
            "tr_centers": tr_centers,
            "tr_restarts": tr_restarts,
            "fit_times": fit_times,
            "gen_times": gen_times,
            "tr_indices": all_tr_indices,
        }
        #Un-normalize the candidates
        lower_bound_WiSe = -15.0
        upper_bound_WiSe = 45.0
        bounds_WiSe = torch.cat((torch.zeros(1, dim) + lower_bound_WiSe, torch.zeros(1, dim) + upper_bound_WiSe))
        Thresholds_WiSe = unnormalize(X_cand, bounds_WiSe)
        ## Save the output
        Obj_WiSe = Y_cand
        Thresholds_WiSe = Thresholds_WiSe.numpy()
        Thresholds_WiSe = tf.convert_to_tensor(Thresholds_WiSe)
        Obj_WiSe = Obj_WiSe.numpy()
        Obj_WiSe = tf.convert_to_tensor(Obj_WiSe)

        data_BO_WiSe = {"Thresholds": Thresholds_WiSe.numpy(),
                   "Obj": Obj_WiSe.numpy()}
        file_name = "2024_01_11_MORBO_GUEs_UAVs_tilts_iteration{}.mat".format(counter_WiSE)
        savemat(file_name, data_BO_WiSe)

        # Increment the counter
        counter_WiSE += 1

    end_time = time.time()
    if trbo_state.tr_hparams.verbose:
        print(f"Total time: {end_time - start_time:.1f} seconds")

    if trbo_state.hv is not None and recompute_all_hvs:
        # Go back and compute all hypervolumes so we don't have to do that later...
        f.record_all_hvs(obj=obj, constraints=trbo_state.constraints)

    output = {
        "n_evals": n_evals,
        "X_history": trbo_state.X_history.cpu(),
        "metric_history": trbo_state.Y_history.cpu(),
        "true_pareto_X": pareto_X,
        "true_pareto_Y": pareto_Y,
        "true_hv": true_hv,
        "n_points_in_tr": n_points_in_tr,
        "n_points_in_tr_collected_by_other": n_points_in_tr_collected_by_other,
        "n_points_in_tr_collected_by_sobol": n_points_in_tr_collected_by_sobol,
        "tr_sizes": tr_sizes,
        "tr_centers": tr_centers,
        "tr_restarts": tr_restarts,
        "fit_times": fit_times,
        "gen_times": gen_times,
        "tr_indices": all_tr_indices,
    }
    if trbo_state.hv is not None and recompute_all_hvs:
        additional_outputs = f.get_outputs()
        output = {**output, **additional_outputs}


run_one_replication(
        seed=123456,
        label="morbo",
        max_evals=2000,
        evalfn="WiSE",
        batch_size=1,
        dim=57,
        n_initial_points=5,
        max_reference_point=[-15.0,-15.0])

