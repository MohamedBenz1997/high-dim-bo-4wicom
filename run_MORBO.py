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
    upper_bound_WiSe = 20.0
    bounds_WiSe = torch.cat((torch.zeros(1, dim) + lower_bound_WiSe, torch.zeros(1, dim) + upper_bound_WiSe))
    new_x = unnormalize(x, bounds_WiSe)
    BS_tilt = tf.constant(new_x.numpy())
    BS_tilt = tf.expand_dims(tf.expand_dims(BS_tilt, axis=0), axis=2)

    # #Specifiying tilts
    # BS_tilt = tf.expand_dims(tf.constant([[
    #     -13.9327, - 13.2903, - 12.8647,   10.5337,    9.9584,   12.8540, - 13.7302, - 12.0697, - 12.2952, - 12.9120,
    #     11.3224, - 13.4506,   12.7093, - 12.6343, - 12.1007, - 12.9844,   11.3501,   12.0331,   11.8313, - 12.1021,
    #     - 13.6658,   10.4538,   13.8567, - 12.9336, - 13.0059, - 12.8360, - 12.8065, - 12.2845, - 13.9383, - 13.4531,
    #     - 13.0222,   12.4988, - 14.2656,   12.3700, - 12.1800, - 13.2468, - 13.7995, - 13.0525, - 12.7080, - 12.7230,
    #     - 12.7104, - 13.1767, - 11.7174, - 13.1188, - 12.4295, - 13.1245,   14.4940, - 13.1295, - 13.9200, - 12.7240,
    #     - 13.2607, - 11.5826, - 12.8363,   13.1386, - 12.6877, - 13.2881, - 11.5168]]), axis=2)

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

    # Import distances
    D = data.D
    D_2d = data.D_2d

    # Import of the UAVs and GUEs LSG and SINR data
    LSG_UAVs_Corridors = data.LSG_UAVs_Corridors
    LSG_GUEs = data.LSG_GUEs
    sinr_TN_UAVs_Corridors = SINR.sinr_TN(LSG_UAVs_Corridors, P_Tx_TN)
    sinr_TN_GUEs = SINR.sinr_TN(LSG_GUEs, P_Tx_TN)

    # BO objective
    # SINR_sumOftheLog_Obj, Rate_sumOftheLog_Obj, sinr_total_UAVs, sinr_total_GUEs = SINR.BO_Multi_Obj_Cooridor(sinr_TN_UAVs_Corridors, sinr_TN_GUEs, alpha=0.0)
    # Rate_sumOftheLog_Obj_GUEs, Coverage_ratio, Rate_GUEs, Rate_UAVs = SINR.BO_Obj_Rates_and_Outage(LSG_GUEs, LSG_UAVs_Corridors, P_Tx_TN, D, D_2d, alpha=0.0)
    Rate_sumOftheLog_Obj_UAVs, Coverage_ratio, _, _ = SINR.BO_Obj_Rates_and_Outage(LSG_GUEs, LSG_UAVs_Corridors, P_Tx_TN, D, D_2d, alpha=1.0)
    Rate_sumOftheLog_Obj_GUEs, Coverage_ratio, _, _ = SINR.BO_Obj_Rates_and_Outage(LSG_GUEs, LSG_UAVs_Corridors, P_Tx_TN, D, D_2d, alpha=0.0)
    Rate_obj_GUEs = Rate_sumOftheLog_Obj_GUEs[0].__float__()
    Rate_obj_UAVs = Rate_sumOftheLog_Obj_UAVs[0].__float__()
    Coverage_ratio = Coverage_ratio.__float__()
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
    # X_init = sobol.draw(n=n_points).to(dtype=dtype)

    ## Bias initialization of MORBO based on TuRBO best observed for \lambda=0.5, 0, and 1
    def bias_initialization(num_samples, feature_dim):

        BS_tilt_tf_GUEs = tf.constant([[
            -14.2892, - 13.3303, - 13.8357,   10.7650,    7.9629,    8.9450, - 11.6216, - 13.5172, - 11.8110, - 11.6466,
            - 14.5059, - 13.6779, - 12.7726, - 13.8389, - 11.6859, - 12.6570, - 12.7154, - 14.1510,    7.0658, - 10.3518,
            - 9.3582, - 13.3084, - 12.9983, - 14.5288, - 14.4601, - 12.7762, - 12.9617, - 13.5321, - 13.6170, - 13.4990,
            - 13.3583, - 13.6499, - 13.0595, - 12.4591, - 12.2421, - 11.1918, - 12.3228, - 11.3664, - 13.9558, - 11.9876,
            - 12.4522, - 12.4918, - 11.3783, - 14.2981, - 12.8244, - 13.6442, - 14.6585, - 13.6940, - 13.0231, - 11.1856,
            - 12.9481, - 11.1927, - 13.4301, - 12.4440, - 12.4617,    7.1116, - 13.5773]])
        BS_tilt_torch_GUEs = torch.tensor(BS_tilt_tf_GUEs.numpy(), dtype=torch.float64)

        BS_tilt_tf_UAVs = tf.constant([[
            -13.4910, - 14.3924, - 13.4719, 9.6376, 11.7109, 11.6339, 11.4240, 13.8082, - 9.5150, - 14.3014,
            13.0297, - 11.2548, 14.3674, 13.0484, - 10.5345, - 9.7556, 13.8271, 12.5596, - 9.2227, - 12.9586,
            11.4705, 14.2217, 12.4362, - 10.1060, - 9.8232, - 13.0684, - 12.5620, - 12.2552, - 13.7850, 13.4906,
            - 13.3329, - 11.3929, - 12.3790, 10.0597, 12.9263, - 10.4925, - 10.7433, - 11.1133, - 14.9433, 14.4689,
            - 14.2804, - 10.3791, - 12.6751, - 9.2092, 14.5388, - 10.0850, 13.5035, - 12.6803, 11.0223, 14.9165,
            - 12.6413, 12.9107, - 14.3168, 14.6330, 10.2188, 10.0585, 13.9559]])
        BS_tilt_torch_UAVs = torch.tensor(BS_tilt_tf_UAVs.numpy(), dtype=torch.float64)

        BS_tilt_tf_both = tf.constant([[
            -13.9327, - 13.2903, - 12.8647, 10.5337, 9.9584, 12.8540, - 13.7302, - 12.0697, - 12.2952, - 12.9120,
            11.3224, - 13.4506, 12.7093, - 12.6343, - 12.1007, - 12.9844, 11.3501, 12.0331, 11.8313, - 12.1021,
            - 13.6658, 10.4538, 13.8567, - 12.9336, - 13.0059, - 12.8360, - 12.8065, - 12.2845, - 13.9383, - 13.4531,
            - 13.0222, 12.4988, - 14.2656, 12.3700, - 12.1800, - 13.2468, - 13.7995, - 13.0525, - 12.7080, - 12.7230,
            - 12.7104, - 13.1767, - 11.7174, - 13.1188, - 12.4295, - 13.1245, 14.4940, - 13.1295, - 13.9200, - 12.7240,
            - 13.2607, - 11.5826, - 12.8363, 13.1386, - 12.6877, - 13.2881, - 11.5168]])

        BS_tilt_torch_both = torch.tensor(BS_tilt_tf_both.numpy(), dtype=torch.float64)

        # Step 2: Generate X_init based on BS_tilt with random +/- 2 adjustments
        adjustments = (torch.rand((num_samples, feature_dim)) * 4) - 2  # Generate random values in [-2, 2]
        adjustments2 = (torch.rand((num_samples+150, feature_dim)) * 4) - 2  # Generate random values in [-2, 2]
        X_init_GUEs = BS_tilt_torch_GUEs + adjustments
        X_init_UAVs = BS_tilt_torch_UAVs + adjustments
        X_init_both = BS_tilt_torch_both + adjustments2
        X_init = torch.cat((X_init_GUEs, X_init_UAVs, X_init_both), dim=0)

        # Step 3: Normalize the values from [-15, 20] to [0, 1]
        old_min, old_max = -15, 20
        new_min, new_max = 0, 1
        X_init = ((X_init - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

        return X_init

    X_init = bias_initialization(num_samples=50, feature_dim=57)

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
        upper_bound_WiSe = 20.0
        bounds_WiSe = torch.cat((torch.zeros(1, dim) + lower_bound_WiSe, torch.zeros(1, dim) + upper_bound_WiSe))
        Thresholds_WiSe = unnormalize(X_cand, bounds_WiSe)
        ## Save the output
        Obj_WiSe = Y_cand
        Thresholds_WiSe = Thresholds_WiSe.numpy()
        Thresholds_WiSe = tf.convert_to_tensor(Thresholds_WiSe)
        Obj_WiSe = Obj_WiSe.numpy()
        Obj_WiSe = tf.convert_to_tensor(Obj_WiSe)

        if counter_WiSE == 0:
            best_value_all = tf.zeros(Obj_WiSe.shape, dtype='float64')
        best_value_all = tf.concat([best_value_all, Obj_WiSe], axis=0)

        data_BO_WiSe = {"Thresholds": Thresholds_WiSe.numpy(),
                   "Obj": Obj_WiSe.numpy(),
                    "Obj_all":best_value_all.numpy()}
        file_name = "2024_02_22_MORBO_corr_50m_tilts_Rate_bias_iteration{}.mat".format(counter_WiSE)
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
        seed=1122,
        label="morbo",
        max_evals=2000,
        evalfn="WiSE",
        batch_size=1,
        dim=57,
        n_initial_points=5*57,
        max_reference_point=[-5.0,-5.0])

# d = {"SINR_UAVs": 10 * np.log10(sinr_total_UAVs.numpy()),
#      "SINR_GUEs": 10 * np.log10(sinr_total_GUEs.numpy()),
#      "Rate_UAVs": Rate_UAVs.numpy(),
#      "Rate_GUEs": Rate_GUEs.numpy()}
# savemat("2023_09_28_IterativeBO_OneTier_GUEs_iteration{}.mat", d)