"""
Created on Thu Nov 16 11:57:05 2023

This is the VS-BO for optimizing wireless networks
@author: Mohamed Benzaghta
"""

from VSBO_class import *
import argparse

parser = argparse.ArgumentParser('VS-BO')
parser.add_argument('--method', type=str,default='VS-BO')
parser.add_argument('--momentum',type=int,default=1)
parser.add_argument('--sampling',type=str,default='CMAES_posterior')

args = parser.parse_args()


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


### Initial parameters
N_FS = 20
acq_optim_method = 'LBFGS'
### use CMAES to sample unimportant variables
less_important_sampling = args.sampling
init_samples = 5
object_dim = 57*2

dim_tilts = 57
lower_bound_tilts = -15.0
upper_bound_tilts = 45.0
bounds_tilts = torch.cat((torch.zeros(1, dim_tilts) + lower_bound_tilts, torch.zeros(1, dim_tilts) + upper_bound_tilts))
dim_vHPBW = 57
lower_bound_vHPBW = 5.0
upper_bound_vHPBW = 30.0
bounds_vHPBW = torch.cat((torch.zeros(1, dim_vHPBW) + lower_bound_vHPBW, torch.zeros(1, dim_vHPBW) + upper_bound_vHPBW))
bounds = torch.cat((bounds_tilts, bounds_vHPBW), dim=1)
total_budget = 800

###Initial simulator parameters
data_size = 2*114
tilts_vector = tf.expand_dims(tf.expand_dims(tf.random.uniform((57,), 0.0, 0.0, tf.float32), axis=0),axis=2)
HPBW_v_vector = tf.expand_dims(tf.expand_dims(tf.random.uniform((57,), 10.0, 10.0, tf.float32), axis=0),axis=2)
Ptx_thresholds_vector = tf.expand_dims(tf.expand_dims(tf.random.uniform((57,), 46.0, 46.0, tf.float32), axis=0),axis=2)
obj_vector = torch.tensor([[-4.6746]], dtype=torch.double)

### Run VS-BO
BO_instance = VSBO(N_FS,object_dim, bounds, data_size, tilts_vector, HPBW_v_vector, Ptx_thresholds_vector, obj_vector)
BO_instance.data_initialize_tilt_vHPBW()
if (less_important_sampling == 'CMAES_posterior'):
    BO_instance.CMAES_initialize()

F_importance_val = []
F_rank = []
F_chosen = []
iter_num = 0

while (iter_num < total_budget):

    # Perform the VS-BO
    iter_num += 1
    try:
        ### GP fitting on important variables
        BO_instance.GP_fitting_active(GP_Matern)
        BO_instance.BO_acq_optim_active(optim_method=acq_optim_method)
        ### sampling on unimportant variables
        BO_instance.data_update_tilt_vHPBW(iter_num, method=less_important_sampling, n_sampling=20)

    except ValueError as e:
        if (e.args[0] == 'Too many cov mat singular!'):
            BO_instance.erase_last_instance()
            iter_num -= 1
            continue
        else:
            raise ValueError(e.args[0])

    # Re-selection of important variables after N_FS is reached (20-iterations)
    if (iter_num % BO_instance.N_FS == 0):
        try:
            BO_instance.GP_fitting(GP_Matern)
            ### We use KLrel: the Grad-IS method introduced in our manuscript for variable seletion
            if args.momentum == 1:
                BO_instance.variable_selection_2('KLrel')
            elif args.momentum == 0:
                BO_instance.variable_selection_nomom('KLrel')
        except ValueError as e:
            if (e.args[0] == 'Too many cov mat singular!'):
                BO_instance.erase_last_instance()
                iter_num -= 1
                continue
            else:
                raise ValueError(e.args[0])
        F_importance_val.append(BO_instance.FS_important_scores)
        F_rank.append(BO_instance.indices)
        F_chosen.append(BO_instance.active_f_list)
        if (less_important_sampling == 'CMAES_posterior'):
            BO_instance.CMAES_update()
        #print(BO_instance.active_f_list)
        print(BO_instance.active_f_dims)
        data_VS = {"variables_list": BO_instance.active_f_list.numpy(),
                   "active_variables": BO_instance.active_f_dims}
        file_name = "2024_04_16_VSBO_Corr_150m_tilts_vHPBW_variables_iteration{}.mat".format(iter_num)
        savemat(file_name, data_VS)
    if (iter_num % 10 == 0):
        print(
            f"Epoch {iter_num:>3} "
            f"Best value: {torch.max(BO_instance.Y).item():>4.3f}")