"""
Created on Thu Nov 16 11:57:05 2023

This is the VS-BO for optimizing wirelees networks
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
object_dim = 57
lower_bound = -20.0
upper_bound = 0.0
bounds = torch.cat((torch.zeros(1, object_dim) + lower_bound, torch.zeros(1, object_dim) + upper_bound))
total_budget = 400

###Initial simulator parameters
data_size = 100
# tilts_vector = tf.expand_dims(tf.expand_dims(tf.random.uniform((57,), 0.0, 0.0, tf.float32), axis=0),axis=2)
tilts_vector = tf.expand_dims(tf.constant([[
    -10.7028, - 13.2502, - 10.8893, - 12.9726, - 11.8176, - 11.4146, - 10.0879, - 10.6533, - 12.6312, - 12.0264,
    - 12.1142, - 12.4132, - 11.1537, - 10.8106, - 12.4998, - 11.4740, - 15.0189, - 11.0204, - 12.9443, - 12.8542,
    - 12.7713, - 12.3419, - 11.4416, - 13.2092, - 11.1103, - 10.3184, - 12.7448, - 12.5280, - 13.7621, - 13.2310,
    - 12.9288, - 10.5971, - 12.1113, - 13.5342, - 12.0637, - 12.9327, - 11.9668, - 14.3249, - 11.6192, - 12.7023,
    - 12.5337, - 12.3194, - 11.2269, - 11.1509, - 14.1286, - 10.3987, - 12.8748, - 10.2173, - 11.7782, - 12.5399,
    - 10.7587, - 10.1067, - 12.4050, - 12.6742, - 13.5555, - 11.6321, - 11.5568]]), axis=2)
HPBW_v_vector = tf.expand_dims(tf.expand_dims(tf.random.uniform((57,), 10.0, 10.0, tf.float32), axis=0),axis=2)
Ptx_thresholds_vector = tf.expand_dims(tf.expand_dims(tf.random.uniform((57,), 46.0, 46.0, tf.float32), axis=0),axis=2)
obj_vector = torch.tensor([[-1.6746]], dtype=torch.double)

### Run VS-BO
BO_instance = VSBO(N_FS,object_dim, bounds, data_size, tilts_vector, HPBW_v_vector, Ptx_thresholds_vector, obj_vector)
BO_instance.data_initialize()
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
        BO_instance.data_update(method=less_important_sampling, n_sampling=20)

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
        # print(BO_instance.active_f_list)
        print(BO_instance.active_f_dims)
    if (iter_num % 10 == 0):
        print(
            f"Epoch {iter_num:>3} "
            f"Best value: {torch.max(BO_instance.Y).item():>4.3f}")