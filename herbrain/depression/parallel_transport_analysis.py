from pathlib import Path
import numpy as np
import pandas as pd

import herbrain.lddmm as lddmm
from herbrain.parallel_transport import main as parallel_transport


output_dir = Path()

# compute atlas from control group at t0
data_set = [
    {'shape': 'path/to/data/control/subject1/t0'},
    {'shape': 'path/to/data/control/subject2/t0'},
]

# fine tune depending on data
registration_args = dict(
    kernel_width=4., regularisation=1, max_iter=2000,
    freeze_control_points=False, metric='varifold', attachment_kernel_width=2.,
    tol=1e-10, filter_cp=True, threshold=0.75)

n_rungs = 10
transport_args = {'kernel_type': 'torch', 'kernel_width': 4, 'n_rungs': n_rungs}

shoot_args = {
    'use_rk2_for_flow': True, 'kernel_width': 4.,
    'number_of_time_steps': n_rungs + 1, 'write_params': False}


atlas_dir = output_dir / 'atlas'
lddmm.deterministic_atlas(
    data_set[0]['shape'], data_set, 'structure', atlas_dir,
    initial_step_size=1e-1, **registration_args)

atlas = atlas_dir / 'atlas'

# parallel transport
subjects = []
data = pd.DataFrame()
for name in subjects:
    source = f'path/to/data/control/{name}/t0'
    target = f'path/to/data/control/{name}/t1'
    shoot_args['source'] = source
    mom_path = parallel_transport(
        source, target, atlas, name, output_dir, registration_args, transport_args,
        shoot_args)

    momenta = np.array(lddmm.read_2D_array(mom_path)[1:])
    series = pd.Series(
        [name] + momenta.flatten().tolist())
    data = data.append(series, ignore_index=True)


data.columns = ['pid'] + [f'mom_{k}_{i}' for k in range(60) for i in range(1, 4)]
all_data = data.set_index('pid')
all_data.to_csv(output_dir / 'transported_momenta.csv')
