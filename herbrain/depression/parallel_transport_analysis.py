from pathlib import Path

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

atlas_dir = output_dir / 'atlas'
lddmm.deterministic_atlas(
    data_set[0]['shape'], data_set, 'structure', atlas_dir,
    initial_step_size=1e-1, **registration_args)

atlas = atlas_dir /

# parallel transport
subjects = []
for name in subjects:
    source = f'path/to/data/control/{name}/t0'
    target = f'path/to/data/control/{name}/t1'
    # parallel_transport(source, target)