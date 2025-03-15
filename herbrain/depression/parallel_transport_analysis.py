from pathlib import Path
import numpy as np
import pandas as pd
import pyvista as pv
import statsmodels.api as sm

import herbrain.lddmm as lddmm
import strings
from herbrain.depression.hotelling import hotelling_t2
from herbrain.parallel_transport import main as parallel_transport
from support.kernels.torch_kernel import TorchKernel


output_dir = Path()

# compute atlas from control group at t0
data_set = [
    {'shape': 'path/to/data/control/subject1/t0'},
    {'shape': 'path/to/data/control/subject2/t0'},
]

# fine tune depending on data
kernel_width = 4
registration_args = dict(
    kernel_width=kernel_width, regularisation=1, max_iter=2000,
    freeze_control_points=False, metric='varifold', attachment_kernel_width=2.,
    tol=1e-10, filter_cp=True, threshold=0.75)

n_rungs = 10
transport_args = {
    'kernel_type': 'torch', 'kernel_width': kernel_width, 'n_rungs': n_rungs}

shoot_args = {
    'use_rk2_for_flow': True, 'kernel_width': kernel_width,
    'number_of_time_steps': n_rungs + 1, 'write_params': False}


atlas_dir = output_dir / 'atlas'
lddmm.deterministic_atlas(
    data_set[0]['shape'], data_set, 'structure', atlas_dir,
    initial_step_size=1e-1, **registration_args)
atlas_cp_path = atlas_dir / strings.cp_str
atlas = atlas_dir / 'atlas'

atlas_cp = lddmm.read_2D_array(atlas_cp_path)
n_cp = atlas_cp.shape[-1]
kernel = TorchKernel(kernel_width=kernel_width, device='auto')

# parallel transport
subjects = []
transport_result = pd.DataFrame()
for name in subjects:
    source = f'path/to/data/control/{name}/t0'
    target = f'path/to/data/control/{name}/t1'
    shoot_args['source'] = source
    cp, mom = parallel_transport(
        source, target, atlas, name, output_dir, registration_args, transport_args,
        shoot_args)

    # Compute the velocity field at atlas cp because all patients have momentas
    # supported at different control points, and velocity is more stable/smooth
    vel = kernel.convolve(atlas_cp, cp, mom)
    series = pd.Series([name] + vel.flatten().tolist())
    transport_result = transport_result.append(series, ignore_index=True)

transport_result.columns = ['pid'] + [
    f'vel_{k}_{i}' for k in range(n_cp) for i in range(1, 4)]
transport_result = transport_result.set_index('pid')
vel_columns = transport_result.columns[transport_result.columns.str.contains('vel')]

# Add the disease column, based on stg
transport_result['disease'] = 'control'
transport_result.to_csv(output_dir / 'transported_vel.csv')

grouped = transport_result.groupby('disease').mean()
controls = transport_result.loc[transport_result.disease == 'control', vel_columns]
patients = transport_result.loc[transport_result.disease == 'PPD', vel_columns]

# Hotelling test
pval, t2 = hotelling_t2(controls, patients)
reject, corrected_pval, _, fwer = sm.stats.multipletests(pval, method='bonferroni')

mean_vel_ppd = grouped.loc['PPD', vel_columns].values
mean_vel_ctl = grouped.loc['control', vel_columns].values
significative_vel = reject[:, None] * mean_vel_ppd
significative_vel_dif = reject[:, None] * (mean_vel_ppd - mean_vel_ctl).numpy()

poly = pv.PolyData(atlas_cp)
poly['SignificantMeanVelocity'] = significative_vel
poly['SignificantMeanVelocityDiff'] = significative_vel_dif
poly['Hotelling_pval'] = np.log(corrected_pval)
poly['Hotelling_reject'] = reject.astype(float)
poly['MeanVelocityPPD'] = mean_vel_ppd
poly['MeanVelocityControl'] = mean_vel_ctl
poly.save(output_dir / 'hotelling_cp.vtk')
