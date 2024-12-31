import pandas as pd
from pathlib import Path
from herbrain.lddmm import spline_regression


def get_data_set(
        data_dir: Path, hemisphere: str, structure: str, covariates: pd.DataFrame):
    data_list = covariates[covariates[structure]]  # remove excluded
    data_list = data_list[
        (1 < data_list['gestWeek']) & (data_list['gestWeek'] < 25)]  # select phase
    data_path = data_dir / structure / 'raw'
    dataset = [{'shape': data_path / f'{hemisphere}_{structure}_t01.vtk'}] + [
        {
            'shape': data_path / f'{hemisphere}_{structure}_t{k:02}.vtk'
         } for k in data_list['day']]
    return dataset, data_list['times']


project_dir = Path('/user/nguigui/home/Documents/UCSB')
output_dir = project_dir / 'meshes_nico'

covariate = pd.read_csv(project_dir / 'covariates.csv')
struct = "PostHipp"

spline_args = {
    'freeze_control_points': False,
    'max_iter': 1000,
    'kernel_width': 5.,
    'initial_step_size': 100,
    'freeze_external_forces': True,
    'use_rk2_for_flow': True,
    'regularisation': 5.,
    'tol': 1e-10, 'geodesic_weight': 1.,
    'metric': 'varifold'}

data_set, times = get_data_set(output_dir, 'left', struct, covariate)
times = (times - times.min()) / (times.max() - times.min())
target_weights = [1 / len(data_set)] * len(data_set)

spline_regression(
    source=data_set[0]['shape'],
    target=data_set,
    output_dir=output_dir / struct / 'regression',
    times=times.tolist(),
    t0=min(times),
    subject_id=[struct],
    target_weights=target_weights, **spline_args)

# plotter = pv.Plotter()
# plotter.add_mesh(target_struct[z], color='red')
# plotter.add_mesh(substruc_mesh_aligned, color='blue')
# plotter.add_mesh(substruc_mesh_unaligned, color='green')
# plotter.show()

