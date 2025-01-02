import pandas as pd
from pathlib import Path

import herbrain.lddmm as lddmm
from strings import cp_str


def get_data_set(
        data_dir: Path, hemisphere: str, structure: str, covariates: pd.DataFrame):
    data_list = covariates[covariates[structure]]  # remove excluded
    data_list = data_list[
        ((1 < data_list['gestWeek']) & (data_list['gestWeek'] < 25))
        | (data_list['day'] == 3)]  # select phase + template at day 3
    data_path = data_dir / structure / 'raw'
    dataset = [
        {
            'shape': data_path / f'{hemisphere}_{structure}_t{k:02}.vtk'
         } for k in data_list['day']]
    return dataset, data_list['times']


project_dir = Path('/user/nguigui/home/Documents/UCSB')
output_dir = project_dir / 'meshes_nico'

covariate = pd.read_csv(project_dir / 'covariates.csv')
struct = "PostHipp"

data_set, times = get_data_set(output_dir, 'left', struct, covariate)
times = (times - times.min()) / (times.max() - times.min())
target_weights = [1 / len(data_set)] * len(data_set)

# registration to optimize control points
source = data_set[0]['shape']
target = data_set[-1]['shape']
registration_args = {
    'kernel_width': 4.,
    'regularisation': 1,
    'max_iter': 2000,
    'freeze_control_points': False,
    'metric': 'varifold',
    'tol': 1e-10,
    'filter_cp': True,
    'threshold': .75}

registration_dir = output_dir / struct / 'inital_registration'
lddmm.registration(source, target, registration_dir, **registration_args)


spline_args = registration_args.copy()
spline_args.update({
    'initial_step_size': 100,
    'regularisation': 1.,
    'freeze_external_forces': True,
    'freeze_control_points': True,
    'initial_control_points': registration_dir / cp_str,
})

regression_dir = output_dir / struct / 'regression'
lddmm.spline_regression(
    source=data_set[0]['shape'],
    target=data_set,
    output_dir=regression_dir,
    times=times.tolist(),
    t0=min(times),
    subject_id=[''],
    target_weights=target_weights, **spline_args)

# plotter = pv.Plotter()
# plotter.add_mesh(target_struct[z], color='red')
# plotter.add_mesh(substruc_mesh_aligned, color='blue')
# plotter.add_mesh(substruc_mesh_unaligned, color='green')
# plotter.show()

