import pandas as pd
import numpy as np
from pathlib import Path

import lddmm
from herbrain.pregnancy.configurations import configurations
from herbrain.regression import main as regression
from preprocessing import main as preprocess
from preprocessing import swap_left_right
from strings import residual_str, residual_str_spline, template_str


def get_data_set(
        data_dir: Path, covariate: pd.DataFrame, hemisphere='left', structure='ERC',
        tmin=0, tmax=10, time_var='gestWeek', day_ref=0, variable='times'):
    data_list = covariate[covariate[structure]]  # remove excluded
    data_list = data_list[
        ((tmin < data_list[time_var]) & (data_list[time_var] < tmax))
        | (data_list.day.astype(int) == day_ref)]  # select phase + template at day ref
    data_list = data_list.dropna(subset=variable)
    data_path = data_dir / structure / 'raw'
    dataset = [
        {
            'shape': data_path / f'{hemisphere}_{structure}_t{k:02}.vtk'
         } for k in data_list['day']]
    return dataset, data_list[variable]


project_dir = Path('/user/nguigui/home/Documents/UCSB')
data_dir_ = project_dir / 'meshes_adele' / 'a_meshed'
out_dir = project_dir / 'pregnancy' / 'meshes_nico'

# The following left and right segmentations are swapped - Do only once
swap_list = [1, 2, 3, 4, 5, 6, 18, 19]
swap_left_right(swap_list, data_dir_)

# preprocess: align, smooth and decimate + extract subregions
for side_ in ["left"]:  # , "right"]:
    preprocess(1, 26, 1, side_, data_dir_, out_dir)

# compute fraction of gest time and add to covariates
covariates = pd.read_csv(project_dir / '28Baby_Hormones.csv')
covariates['day'] = covariates.sessionID.str.split('-').apply(lambda k: k[
    -1]).astype(int)
covariates['times'] = covariates.gestWeek / 40

nice_zones = ["PRC", "PHC", "PostHipp", "CA2+3", "ERC"]
covariates[nice_zones] = True

# some subregions are not segmented on a few sessions
covariates.loc[covariates.sessionID == 'ses-13', 'PostHipp'] = False
covariates.loc[covariates.sessionID == 'ses-14', 'PostHipp'] = False
covariates.loc[covariates.sessionID == 'ses-15', nice_zones] = False
covariates.loc[covariates.sessionID == 'ses-27', nice_zones] = False
covariates.to_csv(project_dir / 'covariates.csv', index=False)

results_csv = (out_dir / 'results.csv')
if results_csv.exists():
    results = pd.read_csv(results_csv)
else:
    results = pd.DataFrame(columns=['config', 'structure', 'r2'])
for config in configurations[8:]:
    data_set, times = get_data_set(out_dir, covariates, **config['dataset'])
    structure = config['dataset']['structure']
    config_id = config['config_id']
    del config['dataset']

    regression(data_set, times, out_dir, **config)
    atlas_dir = out_dir / structure / config_id / 'atlas'


    lddmm.deterministic_atlas(
        data_set[0]['shape'], data_set,
        structure, atlas_dir, initial_step_size=1e-1, **config['registration_args'])

    # Compute varifold distance between atlas and all datapoints
    atlas_dir_ = out_dir / structure / config_id / 'atlas_frozen'
    del config['registration_args']['max_iter']
    lddmm.deterministic_atlas(
        atlas_dir / template_str, data_set,
        structure, atlas_dir_, initial_step_size=1e-10, max_iter=0,
        **config['registration_args'])

    regression_dir = out_dir / structure / config_id / 'regression'
    residues_reg = lddmm.read_2D_array(regression_dir / residual_str_spline)
    residues_atlas = lddmm.read_2D_array(atlas_dir_ / residual_str)

    r2 = 1 - np.sum(residues_reg) / np.sum(residues_atlas)
    new_row = pd.DataFrame([{"structure": structure, "config": config_id, "r2": r2}])
    results = pd.concat([results, new_row], ignore_index=True)
    print('===========================================================================')
    print(config_id, r2)
    print('===========================================================================')

results.to_csv(out_dir / 'results.csv')
