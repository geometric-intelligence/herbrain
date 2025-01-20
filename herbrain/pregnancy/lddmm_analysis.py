import pandas as pd
from pathlib import Path

from herbrain.pregnancy.configurations import configurations
from herbrain.regression import main as regression
from preprocessing import main as preprocess
from preprocessing import swap_left_right


def get_data_set(
        data_dir: Path, covariate: pd.DataFrame, hemisphere='left', structure='ERC',
        tmin=0, tmax=10, time_var='gestWeek', day_ref=0, variable='times'):
    data_list = covariate[covariate[structure]]  # remove excluded
    data_list = data_list[
        ((tmin < data_list[time_var]) & (data_list[time_var] < tmax))
        | (data_list['day'] == day_ref)]  # select phase + template at day ref
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

# The following left and right segmentations are swapped
swap_list = [1, 2, 3, 4, 5, 6, 18, 19]
# swap_left_right(swap_list, data_dir_)

# preprocess: align, smooth and decimate + extract subregions
# for side_ in ["left"]:  # , "right"]:
#     preprocess(1, 26, 1, side_, data_dir_, out_dir)

# compute fraction of gest time and add to covariates
covariates = pd.read_csv(project_dir / '28Baby_Hormones.csv')
covariates['day'] = covariates.sessionID.str.split('-').apply(lambda k: k[-1])
covariates['times'] = covariates.gestWeek / 40

nice_zones = ["PRC", "PHC", "PostHipp", "CA2+3", "ERC"]
covariates[nice_zones] = True

# some subregions are not segmented on a few sessions
covariates.loc[covariates.sessionID == 'ses-13', 'PostHipp'] = False
covariates.loc[covariates.sessionID == 'ses-14', 'PostHipp'] = False
covariates.loc[covariates.sessionID == 'ses-15', nice_zones] = False
covariates.loc[covariates.sessionID == 'ses-27', nice_zones] = False
covariates.to_csv(project_dir / 'covariates.csv', index=False)

for config in configurations[0:1]:
    data_set, times = get_data_set(out_dir, covariates, **config['dataset'])
    del config['dataset']
    regression(data_set, times, out_dir, **config)
