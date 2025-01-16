import pandas as pd
from pathlib import Path

from herbrain.pregnancy.configurations import configurations
from herbrain.regression import main as regression
from preprocessing import main as preprocess

project_dir = Path('/user/nguigui/home/Documents/UCSB')
data_dir_ = project_dir / 'meshes_adele' / 'a_meshed'
out_dir = project_dir / 'pregnancy' / 'meshes_nico'

# The following left and right segmentations are swapped
swap_list = [1, 2, 3, 4, 5, 6, 18, 19]
tmp_dir = data_dir_ / 'tmp'
tmp_dir.mkdir(exist_ok=True)
for day in swap_list:
    file_left = data_dir_ / f'left_structure_-1_day{day:02}.ply'
    file_left_tmp = tmp_dir / file_left.name
    file_left.rename(file_left_tmp)
    file_right = data_dir_ / f'right_structure_-1_day{day:02}.ply'
    file_right.rename(data_dir_ / f'left_structure_-1_day{day:02}.ply')
    file_left_tmp.rename(file_right)

# preprocess: align, smooth and decimate + extract subregions
for side_ in ["left"]:  # , "right"]:
    preprocess(1, 26, 1, side_, data_dir_, out_dir)

# compute fraction of gest time and add to covariates
covariates = pd.read_csv(project_dir / '28Baby_Hormones.csv')
covariates['day'] = covariates.sessionID.str.split('-').apply(lambda k: k[-1])
covariates['times'] = covariates.gestWeek / 40

nice_zones = ["PRC", "PHC", "PostHipp", "CA2+3", "ERC"]
for zo in nice_zones:
    covariates[zo] = True

# some subregions are not segmented on a few sessions
covariates.loc[covariates.sessionID == 'ses-13', 'PostHipp'] = False
covariates.loc[covariates.sessionID == 'ses-14', 'PostHipp'] = False
covariates.loc[covariates.sessionID == 'ses-15', nice_zones] = False
covariates.to_csv(project_dir / 'covariates.csv', index=False)

for config in configurations[-1]:
    regression(covariates, out_dir, **config)
