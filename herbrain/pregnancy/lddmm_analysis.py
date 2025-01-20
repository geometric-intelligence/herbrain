import pandas as pd
from pathlib import Path

from herbrain.pregnancy.configurations import configurations
from herbrain.regression import main as regression
from preprocessing import main as preprocess
from preprocessing import swap_left_right

project_dir = Path('/user/nguigui/home/Documents/UCSB')
data_dir_ = project_dir / 'meshes_adele' / 'a_meshed'
out_dir = project_dir / 'pregnancy' / 'meshes_nico'

# The following left and right segmentations are swapped
swap_list = [1, 2, 3, 4, 5, 6, 18, 19]
swap_left_right(swap_list, data_dir_)

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
covariates.loc[covariates.sessionID == 'ses-27', nice_zones] = False
covariates.to_csv(project_dir / 'covariates.csv', index=False)

for config in configurations[4:]:
    regression(covariates, out_dir, **config)
