import preprocessing
import herbrain.lddmm as lddmm


from pathlib import Path

project_dir = Path('/user/nguigui/home/Documents/UCSB/menstrual')
data_dir = project_dir / 'a_meshed'
output_dir = project_dir / 'meshes_nico'

ref_labels = preprocessing.get_ref_labels()

side = 'left'
# preprocessing.main(1, 26, 1, side, data_dir, output_dir)

# registration of day 1 - main geodesic
structure = 'PRC'
source_day = 1
struc_name = f'{side}_{structure}_t{source_day:02}.vtk'
source = output_dir / structure / 'raw' / struc_name

target_day = 30
struc_name = f'{side}_{structure}_t{target_day:02}.vtk'
target = output_dir / structure / 'raw' / struc_name

registration_dir = output_dir / structure / 'initial_registration'
registration_args = dict(
    kernel_width=4., regularisation=1., max_iter=2000,
    freeze_control_points=False, attachment_kernel_width=1., metric='varifold',
    tol=1e-10, filter_cp=True, threshold=0.75)

lddmm.registration(source, target, registration_dir, **registration_args)


