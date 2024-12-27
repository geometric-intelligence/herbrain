import pyvista as pv
from pathlib import Path
from herbrain.lddmm import registration, spline_regression

project_dir = Path('/user/nguigui/home/Documents/UCSB')
data_dir = project_dir / 'meshes_adele' / 'b_centered'
output_dir = project_dir / 'meshes_nicolas' / 'b_centered'
tmp_dir = Path('/tmp')

# for day in range(1, 26):
#     name = f'left_structure_-1_day{day:02}.ply'
#     new_name = f'left_structure_t{day:02}'
#     try:
#         mesh = pv.read(data_dir / name)
#         mesh.save((output_dir / new_name).with_suffix('.vtk'))
#     except FileNotFoundError:
#         continue

day = 7
new_name = f'left_structure_t{day:02}.vtk'
source = output_dir / new_name
day = 18
new_name = f'left_structure_t{day:02}.vtk'
target = output_dir / new_name

registration(
    source, target, project_dir / 'registration', kernel_width=5., metric='varifold')
