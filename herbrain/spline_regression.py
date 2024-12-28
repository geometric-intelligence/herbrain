import pandas as pd
import pyvista as pv
from pathlib import Path
from herbrain.lddmm import preprocess_full_structure, preprocess_substructure

project_dir = Path('/user/nguigui/home/Documents/UCSB')
data_dir = project_dir / 'meshes_adele' / 'b_centered'
output_dir = project_dir / 'meshes_nico'
output_dir.mkdir(exist_ok=True)
smooth_dir = output_dir / 'full_structure' / 'raw'
smooth_dir.mkdir(parents=True, exist_ok=True)

zones = [0, 1, 5]
for z in zones:
    (output_dir / f'structure_{z}' / 'raw').mkdir(parents=True, exist_ok=True)

ref_mesh = pv.read(data_dir / 'left_structure_-1_day01.ply')
df = pd.DataFrame(ref_mesh.get_array('RGBA'))
unique_rows = df.value_counts().index
ref_labels = {row: i for i, row in enumerate(unique_rows)}

for day in range(1, 26):
    name = f'left_structure_-1_day{day:02}.ply'
    decimate_name = f'full_structure_t{day:02}.vtk'
    try:
        day_mesh = pv.read(data_dir / name)
        preproc_mesh = preprocess_full_structure(day_mesh)
        preproc_mesh.save(smooth_dir / decimate_name)

        for z in [0, 1, 5]:
            struc_name = f'structure_{z}_t{day:02}.vtk'
            substruc_mesh = preprocess_substructure(day_mesh, z, ref_labels)
            substruc_mesh.save(output_dir / f'structure_{z}' / 'raw' / struc_name)

    except FileNotFoundError:
        continue

# day = 7
# new_name = f'left_structure_t{day:02}.vtk'
# source = output_dir.parent / 'b_centered' / new_name
# day = 18
# new_name = f'left_structure_t{day:02}.vtk'
# target = output_dir.parent / 'b_centered' / new_name
#
# registration(
#     source, target, project_dir / 'registration', kernel_width=5., metric='varifold',
#     max_iter=20, print_every=1)
