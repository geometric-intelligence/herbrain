import pandas as pd
import pyvista as pv


def preprocess_full_structure(mesh: pv.DataSet, target_mesh=None):
    aligned = mesh.align(target_mesh, max_iterations=100)
    new_mesh = aligned.smooth_taubin(n_iter=20)
    target_reduction = 1 - (5000 / mesh.n_points)
    new_mesh = new_mesh.decimate(target_reduction)
    return new_mesh, aligned


def preprocess_substructure(
        mesh: pv.DataSet, label: str, labels: dict, align=False,
        target_mesh=None):
    new_mesh = mesh
    labels_df = pd.DataFrame(new_mesh.get_array('RGBA'))
    labels_df['label'] = labels_df.apply(tuple, axis=1).map(labels)
    new_mesh['label'] = labels_df['label']
    structure = new_mesh.extract_points(
        labels_df.loc[labels_df.label == label].index, adjacent_cells=False)
    if align and target_mesh is not None:
        structure = structure.align(target_mesh, max_iterations=10)
    return structure.extract_surface().smooth_taubin(20)


if __name__ == '__main__':
    from pathlib import Path

    project_dir = Path('/user/nguigui/home/Documents/UCSB')
    data_dir = project_dir / 'meshes_adele' / 'a_meshed'
    output_dir = project_dir / 'meshes_nico'
    output_dir.mkdir(exist_ok=True)
    smooth_dir = output_dir / 'full_structure' / 'raw'
    smooth_dir.mkdir(parents=True, exist_ok=True)

    zones = ["PRC", "PHC", "PostHipp", "CA2+3", "ERC"]
    for z in zones:
        (output_dir / z / 'raw').mkdir(parents=True, exist_ok=True)

    ref_labels = {
        (255, 0, 255, 255): "PRC",
        (0, 255, 255, 255): "PHC",
        (255, 215, 0, 255): "AntHipp",
        (255, 255, 0, 255): "ERC",
        (80, 179, 221, 255): "SUB",
        (184, 115, 51, 255): "PostHipp",
        (255, 0, 0, 255): "CA1",
        (0, 0, 255, 255): "DG",
        (0, 255, 0, 255): "CA2+3"}

    for side in ["left"]:  # , "right"]:
        ref_mesh = pv.read(data_dir / f'{side}_structure_-1_day01.ply')
        target_struct = {k: None for k in zones}

        for day in range(1, 26):
            name = f'{side}_structure_-1_day{day:02}.ply'
            decimate_name = f'{side}_full_{day:02}.vtk'

            try:
                day_mesh = pv.read(data_dir / name)
                preproc, aligned_day = preprocess_full_structure(day_mesh, ref_mesh)
                preproc.save(smooth_dir / decimate_name)

                for z in zones:
                    struc_name = f'{side}_{z}_t{day:02}.vtk'
                    substruc_mesh = preprocess_substructure(
                        aligned_day, z, ref_labels, align=(day > 1),
                        target_mesh=target_struct[z])
                    substruc_mesh.save(output_dir / z / 'raw' / struc_name)
                    if day == 1:
                        target_struct[z] = substruc_mesh

            except FileNotFoundError:
                continue
