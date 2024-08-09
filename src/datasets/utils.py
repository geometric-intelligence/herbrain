"""Utils to import data."""

import glob
import os

import geomstats.backend as gs
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import trimesh
from geomstats.geometry.discrete_surfaces import (
    DiscreteSurfaces,
    DiscreteSurfacesExpSolver,
    ElasticMetric,
)
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.hyperbolic import Hyperbolic
from geomstats.geometry.hypersphere import Hypersphere

import H2_SurfaceMatch.utils.input_output as h2_io
import src.datasets.synthetic as synthetic
from src.import_project_config import import_default_config
from src.regression.geodesic_regression import RiemannianGradientDescent


def get_optimizer(use_cuda, n_vertices, max_iter=100, tol=1e-5):
    """Determine Optimizer based on use_cuda.

    If we are running on GPU, we use RiemannianGradientDescent.

    Parameters
    ----------
    use_cuda : bool. Whether to use GPU.
    n_vertices : int
    max_iter : int
    tol : float
    """
    if use_cuda:
        embedding_space_dim = 3 * n_vertices
        print("embedding_space_dim", embedding_space_dim)
        embedding_space = Euclidean(dim=embedding_space_dim)
        optimizer = RiemannianGradientDescent(
            max_iter=max_iter,
            init_step_size=0.1,
            tol=tol,
            verbose=False,
            space=embedding_space,
        )
    else:
        optimizer = None
    return optimizer


def load_synthetic_data(config):
    """Load synthetic data according to values in config file."""
    if config.device_id is None:
        torchdeviceId = torch.device("cuda:0") if config.use_cuda else "cpu"
    else:
        torchdeviceId = (
            torch.device(f"cuda:{config.device_id}") if config.use_cuda else "cpu"
        )

    project_dir = config.project_dir
    project_config = import_default_config(project_dir)
    if config.dataset_name == "synthetic_mesh":
        print("Using synthetic mesh data")
        data_dir = project_config.synthetic_data_dir
        start_shape, end_shape = config.start_shape, config.end_shape
        n_X = config.n_X
        n_subdivisions = config.n_subdivisions
        noise_factor = config.noise_factor
        project_linear_noise = config.project_linear_noise
        linear_noise = config.linear_noise

        mesh_dir = os.path.join(
            data_dir,
            f"geodesic_{start_shape}_{end_shape}_{n_X}_subs{n_subdivisions}"
            f"_noise{noise_factor}_projected{project_linear_noise}_linear_noise{linear_noise}",
        )

        mesh_sequence_vertices_path = os.path.join(
            mesh_dir, "mesh_sequence_vertices.npy"
        )
        mesh_faces_path = os.path.join(mesh_dir, "mesh_faces.npy")
        X_path = os.path.join(mesh_dir, "X.npy")
        true_intercept_path = os.path.join(mesh_dir, "true_intercept.npy")
        true_coef_path = os.path.join(mesh_dir, "true_coef.npy")

        noiseless_mesh_dir = os.path.join(
            data_dir,
            f"geodesic_{start_shape}_{end_shape}_{n_X}_subs{n_subdivisions}"
            f"_noise{0.0}",
        )

        noiseless_mesh_sequence_vertices_path = os.path.join(
            noiseless_mesh_dir, "mesh_sequence_vertices.npy"
        )
        noiseless_mesh_faces_path = os.path.join(noiseless_mesh_dir, "mesh_faces.npy")
        noiseless_X_path = os.path.join(noiseless_mesh_dir, "X.npy")
        noiseless_true_intercept_path = os.path.join(
            noiseless_mesh_dir, "true_intercept.npy"
        )
        noiseless_true_coef_path = os.path.join(noiseless_mesh_dir, "true_coef.npy")

        if os.path.exists(noiseless_mesh_dir):
            print(f"Noiseless geodesic exists in {mesh_dir}. Loading now.")
            noiseless_mesh_sequence_vertices = gs.array(
                np.load(noiseless_mesh_sequence_vertices_path)
            )
            mesh_faces = gs.array(np.load(noiseless_mesh_faces_path))
            X = gs.array(np.load(noiseless_X_path))
            true_intercept = gs.array(np.load(noiseless_true_intercept_path))
            true_coef = gs.array(np.load(noiseless_true_coef_path))
        else:
            print(
                f"Noiseless geodesic does not exist in {noiseless_mesh_dir}. Creating one."
            )
            start_mesh = load_mesh(start_shape, n_subdivisions, config)
            end_mesh = load_mesh(end_shape, n_subdivisions, config)

            (
                noiseless_mesh_sequence_vertices,
                mesh_faces,
                X,
                true_intercept,
                true_coef,
            ) = synthetic.generate_parameterized_mesh_geodesic(
                start_mesh, end_mesh, config, n_X, config.n_steps
            )

            os.makedirs(noiseless_mesh_dir)
            np.save(
                noiseless_mesh_sequence_vertices_path, noiseless_mesh_sequence_vertices
            )
            np.save(noiseless_mesh_faces_path, mesh_faces)
            np.save(noiseless_X_path, X)
            np.save(noiseless_true_intercept_path, true_intercept)
            np.save(noiseless_true_coef_path, true_coef)

        y_noiseless = noiseless_mesh_sequence_vertices

        faces = gs.array(mesh_faces)
        print("config.use_cuda: ", config.use_cuda)
        print("config.torch_dtype: ", config.torch_dtype)
        print("config.torchdeviceId: ", torchdeviceId)
        if config.use_cuda:
            faces = faces.to(torchdeviceId)

        space = DiscreteSurfaces(faces=gs.array(mesh_faces))
        elastic_metric = ElasticMetric(
            space=space,
            a0=project_config.a0,
            a1=project_config.a1,
            b1=project_config.b1,
            c1=project_config.c1,
            d1=project_config.d1,
            a2=project_config.a2,
        )
        optimizer = get_optimizer(
            config.use_cuda, n_vertices=len(y_noiseless[0]), max_iter=100, tol=1e-5
        )
        elastic_metric.exp_solver = DiscreteSurfacesExpSolver(
            space=space, n_steps=config.n_steps, optimizer=optimizer
        )
        space.metric = elastic_metric

        if os.path.exists(mesh_dir):
            print(f"Synthetic geodesic exists in {mesh_dir}. Loading now.")
            mesh_sequence_vertices = gs.array(np.load(mesh_sequence_vertices_path))
            mesh_faces = gs.array(np.load(mesh_faces_path))
            X = gs.array(np.load(X_path))
            true_intercept = gs.array(np.load(true_intercept_path))
            true_coef = gs.array(np.load(true_coef_path))

            y = mesh_sequence_vertices
            return space, y, y_noiseless, X, true_intercept, true_coef

        print(f"No noisy synthetic geodesic found in {mesh_dir}. Creating one.")
        # projecting linear noise does not apply to meshes
        # project_linear_noise = config.project_linear_noise
        # mesh_sequence_vertices = synthetic.add_linear_noise(
        #     space,
        #     noiseless_mesh_sequence_vertices,
        #     config.dataset_name,
        #     project_linear_noise,
        #     noise_factor=noise_factor,
        # )

        space = DiscreteSurfaces(faces=gs.array(mesh_faces))
        print(f"space faces: {space.faces.shape}")
        elastic_metric = ElasticMetric(
            space=space,
            a0=project_config.a0,
            a1=project_config.a1,
            b1=project_config.b1,
            c1=project_config.c1,
            d1=project_config.d1,
            a2=project_config.a2,
        )
        optimizer = get_optimizer(
            config.use_cuda, n_vertices=len(y_noiseless[0]), max_iter=100, tol=1e-5
        )
        elastic_metric.exp_solver = DiscreteSurfacesExpSolver(
            space=space, n_steps=config.n_steps, optimizer=optimizer
        )
        space.metric = elastic_metric

        if config.linear_noise:
            print(f"noise factor: {config.noise_factor}")
            print(f"dataset name: {config.dataset_name}")
            print(f"y noiseless shape: {y_noiseless.shape}")
            mesh_sequence_vertices = synthetic.add_linear_noise(
                space,
                noiseless_mesh_sequence_vertices,
                config.dataset_name,
                config.project_linear_noise,
                noise_factor=config.noise_factor,
                random_seed=config.random_seed,
            )
        else:
            mesh_sequence_vertices = synthetic.add_geodesic_noise(
                space,
                noiseless_mesh_sequence_vertices,
                config.dataset_name,
                noise_factor=config.noise_factor,
                random_seed=config.random_seed,
            )

        print("Noisy mesh_sequence vertices: ", mesh_sequence_vertices.shape)
        print("Noisy mesh faces: ", mesh_faces.shape)
        print("X: ", X.shape)

        os.makedirs(mesh_dir)
        np.save(mesh_sequence_vertices_path, mesh_sequence_vertices)
        np.save(mesh_faces_path, mesh_faces)
        np.save(X_path, X)
        np.save(true_intercept_path, true_intercept)
        np.save(true_coef_path, true_coef)

        y = mesh_sequence_vertices
        return space, y, y_noiseless, X, true_intercept, true_coef

    elif config.dataset_name in ["hyperboloid", "hypersphere"]:
        print(f"Creating synthetic dataset on {config.dataset_name}")
        if config.dataset_name == "hyperboloid":
            space = Hyperbolic(dim=config.space_dimension, coords_type="extrinsic")
        else:
            space = Hypersphere(dim=config.space_dimension)

        # X, y_noiseless, y_noisy, true_intercept, true_coef = synthetic.generate_noisy_benchmark_data(space = space, linear_noise=config.linear_noise, dataset_name=config.dataset_name, n_samples=config.n_X, noise_factor=config.noise_factor)
        X, y_noiseless, true_intercept, true_coef = synthetic.generate_general_geodesic(
            space, config.n_X, config.synthetic_tan_vec_length
        )
        if config.linear_noise:
            print(f"noise factor: {config.noise_factor}")
            print(f"dataset name: {config.dataset_name}")
            print(f"space dimension: {config.space_dimension}")
            print(f"y noiseless shape: {y_noiseless.shape}")
            y_noisy = synthetic.add_linear_noise(
                space,
                y_noiseless,
                config.dataset_name,
                config.project_linear_noise,
                noise_factor=config.noise_factor,
                random_seed=config.random_seed,
            )
        else:
            y_noisy = synthetic.add_geodesic_noise(
                space,
                y_noiseless,
                config.dataset_name,
                noise_factor=config.noise_factor,
                random_seed=config.random_seed,
            )
        return space, y_noisy, y_noiseless, X, true_intercept, true_coef
    else:
        raise ValueError(f"Unknown dataset name {config.dataset_name}")


def load_mesh_data_from_path(mesh_dir, config):
    """Load mesh vertices, faces, vertex colors from a specified directory.

    This function assumes the files are named in the following format:
    {hemisphere}_structure_{structure_id}_day{day}_at_{area_threshold}.ply
    """
    project_dir = config.project_dir
    project_config = import_default_config(project_dir)

    # make sure there are meshes in the directory
    mesh_string_base = os.path.join(
        mesh_dir,
        f"{config.hemispheres[0]}_structure_{config.structure_ids[0]}**.ply",
    )
    mesh_paths = sorted(glob.glob(mesh_string_base))
    print(
        f"\nFound {len(mesh_paths)} .plys for ({config.hemispheres[0]}, {config.structure_ids[0]}) in {mesh_dir}"
    )

    # load meshes
    mesh_sequence_vertices, mesh_sequence_faces = [], []
    first_day = int(project_config.day_range[0])
    last_day = int(project_config.day_range[1])

    days_to_ignore = []
    for day in range(first_day, last_day + 1):
        mesh_path = os.path.join(
            mesh_dir,
            f"{config.hemispheres[0]}_structure_{config.structure_ids[0]}_day{day:02d}"
            f"_at_{config.area_thresholds[0]}.ply",
        )

        if not os.path.exists(mesh_path):
            print(f"Day {day} has no data. Skipping.")
            print(f"DayID not to use: {day}")
            days_to_ignore.append(day)
            continue

        vertices, faces, vertex_colors = h2_io.loadData(mesh_path)

        if vertices.shape[0] == 0:
            print(f"Day {day} has no data. Skipping.")
            print(f"DayID not to use: {day}")
            days_to_ignore.append(day)
            continue

        mesh_sequence_vertices.append(vertices)
        mesh_sequence_faces.append(faces)

    days_to_ignore = gs.array(days_to_ignore)

    mesh_sequence_vertices = gs.array(mesh_sequence_vertices)

    for faces in mesh_sequence_faces:
        if (faces != mesh_sequence_faces[0]).all():
            raise ValueError("Meshes are not parameterized: not the same faces.")

    mesh_faces = gs.array(mesh_sequence_faces[0])

    return mesh_sequence_vertices, mesh_faces, vertex_colors, days_to_ignore


def load_hormone_sorted_mesh_data_from_path(mesh_dir, config):
    """Load mesh vertices, faces, vertex colors from a the hormone sorted dir.

    This function assumes the files end with the following format:
    hormone_level{hormone_level}.ply
    """
    days_to_ignore = None
    print("Using mesh data from progesterone sorted directory")

    sorted_hormone_levels_path = os.path.join(mesh_dir, "sorted_hormone_levels.npy")
    sorted_hormone_levels = np.loadtxt(sorted_hormone_levels_path, delimiter=",")

    mesh_sequence_vertices = []
    mesh_sequence_faces = []

    for i, hormone_level in enumerate(sorted_hormone_levels):
        file_suffix = f"hormone_level{hormone_level}.ply"

        # List all files in the directory
        files_in_directory = os.listdir(mesh_dir)

        # Filter files that end with the specified format
        matching_files = [
            file for file in files_in_directory if file.endswith(file_suffix)
        ]

        # Construct the full file paths using os.path.join
        mesh_paths = [os.path.join(mesh_dir, file) for file in matching_files]

        # Print the result
        for mesh_path in mesh_paths:
            print(f"Mesh Path {i + 1}: {mesh_path}")
            vertices, faces, _ = h2_io.loadData(mesh_path)
            mesh_sequence_vertices.append(vertices)
            mesh_sequence_faces.append(faces)

    mesh_sequence_vertices = gs.array(mesh_sequence_vertices)

    for faces in mesh_sequence_faces:
        if (faces != mesh_sequence_faces[0]).all():
            raise ValueError("Meshes are not parameterized: not the same faces.")

    mesh_faces = gs.array(mesh_sequence_faces[0])

    return mesh_sequence_vertices, mesh_faces, None, days_to_ignore


def load_real_data(config, return_og_segmentation=False):
    """Load real brain meshes according to values in config file."""
    project_dir = config.project_dir
    project_config = import_default_config(project_dir)

    if project_config.sort:
        mesh_dir = project_config.sorted_dir
        (
            mesh_sequence_vertices,
            mesh_faces,
            vertex_colors,
            days_to_ignore,
        ) = load_hormone_sorted_mesh_data_from_path(mesh_dir, config)
    else:
        print("Using mesh data from (unsorted) reparameterized directory")
        mesh_dir = project_config.reparameterized_dir

        (
            mesh_sequence_vertices,
            mesh_faces,
            vertex_colors,
            days_to_ignore,
        ) = load_mesh_data_from_path(mesh_dir, config)

    if return_og_segmentation:
        mesh_dir = project_config.nondegenerate_dir
        (
            raw_mesh_sequence_vertices,
            raw_mesh_faces,
            raw_vertex_colors,
            _,
        ) = load_mesh_data_from_path(mesh_dir, config)

    #     # make sure there are meshes in the directory
    #     mesh_string_base = os.path.join(
    #         mesh_dir,
    #         f"{config.hemispheres[0]}_structure_{config.structure_ids[0]}**.ply",
    #     )
    #     mesh_paths = sorted(glob.glob(mesh_string_base))
    #     print(
    #         f"\nFound {len(mesh_paths)} .plys for ({config.hemispheres[0]}, {config.structure_ids[0]}) in {mesh_dir}"
    #     )

    #     # load meshes
    #     mesh_sequence_vertices, mesh_sequence_faces = [], []
    #     first_day = int(project_config.day_range[0])
    #     last_day = int(project_config.day_range[1])

    #     days_to_ignore = []
    #     for day in range(first_day, last_day + 1):
    #         mesh_path = os.path.join(
    #             mesh_dir,
    #             f"{config.hemispheres[0]}_structure_{config.structure_ids[0]}_day{day:02d}"
    #             f"_at_{config.area_thresholds[0]}.ply",
    #         )

    #         if not os.path.exists(mesh_path):
    #             print(f"Day {day} has no data. Skipping.")
    #             print(f"DayID not to use: {day}")
    #             days_to_ignore.append(day)
    #             continue

    #         vertices, faces, vertex_colors = h2_io.loadData(mesh_path)

    #         if vertices.shape[0] == 0:
    #             print(f"Day {day} has no data. Skipping.")
    #             print(f"DayID not to use: {day}")
    #             days_to_ignore.append(day)
    #             continue

    #         mesh_sequence_vertices.append(vertices)
    #         mesh_sequence_faces.append(faces)

    #     days_to_ignore = gs.array(days_to_ignore)

    # mesh_sequence_vertices = gs.array(mesh_sequence_vertices)

    # for faces in mesh_sequence_faces:
    #     if (faces != mesh_sequence_faces[0]).all():
    #         raise ValueError("Meshes are not parameterized: not the same faces.")

    # mesh_faces = gs.array(mesh_sequence_faces[0])

    space = DiscreteSurfaces(faces=mesh_faces)
    elastic_metric = ElasticMetric(
        space=space,
        a0=project_config.a0,
        a1=project_config.a1,
        b1=project_config.b1,
        c1=project_config.c1,
        d1=project_config.d1,
        a2=project_config.a2,
    )
    optimizer = get_optimizer(
        config.use_cuda,
        n_vertices=len(mesh_sequence_vertices[0]),
        max_iter=100,
        tol=1e-5,
    )
    elastic_metric.exp_solver = DiscreteSurfacesExpSolver(
        space=space, n_steps=config.n_steps, optimizer=optimizer
    )
    space.metric = elastic_metric

    if project_config.dataset_name == "menstrual_mesh":
        hormones_path = os.path.join(project_config.data_dir, "hormones.csv")
        hormones_df = pd.read_csv(hormones_path, delimiter=",")
    if project_config.dataset_name == "pregnancy_mesh":
        hormones_path = "/home/data/pregnancy/28Baby_Hormones.csv"
        hormones_df = pd.read_csv(hormones_path, delimiter=",")
        hormones_df["dayID"] = [
            int(entry.split("-")[1]) for entry in hormones_df["sessionID"]
        ]
        hormones_df = hormones_df.drop(
            hormones_df[hormones_df["dayID"] == 27].index
        )  # sess 27 is a repeat of sess 26
        # df = df[df["dayID"] != 27]  # sess 27 is a repeat of sess 26

    full_hormones_df = hormones_df
    hormones_df = hormones_df[hormones_df["dayID"] < project_config.day_range[1] + 1]
    hormones_df = hormones_df[hormones_df["dayID"] > project_config.day_range[0] - 1]
    if days_to_ignore is not None:
        for day in days_to_ignore:
            day = int(day)
            hormones_df = hormones_df.drop(
                hormones_df[hormones_df["dayID"] == day].index
            ).reset_index(drop=True)
            print("Hormones excluded from day: ", day)

    if project_config.dataset_name == "pregnancy_mesh":
        print("df index: ", hormones_df.index)
        missing_days = hormones_df[hormones_df.isnull().any(axis=1)].index
        print(f"Missing days: {missing_days}")

        # Remove rows with missing hormone values from the dataframe
        hormones_df = hormones_df.dropna()

        # Remove corresponding brain meshes from the array
        mesh_sequence_vertices = np.delete(mesh_sequence_vertices, missing_days, axis=0)

    print(f"space faces: {space.faces.shape}")
    print(f"mesh_sequence_vertices shape: {mesh_sequence_vertices.shape}")
    print(f"hormones_df shape: {hormones_df.shape}")

    if return_og_segmentation:
        return (
            space,
            mesh_sequence_vertices,
            mesh_faces,
            vertex_colors,
            hormones_df,
            full_hormones_df,
            raw_mesh_sequence_vertices,
            raw_mesh_faces,
            raw_vertex_colors,
        )

    return space, mesh_sequence_vertices, vertex_colors, hormones_df, full_hormones_df


def load_raw_mri_data(mri_dir):
    """Load raw MRI data according to values in config file.

    Parameters
    ----------
    config : dict
        Configuration dictionary.

    Returns
    -------
    nii_data : np.ndarray
        3D MRI data.
    """
    mri_dict = {}
    print(f"Looking into: {mri_dir}")
    # for i_day, day_dir in enumerate(mri_dir):
    for i_session, day_dir in enumerate(os.listdir(mri_dir)):
        # Construct the full path to the day directory
        full_day_dir = os.path.join(mri_dir, day_dir)

        file_found = False
        for file_name in os.listdir(full_day_dir):
            if file_name.startswith("BrainNormalized") and file_name.endswith(
                ".nii.gz"
            ):
                file_found = True
                img_path = os.path.join(full_day_dir, file_name)
                img = nib.load(img_path)
                img_data = img.get_fdata()
                mri_dict[i_session] = img_data
                print(f"Loaded MRI data for sess {i_session}")
                break
        if not file_found:
            print(f"File not found in {day_dir}")

    return mri_dict


def load_mesh(mesh_type, n_subdivisions, config):
    """Load a mesh from the synthetic dataset.

    If the mesh does not exist, create it.

    Parameters
    ----------
    mesh_type : str, {"sphere", "ellipsoid", "pill", "cube"}
    """
    project_dir = config.project_dir
    project_config = import_default_config(project_dir)
    data_dir = project_config.synthetic_data_dir
    shape_dir = os.path.join(data_dir, f"{mesh_type}_subs{n_subdivisions}")
    vertices_path = os.path.join(shape_dir, "vertices.npy")
    faces_path = os.path.join(shape_dir, "faces.npy")

    if os.path.exists(shape_dir):
        print(f"{mesh_type} mesh exists in {shape_dir}. Loading now.")
        vertices = np.load(vertices_path)
        faces = np.load(faces_path)
        return trimesh.Trimesh(vertices=vertices, faces=faces)

    print(f"Creating {mesh_type} mesh in {shape_dir}")
    mesh = synthetic.generate_mesh(mesh_type, n_subdivisions)
    os.makedirs(shape_dir)
    np.save(vertices_path, mesh.vertices)
    np.save(faces_path, mesh.faces)
    return mesh


def mesh_diameter(mesh_vertices):
    """Compute the diameter of a mesh."""
    max_distance = 0
    for i_vertex in range(mesh_vertices.shape[0]):
        for j_vertex in range(i_vertex + 1, mesh_vertices.shape[0]):
            distance = gs.linalg.norm(mesh_vertices[i_vertex] - mesh_vertices[j_vertex])
            if distance > max_distance:
                max_distance = distance
    return max_distance
