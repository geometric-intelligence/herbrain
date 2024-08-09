"""Functions perform calculations necessary for the dash app."""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

os.environ["GEOMSTATS_BACKEND"] = "pytorch"  # noqa: E402
import geomstats.backend as gs
import nibabel as nib
import pandas as pd

import project_pregnancy.default_config as default_config
import src.datasets.utils as data_utils
import src.setcwd
import src.viz as viz
from src.preprocessing import smoothing
from src.regression import training

src.setcwd.main()


def train_lr_model(X, mesh_sequence_vertices, n_X, p_values=False):
    """Train a linear regression model on the data."""
    mean_mesh = mesh_sequence_vertices.mean(axis=0)

    # Compute neighbors once and for all from the mean mesh
    k_neighbors = 10
    mesh_neighbors = smoothing.compute_neighbors(mean_mesh, k=k_neighbors)

    n_meshes, n_vertices, _ = mesh_sequence_vertices.shape

    y = mesh_sequence_vertices.reshape(n_meshes, -1)
    y_mean = y.mean(axis=0)
    y = y - y_mean

    # Define the number of principal components
    n_components = 4  # Adjust based on variance explanation: see notebook 02
    pca = PCA(n_components=n_components)
    y_pca = pca.fit_transform(y)
    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"Cumul. variance explained w/ {n_components} components: {explained_var}")

    lr = LinearRegression()
    lr.fit(X, y_pca)
    y_pca_pred = lr.predict(X)
    r2 = r2_score(y_pca, y_pca_pred)
    adjusted_r2 = 1 - (1 - r2) * (n_meshes - 1) / (n_meshes - n_X - 1)
    print(f"Adjusted R2 score (adjusted for several inputs): {adjusted_r2:.2f}")

    if p_values:
        p_values = [
            0.0,
            0.0,
            0.0,
        ]  # placeholder, else: training.calculate_p_values(X_multiple, y_pca, lr)
        estrogen_p_value = p_values[0]
        progesterone_p_value = p_values[1]
        lh_p_value = p_values[2]

        return (
            lr,
            pca,
            y_mean,
            n_vertices,
            mesh_neighbors,
            adjusted_r2,
            estrogen_p_value,
            progesterone_p_value,
            lh_p_value,
        )

    return lr, pca, y_mean, n_vertices, mesh_neighbors


def predict_mesh(
    X,
    lr,
    pca,
    y_mean,
    n_vertices,
    mesh_neighbors,
    space,
    vertex_colors,
    current_figure=None,
    relayoutData=None,
):
    """Predict the mesh based on the hormone values."""
    y_pca_pred = lr.predict(X)

    y_pred = pca.inverse_transform(y_pca_pred) + y_mean.numpy()
    mesh_pred = y_pred.reshape(n_vertices, 3)
    mesh_pred = smoothing.median_smooth(mesh_pred, mesh_neighbors)

    # Plot Mesh
    if current_figure and "layout" in current_figure:
        layout = current_figure["layout"]
    else:
        layout = go.Layout(
            margin=go.layout.Margin(
                l=0,
                r=0,
                b=0,
                t=0,
            ),
            width=700,
            height=700,
            scene=dict(
                aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z"
            ),
        )

    faces = gs.array(space.faces).numpy()
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=mesh_pred[:, 0],
                y=mesh_pred[:, 1],
                z=mesh_pred[:, 2],
                colorbar_title="z",
                vertexcolor=vertex_colors,
                # i, j and k give the vertices of triangles
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                name="y",
            )
        ],
        layout=layout,
    )

    # if relayoutData and ("scene.camera" in relayoutData):
    #     scene_camera = relayoutData["scene.camera"]
    # else:
    #     scene_camera = dict(
    #         up=dict(x=0, y=0, z=1),
    #         center=dict(x=0, y=0, z=0),
    #         eye=dict(x=0, y=0, z=2.5),
    #     )
    # fig.update_layout(scene_camera=scene_camera)
    return fig


def plot_slice_as_plotly(
    one_slice, cmap="gray", title="Slice Visualization", x_label="X", y_label="Y"
):
    """Display an image slice as a Plotly figure."""
    # Create heatmap trace for the current slice
    heatmap_trace = go.Heatmap(z=one_slice.T, colorscale=cmap, showscale=False)

    print("One slice shape:", one_slice.shape)

    width = int(len(one_slice[:, 0]) * 1.5)
    height = int(len(one_slice[0]) * 1.5)
    print("Width:", width, "Height:", height)

    layout = go.Layout(
        title=title,
        title_x=0.5,
        xaxis=dict(title=x_label),
        yaxis=dict(title=y_label),
        width=width,
        height=height,
    )

    # Create a Plotly figure with the heatmap trace
    fig = go.Figure(data=heatmap_trace, layout=layout)

    # Update layout to adjust appearance
    # fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)

    return fig


def return_nii_plot(sess_number, x, y, z, raw_mri_dict):  # week,
    """Return the nii plot based on the week and the x, y, z coordinates."""
    slice_0 = raw_mri_dict[sess_number][x, :, :]
    slice_1 = raw_mri_dict[sess_number][:, y, :]
    slice_2 = raw_mri_dict[sess_number][:, :, z]

    common_width = max(len(slice_0[:, 0]), len(slice_1[:, 0]), len(slice_2[:, 0]))
    common_height = max(len(slice_0[0]), len(slice_1[0]), len(slice_2[0]))

    slices = [slice_0, slice_1, slice_2]
    for i_slice, slice in enumerate([slice_0, slice_1, slice_2]):
        if len(slice[:, 0]) < common_width:
            diff = common_width - len(slice[:, 0])
            # slice = np.pad(slice, ((0, diff), (0, 0)), mode="constant")
            slice = np.pad(slice, ((diff // 2, diff // 2), (0, 0)), mode="constant")
            slices[i_slice] = slice
        if len(slice[0]) < common_height:
            diff = common_height - len(slice[0])
            # slice = np.pad(slice, ((0, 0), (0, diff)), mode="constant")
            slice = np.pad(slice, ((0, 0), (diff // 2, diff // 2)), mode="constant")
            slices[i_slice] = slice

    side_fig = plot_slice_as_plotly(
        slices[0], cmap="gray", title="Side View", x_label="Y", y_label="Z"
    )
    front_fig = plot_slice_as_plotly(
        slices[1], cmap="gray", title="Front View", x_label="X", y_label="Z"
    )
    top_fig = plot_slice_as_plotly(
        slices[2], cmap="gray", title="Top View", x_label="X", y_label="Y"
    )

    return side_fig, front_fig, top_fig


# def pre_calculate_mri_figs(raw_mri_dict, mri_coordinates_info):
#     """Pre-calculate the slices of the MRI image."""
#     # pre-calculate side view mri figs

#     fig_dict = []
#     for week in raw_mri_dict.keys():
#         print("Calculating MRI figures for week", week)
#         x_values = gs.arange(
#             mri_coordinates_info["x"]["min_value"],
#             mri_coordinates_info["x"]["max_value"],
#             mri_coordinates_info["x"]["step"],
#         )
#         for x in x_values:
#             y_values = gs.arange(
#                 mri_coordinates_info["y"]["min_value"],
#                 mri_coordinates_info["y"]["max_value"],
#                 mri_coordinates_info["y"]["step"],
#             )
#             for y in y_values:
#                 z_values = gs.arange(
#                     mri_coordinates_info["z"]["min_value"],
#                     mri_coordinates_info["z"]["max_value"],
#                     mri_coordinates_info["z"]["step"],
#                 )
#                 for z in z_values:
#                     slice_0 = raw_mri_dict[week][x, :, :]
#                     slice_1 = raw_mri_dict[week][:, y, :]
#                     slice_2 = raw_mri_dict[week][:, :, z]

#                     side_fig = plot_slice_as_plotly(
#                         slice_0,
#                         cmap="gray",
#                         title="Side View",
#                         x_label="Y",
#                         y_label="Z",
#                     )
#                     front_fig = plot_slice_as_plotly(
#                         slice_1,
#                         cmap="gray",
#                         title="Front View",
#                         x_label="X",
#                         y_label="Z",
#                     )
#                     top_fig = plot_slice_as_plotly(
#                         slice_2, cmap="gray", title="Top View", x_label="X", y_label="Y"
#                     )
#                     print("Week", week, "x", x, "y", y, "z", z)

#                     fig_dict.append(
#                         {
#                             "week": week,
#                             "x": x,
#                             "y": y,
#                             "z": z,
#                             "side_fig": side_fig,
#                             "front_fig": front_fig,
#                             "top_fig": top_fig,
#                         }
#                     )
#     fig_df = pd.DataFrame(fig_dict)
#     return fig_df
