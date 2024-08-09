"""Creates a Dash app where hormone sliders predict hippocampal shape.

Run the app with the following command:
python main_3_dash_app.py

Notes on Dash:
- Dash is a Python framework for building web applications.
- html.H1(children= ...) is an example of a title. You can change H1 to H2, H3, etc.
    to change the size of the title.
"""

import os
import random

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go  # or plotly.express as px
from dash import Dash, Input, Output, callback, dcc, html

os.environ["GEOMSTATS_BACKEND"] = "pytorch"  # noqa: E402
import geomstats.backend as gs

import project_menstrual.default_config as default_config
import src.datasets.utils as data_utils
import src.setcwd
from src.regression import training

src.setcwd.main()

os.environ["GEOMSTATS_BACKEND"] = "pytorch"

# Multiple Linear Regression

(
    space,
    mesh_sequence_vertices,
    vertex_colors,
    all_hormone_levels,
) = data_utils.load_real_data(default_config)

n_vertices = len(mesh_sequence_vertices[0])
n_meshes_in_sequence = len(mesh_sequence_vertices)
faces = gs.array(space.faces).numpy()

# TODO: instead, save these values in main_2, and then load them here.
# or, figure out how to predict the mesh using just the intercept
# and coef learned here, and then load them.

progesterone_levels = gs.array(all_hormone_levels["Prog"].values)
estrogen_levels = gs.array(all_hormone_levels["Estro"].values)
lh_levels = gs.array(all_hormone_levels["LH"].values)
dheas_levels = gs.array(all_hormone_levels["DHEAS"].values)
shbg_levels = gs.array(all_hormone_levels["SHBG"].values)
fsh_levels = gs.array(all_hormone_levels["FSH"].values)
# gest_week = gs.array(all_hormone_levels["gestWeek"].values)

progesterone_average = gs.mean(progesterone_levels)
estrogen_average = gs.mean(estrogen_levels)
lh_average = gs.mean(lh_levels)
dheas_average = gs.mean(dheas_levels)
shbg_average = gs.mean(shbg_levels)
fsh_average = gs.mean(fsh_levels)
# gest_week_average = gs.mean(gest_week)

y = mesh_sequence_vertices
X_multiple = gs.vstack(
    (
        progesterone_levels,
        estrogen_levels,
        lh_levels,
        # gest_week,
        dheas_levels,
        shbg_levels,
        fsh_levels,
    )
).T  # NOTE: copilot thinks this should be transposed.

(
    multiple_intercept_hat,
    multiple_coef_hat,
    mr,
    percent_significant_p_values,
) = training.fit_linear_regression(y, X_multiple, return_p=True)

# NOTE (Nina): this is not really n_train
# since we've just trained on the whole dataset
n_train = int(default_config.train_test_split * n_meshes_in_sequence)

X_indices = np.arange(n_meshes_in_sequence)
# Shuffle the array to get random values
random.shuffle(X_indices)
train_indices = X_indices[:n_train]
train_indices = np.sort(train_indices)
test_indices = X_indices[n_train:]
test_indices = np.sort(test_indices)
mr_score_array = training.compute_R2(y, X_multiple, test_indices, train_indices)

# hormone p values
progesterone_p_value = percent_significant_p_values[0]
estrogen_p_value = percent_significant_p_values[1]
lh_p_value = percent_significant_p_values[2]
dheas_p_value = percent_significant_p_values[3]
shbg_p_value = percent_significant_p_values[4]
fsh_p_value = percent_significant_p_values[5]

# Parameters for sliders

hormones_info = {
    "progesterone": {"min_value": 0, "max_value": 15, "step": 1},
    "FSH": {"min_value": 0, "max_value": 15, "step": 1},
    "LH": {"min_value": 0, "max_value": 50, "step": 5},
    "estrogen": {"min_value": 0, "max_value": 250, "step": 10},
    "DHEAS": {"min_value": 0, "max_value": 300, "step": 10},
    "SHBG": {"min_value": 0, "max_value": 70, "step": 5},
}

app = Dash(
    __name__, external_stylesheets=[dbc.themes.BOOTSTRAP]
)  # , external_stylesheets=external_stylesheets)

sliders = dbc.Card(
    [
        dbc.Stack(
            [
                # html.H6(f"Progesterone ng/ml, p-value: {progesterone_p_value}"),
                dbc.Label(
                    f"Progesterone ng/ml, percent_significant_p_values: {progesterone_p_value:05f}",
                    style={"font-size": 50},
                ),
                dcc.Slider(
                    id="progesterone-slider",
                    min=hormones_info["progesterone"]["min_value"],
                    max=hormones_info["progesterone"]["max_value"],
                    step=hormones_info["progesterone"]["step"],
                    value=progesterone_average,
                    marks={
                        hormones_info["progesterone"]["min_value"]: {"label": "min"},
                        hormones_info["progesterone"]["max_value"]: {"label": "max"},
                    },
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                        "style": {"fontSize": "30px"},
                    },
                ),
                # html.H6(f"Estrogen pg/ml, p-value: {estrogen_p_value}"),
                dbc.Label(
                    f"Estrogen pg/ml, percent_significant_p_values: {estrogen_p_value:05f}",
                    style={"font-size": 50},
                ),
                dcc.Slider(
                    id="estrogen-slider",
                    min=hormones_info["estrogen"]["min_value"],
                    max=hormones_info["estrogen"]["max_value"],
                    step=hormones_info["estrogen"]["step"],
                    value=estrogen_average,
                    marks={
                        hormones_info["estrogen"]["min_value"]: {"label": "min"},
                        hormones_info["estrogen"]["max_value"]: {"label": "max"},
                    },
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                        "style": {"fontSize": "30px"},
                    },
                ),
                # html.H6(f"LH ng/ml, p-value: {lh_p_value}"),
                dbc.Label(
                    f"LH ng/ml, percent_significant_p_values: {lh_p_value:05f}",
                    style={"font-size": 50},
                ),
                dcc.Slider(
                    id="LH-slider",
                    min=hormones_info["LH"]["min_value"],
                    max=hormones_info["LH"]["max_value"],
                    step=hormones_info["LH"]["step"],
                    value=lh_average,
                    marks={
                        hormones_info["LH"]["min_value"]: {"label": "min"},
                        hormones_info["LH"]["max_value"]: {"label": "max"},
                    },
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                        "style": {"fontSize": "30px"},
                    },
                ),
                # html.H6(f"DHEAS percent_significant_p_values: {dheas_p_value}"),
                dbc.Label(
                    f"DHEAS ng/ml, percent_significant_p_values: {dheas_p_value:05f}",
                    style={"font-size": 50},
                ),
                dcc.Slider(
                    id="DHEAS-slider",
                    min=hormones_info["DHEAS"]["min_value"],
                    max=hormones_info["DHEAS"]["max_value"],
                    step=hormones_info["DHEAS"]["step"],
                    value=dheas_average,
                    marks={
                        hormones_info["DHEAS"]["min_value"]: {"label": "min"},
                        hormones_info["DHEAS"]["max_value"]: {"label": "max"},
                    },
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                        "style": {"fontSize": "30px"},
                    },
                ),
                # html.H6(f"FSH ng/ml, percent_significant_p_values {fsh_p_value}"),
                dbc.Label(
                    f"FSH ng/ml, percent_significant_p_values: {fsh_p_value:05f}",
                    style={"font-size": 50},
                ),
                dcc.Slider(
                    id="FSH-slider",
                    min=hormones_info["FSH"]["min_value"],
                    max=hormones_info["FSH"]["max_value"],
                    step=hormones_info["FSH"]["step"],
                    value=fsh_average,
                    marks={
                        hormones_info["FSH"]["min_value"]: {"label": "min"},
                        hormones_info["FSH"]["max_value"]: {"label": "max"},
                    },
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                        "style": {"fontSize": "30px"},
                    },
                ),
                # html.H6(f"SHBG, percent_significant_p_values: {shbg_p_value}"),
                dbc.Label(
                    f"SHBG ng/ml, percent_significant_p_values: {shbg_p_value:05f}",
                    style={"font-size": 50},
                ),
                dcc.Slider(
                    id="SHBG-slider",
                    min=hormones_info["SHBG"]["min_value"],
                    max=hormones_info["SHBG"]["max_value"],
                    step=hormones_info["SHBG"]["step"],
                    value=shbg_average,
                    marks={
                        hormones_info["SHBG"]["min_value"]: {"label": "min"},
                        hormones_info["SHBG"]["max_value"]: {"label": "max"},
                    },
                    tooltip={
                        "placement": "bottom",
                        "always_visible": True,
                        "style": {"fontSize": "30px"},
                    },
                ),
            ],
            style={"width": "60%", "display": "inline-block"},
            gap=3,
        ),
    ],
    body=True,
)

app.layout = dbc.Container(
    [
        html.H1("Brain Shape Prediction with Hormones, Menstrual"),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(sliders, md=6),
                dbc.Col(dcc.Graph(id="mesh-plot"), md=6),
            ],
            align="center",
        ),
    ],
    fluid=True,
)


@callback(
    Output("mesh-plot", "figure"),
    Input("progesterone-slider", "value"),
    Input("LH-slider", "value"),
    Input("estrogen-slider", "value"),
    Input("DHEAS-slider", "value"),
    Input("SHBG-slider", "value"),
    Input("FSH-slider", "value"),
    # Input("gest_week-slider", "value"),
)
def plot_hormone_levels_plotly(progesterone, FSH, LH, estrogen, SHBG, DHEAS):
    """Update the mesh plot based on the hormone levels."""
    progesterone = gs.array(progesterone)
    FSH = gs.array(FSH)
    LH = gs.array(LH)
    estrogen = gs.array(estrogen)
    SHBG = gs.array(SHBG)
    DHEAS = gs.array(DHEAS)

    # Predict Mesh
    X_multiple = gs.vstack(
        (
            progesterone,
            estrogen,
            DHEAS,
            LH,
            FSH,
            SHBG,
        )
    ).T

    X_multiple_predict = gs.array(X_multiple.reshape(len(X_multiple), -1))

    y_pred_for_mr = mr.predict(X_multiple_predict)
    y_pred_for_mr = y_pred_for_mr.reshape([n_vertices, 3])
    # y_pred_for_mr = gaussian_smoothing(y_pred_for_mr, sigma=0.7)

    faces = gs.array(space.faces).numpy()

    x = y_pred_for_mr[:, 0]
    y = y_pred_for_mr[:, 1]
    z = y_pred_for_mr[:, 2]

    i = faces[:, 0]
    j = faces[:, 1]
    k = faces[:, 2]

    layout = go.Layout(
        margin=go.layout.Margin(
            l=0,  # left margin
            r=0,  # right margin
            b=0,  # bottom margin
            t=0,  # top margin
        )
    )

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                colorbar_title="z",
                vertexcolor=vertex_colors,
                # i, j and k give the vertices of triangles
                i=i,
                j=j,
                k=k,
                name="y",
                # showscale=True,
            )
        ],
        layout=layout,
    )

    fig.update_layout(width=1000)
    fig.update_layout(height=1000)

    # rescale the axes to fit the shape
    for axis in ["x", "y", "z"]:
        fig.update_layout(scene=dict(aspectmode="data"))
        fig.update_layout(scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"))

    # Default parameters which are used when `layout.scene.camera` is not provided
    # camera1 = dict(
    #     up=dict(x=0, y=0, z=1),
    #     center=dict(x=0, y=0, z=0),
    #     eye=dict(x=2.5, y=-2.5, z=0.0),
    # )

    camera2 = dict(
        up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=0, y=0, z=2.5)
    )

    fig.update_layout(
        scene_camera=camera2, margin=dict(l=0, r=0, b=0, t=0)
    )  # margin=dict(l=0, r=0, b=0, t=0)

    return fig


if __name__ == "__main__":
    # app.run(debug=True)
    app.run_server(
        debug=True, use_reloader=False
    )  # Turn off reloader if inside Jupyter
