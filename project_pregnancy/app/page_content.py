"""Functions to generate the content of the different pages of the app."""

import os
import random

import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, dcc, html

os.environ["GEOMSTATS_BACKEND"] = "pytorch"  # noqa: E402
import geomstats.backend as gs

import project_pregnancy.default_config as default_config
import src.datasets.utils as data_utils
import src.setcwd
from src.preprocessing import smoothing
from src.regression import training

src.setcwd.main()

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "18rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

margin_side = "20px"
text_fontsize = "24px"
text_fontfamily = "Avenir"
title_fontsize = "40px"
space_between_sections = "70px"
space_between_title_and_content = "30px"

img_herbrain = html.Img(
    src="assets/herbrain.png",
    style={"width": "100%", "height": "auto"},
)

img_study_timeline = html.Img(
    src="assets/study_timeline.png",
    style={"width": "100%", "height": "auto"},
)

instructions_title = dbc.Row(
    [
        dbc.Col(
            html.Img(
                src="assets/instructions_emoji.jpeg",
                style={"width": "50px", "height": "auto"},
            ),
            width=1,
        ),
        dbc.Col(
            html.P("Instructions", style={"fontSize": title_fontsize}),
            width=10,
        ),
    ],
    align="center",
)

overview_title = dbc.Row(
    [
        dbc.Col(
            html.Img(
                src="assets/overview_emoji.jpeg",
                style={"width": "50px", "height": "auto"},
            ),
            width=1,
        ),
        dbc.Col(
            html.P("Overview", style={"fontSize": title_fontsize}),
            width=10,
        ),
    ],
    align="center",
)

acknowledgements_title = dbc.Row(
    [
        dbc.Col(
            html.Img(
                src="assets/acknowledgements_emoji.jpeg",
                style={"width": "50px", "height": "auto"},
            ),
            width=1,
        ),
        dbc.Col(
            html.P("Acknowledgements", style={"fontSize": title_fontsize}),
            width=10,
        ),
    ],
    align="center",
)


def hormone_slider(hormone_name, hormones_info):
    """Return a slider for a hormone."""
    return dcc.Slider(
        id=f"{hormone_name}-slider",
        min=hormones_info[hormone_name]["min_value"],
        max=hormones_info[hormone_name]["max_value"],
        step=hormones_info[hormone_name]["step"],
        value=hormones_info[hormone_name]["mean_value"],
        marks={
            hormones_info[hormone_name]["min_value"]: {"label": "min"},
            hormones_info[hormone_name]["max_value"]: {"label": "max"},
        },
        tooltip={
            "placement": "bottom",
            "always_visible": True,
            "style": {"fontSize": "30px", "fontFamily": text_fontfamily},
        },
    )


def slider(slider_name, slider_info):
    """Return a slider with the info in slider_info."""
    return dcc.Slider(
        id=f"{slider_name}-slider",
        min=slider_info[slider_name]["min_value"],
        max=slider_info[slider_name]["max_value"],
        step=slider_info[slider_name]["step"],
        value=slider_info[slider_name]["mean_value"],
        marks={
            slider_info[slider_name]["min_value"]: {"label": "min"},
            slider_info[slider_name]["max_value"]: {"label": "max"},
        },
        tooltip={
            "placement": "bottom",
            "always_visible": True,
            "style": {"fontSize": "30px", "fontFamily": text_fontfamily},
        },
    )


def sidebar():
    """Return the sidebar of the app."""
    title = dbc.Row(
        [
            dbc.Col(
                html.Img(
                    src="assets/wbhi_logo.png",
                    style={"width": "100px", "height": "auto"},
                ),
                width=2,
            ),
            dbc.Col(width=0.5),
            dbc.Col(
                html.H2("HerBrain", className="display-4"),
                width=10,
            ),
        ],
        align="center",
    )

    home_link = dbc.Row(
        [
            dbc.Col(
                html.Img(
                    src="assets/home_emoji.jpeg",
                    style={"width": "30px", "height": "auto"},
                ),
                width=2,
            ),
            dbc.Col(
                dbc.NavLink("Home", href="/", active="exact"),
                width=10,
            ),
        ],
        align="center",
    )

    mri_link = dbc.Row(
        [
            dbc.Col(
                html.Img(
                    src="assets/brain_emoji.jpeg",
                    style={"width": "40px", "height": "auto"},
                ),
                width=2,
                align="center",
            ),
            dbc.Col(
                dbc.NavLink("Explore MRI Data", href="/page-1", active="exact"),
                width=10,
            ),
        ],
        align="center",
    )

    ai_link = dbc.Row(
        [
            dbc.Col(
                html.Img(
                    src="assets/robot_emoji.jpeg",
                    style={"width": "40px", "height": "auto"},
                ),
                width=2,
            ),
            dbc.Col(
                dbc.NavLink(
                    "AI: Hormones to Hippocampus Shape", href="/page-2", active="exact"
                ),
                width=10,
            ),
        ],
        align="center",
    )

    return html.Div(
        [
            title,
            html.Hr(),
            html.P(
                "Explore how the female brain changes during pregnancy",
                className="lead",
            ),
            dbc.Nav(
                [
                    home_link,
                    mri_link,
                    ai_link,
                ],
                vertical=True,
                pills=True,
            ),
        ],
        style=SIDEBAR_STYLE,
    )


def homepage():
    """Return the content of the homepage."""
    banner = [
        html.Div(style={"height": "20px"}),
        html.Img(
            src="assets/herbrain_logo_text.png",
            style={
                "width": "80%",
                "height": "auto",
                "marginLeft": "10px",
                "marginRight": "10px",
            },
        ),
    ]

    overview_text = html.P(
        [
            "Welcome to HerBrain! This application is a tool to explore how the brain changes during pregnancy. Ovarian hormones, such as estrogen and progesterone, are known to influence the brain, and these hormones are elevated 100-1000 fold during pregnancy.",
            html.Br(),
            html.Br(),
            "The hippocampus and the structures around it are particularly sensitives to hormones. In pregnancy, sex hormones are believed to drive the decline in hippocampal volume that occurs during gestation.",
        ],
        style={"fontSize": text_fontsize, "fontFamily": text_fontfamily},
    )

    instructions_text = html.P(
        [
            "Use the sidebar to navigate between the different pages of the application. The 'Explore MRI Data' page allows you to explore the brain MRIs from the study. The 'AI: Hormones to Hippocampus Shape' page allows you to explore the relationship between hormones and the shape of the hippocampus.",
        ],
        style={"fontSize": text_fontsize, "fontFamily": text_fontfamily},
    )

    acknowledgements_text = html.P(
        [
            "This application was developed by Adele Myers and Nina Miolane and made possible by the support of the Women's Brain Health Initiative. Brain MRI data was collected in the study: Pritschet, Taylor, Cossio, Santander, Grotzinger, Faskowitz, Handwerker, Layher, Chrastil, Jacobs. Neuroanatomical changes observed over the course of a human pregnancy. (2024).",
        ],
        style={"fontSize": text_fontsize, "fontFamily": text_fontfamily},
    )

    brain_image_row = dbc.Row(
        [dbc.Col(md=2), dbc.Col(img_herbrain, md=8), dbc.Col(md=2)],
        style={"marginLeft": margin_side, "marginRight": margin_side},
    )

    contents_container = dbc.Container(
        [
            *banner,
            html.Hr(),
            overview_title,
            html.Div(style={"height": space_between_title_and_content}),
            overview_text,
            brain_image_row,
            html.Div(style={"height": space_between_sections}),
            html.Hr(),
            instructions_title,
            html.Div(style={"height": space_between_title_and_content}),
            instructions_text,
            html.Div(style={"height": space_between_sections}),
            html.Hr(),
            acknowledgements_title,
            html.Div(style={"height": space_between_title_and_content}),
            acknowledgements_text,
        ],
        fluid=True,
    )

    return dbc.Row(
        [
            dbc.Col(sm=1),
            dbc.Col(contents_container, sm=10),
            dbc.Col(sm=1),
        ]
    )


def coordinate_slider(coordinate_name, mri_coordinates_info):
    """Return a slider for a coordinate (coordinate for mri slice)."""
    return dcc.Slider(
        id=f"{coordinate_name}-slider",
        min=mri_coordinates_info[coordinate_name]["min_value"],
        max=mri_coordinates_info[coordinate_name]["max_value"],
        step=mri_coordinates_info[coordinate_name]["step"],
        value=mri_coordinates_info[coordinate_name]["mean_value"],
        marks={
            mri_coordinates_info[coordinate_name]["min_value"]: {"label": "min"},
            mri_coordinates_info[coordinate_name]["max_value"]: {"label": "max"},
        },
        tooltip={
            "placement": "bottom",
            "always_visible": True,
            "style": {"fontSize": "30px", "fontFamily": text_fontfamily},
        },
    )


def explore_data(mri_coordinates_info, hormones_info):
    """Return the content of the data exploration page."""
    study_row = dbc.Row(
        [dbc.Col(md=1), dbc.Col(img_study_timeline, md=10), dbc.Col(md=1)],
        style={"marginLeft": margin_side, "marginRight": margin_side},
    )

    banner = [
        dbc.Row(
            [
                dbc.Col(
                    html.Img(
                        src="assets/brain_emoji.jpeg",
                        style={"width": "100px", "height": "auto"},
                    ),
                    width=1,
                ),
                dbc.Col(
                    html.P(
                        "Explore Brain MRIs Throughout Pregnancy",
                        style={"fontSize": title_fontsize},
                    ),
                    width=10,
                ),
            ],
            align="center",
        ),
    ]

    overview_text = dbc.Row(
        [
            html.P(
                [
                    "MRI data was collected ~ once every 2 weeks throughout pregnancy, showing the structural changes that occur in the brain over the course of a human pregnancy. Estrogen, progesterone, and LH levels were also measured at most sessions.",
                ],
                style={"fontSize": text_fontsize, "fontFamily": text_fontfamily},
            ),
            study_row,
        ],
    )

    instructions_text = dbc.Row(
        [
            html.P(
                [
                    "Use the 'Session Number' slider to flip through T1 brain data from each MRI session. Use the X, Y, Z coordinate sliders choose the MRI slice. Additional information about the session will be displayed to the right of the sliders.",
                ],
                style={
                    "fontSize": text_fontsize,
                    "fontFamily": text_fontfamily,
                    "marginLeft": margin_side,
                    "marginRight": margin_side,
                },
            ),
        ],
    )

    acknowledgements_text = dbc.Row(
        [
            html.P(
                [
                    "Data and study timeline image from: Pritschet, Taylor, Cossio, Santander, Grotzinger, Faskowitz, Handwerker, Layher, Chrastil, Jacobs. Neuroanatomical changes observed over the course of a human pregnancy. (2024)",
                ],
                style={
                    "fontSize": text_fontsize,
                    "fontFamily": text_fontfamily,
                    "marginLeft": margin_side,
                    "marginRight": margin_side,
                },
            ),
        ],
    )

    sliders_card = dbc.Card(
        [
            dbc.Stack(
                [
                    dbc.Label(
                        "Session Number",
                        style={
                            "font-size": text_fontsize,
                            "fontFamily": text_fontfamily,
                        },
                    ),
                    slider("scan-number", hormones_info),
                    dbc.Label(
                        "X Coordinate (Changes Side View)",
                        style={
                            "font-size": text_fontsize,
                            "fontFamily": text_fontfamily,
                        },
                    ),
                    slider("x", mri_coordinates_info),
                    dbc.Label(
                        "Y Coordinate (Changes Front View)",
                        style={
                            "font-size": text_fontsize,
                            "fontFamily": text_fontfamily,
                        },
                    ),
                    slider("y", mri_coordinates_info),
                    dbc.Label(
                        "Z Coordinate (Changes Top View)",
                        style={
                            "font-size": text_fontsize,
                            "fontFamily": text_fontfamily,
                        },
                    ),
                    slider("z", mri_coordinates_info),
                ],
                gap=3,
            )
        ],
        body=True,
    )

    sess_info_card = dbc.Card(
        [
            dbc.Stack(
                [
                    # create a text box that can be adjusted in a callback
                    dbc.Label(
                        "Session Information:",
                        style={
                            "font-size": text_fontsize,
                            "fontFamily": text_fontfamily,
                        },
                    ),
                    html.Div(
                        id="session-number",
                        style={
                            "font-size": text_fontsize,
                            "fontFamily": text_fontfamily,
                        },
                    ),
                    html.Div(
                        id="gest-week",
                        style={
                            "font-size": text_fontsize,
                            "fontFamily": text_fontfamily,
                        },
                    ),
                    html.Div(
                        id="estrogen-level",
                        style={
                            "font-size": text_fontsize,
                            "fontFamily": text_fontfamily,
                        },
                    ),
                    html.Div(
                        id="progesterone-level",
                        style={
                            "font-size": text_fontsize,
                            "fontFamily": text_fontfamily,
                        },
                    ),
                    html.Div(
                        id="LH-level",
                        style={
                            "font-size": text_fontsize,
                            "fontFamily": text_fontfamily,
                        },
                    ),
                    html.Div(
                        id="endo-status",
                        style={
                            "font-size": text_fontsize,
                            "fontFamily": text_fontfamily,
                        },
                    ),
                    html.Div(
                        id="trimester",
                        style={
                            "font-size": text_fontsize,
                            "fontFamily": text_fontfamily,
                        },
                    ),
                ],
                gap=3,
            )
        ],
        body=True,
    )

    sliders_column = [
        dbc.Row(sliders_card),
    ]

    plots_card = dbc.Row(
        [
            dbc.Col(
                html.Div(
                    dcc.Graph(id="nii-plot-side", config={"displayModeBar": False}),
                    style={"paddingTop": "0px"},
                ),
                sm=4,
            ),
            dbc.Col(
                html.Div(
                    dcc.Graph(id="nii-plot-front", config={"displayModeBar": False}),
                    style={"paddingTop": "0px"},
                ),
                sm=4,
            ),
            dbc.Col(
                html.Div(
                    dcc.Graph(id="nii-plot-top", config={"displayModeBar": False}),
                    style={"paddingTop": "0px"},
                ),
                sm=4,
            ),
        ],
        align="center",
        style={
            "marginLeft": margin_side,
            "marginRight": margin_side,
            "marginTop": "50px",
        },
    )

    contents_container = dbc.Container(
        [
            *banner,
            html.Hr(),
            overview_title,
            html.Div(style={"height": space_between_title_and_content}),
            overview_text,
            html.Div(style={"height": space_between_sections}),
            html.Hr(),
            instructions_title,
            html.Div(style={"height": space_between_title_and_content}),
            instructions_text,
            dbc.Row(
                [
                    dbc.Col(plots_card, sm=14),
                ],
                align="center",
                style={
                    "marginLeft": margin_side,
                    "marginRight": margin_side,
                    "marginTop": "50px",
                },
            ),
            dbc.Row(
                [
                    dbc.Col(sliders_column, sm=7, width=700),
                    dbc.Col(sess_info_card, sm=4, width=700),
                ],
                align="center",
                style={
                    "marginLeft": margin_side,
                    "marginRight": margin_side,
                    "marginTop": "50px",
                },
            ),
            html.Div(style={"height": space_between_sections}),
            html.Hr(),
            acknowledgements_title,
            html.Div(style={"height": space_between_title_and_content}),
            acknowledgements_text,
        ],
        fluid=True,
    )

    return dbc.Row(
        [
            dbc.Col(sm=1),
            dbc.Col(contents_container, sm=10),
            dbc.Col(sm=1),
        ]
    )


def ai_hormone_prediction(
    hormones_info,
):
    """Return the content of the AI hormone prediction page."""
    banner = [
        dbc.Row(
            [
                dbc.Col(
                    html.Img(
                        src="assets/robot_emoji.jpeg",
                        style={"width": "100px", "height": "auto"},
                    ),
                    width=1,
                ),
                dbc.Col(
                    html.P(
                        "AI: Hormones to Hippocampus Shape",
                        style={"fontSize": title_fontsize},
                    ),
                    width=10,
                ),
            ],
            align="center",
        ),
    ]

    overview_text = dbc.Row(
        [
            html.P(
                [
                    "The hippocampus is a brain region that is particularly sensitive to hormones. In pregnancy the hippocampus volume is known to decrease, but we find that the shape of the hippocampus changes as well. We have trained an AI to predict the shape of the hippocampus based on hormone levels.",
                    html.Br(),
                ],
                style={"fontSize": text_fontsize, "fontFamily": text_fontfamily},
            ),
        ],
    )

    instructions_text = dbc.Row(
        [
            html.P(
                [
                    "Use the hormone sliders or the gestational week slider to adjust observe the predicted shape changes in the left hippocampal formation.",
                    html.Br(),
                ],
                style={"fontSize": text_fontsize, "fontFamily": text_fontfamily},
            ),
        ],
    )

    acknowledgements_text = dbc.Row(
        [
            html.P(
                [
                    "Our AI was trained on data from the study: Pritschet, Taylor, Cossio, Santander, Grotzinger, Faskowitz, Handwerker, Layher, Chrastil, Jacobs. Neuroanatomical changes observed over the course of a human pregnancy. (2024)",
                ],
                style={"fontSize": text_fontsize, "fontFamily": text_fontfamily},
            ),
        ],
    )

    week_slider_card = dbc.Card(
        [
            html.Div(
                id="gest_week_slider_container",
                style={"display": "block"},
                children=[
                    dbc.Stack(
                        [
                            dbc.Label(
                                "Gestational Week",
                                style={
                                    "display": "block",
                                    "font-size": text_fontsize,
                                    "fontFamily": text_fontfamily,
                                },
                            ),
                            slider("gest-week", hormones_info),
                        ],
                        gap=3,
                    )
                ],
            ),
        ],
        body=True,
    )

    hormone_sliders_card = dbc.Card(
        [
            html.Div(
                id="hormone_slider_container",
                style={"display": "block"},
                children=[
                    dbc.Stack(
                        [
                            dbc.Label(
                                "Estrogen pg/ml",
                                style={
                                    "font-size": 30,
                                    "fontFamily": text_fontfamily,
                                    "display": "block",
                                },
                            ),
                            slider("estrogen", hormones_info),
                            dbc.Label(
                                "Progesterone ng/ml",
                                style={
                                    "font-size": 30,
                                    "fontFamily": text_fontfamily,
                                    "display": "block",
                                },
                            ),
                            slider("progesterone", hormones_info),
                            dbc.Label(
                                "LH ng/ml",
                                style={
                                    "font-size": 30,
                                    "fontFamily": text_fontfamily,
                                    "display": "block",
                                },
                            ),
                            slider("LH", hormones_info),
                        ],
                        gap=3,
                    )
                ],
            ),
        ],
        body=True,
    )

    sliders_column = [
        html.Button(
            "Click Here to Toggle Between Gestational Week vs Hormone Value Prediction",
            id="button",
            n_clicks=0,
        ),
        dbc.Row(week_slider_card),
        dbc.Row(hormone_sliders_card),
    ]

    substructure_legend_row = dbc.Row(
        [
            html.Img(
                src="assets/substructure_legend.png",
                style={"width": "100%", "height": "auto"},
            ),
        ]
    )

    contents_container = dbc.Container(
        [
            *banner,
            html.Hr(),
            overview_title,
            html.Div(style={"height": space_between_title_and_content}),
            overview_text,
            html.Div(style={"height": space_between_sections}),
            html.Hr(),
            instructions_title,
            html.Div(style={"height": space_between_title_and_content}),
            instructions_text,
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            dcc.Graph(id="mesh-plot"),
                            style={"paddingTop": "0px"},
                        ),
                        sm=4,
                        width=700,
                    ),
                    dbc.Col(sm=3, width=100),
                    dbc.Col(sliders_column, sm=4, width=700),
                ],
                align="center",
                style={
                    "marginLeft": margin_side,
                    "marginRight": margin_side,
                    "marginTop": "50px",
                },
            ),
            substructure_legend_row,
            html.Div(style={"height": space_between_sections}),
            html.Hr(),
            acknowledgements_title,
            html.Div(style={"height": space_between_title_and_content}),
            acknowledgements_text,
        ],
        fluid=True,
    )

    return dbc.Row(
        [
            dbc.Col(sm=1),
            dbc.Col(contents_container, sm=10),
            dbc.Col(sm=1),
        ]
    )
