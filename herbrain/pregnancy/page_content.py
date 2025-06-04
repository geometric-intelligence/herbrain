"""Functions to generate the content of the different pages of the app."""

import dash_bootstrap_components as dbc
from dash import dcc, get_asset_url, html
from polpo.dash.style import STYLE as S

from .gpt_chat import gpt_chat_component, set_openai_api_key

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "18rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}


def img_herbrain():
    return html.Img(
        src=get_asset_url("herbrain.png"),
        style={"width": "100%", "height": "auto"},
    )


def img_study_timeline():
    return html.Img(
        src=get_asset_url("study_timeline.png"),
        style={"width": "100%", "height": "auto"},
    )


def background_title():
    return dbc.Row(
        [
            dbc.Col(
                html.P("Background", style={"fontSize": S.title_fontsize}),
                width=10,
            ),
        ],
        align="center",
    )

def digital_twin_title():
    return dbc.Row(
        [
            dbc.Col(
                html.P("Digital Twins", style={"fontSize": S.title_fontsize}),
                width=10,
            ),
        ],
        align="center",
    )


def pregnancy_title():
    return dbc.Row(
        [
            dbc.Col(
                html.Img(
                    src=get_asset_url("pregnancy_logo.png"),
                    style={"width": "70px", "height": "auto"},
                ),
                width=2,
            ),
            dbc.Col(
                html.P("Digital Twin of the Pregnant Brain", style={"fontSize": S.title_fontsize}),
                width=10,
            ),
        ],
        align="center",
    )


def acknowledgements_title():
    return dbc.Row(
        [
            dbc.Col(
                html.P("Acknowledgements", style={"fontSize": S.title_fontsize}),
                width=10,
            ),
        ],
        align="center",
    )


def sidebar(sidebar_elems, page_register):
    """Return the sidebar of the app."""
    title = dbc.Row(
        [
            dbc.Col(
                html.Img(
                    src=get_asset_url("wbhi_logo.png"),
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

    headers = []
    for elem in sidebar_elems:
        if elem.active:
            compns = elem.to_dash(page_register)
            headers.append(compns[0])

    return html.Div(
        [
            title,
            html.Hr(),
            html.P(
                "Explore how the female brain changes during pregnancy",
                className="lead",
            ),
            dbc.Nav(
                headers,
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
            src=get_asset_url("herbrain_logo_text.png"),
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
        style={"fontSize": S.text_fontsize, "fontFamily": S.text_fontfamily},
    )

    instructions_text = html.P(
        [
            "Use the sidebar to navigate between the different pages of the application. The 'Explore MRI Data' page allows you to explore the brain MRIs from the study. The 'AI: Hormones to Hippocampus Shape' page allows you to explore the relationship between hormones and the shape of the hippocampus.",
        ],
        style={"fontSize": S.text_fontsize, "fontFamily": S.text_fontfamily},
    )

    acknowledgements_text = html.P(
        [
            "This application was developed by Adele Myers, Sarah Kushner, Luís F. Pereira, and Nina Miolane and made possible by the support of the Women's Brain Health Initiative. Brain MRI data was collected in the study: Pritschet, Taylor, Cossio, Santander, Grotzinger, Faskowitz, Handwerker, Layher, Chrastil, Jacobs. Neuroanatomical changes observed over the course of a human pregnancy. (2024).",
        ],
        style={"fontSize": S.text_fontsize, "fontFamily": S.text_fontfamily},
    )

    brain_image_row = dbc.Row(
        [dbc.Col(md=2), dbc.Col(img_herbrain(), md=8), dbc.Col(md=2)],
        style={"marginLeft": S.margin_side, "marginRight": S.margin_side},
    )

    contents_container = dbc.Container(
        [
            *banner,
            html.Hr(),
            background_title(),
            html.Div(style={"height": S.space_between_title_and_content}),
            overview_text,
            brain_image_row,
            html.Div(style={"height": S.space_between_sections}),
            html.Hr(),
            digital_twin_title(),
            html.Div(style={"height": S.space_between_title_and_content}),
            instructions_text,
            html.Div(style={"height": S.space_between_sections}),
            html.Hr(),
            acknowledgements_title(),
            html.Div(style={"height": S.space_between_title_and_content}),
            acknowledgements_text,
        ],
        fluid=True,
    )

    return [
        dbc.Row(
            [
                dbc.Col(sm=1),
                dbc.Col(contents_container, sm=10),
                dbc.Col(sm=1),
            ]
        )
    ]


# def mri_page(mri_explorer):
#     """Return the content of the data exploration page."""
#     study_row = dbc.Row(
#         [dbc.Col(md=1), dbc.Col(img_study_timeline(), md=10), dbc.Col(md=1)],
#         style={"marginLeft": S.margin_side, "marginRight": S.margin_side},
#     )

#     banner = [
#         dbc.Row(
#             [
#                 dbc.Col(
#                     html.Img(
#                         src=get_asset_url("brain_emoji.jpeg"),
#                         style={"width": "70px", "height": "auto"},
#                     ),
#                     width=1,
#                 ),
#                 dbc.Col(
#                     html.P(
#                         "Explore Brain MRIs Throughout Pregnancy",
#                         style={"fontSize": S.title_fontsize},
#                     ),
#                     width=10,
#                 ),
#             ],
#             align="center",
#         ),
#     ]

#     overview_text = dbc.Row(
#         [
#             html.P(
#                 [
#                     "MRI data was collected ~ once every 2 weeks throughout pregnancy, showing the structural changes that occur in the brain over the course of a human pregnancy. Estrogen, progesterone, and LH levels were also measured at most sessions.",
#                 ],
#                 style={"fontSize": S.text_fontsize, "fontFamily": S.text_fontfamily},
#             ),
#             study_row,
#         ],
#     )

#     acknowledgements_text = dbc.Row(
#         [
#             html.P(
#                 [
#                     "Data and study timeline image from: Pritschet, Taylor, Cossio, Santander, Grotzinger, Faskowitz, Handwerker, Layher, Chrastil, Jacobs. Neuroanatomical changes observed over the course of a human pregnancy. (2024)",
#                 ],
#                 style={
#                     "fontSize": S.text_fontsize,
#                     "fontFamily": S.text_fontfamily,
#                     "marginLeft": S.margin_side,
#                     "marginRight": S.margin_side,
#                 },
#             ),
#         ],
#     )

#     instructions_text = dbc.Row(
#         [
#             html.P(
#                 [
#                     (
#                         "Use the 'Session Number' slider to flip through T1 brain data from each MRI session. "
#                         "Use the X, Y, Z coordinate sliders to choose the MRI slice. "
#                         "Additional information about the session will be displayed to the right of the sliders."
#                     )
#                 ],
#                 style={
#                     "fontSize": S.text_fontsize,
#                     "fontFamily": S.text_fontfamily,
#                     "marginLeft": S.margin_side,
#                     "marginRight": S.margin_side,
#                 },
#             ),
#         ],
#     )

#     contents_container = dbc.Container(
#         [
#             *banner,
#             html.Hr(),
#             overview_title(),
#             html.Div(style={"height": S.space_between_title_and_content}),
#             overview_text,
#             html.Div(style={"height": S.space_between_sections}),
#             html.Hr(),
#             instructions_title(),
#             html.Div(style={"height": S.space_between_title_and_content}),
#             instructions_text,
#         ]
#         + mri_explorer.to_dash()
#         + [
#             html.Div(style={"height": S.space_between_sections}),
#             html.Hr(),
#             acknowledgements_title(),
#             html.Div(style={"height": S.space_between_title_and_content}),
#             acknowledgements_text,
#         ],
#         fluid=True,
#     )

#     return [
#         dbc.Row(
#             [
#                 dbc.Col(sm=1),
#                 dbc.Col(contents_container, sm=10),
#                 dbc.Col(sm=1),
#             ]
#         )
#     ]


# def ai_hormone_prediction(mesh_explorer, gpt=False):
#     """Return the content of the AI hormone prediction page."""
#     banner = [
#         dbc.Row(
#             [
#                 dbc.Col(
#                     html.Img(
#                         src=get_asset_url("robot_emoji.jpeg"),
#                         style={"width": "70px", "height": "auto"},
#                     ),
#                     width=1,
#                 ),
#                 dbc.Col(
#                     html.P(
#                         "AI: Hormones to Hippocampus Shape",
#                         style={"fontSize": S.title_fontsize},
#                     ),
#                     width=10,
#                 ),
#             ],
#             align="center",
#         ),
#     ]

#     overview_text = dbc.Row(
#         [
#             html.P(
#                 [
#                     "The hippocampus is a brain region that is particularly sensitive to hormones. In pregnancy the hippocampus volume is known to decrease, but we find that the shape of the hippocampus changes as well. We have trained an AI to predict the shape of the hippocampus based on hormone levels.",
#                     html.Br(),
#                 ],
#                 style={"fontSize": S.text_fontsize, "fontFamily": S.text_fontfamily},
#             ),
#         ],
#     )

#     gpt_component = []
#     if gpt:
#         if set_openai_api_key():
#             gpt_component = [gpt_chat_component()]

#     contents_container = dbc.Container(
#         [
#             *banner,
#             html.Hr(),
#             overview_title(),
#             html.Div(style={"height": S.space_between_title_and_content}),
#             overview_text,
#             html.Div(style={"height": S.space_between_sections}),
#             html.Hr(),
#             instructions_title(),
#             html.Div(style={"height": S.space_between_title_and_content}),
#             dbc.Row(
#                 [
#                     html.P(
#                         [
#                             "Use the hormone sliders or the gestational week slider to adjust observe the predicted shape changes in the left hippocampal formation.",
#                             html.Br(),
#                         ],
#                         style={
#                             "fontSize": S.text_fontsize,
#                             "fontFamily": S.text_fontfamily,
#                         },
#                     ),
#                 ],
#             ),
#         ]
#         + mesh_explorer.to_dash()
#         + [html.Div(style={"height": S.space_between_sections}), html.Hr()]
#         + gpt_component,
#         fluid=True,
#     )

#     return [
#         dbc.Row(
#             [
#                 dbc.Col(sm=1),
#                 dbc.Col(contents_container, sm=10),
#                 dbc.Col(sm=1),
#             ]
#         )
#     ]


def pregnancy_page(mesh_mri_image_seq_explorer, gpt=False):
    """Creates the pregnancy page. 
    
    A button will indicate whether the user wants to predict by gestation week or hormones. If the user
    indicates gestation week, this page will display the image seq on the far left, the mri in the middle, 
    and the mri on the right. All of these will be controller by one gestation week slider.
    
    If the user selects hormones, this page will only display the mesh explorer, with sliders for 
    hormones."""
    banner = [
        dbc.Row(
            [
                dbc.Col(
                    html.Img(
                        src=get_asset_url("pregnancy_logo.png"),
                        style={"width": "70px", "height": "auto"},
                    ),
                    width=1,
                ),
                dbc.Col(
                    html.P(
                        "Digital Twin of the Pregnant Brain",
                        style={"fontSize": S.title_fontsize},
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
                    "Overview: The hippocampus is a brain region that is particularly sensitive to hormones. In pregnancy the hippocampus volume is known to decrease, but we find that the shape of the hippocampus changes as well. We have trained an AI to predict the shape of the hippocampus based on hormone levels or gestation week.",
                    html.Br(),
                ],
                style={"fontSize": S.text_fontsize, "fontFamily": S.text_fontfamily},
            ),
        ],
    )

    instructions_text = dbc.Row(
        [
            html.P(
                [
                    (
                        "Instructions: Use the hormone sliders or the gestational week slider to adjust observe the predicted shape changes in the left hippocampal formation. "
                        "Beige color indicates pre-pregnancy shape, blue indicates shrinking compared to pre-pregnancy, and red indicates growth."
                    )
                ],
                style={
                    "fontSize": S.text_fontsize,
                    "fontFamily": S.text_fontfamily,
                },
            ),
        ],
    )


    gpt_component = []
    if gpt:
        if set_openai_api_key():
            gpt_component = [gpt_chat_component()]

    contents_container = dbc.Container(
        [
            *banner,
            html.Hr(),
            overview_text,
            html.Div(style={"height": S.space_between_title_and_content}),
            instructions_text,
            html.Div(style={"height": S.space_between_sections}),
            html.Hr(),
        ]
        + mesh_mri_image_seq_explorer.to_dash()
        + [html.Div(style={"height": S.space_between_sections}), html.Hr()]
        + gpt_component,
        fluid=True,
    )

    return [
        dbc.Row(
            [
                dbc.Col(sm=1),
                dbc.Col(contents_container, sm=10),
                dbc.Col(sm=1),
            ]
        )
    ]


def menstrual_page():
    """Not Implemented."""
    return [
        dbc.Row(
            [
                dbc.Col(sm=1),
                dbc.Col([], sm=10),
                dbc.Col(sm=1),
            ]
        )
    ]


def app_layout(sidebar_elems, page_register):
    # the styles for the main content position it to the right of the sidebar and
    # add some padding.
    CONTENT_STYLE = {
        "margin-left": "18rem",
        "margin-right": "2rem",
        "padding": "2rem 1rem",
    }
    content = html.Div(id="page-content", style=CONTENT_STYLE)

    return html.Div(
        [
            dcc.Location(id="url"),
            sidebar(sidebar_elems, page_register),
            content,
        ]
    )
