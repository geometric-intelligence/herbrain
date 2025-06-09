import abc

import dash_bootstrap_components as dbc
from dash import html

from polpo.dash.style import STYLE as S


class Layout(abc.ABC):
    @abc.abstractmethod
    def to_dash(self, comps):
        pass

class ThreeColumnLayout(Layout):
    def to_dash(self, comps):
        right, middle, left = comps

        left_comp = left.to_dash() # probably the mesh
        middle = middle.to_dash()
        right_comp = right.to_dash()

        return [ # TODO: LATER
            dbc.Row(
                [
                    dbc.Col(
                        html.Div(
                            left_comp,
                            style={"paddingTop": "0px"},
                        ),
                        sm=6,
                        width=900,
                    ),
                    dbc.Col(sm=3, width=100),
                    dbc.Col(right_comp, sm=3, width=500),
                    dbc.Col(sm=3, width=100),
                    dbc.Col(right_comp, sm=3, width=500),
                ],
                align="center",
                style={
                    "marginLeft": S.margin_side,
                    "marginRight": S.margin_side,
                    "marginTop": "50px",
                },
            ),
        ]
    

class MeshLayout(Layout):
    """Create a layout for the mesh explorer.
    
    The mesh explorer has a single column, with the output at the top, and then three
    inputs below."""
    def to_dash(self, comps):
        inputs, mesh_graph = comps

        toggle_button, template_viz_button, gest_slider = inputs

        return dbc.Container(
            [
                dbc.Col(
                    [
                        html.Div(mesh_graph.to_dash(), style={"marginBottom": "40px"}),
                        html.Div(toggle_button.to_dash(), style={"marginBottom": "30px"}),
                        html.Div(template_viz_button.to_dash(), style={"marginBottom": "30px"}),
                        html.Div(
                            gest_slider.to_dash(),
                            style={"marginBottom": "30px", "width": "80%"},
                        ),
                    ],
                    width=12,
                    style={
                        "display": "flex",
                        "flexDirection": "column",
                        "justifyContent": "center",
                        "alignItems": "center",
                        "minHeight": "900px",  # Increased minimum height for more vertical space
                        "paddingTop": "30px",
                        "paddingBottom": "30px",
                    },
                )
            ],
            style={
                # "marginTop": "50px",
                "height": "100%",
                "display": "flex",
                "flexDirection": "column",
                "justifyContent": "center",
                "minHeight": "900px",  # Increased minimum height for more vertical space
            },
            fluid=True,
        )

class MriLayout(Layout):
    """Create a layout for the MRI explorer.
    
    The MRI explorer has a single column, with the output at the top, and then three
    inputs below."""
    def to_dash(self, comps):
        inputs, mri = comps

        radio_button, slider = inputs

        return dbc.Container(
            [
                dbc.Col(
                    [
                        html.Div(mri.to_dash()),
                        html.Div(radio_button.to_dash()),
                        html.Div(
                            slider.to_dash(),
                            style={
                                "width": "100%",
                                "border": "1px solid #e0e0e0",
                                "borderRadius": "8px",
                                "padding": "20px",
                                # "backgroundColor": "#fafbfc",
                                "marginBottom": "30px",
                            },
                        ),
                    ],
                    width=12,
                    style={"display": "flex", "flexDirection": "column", "justifyContent": "center", "alignItems": "center"},
                )
            ],
            style={
                # "marginTop": "50px", 
                "height": "100%", 
                "display": "flex", 
                "flexDirection": "column", 
                "justifyContent": "center"
                },
            fluid=True,
        )
