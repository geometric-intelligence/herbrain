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
        inputs, output = comps

        input1, input2, input3 = inputs

        return dbc.Container(
            [
                dbc.Col(
                    [
                        html.Div(output.to_dash(), style={"marginBottom": "40px"}),
                        html.Div(input1.to_dash(), style={"marginBottom": "30px"}),
                        html.Div(input2.to_dash(), style={"marginBottom": "30px"}),
                        html.Div(input3.to_dash(), style={"marginBottom": "30px"}),
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
                "marginTop": "50px",
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
        inputs, output = comps

        input1, input2 = inputs

        return dbc.Container(
            [
                dbc.Col(
                    [
                        html.Div(output.to_dash()),
                        html.Div(input1.to_dash()),
                        html.Div(input2.to_dash()),
                    ],
                    width=12,
                    style={"display": "flex", "flexDirection": "column", "justifyContent": "center", "alignItems": "center"},
                )
            ],
            style={"marginTop": "50px", "height": "100%", "display": "flex", "flexDirection": "column", "justifyContent": "center"},
            fluid=True,
        )
