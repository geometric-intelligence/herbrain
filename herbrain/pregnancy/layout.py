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

        return dbc.Col(
            [
                dbc.Row(
                    html.Div(output.to_dash()),
                ),
                dbc.Row(
                    html.Div(input1.to_dash()),
                ),
                dbc.Row(
                    html.Div(input2.to_dash()),
                ),
                dbc.Row(
                    html.Div(input3.to_dash()),
                ),
            ],
            align="right",
            style={
                # "marginLeft": S.margin_side,
                # "marginRight": S.margin_side,
                "marginTop": "50px",
            },
        )
