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
    "padding": "2rem 1.5rem",
    "backgroundColor": "#FFFFFF",
    "borderRight": "1px solid #F0F0F0",
}


def sidebar(sidebar_elems, page_register):
    """Return the sidebar of the app."""
    # Header with logo and title
    header = dcc.Link(
        html.Div(
            [
                html.Img(
                    src=get_asset_url("herbrain_logo.png"),
                    style={
                        "width": "50px",
                        "height": "auto",
                        "marginRight": "0.75rem",
                    },
                ),
                html.Span(
                    "HerBrain",
                    style={
                        "fontFamily": "'Playfair Display', Georgia, serif",
                        "fontSize": "1.75rem",
                        "fontWeight": "700",
                        "color": "#2D3436",
                    }
                ),
            ],
            style={
                "display": "flex",
                "alignItems": "center",
                "cursor": "pointer",
            }
        ),
        href="/",
        style={"textDecoration": "none"},
    )
    
    # Subtitle
    subtitle = html.P(
        "Digital twins of women's brains",
        style={
            "fontFamily": "'Inter', -apple-system, sans-serif",
            "fontSize": "0.95rem",
            "color": "#636E72",
            "marginTop": "0.75rem",
            "marginBottom": "0",
            "fontWeight": "400",
        }
    )

    # Register all routes (including inactive ones), but build custom nav items
    nav_items = []
    for elem in sidebar_elems:
        # Register route for all elements (active or not)
        elem.to_dash(page_register)
        
        # Only add to nav if active
        if elem.active:
            # Get the href and text from the tab_header
            href = elem.tab_header.href
            text = elem.tab_header.text
            image_url = elem.tab_header.image_url
            
            # Determine which card class to use for active state
            card_class = "sidebar-nav-item"
            
            nav_item = dcc.Link(
                html.Div(
                    [
                        html.Img(
                            src=get_asset_url(image_url),
                            className="sidebar-nav-icon",
                        ),
                        html.Span(text),
                    ],
                    style={
                        "display": "flex",
                        "alignItems": "center",
                    }
                ),
                href=href,
                className=card_class,
                id=f"nav-{text.lower()}",
            )
            nav_items.append(nav_item)

    return html.Div(
        [
            header,
            subtitle,
            html.Hr(style={"borderColor": "#F0F0F0", "marginTop": "1.5rem", "marginBottom": "1.5rem"}),
            html.Div(nav_items),
        ],
        style=SIDEBAR_STYLE,
    )


def homepage():
    """Return the content of the homepage with card-based design."""
    
    # Color scheme
    accent_color = "#E8927C"  # Coral/salmon color for "Digital Twin" labels
    bg_color = "#F5F6F8"  # Light gray background
    card_bg = "#FFFFFF"
    text_dark = "#2D3436"
    text_muted = "#636E72"
    
    # Main title section
    header = html.Div(
        [
            html.H1(
                "HerBrain",
                style={
                    "fontFamily": "'Playfair Display', Georgia, serif",
                    "fontSize": "4rem",
                    "fontWeight": "700",
                    "color": text_dark,
                    "marginBottom": "0.5rem",
                    "letterSpacing": "-1px",
                }
            ),
            html.P(
                "Digital Twins of Women's Brains",
                style={
                    "fontFamily": "'Inter', -apple-system, sans-serif",
                    "fontSize": "1.25rem",
                    "color": text_muted,
                    "fontWeight": "400",
                }
            ),
        ],
        style={
            "textAlign": "center",
            "paddingTop": "3rem",
            "paddingBottom": "3rem",
        }
    )
    
    # Base card style
    card_style_base = {
        "backgroundColor": card_bg,
        "borderRadius": "12px",
        "padding": "2rem 1.5rem",
        "textAlign": "center",
        "border": "1px solid #E8EAED",
        "height": "100%",
        "display": "flex",
        "flexDirection": "column",
        "alignItems": "center",
    }
    
    # Pregnancy card
    pregnancy_card = dcc.Link(
        html.Div(
            [
                html.Img(
                    src=get_asset_url("pregnancy_logo.png"),
                    style={"width": "70px", "height": "70px", "marginBottom": "1.25rem"}
                ),
                html.H2(
                    "Pregnancy",
                    style={
                        "fontFamily": "'Playfair Display', Georgia, serif",
                        "fontSize": "1.75rem",
                        "fontWeight": "600",
                        "color": text_dark,
                        "marginBottom": "0.4rem",
                    }
                ),
                html.P(
                    "Digital Twin",
                    style={
                        "color": accent_color,
                        "fontFamily": "'Inter', sans-serif",
                        "fontSize": "0.9rem",
                        "fontWeight": "500",
                        "marginBottom": "0.75rem",
                    }
                ),
                html.P(
                    "Explore brain transformations across 40 weeks of pregnancy. Track how subcortical structures respond to hormonal surges.",
                    style={
                        "color": text_muted,
                        "fontFamily": "'Inter', sans-serif",
                        "fontSize": "0.875rem",
                        "lineHeight": "1.6",
                        "maxWidth": "280px",
                    }
                ),
            ],
            style=card_style_base,
            className="pregnancy-card",
        ),
        href="/page-1",
        style={"textDecoration": "none", "display": "block", "height": "100%"},
    )
    
    # Menstruation card  
    menstruation_card = dcc.Link(
        html.Div(
            [
                html.Img(
                    src=get_asset_url("menstrual_logo.png"),
                    style={"width": "70px", "height": "70px", "marginBottom": "1.25rem"}
                ),
                html.H2(
                    "Menstruation",
                    style={
                        "fontFamily": "'Playfair Display', Georgia, serif",
                        "fontSize": "1.75rem",
                        "fontWeight": "600",
                        "color": text_dark,
                        "marginBottom": "0.4rem",
                    }
                ),
                html.P(
                    "Digital Twin",
                    style={
                        "color": accent_color,
                        "fontFamily": "'Inter', sans-serif",
                        "fontSize": "0.9rem",
                        "fontWeight": "500",
                        "marginBottom": "0.75rem",
                    }
                ),
                html.P(
                    "Coming soon: Explore cyclic brain changes throughout the menstrual cycle. Understand how monthly hormonal fluctuations shape neural structure.",
                    style={
                        "color": text_muted,
                        "fontFamily": "'Inter', sans-serif",
                        "fontSize": "0.875rem",
                        "lineHeight": "1.6",
                        "maxWidth": "280px",
                    }
                ),
            ],
            style=card_style_base,
            className="menstruation-card",
        ),
        href="/page-2",
        style={"textDecoration": "none", "display": "block", "height": "100%"},
    )
    
    # Cards row
    cards_row = dbc.Row(
        [
            dbc.Col(pregnancy_card, md=6, lg=5, className="mb-4"),
            dbc.Col(menstruation_card, md=6, lg=5, className="mb-4"),
        ],
        justify="center",
        style={"paddingLeft": "1rem", "paddingRight": "1rem"},
    )
    
    # Footer
    footer = html.Div(
        [
            html.Hr(style={"borderColor": "#E8EAED", "marginTop": "3rem"}),
            html.P(
                "© 2024 Geometric Intelligence",
                style={
                    "textAlign": "center",
                    "color": text_muted,
                    "fontFamily": "'Inter', sans-serif",
                    "fontSize": "0.9rem",
                    "paddingTop": "1.5rem",
                    "paddingBottom": "2rem",
                }
            ),
        ]
    )
    
    # Main container with background
    return [
        html.Div(
            [
                dbc.Container(
                    [
                        header,
                        cards_row,
                        footer,
                    ],
                    fluid=True,
                    style={"maxWidth": "1000px"},
                ),
            ],
            style={
                "backgroundColor": bg_color,
                "minHeight": "100vh",
                "paddingTop": "1rem",
            }
        )
    ]



def pregnancy_page(pregnancy_explorer, gpt=False):
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
                    "Overview: The subcortical structures of the brain are sensitive to sex hormone changes. In pregnancy, hormones experience extreme fluctuations, and subcortical structure volumes are known to decrease. However, we find that the shape of these structures change as well. We have trained an AI to predict shape changes of the subcortical structures based on hormone levels or gestation week. Blue areas indicate growth and red areas indicate shrinkage compared to pre-pregnancy shape. Beige color indicates pre-pregnancy shape.",
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
                        "Instructions: Change the gestational week slider or hormone sliders, and the AI model will predict subcortical structure shape changes for these inputs."
                        " The MRI view will update to show the closest corresponding MRI data."
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
        ]
        + pregnancy_explorer.to_dash()
        + [html.Div(style={"height": 0}), html.Hr()]
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
