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
    "width": "19rem",
    "padding": "1.75rem 1.5rem",
    "backgroundColor": "#FFFFFF",
    "borderRight": "1px solid #EAEAEA",
    "display": "flex",
    "flexDirection": "column",
}


def sidebar(sidebar_elems, page_register):
    """Return the sidebar of the app."""
    # Header with logo
    header = dcc.Link(
        html.Img(
            src=get_asset_url("herbrain_logo.png"),
            style={
                "width": "140px",
                "height": "auto",
                "cursor": "pointer",
            },
        ),
        href="/",
        style={"textDecoration": "none", "display": "block"},
    )
    
    # Subtitle
    subtitle = html.P(
        "Digital twins of women's brains",
        style={
            "fontFamily": "'Inter', -apple-system, sans-serif",
            "fontSize": "0.9rem",
            "color": "#7F8C8D",
            "marginTop": "1rem",
            "marginBottom": "0",
            "fontWeight": "400",
            "letterSpacing": "0.01em",
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
                        html.Span(
                            text,
                            style={
                                "fontFamily": "'Inter', -apple-system, sans-serif",
                                "fontSize": "1.1rem",
                                "fontWeight": "500",
                                "letterSpacing": "0.01em",
                            }
                        ),
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
            html.Hr(style={"borderColor": "#EAEAEA", "marginTop": "1.75rem", "marginBottom": "1.5rem", "opacity": "0.7"}),
            html.Div(nav_items),
        ],
        style=SIDEBAR_STYLE,
    )


def homepage():
    """Return the content of the homepage with card-based design."""
    
    # Color scheme - refined palette
    green_accent = "#4A7C6F"  # Teal green for Pregnancy
    orange_accent = "#E8927C"  # Coral/salmon for Menstruation & labels
    bg_color = "#FAFBFC"  # Subtle off-white background
    card_bg = "#FFFFFF"
    text_dark = "#1A1A2E"  # Deep charcoal
    text_muted = "#6B7280"  # Refined gray
    
    # Main title section - centered logo
    header = html.Div(
        [
            html.Img(
                src=get_asset_url("herbrain_logo.png"),
                style={
                    "width": "200px",
                    "height": "auto",
                    "marginBottom": "1.25rem",
                },
            ),
            html.P(
                "Digital Twins of Women's Brains",
                style={
                    "fontFamily": "'Inter', -apple-system, sans-serif",
                    "fontSize": "1.15rem",
                    "color": text_muted,
                    "fontWeight": "400",
                    "letterSpacing": "0.02em",
                }
            ),
        ],
        style={
            "textAlign": "center",
            "paddingTop": "4rem",
            "paddingBottom": "3.5rem",
        }
    )
    
    # Base card style - premium feel
    card_style_base = {
        "backgroundColor": card_bg,
        "borderRadius": "16px",
        "padding": "2.5rem 2rem",
        "textAlign": "center",
        "border": "1px solid #E5E7EB",
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
                    style={"width": "80px", "height": "80px", "marginBottom": "1.5rem"}
                ),
                html.H2(
                    "Pregnancy",
                    style={
                        "fontFamily": "'Playfair Display', Georgia, serif",
                        "fontSize": "1.85rem",
                        "fontWeight": "600",
                        "color": text_dark,
                        "marginBottom": "0.5rem",
                        "letterSpacing": "-0.01em",
                    }
                ),
                html.P(
                    "Digital Twin",
                    style={
                        "color": orange_accent,
                        "fontFamily": "'Inter', sans-serif",
                        "fontSize": "0.95rem",
                        "fontWeight": "500",
                        "marginBottom": "1rem",
                        "letterSpacing": "0.02em",
                    }
                ),
                html.P(
                    "Explore brain transformations across 40 weeks of pregnancy. Track how subcortical structures respond to hormonal surges.",
                    style={
                        "color": text_muted,
                        "fontFamily": "'Inter', sans-serif",
                        "fontSize": "0.925rem",
                        "lineHeight": "1.7",
                        "maxWidth": "300px",
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
                    style={"width": "80px", "height": "80px", "marginBottom": "1.5rem"}
                ),
                html.H2(
                    "Menstruation",
                    style={
                        "fontFamily": "'Playfair Display', Georgia, serif",
                        "fontSize": "1.85rem",
                        "fontWeight": "600",
                        "color": text_dark,
                        "marginBottom": "0.5rem",
                        "letterSpacing": "-0.01em",
                    }
                ),
                html.P(
                    "Digital Twin",
                    style={
                        "color": orange_accent,
                        "fontFamily": "'Inter', sans-serif",
                        "fontSize": "0.95rem",
                        "fontWeight": "500",
                        "marginBottom": "1rem",
                        "letterSpacing": "0.02em",
                    }
                ),
                html.P(
                    "Coming soon: Explore cyclic brain changes throughout the menstrual cycle. Understand how monthly hormonal fluctuations shape neural structure.",
                    style={
                        "color": text_muted,
                        "fontFamily": "'Inter', sans-serif",
                        "fontSize": "0.925rem",
                        "lineHeight": "1.7",
                        "maxWidth": "300px",
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
            dbc.Col(pregnancy_card, md=6, lg=5, className="mb-4 px-3"),
            dbc.Col(menstruation_card, md=6, lg=5, className="mb-4 px-3"),
        ],
        justify="center",
    )
    
    # Footer
    footer = html.Div(
        [
            html.Hr(style={"borderColor": "#E5E7EB", "marginTop": "4rem", "opacity": "0.6"}),
            html.P(
                "© 2024 Geometric Intelligence",
                style={
                    "textAlign": "center",
                    "color": text_muted,
                    "fontFamily": "'Inter', sans-serif",
                    "fontSize": "0.85rem",
                    "paddingTop": "1.5rem",
                    "paddingBottom": "2rem",
                    "letterSpacing": "0.01em",
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
                    style={"maxWidth": "960px"},
                ),
            ],
            style={
                "backgroundColor": bg_color,
                "minHeight": "100vh",
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
        "marginLeft": "19rem",
        "marginRight": "1.5rem",
        "padding": "1.5rem 1rem",
        "minHeight": "100vh",
        "backgroundColor": "#FAFBFC",
    }
    content = html.Div(id="page-content", style=CONTENT_STYLE)

    return html.Div(
        [
            dcc.Location(id="url"),
            sidebar(sidebar_elems, page_register),
            content,
        ],
        style={"backgroundColor": "#FAFBFC"},
    )
