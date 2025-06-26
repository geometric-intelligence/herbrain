"""Simple GPT chat component for the AI prediction page."""

import base64
import logging
import os

import dash_bootstrap_components as dbc
import openai
import plotly.graph_objects as go
import plotly.io as pio
from dash import Input, Output, State, callback, dcc, html
from polpo.dash.style import STYLE as S


def set_openai_api_key():
    # Set OpenAI API key from text file
    api_key_path = os.environ.get("OPENAI_API_KEY_PATH", "openai_api_key.txt")
    try:
        with open(api_key_path, "r") as f:
            openai.api_key = f.read().strip()
            os.environ["OPENAI_API_KEY"] = openai.api_key

        return True
    except FileNotFoundError:
        logging.warning(
            f"File {api_key_path} not found. GPT chat functionality will not work."
        )
        return False


def gpt_chat_component():
    """Create a GPT chat component with message history and chat-like interface."""
    return html.Div(
        [
            html.Hr(),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4(
                                "Ask the AI neurobot what the changes mean for your brain",
                                style={"fontSize": S.title_fontsize},
                            ),
                            html.P(
                                "Ask questions about the brain changes during pregnancy, hormone levels, or any other aspects of the data shown on this page. The AI neurobot will analyze the 3D visualization of your brain and provide insights.",
                                style={
                                    "fontSize": "0.9em",
                                    "fontFamily": S.text_fontfamily,
                                    "color": "#666",
                                },
                            ),
                            # Input area
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Textarea(
                                                id="gpt-input",
                                                placeholder="Type your question here...",
                                                style={
                                                    "width": "100%",
                                                    "height": "100px",
                                                    "marginBottom": "10px",
                                                },
                                            ),
                                            dbc.Button(
                                                "Ask ChatGPT",
                                                id="gpt-submit",
                                                color="primary",
                                                className="mb-3",
                                            ),
                                        ]
                                    )
                                ]
                            ),
                            # Chat history container
                            html.Div(
                                id="chat-history",
                                style={
                                    "height": "200px",  # Reduced from 400px to 200px
                                    "overflowY": "auto",
                                    "border": "1px solid #ddd",
                                    "borderRadius": "5px",
                                    "padding": "10px",
                                    "marginBottom": "10px",
                                    "backgroundColor": "#f8f9fa",
                                },
                            ),
                            html.P(
                                "Disclaimer: Neurobot is an educational tool designed to spark curiosity and provide general information about neuroscience. It may generate incomplete, outdated, or inaccurate responses, and should not be relied upon for medical or diagnostic purposes. Neurobot is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider with any questions about your health or medical condition.",
                                style={
                                    "fontSize": "0.9em",
                                    "fontFamily": S.text_fontfamily,
                                    "color": "#666",
                                },
                            ),
                            # Store for chat history
                            dcc.Store(id="chat-store", data=[]),
                        ]
                    )
                ]
            ),
        ]
    )


def create_message_bubble(message, is_user=True):
    """Create a message bubble for the chat."""
    return html.Div(
        [
            html.Div(
                message,
                style={
                    "backgroundColor": "#007bff" if is_user else "#e9ecef",
                    "color": "white" if is_user else "black",
                    "padding": "10px",
                    "borderRadius": "10px",
                    "marginBottom": "10px",
                    "maxWidth": "80%",
                    "marginLeft": "auto" if is_user else "0",
                    "marginRight": "0" if is_user else "auto",
                },
            )
        ],
        style={"marginBottom": "10px"},
    )


@callback(
    [Output("chat-history", "children"), Output("gpt-input", "value")],
    Input("gpt-submit", "n_clicks"),
    [
        State("gpt-input", "value"),
        State("chat-store", "data"),
        State("gestWeek-slider", "value"),  # Gestational week slider
        State("estro-slider", "value"),  # Estrogen slider
        State("prog-slider", "value"),  # Progesterone slider
        State("lh-slider", "value"),  # LH slider
        State("mesh-plot", "figure"),
    ],  # Current mesh figure
    prevent_initial_call=True,
)
def update_chat(n_clicks, question, chat_history, gest_week, estro, prog, lh, figure):
    """Update the chat history when a new message is sent."""
    if not question:
        return chat_history, ""

    try:
        # Initialize OpenAI client
        client = openai.OpenAI()

        # Create context string with current slider values
        context = f"""Current hormone levels and gestational week:
- Gestational Week: {gest_week}
- Estrogen: {estro} pg/ml
- Progesterone: {prog} ng/ml
- LH: {lh} ng/ml

Please analyze the attached 3D mesh visualization and use these hormone values to provide context in your response."""

        # Convert Plotly figure to image
        if figure:
            # Create a proper Plotly figure object from the dictionary
            temp_fig = go.Figure(figure)
            # Ensure the figure has the right size and layout
            temp_fig.update_layout(
                width=800, height=600, margin=dict(l=0, r=0, t=0, b=0)
            )
            # Convert to PNG image
            img_bytes = pio.to_image(temp_fig, format="png")
            # Convert to base64
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        else:
            img_base64 = None


        # Prepare messages for the API call
        messages = [
            {
                "role": "system",
                "content": "You are a neuroscientist specializing in the pregnancy and postpartum brain. You answer questions using short, precise sentences. Only respond to questions related to neuroscience of pregnancy, hormones, and motherhood, and women's brains. If a question is outside this scope, politely decline to answer.",
            },
            {
                "role": "system",
                "content": "You are a helpful assistant explaining brain changes during pregnancy. Focus on the relationship between hormones and brain structure.",
            },
            {
                "role": "system",
                "content": "Just above your chat box, you see the rendered 3D hippocampus of a brain of a pregnant woman—this is the image provided in your context. Be prepared to answer questions based on what you observe in this brain image.",
            },
            {"role": "system", "content": context},
            {
                "role": "system",
                "content": "You can refer to the 3D mesh visualization of the brain and the MRI image provided in the chat or by the messages appended below.",
            },
            {"role": "system",
                "content": "You can also refer to the hormone levels and gestation week provided in the context."},
            {
                "role": "system",
                "content": "The user can also see an mri of the brain, which shows the brain during pregnancy at the gestation week indicated in the context. On the larger scale of the whole brain, it is difficult to see the changes that are happening as a result of pregnancy."
            },
            {
                "role": "system",
                "content": "The 3D mesh visualization shown in the appended message shows the hippocampus of the brain during pregnancy. This is a more localized view of the brain, and it is easier to see the changes that are happening as a result of pregnancy. Red indicates areas that are shrinking as a result of pregnancy, and blue shows areas that are getting bigger as a result of pregnancy. Beige areas have not changed from the pre-pregnancy state.",
            },
            {
                "role": "system",
                "content": "When answering questions, try to explain what is happening in the figures appended below. Try to explain that some areas are shrinking and some areas are growing, and that is a result of changing gestation week. Feel free to include any other observations you make."
            },
            {
                "role": "system",
                "content": "If you are unsure about the answer, please say that you don't know.",
            },
            {
                "role": "system",
                "content": "Your job is to be a scientific assistant. Assume that this app is sent to someone with no knowledge of neuroscience, who does not know how to read scientific plots. You are here to help them understand the data and results presented in the app.",
            },
            {
                "role": "system",
                "content": "You are a helpful assistant explaining brain changes during pregnancy. Focus on the relationship between hormones and brain structure.",
            },
            {
                "role": "system",
                "content": "The structures you see in the 3D mesh visualization are subcortical structures. Specifically, they are the accumbens nucleus, Amygdala, Caudate nucleus, Hippocampus, Globus pallidus (Pallidum), Putamen, Thalamus.",
            },

        ]

        # Add the image if available
        if img_base64:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                        },
                    ],
                }
            )
        else:
            messages.append({"role": "user", "content": question})
        

        # Create the chat completion
        response = client.chat.completions.create(
            model="gpt-4o",  # Using GPT-4o-mini to handle mesh visualization
            messages=messages,
            max_tokens=500,
        )

        # Get the response
        answer = response.choices[0].message.content

        # Update chat history
        new_history = chat_history + [
            create_message_bubble(question, is_user=True),
            create_message_bubble(answer, is_user=False),
        ]

        return new_history, ""

    except Exception as e:
        error_message = f"Error: {str(e)}"
        new_history = chat_history + [
            create_message_bubble(question, is_user=True),
            create_message_bubble(error_message, is_user=False),
        ]
        return new_history, ""


@callback(
    Output("gpt-submit", "n_clicks"),
    Input("gpt-input", "n_submit"),
    State("gpt-submit", "n_clicks"),
    prevent_initial_call=True,
)
def handle_enter(n_submit, n_clicks):
    """Handle Enter key press in the textarea."""
    if n_submit is None:
        return n_clicks
    return (n_clicks or 0) + 1
