from polpo.dash.components import (
    ComponentGroup,
    DepVar,
    FunctionComponent,
    Graph,
    MriSliders,
    MultiModelsMeshExplorer,
    SharedInputModelsBasedExplorer,
    MeshExplorer,
    SidebarElem,
    SidebarHeader,
    Slider,
    MriGraphRow,
    BaseComponentGroup,
    RadioButton,
    Component,
    VarDefComponent,
)
from polpo.dash.layout import MultiRowLayout, TwoRowLayout
from polpo.models import (
    MriSlicesLookup,
    PdDfLookup,
)
from polpo.dash.callbacks import (
    create_button_toggler_for_view_model_update,
    create_view_model_update,
)
import polpo.preprocessing.dict as ppdict
import dash_bootstrap_components as dbc
from polpo.dash.variables import VarDef
from polpo.preprocessing import ListSqueeze
from polpo.plot.mesh import MeshesPlotter, MeshPlotter, StaticMeshPlotter
from polpo.plot.plotly import GoPlotter #SlicePlotter
from .data import (
    # HormonesCsvLoader,
    # MaternalRegisteredMeshesLoader,
    # MultipleMaternalMeshesLoader,
    NibImage2Mesh,
    # PilotMriImageLoader,
    # TemplateImageLoader,
)
from .layout import MeshLayout, MriLayout
from .models import MriModel, ClosestImageLookup
from polpo.dash.style import STYLE as S
from dash import Dash, Input, Output, State, callback, dcc, html
import numpy as np
import plotly.graph_objs as go

from polpo.dash.components import Image, Slider
from polpo.dash.style import update_style
from polpo.dash.variables import VarDef
from polpo.models import ListLookup   
import os
import sys
from polpo.preprocessing import Sorter
from polpo.preprocessing.path import FileFinder
from dash import Dash, get_asset_url

    


class PregnancyExplorer:
    def __init__(
        self,
        cfg,
        mri_data,
        hormones_df,
        data_type,
        template_image,
        n_structs,
        week_mesh_model,
        hormones_mesh_model,
        hormones_ordering,
    ):
        # Variable definitions
        
        self.gest_week_var = VarDef(
            id_="gestWeek", name="Gestational Week", min_value=0, max_value=36, default_value=15
        )
        self.estro = VarDef( # makes it easier to have all this info contained in a var, rather than having to type these things every time they are used
            id_="estro",
            name="Estrogen",
            unit="pg/ml",
            min_value=4100,
            max_value=12400,
        )
        self.prog = VarDef(
            id_="prog",
            name="Progesterone",
            unit="ng/ml",
            min_value=54,
            max_value=103,
        )
        self.lh = VarDef(
            id_="lh",
            name="LH",
            unit="ng/ml",
            min_value=0.59,
            max_value=1.45,
        )
        self.template_mesh = NibImage2Mesh()(template_image) # used for whole brain animation

        self.postproc_pred = None
        if data_type == "multiple":
            self.postproc_pred = ppdict.DictMap(step=ListSqueeze()) + ppdict.DictToValuesList()

        self.hormone_label_style = {"fontSize": 30, "display": "block"}
        self.template_visibility = True
        self.n_structs = n_structs
        self.week_mesh_model = week_mesh_model
        self.hormones_mesh_model = hormones_mesh_model
        self.hormones_ordering = hormones_ordering
        self.mri_data = mri_data
        # TODO: create a radio button for the different views of the brain 
        self.hormones_df = hormones_df

        self.gest_week_slider = Slider(self.gest_week_var)

        self.mri_explorer = MriExplorer(
            self.gest_week_slider,
            self.gest_week_var,
            self.mri_data,
            self.hormones_df, # this tells the gest week-session correspondance.
            radio_button_init = 0, # "sagittal", # default view
            id_prefix="",
        )

        self.animation_explorer = AnimationExplorer(cfg.app.assets_folder, self.gest_week_slider)

        self.gest_week_mesh_explorer = MultiModelsMeshExplorer(
            graph=Graph( # dash graph object
                id_="mesh-plot",
                plotter=MeshesPlotter(
                    plotters=[MeshPlotter() for _ in range(self.n_structs)],
                    overlay_plotter=StaticMeshPlotter( # this is for the overall brain
                        mesh=self.template_mesh, visible=self.template_visibility
                    ),
                    bounds=None,  # TODO: check need
                    overlay_bounds=None,  # TODO: check need
                ),
            ),
            models=(self.week_mesh_model, self.hormones_mesh_model),
            inputs=(
                Slider(self.gest_week_var), # not yet a dash slider. will turn into a dash slider when we call .to_dash(). it is just an array of containers or something
                ComponentGroup( # this has a to_dash() too. everything under components has a to_dash()
                    ordering=self.hormones_ordering,
                    components=[
                        Slider(var, step, label_style=self.hormone_label_style)
                        for var, step in [
                            (self.estro, 500),
                            (self.prog, 3),
                            (self.lh, 0.05),
                        ]
                    ],
                ),
            ),
            checkbox_labels=((-1, "Show Full Brain", self.template_visibility),),
            button_label=" Click Here to Toggle Between Gestational Week vs Hormone Value Prediction",
            postproc_pred=self.postproc_pred,
            layout=MeshLayout(),
        )


    def to_dash(self):
        return [
            dbc.Row(
                [
                    # The columns are overlapping because the sum of the widths (2 + 4 + 6 = 12) matches the default Bootstrap grid (12 columns),
                    # but if the content inside the columns is too wide, or if there is not enough padding/margin, they may visually overlap.
                    # To prevent overlap, ensure that the content inside each column is responsive and does not exceed its column width.
                    # You can also add style or className to enforce overflow handling or add padding.
                    # Here is a version with explicit style to help prevent overlap:

                    dbc.Col(self.animation_explorer.to_dash(), width=2, style={"overflow": "auto", "padding": "20px"}),
                    dbc.Col(self.mri_explorer.to_dash(), width=5, style={"overflow": "auto", "padding": "20px"}),
                    dbc.Col(self.gest_week_mesh_explorer.to_dash(), width=5, style={"overflow": "auto", "padding": "20px"}),
                ],
                align="center",
            ),
            # dbc.Row(
            #     [
            #         dbc.Col(self.gest_week_slider.to_dash(), width=6),
            #     ],
            #     align="center",
            # ),
        ]
    



class MriExplorer(BaseComponentGroup): # different from one in polpo because it will intake gest week slider and output one mri slice.
    def __init__(
        self,
        gest_week_slider,
        gest_week_var,
        mri_data,
        hormones_df, # this tells the gest week-session correspondance.
        radio_button_init = "sagittal", # default view
        id_prefix="",
    ):
        graph_object = Graph(id_="mri-plot")
        self.radio_button_init = radio_button_init
        self.radio_button = RadioButton(id_="mri-view-toggle",
                                       options=[(0, "Sagittal"), (1, "Coronal"), (2, "Axial")],
                                       default_value=radio_button_init)
        
        self.mri_data = mri_data

        # Get the dimensions for each view
        sample_volume = mri_data[0]  # Use first volume to get dimensions
        self.view_dims = {
            "sagittal": sample_volume.shape[0],
            "coronal": sample_volume.shape[1],
            "axial": sample_volume.shape[2]
        }

        # Create a VarDef for each view's slice range
        self.mri_slice = VarDef(
            id_="mri_slice",
            name="MRI Slice",
            min_value=0,
            max_value=min(self.view_dims.values()) - 1,  # Use min dimension across all views
            default_value=min(self.view_dims.values()) // 2
        )

        self.mri_slice_slider = Slider(
            self.mri_slice,
            step=1,  # Use step=1 for finer control
            label_style={"fontSize": 20, "display": "block"}
        )
        
        self.graph_object = graph_object
        self.mri_model = MriModel(data=mri_data, hormones_df=hormones_df, index_tar=1, slicer=None)
        self.gest_week_slider = gest_week_slider

        super().__init__([gest_week_slider, self.mri_slice_slider, self.radio_button, graph_object], id_prefix)

    def to_dash(self): # this is where you create the layout of the page
        if hasattr(self.gest_week_slider, "update_lims"):
            self.gest_week_slider.update_lims(self.mri_data)

        # Create a single model for the MRI explorer
        models = [self.mri_model]

        # Set up inputs in the correct order: (gest_week, view_index, slice_index)
        inputs = ComponentGroup(
            components=[
                self.gest_week_slider,  # gest_week
                self.radio_button,      # view_index
                self.mri_slice_slider   # slice_index
            ],
            title="MRI Controls"
        )

        outputs = Graph(
                id_="mri-view",
                plotter=SlicePlotter(title="Selected MRI", just_image=True),
            )

        shown_inputs = ComponentGroup(
            components=[
                self.radio_button,      # view_index
                self.mri_slice_slider,  # slice_index
            ],
            title="Shown MRI controls"
        )

        # Create the explorer with the model, inputs, and output
        mri_explorer = SharedOutputModelsBasedExplorer(
            models=models,
            inputs=inputs,
            outputs=outputs,
            shown_inputs=shown_inputs,
            layout=MriLayout(),
        )

        return dbc.Container(mri_explorer.to_dash())




class AnimationExplorer():
    def __init__(self, assets_folder_path, week_slider):
        self.assets_folder_path = assets_folder_path
        self.image_paths = self._load_pregnancy_images()
        self.week_slider = week_slider # should be a VarDef

    def _load_pregnancy_images(self):
        """ Load pregnancy images from the specified assets folder.
        
        Returns
        -------
        List[str]
            List of URLs for the loaded images.
        """
        # assumes assets at app folder level
        file_path = os.path.dirname(sys.modules[__package__].__file__)
        # removes ./
        short_assets_folder = "/".join(self.assets_folder_path.split("/")[1:])

        assets_folder_abs = os.path.join(file_path, short_assets_folder)

        images = (
            FileFinder(data_dir=os.path.join(assets_folder_abs, "pregnancy_frames")) + Sorter()
        )()

        n_path_assets = len(assets_folder_abs)
        return [get_asset_url(image[n_path_assets + 1 :]) for image in images]


    def to_dash(self):
        # TODO: do version with DictLookup
        model = ClosestImageLookup(self.image_paths) #here, input will be weeks, and output needs to be an image.

        input = self.week_slider
        
        output = Image(id_=f"pregnancy-image", style={"width": "100%", "height": "auto", "maxWidth": "1000px"})

        image_seq_explorer = SingleInputOutputModelsBasedExplorer(model, input, output, shown_input=None)
        return dbc.Container(image_seq_explorer.to_dash())



class SharedOutputModelsBasedExplorer(BaseComponentGroup):
    def __init__(
        self, models, inputs, outputs, shown_inputs=None, id_prefix="", postproc_pred=None, layout=None
    ):
        self.models = models
        self.inputs = inputs
        self.outputs = outputs
        self.postproc_pred = postproc_pred
            
        self.shown_inputs = shown_inputs
        self.layout = layout or MultiRowLayout()  # Use MultiRowLayout as default

        super().__init__([outputs, inputs], id_prefix=id_prefix)

    def to_dash(self):
        # Create a simple layout if none is provided
        if self.shown_inputs is None:
            inputs_col = dbc.Col([], width=6)
        else:
            inputs_col = dbc.Col(self.shown_inputs.to_dash(), width=6)

        if self.layout is None:
            return dbc.Container([
                dbc.Col(
                    [
                        dbc.Row(
                            dbc.Col(self.outputs.to_dash(), width=6)
                            ),
                        # dbc.Row(inputs_col.children, width=6),
                        dbc.Row(inputs_col),
                    ],
                )
            ])
        
        out = self.layout.to_dash([self.shown_inputs, self.outputs,])

        # Create callbacks for each input-model pair
        for model in self.models:
            create_view_model_update(
                output_view=self.outputs,
                input_view=self.inputs,
                model=model,
                postproc_pred=self.postproc_pred,
            )

        return out
    

class SingleInputOutputModelsBasedExplorer(BaseComponentGroup):
    def __init__(
        self, model, input, output, shown_input=None, id_prefix="", postproc_pred=None, layout=None
    ):
        if layout is None:
            layout = TwoRowLayout()

        self.model = model
        self.input = input
        self.output = output
        self.postproc_pred = postproc_pred
        self.layout = layout
        self.shown_input = shown_input

        super().__init__([output, input], id_prefix=id_prefix)

    def to_dash(self):
        if self.shown_input is None:
            out = dbc.Row([
                    dbc.Col(self.output.to_dash(), width=12),
            ])
        else:
            out = dbc.Row([
                    dbc.Col(self.output.to_dash(), width=12),
                    dbc.Col(self.shown_input.to_dash(), width=12),
            ])

        create_view_model_update(
            output_view=self.output,
            input_view=self.input,
            model=self.model,
            postproc_pred=self.postproc_pred,
        )

        return out
    


class SlicePlotter(GoPlotter): # need to eventually integrate with polpo, but for now putting here so that i can remove the x and y ticks.
    def __init__(
        self, cmap="gray", title="Slice Visualization", x_label="X", y_label="Y", just_image=False
    ):
        self.cmap = cmap
        self.title = title
        self.x_label = x_label
        self.y_label = y_label

        if not just_image:
            self.layout = go.Layout(
                title=self.title,
                title_x=0.5,
                xaxis=dict(title=self.x_label),
                yaxis=dict(title=self.y_label),
                uirevision="constant",
            )
        else:
            self.layout = go.Layout(
                title=None,  # No title
                title_x=0.5,  # (optional, has no effect if title=None)
                xaxis=dict(
                    title=None,         # No x axis label
                    showticklabels=False,  # No tick numbers
                    ticks='',              # No ticks
                    showgrid=False,        # No grid lines
                    zeroline=False,        # No zero line
                    showline=False,        # No axis line
                ),
                yaxis=dict(
                    title=None,         # No y axis label
                    showticklabels=False,  # No tick numbers
                    ticks='',              # No ticks
                    showgrid=False,        # No grid lines
                    zeroline=False,        # No zero line
                    showline=False,        # No axis line
                ),
                uirevision="constant",
            )
            

    def transform_data(self, data):
        return [go.Heatmap(z=data.T, colorscale=self.cmap, showscale=False)]

    def plot(self, data=None):
        # Create heatmap trace for the current slice
        if data is None:
            return go.Figure(
                layout=self.layout,
            )

        fig = go.Figure(data=self.transform_data(data), layout=self.layout)

        width = int(len(data[:, 0]) * 1.5)
        height = int(len(data[0]) * 1.5)

        fig.update_layout(
            width=width,
            height=height,
        )

        print(f"self.x_label: {self.x_label}, self.y_label: {self.y_label}")

        
        return fig
    

class RadioButton(Component): # the one in polpo had a bug
    """Radio button group.

    Parameters
    ----------
    id_ : str
        The unique ID for the radio button group.
    options : list of tuple
        A list of (value, label) tuples for the options.
    default_value : str
        The default selected value.
    inline : bool
        Whether to display options inline (horizontally).
    """

    def __init__(self, id_, options, default_value=None, inline=True):
        super().__init__(id_prefix=id_) # initialize Component with id_prefix
        self.options = options
        self.default_value = default_value or options[0][0]
        self.inline = inline
        self.id_ = id_  # store the id for later use

    def to_dash(self):
        """Convert the component into a Dash UI element."""
        return html.Div(
            [
                html.Span("MRI View", style={"marginRight": "16px", "fontWeight": "bold"}),
                dcc.RadioItems(
                    id=self.id_,
                    options=[
                        {"label": label, "value": value}
                        for value, label in self.options
                    ],
                    value=self.default_value,
                    inline=self.inline,
                    style={'margin': '20px 0'}
                ),
            ],
            style={"display": "flex", "alignItems": "center"}
        )

    def as_input(self):
        return [Input(self.id_, "value")]

    def as_output(self, component_property="value", allow_duplicate=False):
        return [Output(self.id_, component_property, allow_duplicate=allow_duplicate)]


# class Slider(VarDefComponent):
#     """Slider."""

#     def __init__(self, var_def, step=1, id_prefix="", label_style=None):
#         super().__init__(var_def=var_def, id_prefix=id_prefix, id_suffix="-slider")
#         # TODO: think more about this design

#         self.step = step

#         # TODO: can default be set for the general app instead?
#         default_label_style = {
#             "fontSize": S.text_fontsize,
#             "fontFamily": S.text_fontfamily,
#         }
#         self.label_style = (label_style or {}).update(default_label_style)

#     def __repr__(self):
#         return f"Slider({self.id})"

#     def to_dash(self):
#         # TODO: allow to config from config file, e.g. label_style
#         label = dbc.Label(
#             self.var_def.label,
#             style=self.label_style,
#         )

#         # ensure default value is on the slider
#         min_value, max_value = self.var_def.min_value, self.var_def.max_value
#         step = self.step
#         value = min(max_value, self.var_def.default_value)
#         value = max(min_value, value)
#         n_steps = round((value - min_value) / step)
#         value = min_value + step * n_steps

#         # slider = dcc.Slider(
#         #     id=self.id,
#         #     min=min_value,
#         #     max=max_value,
#         #     step=step,
#         #     value=value,
#         #     marks={
#         #         self.var_def.min_value: {"label": "min"},
#         #         self.var_def.max_value: {"label": "max"},
#         #     },
#         #     tooltip={
#         #         "placement": "bottom",
#         #         "always_visible": True,
#         #         "style": {"fontSize": "25px", "fontFamily": S.text_fontfamily},
#         #     },
#         # )
#         slider = html.Div(
#             [
#                 html.Span(self.var_def.label, style={"marginRight": "16px", "fontWeight": "bold"}),
#                 dcc.Slider(
#                 id=self.id,
#                 min=min_value,
#                 max=max_value,
#                 step=step,
#                 value=value,
#                 marks={
#                     self.var_def.min_value: {"label": "min"},
#                     self.var_def.max_value: {"label": "max"},
#                 },
#                 tooltip={
#                     "placement": "bottom",
#                     "always_visible": True,
#                     "style": {"fontSize": "25px", "fontFamily": S.text_fontfamily},
#                 },
#             )
#             ],
#             style={"display": "flex", "alignItems": "center"}
#         )

#         # return [label, slider]
#         return slider

#     def as_input(self):
#         return [Input(self.id, "drag_value")]
