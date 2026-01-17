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

from polpo.dash.components import Image, Slider as PolpoSlider
from polpo.dash.style import update_style
from polpo.dash.variables import VarDef
from polpo.models import ListLookup   
import os
import sys
from polpo.preprocessing import Sorter


class DebouncedSlider(PolpoSlider):
    """Optimized slider that only fires callbacks on drag release for better performance."""
    
    def to_dash(self):
        """Create a dcc.Slider with drag-release mode for instant feedback."""
        label = dbc.Label(
            self.var_def.label,
            style=self.label_style,
        )

        min_value, max_value = self.var_def.min_value, self.var_def.max_value
        step = self.step
        value = min(max_value, self.var_def.default_value)
        value = max(min_value, value)
        n_steps = round((value - min_value) / step)
        value = min_value + step * n_steps

        slider = dcc.Slider(
            id=self.id,
            min=min_value,
            max=max_value,
            step=step,
            value=value,
            marks={
                min_value: {"label": "min"},
                max_value: {"label": "max"},
            },
            tooltip={
                "placement": "bottom",
                "always_visible": True,
                "style": {"fontSize": "25px", "fontFamily": S.text_fontfamily},
            },
            updatemode='drag',  # Only fire on drag release, not during drag
        )

        return [label, slider]
    
    def as_input(self):
        """Override to return 'value' instead of 'drag_value' for consistency with clientside callbacks."""
        return [Input(self.id, "value")]


# Replace Slider with DebouncedSlider for better performance
Slider = DebouncedSlider
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
        prerendered_week_figures=None,
    ):
        """PregnancyExplorer class. 
        
        Parameters
        ----------
        cfg : Config
            Configuration object containing application settings.
        mri_data : list of np.ndarray (TODO: check type)
            List of MRI data arrays for different gestational weeks.
        hormones_df : pd.DataFrame
            DataFrame containing hormone levels corresponding to gestational weeks.
        data_type : str
            Type of data to be processed, e.g., "single" or "multiple".
        template_image : np.ndarray
            Template image, which appears overlaying the subcortical structures when ``show whole brain'' button is clicked.
        n_structs : int
            Number of structures to be visualized in the mesh explorer.
        week_mesh_model : PostTransformingEstimator from Polpo
            Model for the mesh corresponding to gestational weeks.
        hormones_mesh_model : PostTransformingEstimator from Polpo
            Model for the mesh corresponding to hormone values.
        hormones_ordering : list of str
            List defining the order of hormones for visualization.
        prerendered_week_figures : dict, optional
            Pre-rendered Plotly figures for each gestational week (0-45).
            Used for clientside switching to eliminate network traffic.
        """
        self.prerendered_week_figures = prerendered_week_figures or {}
        # Variable definitions
        
        self.gest_week_var = VarDef(
            id_="gestWeek", name="Gestational Week", min_value=0, max_value=40, default_value=15 # max value was 36
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
        """Convert the PregnancyExplorer to a Dash layout.
        
        Returns
        -------
        Dash layout
            A Dash layout containing the MRI explorer, animation explorer, and mesh explorer.
        """
        return [
            # Store prerendered figures for clientside switching (eliminates network traffic)
            dcc.Store(id="prerendered-mesh-figures", data=self.prerendered_week_figures),
            dbc.Row(
                [
                    dbc.Col(self.animation_explorer.to_dash(), width=2, style={"overflow": "auto", "padding": "20px"}),
                    dbc.Col(self.mri_explorer.to_dash(), width=5, style={"overflow": "auto", "padding": "20px"}),
                    dbc.Col(self.gest_week_mesh_explorer.to_dash(), width=5, style={"overflow": "auto", "padding": "20px"}),
                ],
                align="center",
            ),
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
        """ MriExplorer class for visualizing MRI slices.

        Parameters
        ----------
        gest_week_slider : Polpo Slider
            Polpo Slider object for selecting the gestational week (not yet a dash slider).
        gest_week_var : Polpo VarDef
            Variable definition for gestational week.
        mri_data : list of np.ndarray
            List of MRI data arrays for different gestational weeks.
        hormones_df : pd.DataFrame
            DataFrame containing hormone levels corresponding to gestational weeks.
        radio_button_init : str
            Initial radio button value for selecting the MRI view (default is "sagittal").
        id_prefix : str
            Prefix for the component IDs.
        """
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
        """Convert the MriExplorer to a Dash layout.

        Returns
        -------
        Dash layout
            A Dash layout containing the MRI explorer with controls for gestational week, view selection, and slice selection.
        """
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
        """AnimationExplorer class for visualizing pregnancy animation via video.

        Uses a pre-rendered video file with clientside frame seeking for instant
        frame switching without network requests.

        Parameters
        ----------
        assets_folder_path : str
            Path to the folder containing the pregnancy animation video.
        week_slider : Polpo Slider
            Polpo object. Precursor for the gestational week slider.
        """
        self.assets_folder_path = assets_folder_path
        self.week_slider = week_slider
        self.video_url = self._get_video_url()

    def _get_video_url(self):
        """Get the URL for the pregnancy animation video.
        
        Returns
        -------
        str
            URL for the pregnancy animation video.
        """
        return get_asset_url("pregnancy_animation.mp4")

    def to_dash(self):
        """Create the Dash layout with video and clientside frame seeking.
        
        The video is loaded once and frame seeking happens entirely in the browser,
        eliminating network requests when the gestational week slider changes.
        """
        # Video element - preloaded, muted, no controls
        video = html.Video(
            id="pregnancy-video",
            src=self.video_url,
            style={"width": "100%", "height": "auto", "maxWidth": "200px"},
            preload="auto",  # Preload entire video for instant seeking
            muted=True,  # Required for autoplay policies
            **{"data-testid": "pregnancy-video"}  # For testing
        )
        
        return dbc.Container([
            video,
            # Hidden output for the clientside callback (video currentTime is set via JS)
            html.Div(id="pregnancy-video-time-setter", style={"display": "none"}),
        ])



class SharedOutputModelsBasedExplorer(BaseComponentGroup):
    # Class-level set to track which callbacks have been registered
    _registered_callbacks = set()
    
    def __init__(
        self, models, inputs, outputs, shown_inputs=None, id_prefix="", postproc_pred=None, layout=None
    ):
        """SharedOutputModelsBasedExplorer class for managing multiple models with shared output.

        Parameters
        ----------
        models : list of Polpo models
            List of Polpo models to be used for predictions.
        inputs : Polpo ComponentGroup
            Polpo ComponentGroup containing input components.
        outputs : Polpo Graph
            Polpo Graph object for displaying the output.
        shown_inputs : Polpo ComponentGroup, optional
            Polpo ComponentGroup containing inputs to be shown in the layout (default is None).
        id_prefix : str, optional
            Prefix for the component IDs (default is an empty string).
        postproc_pred : Polpo PostTransformingEstimator, optional
            Post-processing model for predictions (default is None).
        layout : Polpo Layout, optional
            Layout for the explorer (default is MultiRowLayout).
        """
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
            # Create a unique key for this callback to prevent duplicate registration
            callback_key = (
                self.outputs.id_ if hasattr(self.outputs, 'id_') else str(self.outputs),
                type(model).__name__,
                tuple(str(inp) for inp in self.inputs.as_input())
            )
            
            # Skip if this callback has already been registered
            if callback_key in SharedOutputModelsBasedExplorer._registered_callbacks:
                continue
            
            try:
                create_view_model_update(
                    output_view=self.outputs,
                    input_view=self.inputs,
                    model=model,
                    postproc_pred=self.postproc_pred,
                )
                # Mark this callback as registered
                SharedOutputModelsBasedExplorer._registered_callbacks.add(callback_key)
            except Exception as e:
                import traceback
                print(f"ERROR: Failed to create callback for model {model}: {e}")
                traceback.print_exc()
                # Don't re-raise - allow page to load even if callback creation fails
                continue

        return out
    

class SingleInputOutputModelsBasedExplorer(BaseComponentGroup):
    def __init__(
        self, model, input, output, shown_input=None, id_prefix="", postproc_pred=None, layout=None
    ):
        """SingleInputOutputModelsBasedExplorer class for managing a single model with shared input and output.

        Parameters
        ----------
        model : Polpo model
            Polpo model to be used for predictions.
        input : Polpo ComponentGroup
            Polpo ComponentGroup containing input components.
        output : Polpo Graph
            Polpo Graph object for displaying the output.
        shown_input : Polpo ComponentGroup, optional
            Polpo ComponentGroup containing inputs to be shown in the layout (default is None).
        id_prefix : str, optional
            Prefix for the component IDs (default is an empty string).
        postproc_pred : Polpo PostTransformingEstimator, optional
            Post-processing model for predictions (default is None).
        layout : Polpo Layout, optional
            Layout for the explorer (default is TwoRowLayout).
        """
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
    


class SlicePlotter(GoPlotter):
    """OPTIMIZED SlicePlotter for fast MRI slice visualization."""
    
    def __init__(
        self, cmap="gray", title="Slice Visualization", x_label="X", y_label="Y", just_image=False
    ):
        self.cmap = cmap
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.just_image = just_image
        self._cached_layout = None
        self._cached_size = None

    def _get_layout(self, width, height):
        """Get cached layout or create new one."""
        size_key = (width, height)
        if self._cached_layout is not None and self._cached_size == size_key:
            return self._cached_layout
        
        if self.just_image:
            layout = go.Layout(
                width=width,
                height=height,
                xaxis=dict(
                    visible=False,
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                ),
                yaxis=dict(
                    visible=False,
                    showticklabels=False,
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    scaleanchor="x",
                ),
                margin=dict(l=0, r=0, t=0, b=0),
                uirevision="constant",  # Prevents reset on update
            )
        else:
            layout = go.Layout(
                title=self.title,
                width=width,
                height=height,
                xaxis=dict(title=self.x_label),
                yaxis=dict(title=self.y_label),
                uirevision="constant",
            )
        
        self._cached_layout = layout
        self._cached_size = size_key
        return layout

    def transform_data(self, data):
        # Use Heatmap with optimized settings
        return [go.Heatmap(
            z=data.T,
            colorscale=self.cmap,
            showscale=False,
            hoverongaps=False,
            hoverinfo='skip',  # Disable hover for speed
        )]

    def plot(self, data=None):
        if data is None:
            return go.Figure(layout=self._get_layout(300, 300))
        
        # Calculate size once
        width = int(len(data[:, 0]) * 1.5)
        height = int(len(data[0]) * 1.5)
        
        # Create figure with cached layout
        return go.Figure(
            data=self.transform_data(data),
            layout=self._get_layout(width, height)
        )
    

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
                        {"label": f"{label}     ", "value": value}
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

