from polpo.dash.components import (
    ComponentGroup,
    DepVar,
    FunctionComponent,
    Graph,
    MriExplorer,
    MriSliders,
    MultiModelsMeshExplorer,
    MeshExplorer,
    SidebarElem,
    SidebarHeader,
    Slider,
    MriGraphRow,
    BaseComponentGroup
)
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
from .data import (
    # HormonesCsvLoader,
    # MaternalRegisteredMeshesLoader,
    # MultipleMaternalMeshesLoader,
    NibImage2Mesh,
    # PilotMriImageLoader,
    # TemplateImageLoader,
)
from polpo.dash.style import STYLE as S
from dash import Dash, Input, Output, State, callback, dcc, html
import numpy as np
import plotly.graph_objs as go

    


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
        
        self.gest_week = VarDef(
            "gestWeek", name="Gestational Week", min_value=0, max_value=36, default_value=15
        )
        self.estro = VarDef( # makes it easier to have all this info contained in a var, rather than having to type these things every time they are used
            "estro",
            name="Estrogen",
            unit="pg/ml",
            min_value=4100,
            max_value=12400,
        )
        self.prog = VarDef(
            "prog",
            name="Progesterone",
            unit="ng/ml",
            min_value=54,
            max_value=103,
        )
        self.lh = VarDef(
            "lh",
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
        self.hormones_df = hormones_df

        self.mri_explorer = self._mri_explorer()
        self.gest_week_mesh_explorer = self._gest_week_mesh_explorer()
        # self.hormones_mesh_explorer = self._hormones_mesh_explorer()

        self.animation_explorer = AnimationExplorer(cfg.app.assets_folder)
    
    def _mri_explorer(self):
        self.session_id = VarDef("sessionID", name="Session Number", min_value=1, max_value=26)
        self.mri_vars = [self.session_id] + [
            VarDef(id_, name=name)
            for id_, name in [
                ("mri_x", "X Coordinate (Changes Side View)"),
                ("mri_y", "Y Coordinate (Changes Front View)"),
                ("mri_z", "Z Coordinate (Changes Top View)"),
            ]
        ]
        self.endo_status = VarDef("EndoStatus", name="Pregnancy status")
        self.trimester = VarDef("trimester", name="trimester")

        self.mri_steps = [1] + [5] * 3
        self.mri_sliders = MriSliders(
            [Slider(var, step) for var, step in zip(self.mri_vars, self.mri_steps)],
            trims=((20, 40), 50, 70),
        )

        session_info = ComponentGroup(
            components=[
                DepVar(var)
                for var in (
                    self.session_id,
                    self.gest_week,
                    self.estro,
                    self.lh,
                    self.endo_status,
                    self.trimester,
                )
            ],
            title="Session information",
        )

        return MriExplorer(
            self.mri_data, self.hormones_df, self.mri_sliders, session_info, id_prefix="mri-"
        )

    def _gest_week_mesh_explorer(self):
        return MultiModelsMeshExplorer(
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
                Slider(self.gest_week), # not yet a dash slider. will turn into a dash slider when we call .to_dash(). it is just an array of containers or something
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
        )

        # return MultiModelsMeshExplorer(
        #     graph=Graph(
        #         id_="mesh-plot",
        #         plotter=MeshesPlotter(
        #             plotters=[MeshPlotter() for _ in range(self.n_structs)],
        #             overlay_plotter=StaticMeshPlotter(
        #                 mesh=self.template_mesh, visible=self.template_visibility
        #             ),
        #             bounds=None,  # TODO: check need
        #             overlay_bounds=None,  # TODO: check need
        #         ),
        #     ),
        #     models=[self.week_mesh_model],
        #     inputs=(
        #         Slider(self.gest_week),
        #     ),
        #     checkbox_labels=((-1, "Show Full Brain", self.template_visibility),),
        #     postproc_pred=self.postproc_pred,
        # )
    

    # def _hormones_mesh_explorer(self):
    #     return MultiModelsMeshExplorer(
    #         graph=Graph(
    #             id_="mesh-plot",
    #             plotter=MeshesPlotter(
    #                 plotters=[MeshPlotter() for _ in range(self.n_structs)],
    #                 overlay_plotter=StaticMeshPlotter(
    #                     mesh=self.template_mesh, visible=self.template_visibility
    #                 ),
    #                 bounds=None,  # TODO: check need
    #                 overlay_bounds=None,  # TODO: check need
    #             ),
    #         ),
    #         models=[self.hormones_mesh_model],
    #         inputs=(
    #             ComponentGroup(
    #                 ordering=self.hormones_ordering,
    #                 components=[
    #                     Slider(var, step, label_style=self.hormone_label_style)
    #                     for var, step in [
    #                         (self.estro, 500),
    #                         (self.prog, 3),
    #                         (self.lh, 0.05),
    #                     ]
    #                 ],
    #             ),
    #         ),
    #         checkbox_labels=((-1, "Show Full Brain", self.template_visibility),),
    #         postproc_pred=self.postproc_pred,
    #     )


    def to_dash(self):
        return [
            dbc.Row(
                [
                    dbc.Col(self.animation_explorer.to_dash(), width=6),
                    dbc.Col(self.mri_explorer.to_dash(), width=6),
                    dbc.Col(self.gest_week_mesh_explorer.to_dash(), width=6),
                ],
                align="center",
            ),
            # dbc.Row(
            #     [
            #         dbc.Col(self.hormones_mesh_explorer.to_dash(), width=6),
            #     ],
            #     align="center",
            # ),
        ]
    

class MriExplorer(BaseComponentGroup): # different from one in polpo because it will intake gest week slider and output one mri slice.
    # data
    # plots
    # sliders
    # session info card

    # also instructions?

    # TODO: check if multiple callbacks can be defined
    def __init__(
        self,
        mri_data,
        hormones_df,
        sliders,
        session_info,
        graph_row=None,
        id_prefix="",
    ):
        if graph_row is None:
            graph_row = MriGraphRow(index_ordering=list(range(len(sliders) - 1))) # specifically designed assuming we get 3 mris

        # TODO: used to train the model and to update the controller
        self.mri_data = mri_data
        self.hormones_df = hormones_df

        # NB: an input view
        self.sliders = sliders
        # NB: an output view of the brain data
        self.graph_row = graph_row
        # NB: a model of the brain data
        self.mri_model = MriSlicesLookup(self.mri_data)

        # NB: a view of the hormones data
        self.session_info = session_info
        # NB: a model of the hormones data
        self.session_info_model = PdDfLookup(
            df=hormones_df,
            output_keys=[elem.var_def.id for elem in session_info],
            tar=1,
        )

        super().__init__([sliders, graph_row, session_info], id_prefix)

    def _create_callbacks(self):
        create_view_model_update(self.sliders, self.graph_row, self.mri_model) # input can be a slider object (polpo) or a componentgroup (aka, something the user changes)
        create_view_model_update(
            self.sliders[0], self.session_info, self.session_info_model
        )

    def to_dash(self): # this is where you create the layout of the page
        if hasattr(self.sliders, "update_lims"):
            self.sliders.update_lims(self.mri_data)

        plots_card = self.graph_row.to_dash()
        plots = dbc.Row(
            [
                dbc.Col(plots_card, sm=14),
            ],
            align="center",
            style={
                "marginLeft": S.margin_side,
                "marginRight": S.margin_side,
                "marginTop": "50px",
            },
        )

        sliders_card = dbc.Card(
            [
                dbc.Stack(
                    self.sliders.to_dash(),
                    gap=3,
                )
            ],
            body=True,
        )
        sliders_column = [
            dbc.Row(sliders_card),
        ]

        session_info = self.session_info.to_dash()
        sess_info_card = dbc.Card(
            [
                dbc.Stack(
                    session_info,
                    gap=0,
                )
            ],
            body=True,
        )

        sliders_and_session = dbc.Row(
            [
                dbc.Col(sliders_column, sm=7, width=700),
                dbc.Col(sess_info_card, sm=4, width=700),
            ],
            align="center",
            style={
                "marginLeft": S.margin_side,
                "marginRight": S.margin_side,
                "marginTop": "50px",
            },
        )

        self._create_callbacks()

        return [plots, sliders_and_session]


from polpo.dash.components import Image, SharedInputModelsBasedExplorer, Slider
from polpo.dash.style import update_style
from polpo.dash.variables import VarDef
from polpo.models import ListLookup   
import os
import sys
from polpo.preprocessing import Sorter
from polpo.preprocessing.path import FileFinder
from dash import Dash, get_asset_url

class AnimationExplorer():
    def __init__(self, assets_folder_path, weeks):
        self.assets_folder_path = assets_folder_path
        self.image_paths = self._load_pregnancy_images(assets_folder_path)
        self.weeks = weeks # should be a VarDef

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
        short_assets_folder = "/".join(assets_folder_path.split("/")[1:])

        assets_folder_abs = os.path.join(file_path, short_assets_folder)

        images = (
            FileFinder(data_dir=os.path.join(assets_folder_abs, "pregnancy_frames")) + Sorter()
        )()

        n_path_assets = len(assets_folder_abs)
        return [get_asset_url(image[n_path_assets + 1 :]) for image in images]


    def _create_layout(self):
        # TODO: do version with DictLookup
        models = [ListLookup(self.images)] #here, input will be weeks, and output needs to be an image.

        inputs = Slider(self.weeks)

        image_style = {"width": "50%"}
        outputs = [
            Image(id_=f"week_{index:02}", style=image_style)
            for index in range(len(models))
        ]

        image_seq_explorer = SharedInputModelsBasedExplorer(models, inputs, outputs)
        return dbc.Container(image_seq_explorer.to_dash())
