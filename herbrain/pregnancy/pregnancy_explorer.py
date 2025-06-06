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
from polpo.style import STYLE as S
from dash import Dash, Input, Output, State, callback, dcc, html
import numpy as np
    


class PregnancyExplorer1:
    def __init__(
        self,
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
        self.estro = VarDef(
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
        self.template_mesh = NibImage2Mesh()(template_image)

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

    def _gest_week_mesh_explorer(self,):
        return MultiModelsMeshExplorer(
            graph=Graph(
                id_="mesh-plot",
                plotter=MeshesPlotter(
                    plotters=[MeshPlotter() for _ in range(self.n_structs)],
                    overlay_plotter=StaticMeshPlotter(
                        mesh=self.template_mesh, visible=self.template_visibility
                    ),
                    bounds=None,  # TODO: check need
                    overlay_bounds=None,  # TODO: check need
                ),
            ),
            models=(self.week_mesh_model, self.hormones_mesh_model),
            inputs=(
                Slider(self.gest_week),
                ComponentGroup(
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
    


class MriExplorer():
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
        slider,
        session_info,
        graph_row=None,
        id_prefix="",
    ):
        if graph_row is None:
            graph_row = MriGraphRow(index_ordering=list(range(len(sliders) - 1)))

        # TODO: used to train the model and to update the controller
        self.mri_data = mri_data
        self.hormones_df = hormones_df

        # NB: an input view
        self.slider = slider
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
        create_view_model_update(self.sliders, self.graph_row, self.mri_model)
        create_view_model_update(
            self.sliders[0], self.session_info, self.session_info_model
        )

    def to_dash(self):
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
    


class PregnancyExplorer():
    def __init__(
        self,
        app,
        mri_data,
        hormones_df,
        data_type,
        template_image,
        n_structs,
        week_mesh_model,
        hormones_mesh_model,
        hormones_ordering,
    ):
        self.app = app
        self.gest_week = VarDef(
            "gestWeek", name="Gestational Week", min_value=0, max_value=36, default_value=15
        )
        self.estro = VarDef(
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
        self.template_mesh = NibImage2Mesh()(template_image)

        self.gest_week_slider = Slider(self.gest_week)
        self.hormone_slider = ComponentGroup(
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

    
    # def create_pregnancy_callback(self, input, output, mesh_model, mri_model, pic_model):
    #     """Callback for updating the pregnancy explorer based on input changes.

    #     input is gestation week, output is MRI slice, and a mesh graph.
    #     """
    #     @self.app.callback(
    #         [
    #             Output("mesh-plot", "figure"),
    #             Output("gest_week_slider_container", component_property="style"),
    #             Output("hormone_slider_container", component_property="style"),
    #         ],
    #         Input("gest-week-slider", "drag_value"),
    #         Input("estrogen-slider", "drag_value"),
    #         Input("progesterone-slider", "drag_value"),
    #         Input("LH-slider", "drag_value"),
    #         State("mesh-plot", "figure"),
    #         State("mesh-plot", "relayoutData"),
    #         Input("button", "n_mesh_clicks"),
    #         Input("mri-button", "n_mri_clicks"),
    #     )
    #     def update(
    #         gest_week, hormones_df, estrogen, progesterone, LH, current_figure, relayoutData, n_mesh_clicks=0, n_mri_clicks=0
    #     ):
    #         """Update the mesh plot based on the hormone levels."""
    #         if (n_mesh_clicks % 2) == 0:
    #             gest_week_slider_style = {"display": "none"}
    #             hormone_week_slider_style = {"display": "block"}

    #         else:
    #             gest_week_slider_style = {"display": "block"}
    #             hormone_week_slider_style = {"display": "none"}

    #             print("hiding hormone sliders")

    #             progesterone = interpolate_or_return(
    #                 hormones_df, gest_week, x_label="gestWeek", y_label="prog"
    #             )
    #             estrogen = interpolate_or_return(
    #                 hormones_df, gest_week, x_label="gestWeek", y_label="estro"
    #             )
    #             LH = interpolate_or_return(
    #                 hormones_df, gest_week, x_label="gestWeek", y_label="lh"
    #             )
    #             print("progesterone", progesterone)
    #             print("estrogen", estrogen)
    #             print("LH", LH)
    #             print("gest_week", gest_week)

    #         # Cycle through sagittal, axial, and coronal plane views with each MRI button click
    #         plane_views = ["sagittal", "axial", "coronal"]
    #         if n_mri_clicks is not None:
    #             current_plane = plane_views[n_mri_clicks % len(plane_views)]
    #         else:
    #             current_plane = plane_views[0]
    #         print(f"Current MRI plane view: {current_plane}")

    #         X_multiple = np.array([[estrogen, progesterone, LH]])

    #         mesh_plot = mesh_model.predict(
    #             X_multiple,
    #             lr_hormones,
    #             pca_hormones,
    #             y_mean_hormones,
    #             n_vertices_hormones,
    #             mesh_neighbors_hormones,
    #             space,
    #             vertex_colors,
    #             current_figure=current_figure,
    #             relayoutData=relayoutData,
    #         )

    #         mri_view = mri_model.predict(
    #             gest_week
    #         )

    #         pic_view = pic_model.predict(
    #             gest_week
    #         )

    #         return mesh_plot, gest_week_slider_style, hormone_week_slider_style

    def create_mesh_explorer(self):
        return MultiModelsMeshExplorer(
            graph=Graph(
                id_="mesh-plot",
                plotter=MeshesPlotter(
                    plotters=[MeshPlotter() for _ in range(self.n_structs)],
                    overlay_plotter=StaticMeshPlotter(
                        mesh=self.template_mesh, visible=self.template_visibility
                    ),
                    bounds=None,  # TODO: check need
                    overlay_bounds=None,  # TODO: check need
                ),
            ),
            models=(self.week_mesh_model, self.hormones_mesh_model),
            inputs=(
                self.gest_week_slider,
                self.hormone_slider 
            ),
            checkbox_labels=((-1, "Show Full Brain", self.template_visibility),),
            button_label=" Click Here to Toggle Between Gestational Week vs Hormone Value Prediction",
            postproc_pred=self.postproc_pred,
        )

    def create_model(gest_week):
        # Placeholder for model creation logic
        # This would typically involve loading a pre-trained model or training a new one
        pass


    def create_graph_row():
        pass


    def create_callback(self):
        create_view_model_update(self.sliders, self.graph_row, self.pregnancy_model)
        
    def to_dash(self):

        self.create_callback()


            


