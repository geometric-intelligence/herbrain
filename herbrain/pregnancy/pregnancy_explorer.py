from polpo.dash.components import (
    ComponentGroup,
    DepVar,
    FunctionComponent,
    Graph,
    MriExplorer,
    MriSliders,
    MultiModelsMeshExplorer,
    SidebarElem,
    SidebarHeader,
    Slider,
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
    


class PregnancyExplorer:
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
        self.session_id = VarDef("sessionID", name="Session Number", min_value=1, max_value=26)
        self.mri_vars = [self.session_id] + [
            VarDef(id_, name=name)
            for id_, name in [
                ("mri_x", "X Coordinate (Changes Side View)"),
                ("mri_y", "Y Coordinate (Changes Front View)"),
                ("mri_z", "Z Coordinate (Changes Top View)"),
            ]
        ]
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
        self.endo_status = VarDef("EndoStatus", name="Pregnancy status")
        self.trimester = VarDef("trimester", name="trimester")

        self.mri_steps = [1] + [5] * 3
        self.mri_sliders = MriSliders(
            [Slider(var, step) for var, step in zip(self.mri_vars, self.mri_steps)],
            trims=((20, 40), 50, 70),
        )

        self.session_info = ComponentGroup(
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
        self.mri_explorer = MriExplorer(
            mri_data, hormones_df, self.mri_sliders, self.session_info, id_prefix="mri-"
        )

        self.template_mesh = NibImage2Mesh()(template_image)

        self.postproc_pred = None
        if data_type == "multiple":
            self.postproc_pred = ppdict.DictMap(step=ListSqueeze()) + ppdict.DictToValuesList()

        hormone_label_style = {"fontSize": 30, "display": "block"}
        template_visibility = True
        self.mesh_explorer = MultiModelsMeshExplorer(
            graph=Graph(
                id_="mesh-plot",
                plotter=MeshesPlotter(
                    plotters=[MeshPlotter() for _ in range(n_structs)],
                    overlay_plotter=StaticMeshPlotter(
                        mesh=self.template_mesh, visible=template_visibility
                    ),
                    bounds=None,  # TODO: check need
                    overlay_bounds=None,  # TODO: check need
                ),
            ),
            models=(week_mesh_model, hormones_mesh_model),
            inputs=(
                Slider(self.gest_week),
                ComponentGroup(
                    ordering=hormones_ordering,
                    components=[
                        Slider(var, step, label_style=hormone_label_style)
                        for var, step in [
                            (self.estro, 500),
                            (self.prog, 3),
                            (self.lh, 0.05),
                        ]
                    ],
                ),
            ),
            checkbox_labels=((-1, "Show Full Brain", template_visibility),),
            button_label=" Click Here to Toggle Between Gestational Week vs Hormone Value Prediction",
            postproc_pred=self.postproc_pred,
        )

    def to_dash(self):
        return [
            dbc.Row(
                [
                    dbc.Col(self.mri_explorer, width=6),
                    dbc.Col(self.mesh_explorer, width=6),
                ],
                align="center",
            )
        ]