"""Creates a Dash app where week/hormone sliders predict brain shape."""

import os
import socket

import dash_bootstrap_components as dbc
import numpy as np
import polpo.preprocessing.dict as ppdict
import polpo.preprocessing.pd as ppd
from dash import Dash
from polpo.dash.callbacks import PageRegister
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
from polpo.dash.style import update_style
from polpo.dash.variables import VarDef
from polpo.models import DictMeshColorizer, MeshColorizer
from polpo.plot.mesh import MeshesPlotter, MeshPlotter, StaticMeshPlotter
from polpo.preprocessing import ListSqueeze
from polpo.preprocessing.learning import DictsToXY, NestedDictsToXY
from polpo.sklearn.compose import PostTransformingEstimator

import herbrain.pregnancy.page_content as page_content
from herbrain.pregnancy.pregnancy_explorer import PregnancyExplorer

from .data import (
    HormonesCsvLoader,
    MaternalRegisteredMeshesLoader,
    MultipleMaternalMeshesLoader,
    NibImage2Mesh,
    PilotMriImageLoader,
    TemplateImageLoader,
)
from .models import MeshPCR
from .page_content import pregnancy_page, menstrual_page, homepage


def my_app(cfg, data, gpt):
    data_type = data
    style = cfg.style
    update_style(style)

    # TODO: homogenize
    data_dir = os.environ.get("HERBRAIN_DATA_DIR", None)
    if data_dir is None:
        in_frank = socket.gethostname() == "frank"
        data_dir = "/home/data/" if in_frank else "~/.herbrain/data/"

    pregnancy_data_dir = os.path.join(data_dir, "pregnancy")
    maternal_data_dir = os.path.join(data_dir, "maternal")

    hormones_ordering = ["estro", "prog", "lh"]

    mri_data = PilotMriImageLoader(
        data_dir=pregnancy_data_dir, debug=cfg.server.debug
    )()
    hormones_df = HormonesCsvLoader(
        data_dir=pregnancy_data_dir  # TODO: update when other subjects
    )()

    hormones_for_pred = ppd.ColumnsToDict(hormones_ordering)(hormones_df)
    hormones_gest_week = ppd.ColumnToDict("gestWeek")(hormones_df)

    if data_type == "multiple":
        dicts_to_xy = NestedDictsToXY()
    else:
        dicts_to_xy = DictsToXY()

    template_image = TemplateImageLoader(data_dir=pregnancy_data_dir)()

    n_structs = 1
    affine_transform = np.array(
        [
            [1.0, 0.0, 0.0, 25.0],
            [0.0, 1.0, 0.0, 28.0],
            [0.0, 0.0, 1.0, 23.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    if data_type == "multiple":
        structs = [
            "BrStem",
            "L_Thal",
            "R_Thal",
            "L_Caud",
            "R_Caud",
            "L_Puta",
            "R_Puta",
            "L_Pall",
            "R_Pall",
            "L_Hipp",
            "R_Hipp",
            "L_Amyg",
            "R_Amyg",
            "L_Accu",
            "R_Accu",
        ]
        if cfg.server.debug:
            structs = structs[2:]

        n_structs = len(structs)
        registered_meshes = MultipleMaternalMeshesLoader(
            data_dir=maternal_data_dir, max_iterations=500
        )(structs)

    else:
        registered_meshes = MaternalRegisteredMeshesLoader(
            data_dir=maternal_data_dir, max_iterations=500
        )()

    n_pipes = n_structs if data_type == "multiple" else None
    week_mesh_model = MeshPCR(
        model=None, affine_transform=affine_transform, n_pipes=n_pipes
    )
    # TODO: scale input?
    hormones_mesh_model = MeshPCR(
        model=None, affine_transform=affine_transform, n_pipes=n_pipes
    )

    Colorizer = DictMeshColorizer if data_type == "multiple" else MeshColorizer

    week_colorizer = Colorizer(x_ref=np.asarray(0.5), delta_lim=np.asarray(15.0))
    week_mesh_model = PostTransformingEstimator(week_mesh_model, week_colorizer)

    hormones_colorizer = Colorizer(scaling_factor=50.0)
    hormones_mesh_model = PostTransformingEstimator(
        hormones_mesh_model, hormones_colorizer
    )

    X, y = dicts_to_xy([hormones_gest_week, registered_meshes])
    week_mesh_model.fit(X, y)

    X, y = dicts_to_xy([hormones_for_pred, registered_meshes])
    hormones_mesh_model.fit(X, y)

    pregnancy_explorer = PregnancyExplorer(
        mri_data, 
        hormones_df, 
        data_type, 
        template_image,
        n_structs,
        week_mesh_model,
        hormones_mesh_model,
        hormones_ordering,
    )

    sidebar_elems = [
        # home
        SidebarElem(
            active=True,
            tab_header=SidebarHeader(
                href="/", 
                text="Homepage", 
                image_url="homepage_logo.png",
                image_width=40,
            ),
            page=FunctionComponent(homepage),
        ),
        # mri explorer
        SidebarElem(
            active=True,
            tab_header=SidebarHeader(
                href="/page-1",
                text="Digital Twin: Pregnancy",
                image_url="pregnancy_logo.png",
                image_width=40,
            ),
            page=FunctionComponent(
                pregnancy_page,
                pregnancy_explorer=pregnancy_explorer,
                gpt=gpt,
            ),
        ),
        # mesh explorer
        SidebarElem(
            active=True,
            tab_header=SidebarHeader(
                href="/page-2",
                text="Digital Twin: Menstruation",
                image_url="menstrual_logo.png",
                image_width=40,
            ),
            page=FunctionComponent(
                menstrual_page
            ),
        ),
    ]

    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        assets_folder=cfg.app.assets_folder,
    )

    page_register = PageRegister()

    app.layout = page_content.app_layout(sidebar_elems, page_register)

    app.title = cfg.app.title

    server_cfg = cfg.server
    app.run(
        debug=server_cfg.debug,
        use_reloader=server_cfg.use_reloader,
        host=server_cfg.host,
        port=server_cfg.port,
    )
