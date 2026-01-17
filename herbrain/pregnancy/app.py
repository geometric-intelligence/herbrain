"""Creates a Dash app where week/hormone sliders predict brain shape."""

import os
import socket

import dash_bootstrap_components as dbc
import numpy as np
import polpo.preprocessing.dict as ppdict
import polpo.preprocessing.pd as ppd
from dash import Dash, Input, Output, State, clientside_callback
from flask_compress import Compress
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
from .models import MeshPCR, CachingMeshModel
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
    # Affine transformation to center the subcortical structures within the brain template
    # The ENIGMA meshes need to be translated to align with the template brain coordinates
    affine_transform = np.array(
        [
            [1.0, 0.0, 0.0, -23.0],  # translate x to center structures
            [0.0, 1.0, 0.0, -9.0],   # translate y
            [0.0, 0.0, 1.0, 27.0],   # translate z
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    if data_type == "multiple":
        # Note: BrStem is not supported in the enigma derivative, so we exclude it
        structs = [
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
            structs = structs[:2]

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

    # Pre-render all mesh figures for clientside switching (eliminates network traffic)
    print("Pre-rendering mesh figures for clientside switching...")
    
    # Create the template mesh for the brain overlay
    template_mesh = NibImage2Mesh()(template_image)
    
    # Set up postprocessing if needed
    postproc_pred = None
    if data_type == "multiple":
        postproc_pred = ppdict.DictMap(step=ListSqueeze()) + ppdict.DictToValuesList()
    
    # Create the mesh plotter (same configuration as MultiModelsMeshExplorer)
    mesh_plotter = MeshesPlotter(
        plotters=[MeshPlotter() for _ in range(n_structs)],
        overlay_plotter=StaticMeshPlotter(mesh=template_mesh, visible=True),
        bounds=None,
        overlay_bounds=None,
    )
    
    # Pre-render all figures for gestational weeks 0-45
    prerendered_week_figures = {}
    for week in range(0, 46):
        try:
            result = week_mesh_model.predict(np.array([[week]]))
            # Extract the prediction - handle both list/array and direct dict returns
            if isinstance(result, (list, np.ndarray)) and len(result) > 0:
                mesh_data = result[0]
            else:
                mesh_data = result
            # Apply postprocessing if needed
            if postproc_pred is not None:
                mesh_data = postproc_pred(mesh_data)
            fig = mesh_plotter.plot(mesh_data)
            prerendered_week_figures[str(week)] = fig.to_dict()  # Use string keys for JSON
        except Exception as e:
            import traceback
            print(f"Warning: Could not pre-render week {week}: {type(e).__name__}: {e}")
            traceback.print_exc()
    
    print(f"Pre-rendered {len(prerendered_week_figures)} mesh figures for gestational weeks")

    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
        assets_folder=cfg.app.assets_folder,
    )
    
    # Enable gzip compression for all responses (reduces ~7MB to ~500KB)
    Compress(app.server)

    pregnancy_explorer = PregnancyExplorer(
        cfg,
        mri_data, 
        hormones_df, 
        data_type, 
        template_image,
        n_structs,
        week_mesh_model,
        hormones_mesh_model,
        hormones_ordering,
        prerendered_week_figures=prerendered_week_figures,
    )

    sidebar_elems = [
        # home (inactive - not shown in nav, but route still registered)
        SidebarElem(
            active=False,  # Don't show in sidebar nav, but still register the route
            tab_header=SidebarHeader(
                href="/", 
                text="Homepage", 
                image_url="gi-logo.png",
                image_width=40,
            ),
            page=FunctionComponent(homepage),
        ),
        # mri explorer
        SidebarElem(
            active=True,
            tab_header=SidebarHeader(
                href="/page-1",
                text="Pregnancy",
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
                text="Menstruation",
                image_url="menstrual_logo.png",
                image_width=40,
            ),
            page=FunctionComponent(
                menstrual_page
            ),
        ),
    ]

    page_register = PageRegister()

    app.layout = page_content.app_layout(sidebar_elems, page_register)

    app.title = cfg.app.title

    # Clientside callback for instant mesh figure switching (zero network traffic)
    # This callback runs entirely in the browser using prerendered figures
    app.clientside_callback(
        """
        function(week, figures, currentFigure) {
            // Round to nearest integer week
            const weekInt = Math.round(week).toString();
            
            // Return prerendered figure if available
            if (figures && figures[weekInt]) {
                return figures[weekInt];
            }
            
            // Fall back to current figure if no prerendered version
            return window.dash_clientside.no_update;
        }
        """,
        Output("mesh-plot", "figure", allow_duplicate=True),
        Input("gestWeek-slider", "value"),
        State("prerendered-mesh-figures", "data"),
        State("mesh-plot", "figure"),
        prevent_initial_call=True,
    )

    # Clientside callback for instant pregnancy animation frame switching
    # The video has 10 frames (weeks 00, 05, 10, 15, 20, 25, 30, 35, 40, 41) at 1fps
    # Frame seeking happens entirely in the browser - no network requests
    app.clientside_callback(
        """
        function(week) {
            // Get the video element
            const video = document.getElementById('pregnancy-video');
            if (!video) {
                return window.dash_clientside.no_update;
            }
            
            // Map gestational week to video time (frame number at 1fps)
            // Frames: 0=week0-4, 1=week5-9, 2=week10-14, ..., 8=week40, 9=week41+
            let frameTime;
            if (week >= 41) {
                frameTime = 9;
            } else if (week >= 40) {
                frameTime = 8;
            } else {
                frameTime = Math.floor(week / 5);
            }
            
            // Seek to the frame (add small offset to ensure we're in the frame)
            video.currentTime = frameTime + 0.001;
            
            return '';  // Return empty string to satisfy the callback
        }
        """,
        Output("pregnancy-video-time-setter", "children"),
        Input("gestWeek-slider", "value"),
        prevent_initial_call=False,  # Run on initial load too
    )

    server_cfg = cfg.server
    app.run(
        debug=server_cfg.debug,
        use_reloader=server_cfg.use_reloader,
        host=server_cfg.host,
        port=8888, #8888 server_cfg.port
    )
