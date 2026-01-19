"""Creates a Dash app where week/hormone sliders predict brain shape."""

import json
import os
import socket

import dash_bootstrap_components as dbc
import numpy as np
import polpo.preprocessing.dict as ppdict
import polpo.preprocessing.pd as ppd
from dash import Dash, Input, Output, State
from flask_compress import Compress
from polpo.dash.callbacks import PageRegister
from polpo.dash.components import FunctionComponent, SidebarElem, SidebarHeader
from polpo.dash.style import update_style
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
from .page_content import homepage, menstrual_page, pregnancy_page


# Weeks to prerender - every 5 weeks for efficient coverage
PRERENDER_WEEKS = [0, 5, 10, 15, 20, 25, 30, 35, 40]


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

    # Prerender mesh figures for fast clientside switching
    # Only render key weeks (every 5 weeks) to reduce file size
    # Figures are saved to a static JSON file that loads asynchronously
    # Use absolute path based on this module's location
    module_dir = os.path.dirname(os.path.abspath(__file__))
    assets_folder = os.path.join(module_dir, "assets")
    prerendered_figures_path = os.path.join(assets_folder, "prerendered_meshes.json")
    
    # Check if prerendered figures already exist and are up-to-date
    should_prerender = not os.path.exists(prerendered_figures_path)
    
    if should_prerender:
        print("Pre-rendering mesh figures for clientside switching...")
        
        # Set up postprocessing if needed
        postproc_pred = None
        if data_type == "multiple":
            postproc_pred = ppdict.DictMap(step=ListSqueeze()) + ppdict.DictToValuesList()
        
        # Create mesh plotter WITHOUT overlay (overlay is static and added separately)
        # This significantly reduces file size since overlay isn't duplicated for each week
        mesh_plotter = MeshesPlotter(
            plotters=[MeshPlotter() for _ in range(n_structs)],
            overlay_plotter=None,  # No overlay in prerendered figures
            bounds=None,
            overlay_bounds=None,
        )
        
        # Pre-render figures for key gestational weeks only
        prerendered_figures = {}
        for week in PRERENDER_WEEKS:
            try:
                result = week_mesh_model.predict(np.array([[week]]))
                if isinstance(result, (list, np.ndarray)) and len(result) > 0:
                    mesh_data = result[0]
                else:
                    mesh_data = result
                if postproc_pred is not None:
                    mesh_data = postproc_pred(mesh_data)
                fig = mesh_plotter.plot(mesh_data)
                
                # Optimize figure data: remove unnecessary fields to reduce size
                fig_dict = fig.to_dict()
                for trace in fig_dict.get('data', []):
                    # Remove colorbar (not needed for quick switching)
                    if 'colorbar' in trace:
                        del trace['colorbar']
                    # Remove hover info (not needed for quick preview)
                    if 'hoverinfo' not in trace:
                        trace['hoverinfo'] = 'skip'
                
                prerendered_figures[str(week)] = fig_dict
            except Exception as e:
                print(f"Warning: Could not pre-render week {week}: {e}")
        
        # Save to static JSON file (gzip compression happens via flask-compress)
        with open(prerendered_figures_path, 'w') as f:
            json.dump(prerendered_figures, f, separators=(',', ':'))  # Compact JSON
        
        file_size_mb = os.path.getsize(prerendered_figures_path) / (1024 * 1024)
        print(f"Pre-rendered {len(prerendered_figures)} figures, saved to {prerendered_figures_path} ({file_size_mb:.1f} MB)")
    else:
        print(f"Using cached prerendered figures from {prerendered_figures_path}")

    # Google Fonts for homepage typography
    google_fonts = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Playfair+Display:wght@600;700&display=swap"
    
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP, google_fonts],
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

    # Clientside callback for instant mesh figure switching using prerendered figures
    # The figures are loaded asynchronously from a static JSON file (cached by browser)
    # Maps any week to the nearest prerendered week for instant switching
    # Prerendered figures don't include the brain overlay - it's merged from current figure
    app.clientside_callback(
        """
        function(week, currentFigure) {
            // Available prerendered weeks
            const prerenderWeeks = [0, 5, 10, 15, 20, 25, 30, 35, 40];
            
            // Find nearest prerendered week
            let nearestWeek = prerenderWeeks[0];
            let minDiff = Math.abs(week - nearestWeek);
            for (let w of prerenderWeeks) {
                const diff = Math.abs(week - w);
                if (diff < minDiff) {
                    minDiff = diff;
                    nearestWeek = w;
                }
            }
            const weekKey = nearestWeek.toString();
            
            // Check if figures are already cached in window
            if (window._prerenderedMeshFigures && window._prerenderedMeshFigures[weekKey]) {
                const prerenderedFig = window._prerenderedMeshFigures[weekKey];
                
                // Merge with current figure to preserve overlay brain and layout
                if (currentFigure && currentFigure.data) {
                    // Find the overlay trace (last trace, usually the brain template)
                    const overlayTrace = currentFigure.data.find(t => t.name === 'overlay' || t.opacity < 1);
                    
                    // Create merged figure with prerendered data + overlay from current
                    const mergedData = [...prerenderedFig.data];
                    if (overlayTrace) {
                        mergedData.push(overlayTrace);
                    }
                    
                    return {
                        data: mergedData,
                        layout: currentFigure.layout || prerenderedFig.layout
                    };
                }
                
                return prerenderedFig;
            }
            
            // Load figures asynchronously if not already loading
            if (!window._loadingMeshFigures) {
                window._loadingMeshFigures = true;
                fetch('/assets/prerendered_meshes.json')
                    .then(response => response.json())
                    .then(data => {
                        window._prerenderedMeshFigures = data;
                        console.log('Prerendered mesh figures loaded (' + Object.keys(data).length + ' weeks)');
                    })
                    .catch(err => console.error('Failed to load prerendered figures:', err));
            }
            
            // Return no_update while loading - server callback will handle initial render
            return window.dash_clientside.no_update;
        }
        """,
        Output("mesh-plot", "figure", allow_duplicate=True),
        Input("gestWeek-slider", "value"),
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

    # Clientside callback for sidebar active state highlighting
    app.clientside_callback(
        """
        function(pathname) {
            // Update sidebar nav items based on current pathname
            const pregnancyNav = document.getElementById('nav-pregnancy');
            const menstruationNav = document.getElementById('nav-menstruation');
            
            if (pregnancyNav) {
                if (pathname === '/page-1') {
                    pregnancyNav.className = 'sidebar-nav-item active-pregnancy';
                } else {
                    pregnancyNav.className = 'sidebar-nav-item';
                }
            }
            
            if (menstruationNav) {
                if (pathname === '/page-2') {
                    menstruationNav.className = 'sidebar-nav-item active-menstruation';
                } else {
                    menstruationNav.className = 'sidebar-nav-item';
                }
            }
            
            return window.dash_clientside.no_update;
        }
        """,
        Output("page-content", "className"),
        Input("url", "pathname"),
        prevent_initial_call=False,
    )

    server_cfg = cfg.server
    app.run(
        debug=server_cfg.debug,
        use_reloader=server_cfg.use_reloader,
        host=server_cfg.host,
        port=8888, #8888 server_cfg.port
    )
