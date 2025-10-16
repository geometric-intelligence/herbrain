"""Sphinx configuration file."""

from polpo.nbconvert.exporters import Exporter  # noqa: F401

import herbrain

project = "herbrain"
copyright = "2024-, Geometric Intelligence Lab @ UC Santa Barbara"
author = "Geometric Intelligence Lab @ UC Santa Barbara"
release = version = herbrain.__version__

extensions = [
    "nbsphinx",
    "nbsphinx_link",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "polpo.sphinx.ext.nbsymlink",
    "polpo.sphinx.ext.nbtaggallery",
]

nbsymlink_notebooks_dir = "../notebooks"
nbsymlink_renamings = {}

nbtaggallery_tags = ["volume", "lddmm"]
nbtaggallery_tag_captions = {"lddmm": "LDDMM"}


autosummary_imported_members = True
autosummary_generate_overwrite = False
autosummary_mock_imports = ["sklearn", "tqdm", "joblib"]


# Configure napoleon for numpy docstring
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_ivar = False
napoleon_use_rtype = False
napoleon_include_init_with_doc = False

# intersphinx options
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sklearn": ("https://scikit-learn.org/", None),
}

# Configure nbsphinx for notebooks execution
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]
nbsphinx_execute = "never"
nbsphinx_allow_errors = True

templates_path = ["_templates"]

source_suffix = [".rst"]

main_doc = "index"

language = "en"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

pygments_style = None

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "show_prev_next": False,
    "search_bar_text": "Search the docs ...",
    "navigation_with_keys": False,
    "collapse_navigation": False,
    "show_nav_level": 1,
    "show_toc_level": 2,
    "navigation_depth": 2,
    "navbar_align": "left",
    "header_links_before_dropdown": 5,
    "header_dropdown_text": "More",
    "logo": {"text": "HerBrain @ GI lab"},
}
html_copy_source = html_show_sourcelink = False

html_baseurl = "https://geometric-intelligence.github.io/herbrain"
