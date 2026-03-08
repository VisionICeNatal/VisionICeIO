"""Sphinx configuration for VisionICeIO."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # pip install tomli  (only needed for Python 3.10)

_pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
with open(_pyproject, "rb") as _f:
    _meta = tomllib.load(_f)

project = "VisionICeIO"
copyright = "2026, ICe Vision Lab"
author = _meta["project"]["authors"][0]["name"]
release = _meta["project"]["version"]
version = ""  # hide version from the navbar title
html_title = project  # override default "{project} v{release}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "sphinxcontrib.mermaid",
    "numpydoc",
    "myst_parser",
]

mermaid_d3_zoom = False
mermaid_init_js = (
    "mermaid.initialize({"
    "startOnLoad:true,"
    "flowchart:{useMaxWidth:true}"
    "});"
)
html_css_files = ["custom.css"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "**.ipynb_checkpoints"]
autosummary_generate = True
numpydoc_show_class_members = False

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/VisionICeNatal/VisionICeIO",
    "navbar_align": "left",
    "show_toc_level": 2,
}
html_static_path = ["_static"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}
