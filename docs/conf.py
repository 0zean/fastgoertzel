# SPDX-License-Identifier: GPL-3.0-or-later
"""Sphinx configuration.

Build with ``make docs`` (``sphinx-build -W docs docs/_build/html``). The API page is
generated from the docstrings of the compiled module. The signatures it shows are read
from ``src/fastgoertzel/_core.pyi`` rather than from pybind11's generated ones, so the
documentation, the type stubs, and the runtime module (which stubtest checks against each
other in CI) cannot drift apart.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import fastgoertzel

ROOT = Path(__file__).resolve().parent.parent

project = "fastgoertzel"
author = "Nicholas Picini"
copyright = "%Y, Nicholas Picini. GPL-3.0-or-later"
release = fastgoertzel.__version__
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Markdown sources (MyST).
myst_enable_extensions = ["colon_fence", "dollarmath"]
myst_heading_anchors = 3

# Docstrings are NumPy style.
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False

autodoc_docstring_signature = True  # strips pybind11's signature line from the docstring
autodoc_typehints = "none"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

html_theme = "furo"
html_title = f"fastgoertzel {release}"
html_static_path = ["_static"]
html_theme_options = {
    # The "dark" logo is dark text for light backgrounds, as in the README.
    "light_logo": "dark logo.png",
    "dark_logo": "light logo.png",
    "source_repository": "https://github.com/0zean/fastgoertzel/",
    "source_branch": "master",
    "source_directory": "docs/",
}


# ---------------------------------------------------------------------------
# Signatures from the type stub
# ---------------------------------------------------------------------------
def _stub_signatures() -> dict[str, tuple[str, str | None]]:
    """Map ``fastgoertzel.<qualname>`` to ``(args, return annotation)`` from ``_core.pyi``."""
    tree = ast.parse((ROOT / "src" / "fastgoertzel" / "_core.pyi").read_text("utf-8"))
    out: dict[str, tuple[str, str | None]] = {}

    def record(qualname: str, fn: ast.FunctionDef, *, method: bool) -> None:
        args = fn.args
        if method and args.args:
            args = ast.arguments(
                posonlyargs=args.posonlyargs,
                args=args.args[1:],  # drop self
                vararg=args.vararg,
                kwonlyargs=args.kwonlyargs,
                kw_defaults=args.kw_defaults,
                kwarg=args.kwarg,
                defaults=args.defaults,
            )
        text = ast.unparse(args).replace("_Scale", "str")
        ret = ast.unparse(fn.returns).replace("_Scale", "str") if fn.returns else None
        out[f"{project}.{qualname}"] = (f"({text})", ret)

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            record(node.name, node, method=False)
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if not isinstance(item, ast.FunctionDef):
                    continue
                if any(isinstance(d, ast.Name) and d.id == "property" for d in item.decorator_list):
                    continue
                if item.name == "__init__":
                    record(node.name, item, method=True)
                    out[f"{project}.{node.name}"] = (out[f"{project}.{node.name}"][0], None)
                else:
                    record(f"{node.name}.{item.name}", item, method=True)
    return out


SIGNATURES = _stub_signatures()


def _process_signature(
    app: Any,
    what: str,
    name: str,
    obj: Any,
    options: Any,
    signature: str | None,
    return_annotation: str | None,
) -> tuple[str, str | None] | None:
    # Sphinx 9 raises IndexError if this event supplies a signature where autodoc found
    # none, which is the case for the pybind11 class; api.md therefore gives the class an
    # explicit signature in the directive and this handler only replaces existing ones.
    if signature is None:
        return None
    return SIGNATURES.get(name)


def setup(app: Any) -> None:
    app.connect("autodoc-process-signature", _process_signature)
