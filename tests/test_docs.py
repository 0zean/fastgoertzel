"""The examples in the documentation run and say what they claim.

Two sources are checked: the ``>>>`` examples in the docstrings of the compiled module, which
the API reference renders, and the ``python`` code blocks of ``README.md``.
"""

import doctest
import re
from pathlib import Path

import pytest

import fastgoertzel._core as core

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"


def test_docstring_examples():
    result = doctest.testmod(core, optionflags=doctest.NORMALIZE_WHITESPACE, verbose=False)
    assert result.attempted >= 5, "doctest did not find the docstring examples"
    assert result.failed == 0


def _readme_blocks() -> list[str]:
    if not README.is_file():  # tests run from an installed wheel without the repository
        return []
    return re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.S)


@pytest.mark.parametrize("block", _readme_blocks(), ids=lambda b: b.splitlines()[0][:40])
def test_readme_python_blocks_run(block, capsys):
    exec(compile(block, str(README), "exec"), {"__name__": "__readme__"})
    capsys.readouterr()
