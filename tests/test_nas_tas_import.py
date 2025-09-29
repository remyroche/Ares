"""Smoke tests for the ``src.utils.nas_tas`` package."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_from_src_utils_import_nas_tas_succeeds():
    """``from src.utils import nas_tas`` should import without raising."""
    utils_pkg = __import__("src.utils", fromlist=["nas_tas"])
    nas_tas = utils_pkg.nas_tas

    # Basic sanity check – the module should identify itself correctly.
    assert nas_tas.__name__ == "src.utils.nas_tas"
