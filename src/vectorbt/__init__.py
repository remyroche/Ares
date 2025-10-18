"""
VectorBT import stub for environments without native VectorBT support.

By default this stub prevents importing the real ``vectorbt`` package, allowing
the rest of the codebase to detect the missing dependency and fall back to
safe pandas/numpy implementations without crashing the interpreter.

To enable the real VectorBT package (if it is installed and compatible with
the current platform) set the environment variable ``ARES_ENABLE_VECTORBT=1``
before launching Python. When enabled, this stub delegates the import to the
next ``vectorbt`` distribution found on ``sys.path`` (excluding the project
workspace) so the genuine package can be used.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import os
import sys

_ENV_FLAG = os.environ.get("ARES_ENABLE_VECTORBT", "").strip().lower()
_ALLOW_REAL_VECTORBT = _ENV_FLAG in {"1", "true", "yes", "on"}

# Determine the workspace root (the path entry that contains this stub).
_STUB_DIR = os.path.dirname(__file__)
_WORKSPACE_ROOT = os.path.abspath(os.path.join(_STUB_DIR, os.pardir))

if _ALLOW_REAL_VECTORBT:
    # Search for the real vectorbt package on the rest of sys.path.
    _search_paths = [path for path in sys.path if os.path.abspath(path) != _WORKSPACE_ROOT]

    spec = importlib.machinery.PathFinder.find_spec("vectorbt", _search_paths)
    if spec is None or spec.loader is None:
        raise ImportError(
            "ARES_ENABLE_VECTORBT is set, but the real 'vectorbt' package could not be located "
            "outside the project workspace. Install vectorbt in your environment or unset "
            "ARES_ENABLE_VECTORBT to use the stub fallback."
        )

    # Load the genuine vectorbt module and replace this stub in sys.modules.
    module = importlib.util.module_from_spec(spec)
    sys.modules[__name__] = module
    spec.loader.exec_module(module)
    globals().update(module.__dict__)
else:
    raise ImportError(
        "VectorBT is disabled in this environment. Set ARES_ENABLE_VECTORBT=1 to enable the real "
        "package if it is installed."
    )
