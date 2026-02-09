from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_BASE = Path(__file__).parent


def load_step_module(filename: str):
    path = _BASE / filename
    spec = spec_from_file_location(f"tpsl_optimiser_{filename.replace('.', '_')}", path)
    module = module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module
