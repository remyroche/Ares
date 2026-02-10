from __future__ import annotations

import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_BASE = Path(__file__).parent


def load_step_module(filename: str):
    path = _BASE / filename
    mod_name = f"tpsl_optimiser_{filename.replace('.', '_')}"
    spec = spec_from_file_location(mod_name, path)
    module = module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[mod_name] = module  # register so @dataclass can resolve __module__
    spec.loader.exec_module(module)
    return module
