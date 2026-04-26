from __future__ import annotations

import sys

from . import ebm_on_lgbm as _impl

sys.modules[__name__] = _impl
