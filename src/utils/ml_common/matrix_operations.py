from __future__ import annotations

"""Matrix Operations utilities (moved from Step 07).

This module exposes the enhanced, GPU-aware matrix / linear-algebra helpers that
were originally packaged as *step07_enhanced_matrix_operations* in the training
pipeline.  They have been promoted to *ml_common* so that they can be reused by
any component without pulling in the full training-step infrastructure.

We keep the public API identical to the former ``src.utils.enhanced_matrix_operations``
module by re-exporting the same symbols.  Downstream code should now import from
``src.utils.ml_common.matrix_operations``.

>>> from src.utils.ml_common.matrix_operations import get_enhanced_matrix_operations
>>> ops = get_enhanced_matrix_operations()
>>> result = ops.matrix_multiply(A, B)

When the old module is imported a *DeprecationWarning* will be issued pointing
here, so gradual migration is possible.
"""

from warnings import warn

# Re-export everything from the original helper to avoid code duplication.
# NOTE: The original file lives at ``src/utils/enhanced_matrix_operations.py``.

try:
    from ..enhanced_matrix_operations import *  # type: ignore  # noqa: F401,F403
except Exception as exc:  # pragma: no cover – must never fail silently
    raise ImportError(
        "Unable to import 'enhanced_matrix_operations' – ensure utilities package is intact"
    ) from exc

warn(
    "`src.utils.ml_common.matrix_operations` is the new canonical import path for the"
    " enhanced matrix-operation helpers (formerly Step07).  Please update your imports.",
    category=DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    # pull symbols defined in the re-export above
    *globals().keys()  # type: ignore[arg-type]
]