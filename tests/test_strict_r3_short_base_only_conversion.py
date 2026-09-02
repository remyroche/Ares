"""Guard the explicit short conversion fallback after a failed head gate."""

from __future__ import annotations

import pytest

from scripts.run_strict_r3_short_p0_same_model_conversion_oof import (
    _selected_columns,
)


def test_base_only_contract_requires_an_explicit_opt_in() -> None:
    contract = {"heads": []}
    assert _selected_columns(None, contract, allow_base_only=True) == ()
    with pytest.raises(ValueError, match="selector OOF manifest"):
        _selected_columns(None, contract, allow_base_only=False)
