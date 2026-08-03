from scripts import run_mayjun2025_canonical_base_continuation as base
from scripts import run_mayjun2025_canonical_residual_continuation as residual


def test_base_continuation_is_separate_and_has_fixed_month_scope() -> None:
    assert base.SCHEMA == "mayjun2025_canonical_base_continuation_v1"
    assert "continuation" in str(base.DEFAULT_OUTPUT)
    assert tuple(base.SIDES) == ("long", "short")


def test_residual_continuation_uses_separate_contract() -> None:
    assert residual.SCHEMA == "mayjun2025_canonical_residual_continuation_v1"
    assert residual.BASE.name == "mayjun2025_canonical_base_continuation_20260730_v1"
