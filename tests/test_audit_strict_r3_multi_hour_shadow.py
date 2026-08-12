from __future__ import annotations

import ast
from pathlib import Path

import pytest

from scripts.audit_strict_r3_multi_hour_shadow import _utc_hour


ROOT = Path(__file__).resolve().parents[1]


def test_multi_hour_auditor_has_no_exchange_or_order_imports() -> None:
    source = (ROOT / "scripts" / "audit_strict_r3_multi_hour_shadow.py").read_text()
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("ccxt" in name or "kraken" in name for name in imports)
    assert "create_order" not in source
    assert "submit_order" not in source


def test_multi_hour_boundaries_require_exact_hour() -> None:
    assert _utc_hour("2026-08-12T09:00:00Z").hour == 9
    with pytest.raises(ValueError, match="exact UTC hours"):
        _utc_hour("2026-08-12T09:00:01Z")


def test_feature_parity_gate_precedes_model_scoring() -> None:
    source = (ROOT / "scripts" / "audit_strict_r3_multi_hour_shadow.py").read_text()
    assert source.index("feature_parity_audit =") < source.index(
        'str(ROOT / "scripts" / "score_strict_r3_forward.py")',
    )
