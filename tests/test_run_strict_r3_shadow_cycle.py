from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "run_strict_r3_shadow_cycle.py"


def test_shadow_runner_has_no_exchange_or_order_capable_imports() -> None:
    tree = ast.parse(PATH.read_text())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    forbidden = ("ccxt", "kraken", "exchange", "order", "portfolio_manager")
    assert not any(token in name.lower() for name in imported for token in forbidden)


def test_shadow_runner_exposes_only_shadow_mode() -> None:
    source = PATH.read_text()
    assert 'choices=("shadow-only",)' in source
    assert '"order_submission_enabled": False' in source
    assert '"exchange_calls": 0' in source
    assert '"--portfolio-state-json"' in source
    assert '"--inference-bundle"' in source
    assert '"--portfolio-policy-json"' not in source
    assert '"--conversion-bundle-dir"' not in source
