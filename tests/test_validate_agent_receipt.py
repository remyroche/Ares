import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "validate_agent_receipt.py"
SPEC = importlib.util.spec_from_file_location("validate_agent_receipt", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validator)


def _receipt(*, model="luna"):
    return {
        "task": "receipt-test",
        "scope": "Validate the receipt schema.",
        "changed_paths": ["extreme_price_movements/config.py"],
        "contracts": [
            {
                "path": "agents/dataset_contract.md",
                "sha256": validator.sha256(validator.ROOT / "agents/dataset_contract.md"),
            }
        ],
        "validation": {
            "status": "passed",
            "commands": ["pytest -q"],
            "not_run_reason": "",
        },
        "agent_plan": {"subagents": [{"model": model}]},
    }


def test_governed_paths_exclude_tests_and_validator():
    assert validator.is_governed(Path("extreme_price_movements/config.py"))
    assert validator.is_governed(Path("scripts/run_experiment.py"))
    assert not validator.is_governed(Path("tests/test_config.py"))
    assert not validator.is_governed(Path("scripts/validate_agent_receipt.py"))


def test_luna_receipt_validates_for_standard_governed_change():
    assert validator.validate_receipt(
        validator.ROOT / "agents/receipts/example.json",
        _receipt(),
        {"extreme_price_movements/config.py"},
    ) == []


def test_terra_requires_documented_exception():
    errors = validator.validate_receipt(
        validator.ROOT / "agents/receipts/example.json",
        _receipt(model="gpt-5.6-terra"),
        {"extreme_price_movements/config.py"},
    )
    assert any("Terra exception" in error for error in errors)
