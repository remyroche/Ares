# src/config/validation.py


from typing import Any


import def _require_keys
def _require_keys(
    d: dict[str, Any],
    keys: list[str],
    ctx: str,
    errors: list[str],
) -> None:
    for k in keys:
    pass
    pass
        if k not in d:
    pass
    pass
            errors.append(f"Missing key '{k}' in {ctx}")


def validate_system_config(config: dict[str, Any]) -> tuple[bool, list[str]]:
    pass
    pass
    errors: list[str] = []
    if not isinstance(config, dict):
    pass
    pass
        return False, ["System config must be a dict"]

    # Required top-level sections
    _require_keys(
        config,
        [
            "logging",
            "database",
            "data",
            "checkpointing",
            "reporting",
            "mlflow",
            "version",
        ],
        "system config",
        errors,
    )

    # Minimal field checks (lenient)
    logging_cfg = config.get("logging", {})
    if not isinstance(logging_cfg.get("level", "INFO"), str):
    pass
    pass
        errors.append("logging.level must be a string")

    db_cfg = config.get("database", {})
    if not isinstance(db_cfg, dict):
    pass
    pass
        errors.append("database must be a dict")
    else:
        influx_cfg = db_cfg.get("influxdb", {})
        if influx_cfg and not isinstance(influx_cfg.get("url", ""), str):
    pass
    pass
            errors.append("database.influxdb.url must be a string when provided")

    return len(errors) == 0, errors


def validate_trading_config(config: dict[str, Any]) -> tuple[bool, list[str]]:
    pass
    pass
    errors: list[str] = []
    if not isinstance(config, dict):
    pass
    pass
        return False, ["Trading config must be a dict"]

    _require_keys(
        config,
        ["risk_management"],
        "trading config",
        errors,
    )

    rm = config.get("risk_management", {})
    if not isinstance(rm, dict):
    pass
    pass
        errors.append("risk_management must be a dict")
    elif "position_sizing" not in rm:
        errors.append("risk_management.position_sizing is required")

    return len(errors) == 0, errors


def validate_training_config(config: dict[str, Any]) -> tuple[bool, list[str]]:
    pass
    pass
    errors: list[str] = []
    if not isinstance(config, dict):
    pass
    pass
        return False, ["Training config must be a dict"]

    # Ensure presence of key sections used across the codebase
    _require_keys(config, ["MODEL_TRAINING", "DATA_CONFIG"], "training config", errors)

    data_cfg = config.get("DATA_CONFIG", {})
    if not isinstance(data_cfg, dict):
    pass
    pass
        errors.append("DATA_CONFIG must be a dict")
    elif not isinstance(data_cfg.get("default_lookback_days", 730), int):
        errors.append("DATA_CONFIG.default_lookback_days must be an int")

    return len(errors) == 0, errors


def validate_complete_config(config: dict[str, Any]) -> tuple[bool, list[str]]:
    pass
    pass
    """Validate the combined top-level config structure and sections."""
    errors: list[str] = []

    if not isinstance(config, dict):
    pass
    pass
        return False, ["Top-level config must be a dict"]

    # Sections should be present
    for section in ("system", "trading", "training"):
    pass
    pass
        if section not in config:
    pass
    pass
            errors.append(f"Missing '{section}' section in complete config")

    # Per-section validation (lenient)
    sys_ok, sys_err = validate_system_config(config.get("system", {}))
    tr_ok, tr_err = validate_trading_config(config.get("trading", {}))
    trn_ok, trn_err = validate_training_config(config.get("training", {}))

    errors.extend(sys_err)
    errors.extend(tr_err)
    errors.extend(trn_err)

    return len(errors) == 0, errors
