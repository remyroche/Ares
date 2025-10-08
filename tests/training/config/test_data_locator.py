from pathlib import Path

import pytest

from src.training.config.data_locator import DataLocator, DataLocatorConfig


def test_data_locator_resolves_defaults(tmp_path: Path) -> None:
    locator = DataLocator(root=tmp_path)

    assert locator.base_data_dir == (tmp_path / "historical_data").resolve()
    assert locator.data_path("market_data") == (tmp_path / "historical_data").resolve()
    assert locator.generated_path("market_analysis") == (tmp_path / "generated" / "market_analysis").resolve()


def test_data_locator_honours_env_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_root = tmp_path / "custom_data"
    cache_root = tmp_path / "custom_cache"
    monkeypatch.setenv("ARES_DATA_DIR", str(data_root))
    monkeypatch.setenv("ARES_CACHE_DIR", str(cache_root))

    locator = DataLocator(root=tmp_path)

    assert locator.base_data_dir == data_root.resolve()
    assert locator.data_path("market_data") == data_root.resolve()
    assert locator.cache_path("default") == cache_root.resolve()


def test_data_locator_allows_custom_mapping(tmp_path: Path) -> None:
    config = DataLocatorConfig(
        data={"custom": "market"},
        generated={"reports": "reports"},
    )
    locator = DataLocator(config=config, root=tmp_path)

    assert locator.data_path("custom") == (tmp_path / "historical_data" / "market").resolve()
    assert locator.generated_path("reports") == (tmp_path / "generated" / "reports").resolve()

    target = locator.artifacts_path("multi_horizon_outcomes", ensure_exists=True)
    assert target.exists()
