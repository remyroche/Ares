from pathlib import Path

from src.training.config.data_locator import DataLocator, DataLocatorConfig
from src.training.steps.market_analysis.sub_pipeline import MarketAnalysisSubPipeline, SubPipelineConfig


def test_market_analysis_config_exposes_locator_paths(tmp_path: Path) -> None:
    config = SubPipelineConfig(
        data_locator_config=DataLocatorConfig(
            base_data_dir=str(tmp_path / "data"),
            base_cache_dir=str(tmp_path / "cache"),
            base_artifacts_dir=str(tmp_path / "artifacts"),
            base_generated_dir=str(tmp_path / "generated"),
            base_config_dir=str(tmp_path / "config"),
        )
    )

    locator = DataLocator(config.data_locator_config)
    config.attach_locator(locator)

    assert config.data.root == locator.base_data_dir
    assert config.cache.root == locator.base_cache_dir
    assert config.artifacts.root == locator.base_artifacts_dir
    assert config.generated.market_analysis == locator.generated_path("market_analysis")
    assert config.config.multi_horizon_labeling == locator.config_path("multi_horizon_labeling")


def test_market_analysis_pipeline_uses_locator_directories(tmp_path: Path, monkeypatch) -> None:
    locator_config = DataLocatorConfig(
        base_data_dir=str(tmp_path / "data"),
        base_cache_dir=str(tmp_path / "cache"),
        base_artifacts_dir=str(tmp_path / "artifacts"),
        base_generated_dir=str(tmp_path / "generated"),
        base_config_dir=str(tmp_path / "config"),
    )
    config = SubPipelineConfig(data_locator_config=locator_config)

    monkeypatch.setattr(
        "src.training.steps.market_analysis.sub_pipeline.FEATURE_IMPORTANCE_AVAILABLE",
        False,
        raising=False,
    )

    pipeline = MarketAnalysisSubPipeline(config)

    expected_data_dir = config.data.path(config.data_dir_key)
    assert Path(pipeline.config.data_dir) == expected_data_dir
    assert pipeline.config.paths.summary()["data"]["root"] == str(expected_data_dir)
