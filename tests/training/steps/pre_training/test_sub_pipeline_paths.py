from pathlib import Path

from src.training.config.data_locator import DataLocator, DataLocatorConfig
from src.training.steps.pre_training.sub_pipeline import SubPipelineConfig


def test_sub_pipeline_config_exposes_resolved_paths(tmp_path: Path) -> None:
    config = SubPipelineConfig(
        data_locator_config=DataLocatorConfig(
            base_data_dir=str(tmp_path / "data"),
            base_cache_dir=str(tmp_path / "cache"),
            base_artifacts_dir=str(tmp_path / "artifacts"),
            base_generated_dir=str(tmp_path / "generated"),
            base_config_dir=str(tmp_path / "configs"),
        )
    )

    locator = DataLocator(config.data_locator_config)
    config.attach_locator(locator)

    assert config.data.root == locator.base_data_dir
    assert config.cache.root == locator.base_cache_dir
    assert config.artifacts.root == locator.base_artifacts_dir
    assert config.generated.market_analysis == locator.generated_path("market_analysis")
    assert config.config.multi_horizon_labeling == locator.config_path("multi_horizon_labeling")
    assert "market_analysis" in config.paths.summary()["generated"]
