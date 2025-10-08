import asyncio
from types import SimpleNamespace

from src.launcher.ares_launcher import AresLauncher, ExecutionModeType, LauncherMode
from src.training.steps.main_training_pipeline import PipelineStage


def test_pre_training_sub_pipelines_resolve(monkeypatch):
    """Ensure each advertised pre-training sub-pipeline resolves through the launcher."""
    launcher = AresLauncher()
    executed = []

    async def fake_execute(self, sub_pipeline: str, config):
        executed.append((sub_pipeline, config))
        return SimpleNamespace(sub_pipeline=sub_pipeline)

    monkeypatch.setattr(AresLauncher, "_execute_sub_pipeline", fake_execute, raising=True)

    pre_training_steps = [
        "multi_horizon_profit_labeler",
        "feature_lookback_optimization",
        "interactive_feature_generation",
        "final_feature_selection",
    ]

    launcher.pipeline.get_available_sub_pipelines = lambda stage: pre_training_steps if stage == PipelineStage.PRE_TRAINING else []

    for sub_pipeline in pre_training_steps:
        result = asyncio.run(
            launcher.execute_pipeline(
                mode=LauncherMode.SUB_PIPELINE,
                sub_pipeline=sub_pipeline,
                symbol="ETHUSDT",
                exchange="binance",
                timeframe="15m",
                data_dir="historical_data",
                execution_mode=ExecutionModeType.FULL,
            )
        )
        assert result.sub_pipeline == sub_pipeline

    resolved_names = [name for name, _ in executed]
    assert resolved_names == pre_training_steps
