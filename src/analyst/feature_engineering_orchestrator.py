# src/analyst/feature_engineering_orchestrator.py

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from src.analyst.advanced_feature_engineering import AdvancedFeatureEngineering
from src.analyst.autoencoder_feature_generator import \
    AutoencoderFeatureGenerator
from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger


class FeatureEngineeringOrchestrator:
    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.logger = system_logger.getChild("FeatureEngineeringOrchestrator")
        self.is_initialized = False
        self.advanced_feature_engineering = AdvancedFeatureEngineering(self.config)
        self.autoencoder_generator = AutoencoderFeatureGenerator(self.config)

    @handle_errors(
        exceptions=(Exception,), default_return=False, context="orchestrator init"
    )
    async def initialize(self) -> bool:
        await self.advanced_feature_engineering.initialize()
        await self.autoencoder_generator.initialize()
        self.is_initialized = True
        self.logger.info("FeatureEngineeringOrchestrator initialized")
        return True

    @handle_errors(
        exceptions=(Exception,),
        default_return=pd.DataFrame(),
        context="orchestrated feature generation",
    )
    async def generate_all_features(self, klines_df: pd.DataFrame) -> pd.DataFrame:
        if klines_df is None or klines_df.empty:
            return pd.DataFrame()
        features_df = klines_df.copy()
        features_df = self.autoencoder_generator.generate_features(features_df)
        return features_df
