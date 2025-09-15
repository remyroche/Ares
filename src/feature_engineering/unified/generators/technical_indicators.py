"""
Technical Indicators Feature Generator

This module provides a comprehensive technical indicators generator
that demonstrates the unified feature generation system.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import pandas_ta as ta

from ..core import FeatureGenerator, FeatureGeneratorConfig, FeatureGenerationResult, FeatureCategory, FeaturePriority
from ...utils.logger import system_logger
from ...core.decorators import handles_errors


class TechnicalIndicatorsGenerator(FeatureGenerator):
    """
    Generator for technical indicators using pandas_ta.
    
    This generator demonstrates how to implement a feature generator
    using the unified system with comprehensive technical analysis.
    """
    
    def __init__(self, config: Optional[FeatureGeneratorConfig] = None):
        """Initialize the technical indicators generator."""
        if config is None:
            config = FeatureGeneratorConfig(
                name="technical_indicators",
                category=FeatureCategory.TECHNICAL_INDICATORS,
                priority=FeaturePriority.HIGH,
                enabled=True,
                parameters={
                    "indicators": [
                        "sma", "ema", "rsi", "macd", "bbands", "stoch", "atr",
                        "adx", "cci", "williams_r", "ultimate_oscillator"
                    ],
                    "periods": {
                        "sma": [5, 10, 20, 50],
                        "ema": [12, 26],
                        "rsi": [14],
                        "macd": [12, 26, 9],
                        "bbands": [20, 2],
                        "stoch": [14, 3, 3],
                        "atr": [14],
                        "adx": [14],
                        "cci": [20],
                        "williams_r": [14],
                        "ultimate_oscillator": [7, 14, 28]
                    }
                }
            )
        
        super().__init__(config)
        self._indicators = self.config.parameters.get("indicators", [])
        self._periods = self.config.parameters.get("periods", {})
    
    async def initialize(self) -> bool:
        """Initialize the generator."""
        try:
            self.logger.info("Initializing technical indicators generator...")
            
            # Validate required columns
            required_cols = self.get_required_columns()
            if not required_cols:
                self.logger.error("No required columns defined")
                return False
            
            self._is_initialized = True
            self.logger.info(f"Technical indicators generator initialized with {len(self._indicators)} indicators")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing technical indicators generator: {e}")
            return False
    
    async def generate_features(
        self, 
        data: pd.DataFrame,
        context: Optional[Dict[str, Any]] = None
    ) -> FeatureGenerationResult:
        """Generate technical indicators."""
        try:
            if not self._is_initialized:
                return FeatureGenerationResult(
                    success=False,
                    errors=["Generator not initialized"]
                )
            
            # Validate input
            is_valid, errors = self.validate_input(data)
            if not is_valid:
                return FeatureGenerationResult(
                    success=False,
                    errors=errors
                )
            
            # Generate indicators
            features = data.copy()
            generated_indicators = []
            
            for indicator in self._indicators:
                try:
                    indicator_features = await self._generate_indicator(features, indicator)
                    if indicator_features is not None:
                        features = pd.concat([features, indicator_features], axis=1)
                        generated_indicators.append(indicator)
                except Exception as e:
                    self.logger.warning(f"Failed to generate {indicator}: {e}")
            
            # Remove original OHLCV columns to keep only indicators
            indicator_columns = [col for col in features.columns if col not in data.columns]
            indicator_features = features[indicator_columns]
            
            # Validate output
            is_valid, errors = self.validate_output(indicator_features)
            if not is_valid:
                return FeatureGenerationResult(
                    success=False,
                    features=indicator_features,
                    errors=errors
                )
            
            return FeatureGenerationResult(
                success=True,
                features=indicator_features,
                metadata={
                    "generated_indicators": generated_indicators,
                    "indicator_count": len(generated_indicators)
                },
                performance_metrics={
                    "indicators_generated": len(generated_indicators),
                    "total_indicators": len(self._indicators)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error generating technical indicators: {e}")
            return FeatureGenerationResult(
                success=False,
                errors=[f"Technical indicators error: {str(e)}"]
            )
    
    async def _generate_indicator(self, data: pd.DataFrame, indicator: str) -> Optional[pd.DataFrame]:
        """Generate a specific technical indicator."""
        try:
            if indicator == "sma":
                return self._generate_sma(data)
            elif indicator == "ema":
                return self._generate_ema(data)
            elif indicator == "rsi":
                return self._generate_rsi(data)
            elif indicator == "macd":
                return self._generate_macd(data)
            elif indicator == "bbands":
                return self._generate_bbands(data)
            elif indicator == "stoch":
                return self._generate_stoch(data)
            elif indicator == "atr":
                return self._generate_atr(data)
            elif indicator == "adx":
                return self._generate_adx(data)
            elif indicator == "cci":
                return self._generate_cci(data)
            elif indicator == "williams_r":
                return self._generate_williams_r(data)
            elif indicator == "ultimate_oscillator":
                return self._generate_ultimate_oscillator(data)
            else:
                self.logger.warning(f"Unknown indicator: {indicator}")
                return None
                
        except Exception as e:
            self.logger.warning(f"Error generating {indicator}: {e}")
            return None
    
    def _generate_sma(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Simple Moving Averages."""
        features = pd.DataFrame(index=data.index)
        periods = self._periods.get("sma", [5, 10, 20, 50])
        
        for period in periods:
            if len(data) >= period:
                features[f"sma_{period}"] = ta.sma(data["close"], length=period)
        
        return features
    
    def _generate_ema(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Exponential Moving Averages."""
        features = pd.DataFrame(index=data.index)
        periods = self._periods.get("ema", [12, 26])
        
        for period in periods:
            if len(data) >= period:
                features[f"ema_{period}"] = ta.ema(data["close"], length=period)
        
        return features
    
    def _generate_rsi(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Relative Strength Index."""
        features = pd.DataFrame(index=data.index)
        periods = self._periods.get("rsi", [14])
        
        for period in periods:
            if len(data) >= period:
                features[f"rsi_{period}"] = ta.rsi(data["close"], length=period)
        
        return features
    
    def _generate_macd(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate MACD."""
        features = pd.DataFrame(index=data.index)
        params = self._periods.get("macd", [12, 26, 9])
        
        if len(params) >= 3 and len(data) >= max(params):
            macd_result = ta.macd(data["close"], fast=params[0], slow=params[1], signal=params[2])
            if macd_result is not None:
                features["macd"] = macd_result[f"MACD_{params[0]}_{params[1]}_{params[2]}"]
                features["macd_signal"] = macd_result[f"MACDs_{params[0]}_{params[1]}_{params[2]}"]
                features["macd_histogram"] = macd_result[f"MACDh_{params[0]}_{params[1]}_{params[2]}"]
        
        return features
    
    def _generate_bbands(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Bollinger Bands."""
        features = pd.DataFrame(index=data.index)
        params = self._periods.get("bbands", [20, 2])
        
        if len(params) >= 2 and len(data) >= params[0]:
            bb_result = ta.bbands(data["close"], length=params[0], std=params[1])
            if bb_result is not None:
                features["bb_upper"] = bb_result[f"BBU_{params[0]}_{params[1]}"]
                features["bb_middle"] = bb_result[f"BBM_{params[0]}_{params[1]}"]
                features["bb_lower"] = bb_result[f"BBL_{params[0]}_{params[1]}"]
                features["bb_width"] = (bb_result[f"BBU_{params[0]}_{params[1]}"] - bb_result[f"BBL_{params[0]}_{params[1]}"]) / bb_result[f"BBM_{params[0]}_{params[1]}"]
        
        return features
    
    def _generate_stoch(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Stochastic Oscillator."""
        features = pd.DataFrame(index=data.index)
        params = self._periods.get("stoch", [14, 3, 3])
        
        if len(params) >= 3 and len(data) >= params[0]:
            stoch_result = ta.stoch(data["high"], data["low"], data["close"], k=params[0], d=params[1], smooth_k=params[2])
            if stoch_result is not None:
                features["stoch_k"] = stoch_result[f"STOCHk_{params[0]}_{params[1]}_{params[2]}"]
                features["stoch_d"] = stoch_result[f"STOCHd_{params[0]}_{params[1]}_{params[2]}"]
        
        return features
    
    def _generate_atr(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Average True Range."""
        features = pd.DataFrame(index=data.index)
        periods = self._periods.get("atr", [14])
        
        for period in periods:
            if len(data) >= period:
                features[f"atr_{period}"] = ta.atr(data["high"], data["low"], data["close"], length=period)
        
        return features
    
    def _generate_adx(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Average Directional Index."""
        features = pd.DataFrame(index=data.index)
        periods = self._periods.get("adx", [14])
        
        for period in periods:
            if len(data) >= period:
                adx_result = ta.adx(data["high"], data["low"], data["close"], length=period)
                if adx_result is not None:
                    features[f"adx_{period}"] = adx_result[f"ADX_{period}"]
                    features[f"di_plus_{period}"] = adx_result[f"DMP_{period}"]
                    features[f"di_minus_{period}"] = adx_result[f"DMN_{period}"]
        
        return features
    
    def _generate_cci(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Commodity Channel Index."""
        features = pd.DataFrame(index=data.index)
        periods = self._periods.get("cci", [20])
        
        for period in periods:
            if len(data) >= period:
                features[f"cci_{period}"] = ta.cci(data["high"], data["low"], data["close"], length=period)
        
        return features
    
    def _generate_williams_r(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Williams %R."""
        features = pd.DataFrame(index=data.index)
        periods = self._periods.get("williams_r", [14])
        
        for period in periods:
            if len(data) >= period:
                features[f"williams_r_{period}"] = ta.willr(data["high"], data["low"], data["close"], length=period)
        
        return features
    
    def _generate_ultimate_oscillator(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate Ultimate Oscillator."""
        features = pd.DataFrame(index=data.index)
        params = self._periods.get("ultimate_oscillator", [7, 14, 28])
        
        if len(params) >= 3 and len(data) >= max(params):
            uo_result = ta.uo(data["high"], data["low"], data["close"], fast=params[0], medium=params[1], slow=params[2])
            if uo_result is not None:
                features["ultimate_oscillator"] = uo_result
        
        return features
    
    def get_required_columns(self) -> List[str]:
        """Get required columns for technical indicators."""
        return ["open", "high", "low", "close", "volume"]
    
    def get_output_columns(self) -> List[str]:
        """Get output columns that will be generated."""
        output_columns = []
        
        # SMA columns
        periods = self._periods.get("sma", [5, 10, 20, 50])
        output_columns.extend([f"sma_{p}" for p in periods])
        
        # EMA columns
        periods = self._periods.get("ema", [12, 26])
        output_columns.extend([f"ema_{p}" for p in periods])
        
        # RSI columns
        periods = self._periods.get("rsi", [14])
        output_columns.extend([f"rsi_{p}" for p in periods])
        
        # MACD columns
        output_columns.extend(["macd", "macd_signal", "macd_histogram"])
        
        # Bollinger Bands columns
        output_columns.extend(["bb_upper", "bb_middle", "bb_lower", "bb_width"])
        
        # Stochastic columns
        output_columns.extend(["stoch_k", "stoch_d"])
        
        # ATR columns
        periods = self._periods.get("atr", [14])
        output_columns.extend([f"atr_{p}" for p in periods])
        
        # ADX columns
        periods = self._periods.get("adx", [14])
        output_columns.extend([f"adx_{p}" for p in periods])
        output_columns.extend([f"di_plus_{p}" for p in periods])
        output_columns.extend([f"di_minus_{p}" for p in periods])
        
        # CCI columns
        periods = self._periods.get("cci", [20])
        output_columns.extend([f"cci_{p}" for p in periods])
        
        # Williams %R columns
        periods = self._periods.get("williams_r", [14])
        output_columns.extend([f"williams_r_{p}" for p in periods])
        
        # Ultimate Oscillator
        output_columns.append("ultimate_oscillator")
        
        return output_columns