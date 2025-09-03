# src/analyst/market_health_analyzer.py

from __future__ import annotations

import logging
from typing import Any, Dict

import pandas as pd
import numpy as np

from src.core.decorators import handles_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import error, warning


class MarketHealthAnalyzer:
    """
    Analyzes overall market health indicators including:
    - Volume patterns and anomalies
    - Price volatility and stability
    - Market microstructure quality
    - Liquidity metrics
    """
    
    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize Market Health Analyzer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger = system_logger.getChild("MarketHealthAnalyzer")
        
        # Analysis configuration
        self.health_config = config.get("analyst", {}).get("market_health_analyzer", {})
        self.lookback_periods = self.health_config.get("lookback_periods", [20, 50, 100])
        self.volatility_threshold = self.health_config.get("volatility_threshold", 2.0)
        self.volume_anomaly_threshold = self.health_config.get("volume_anomaly_threshold", 3.0)
        
        # State
        self.analysis_results: dict[str, Any] = {}
        self.is_initialized = False
        
        self.logger.info("MarketHealthAnalyzer initialized")
    
    async def initialize(self) -> bool:
        """Initialize the analyzer."""
        self.is_initialized = True
        return True
    
    @handles_errors(
        exceptions=(Exception,),
        default_return={},
        context="market health analysis",
    )
    async def execute_market_health_analysis(self, analysis_input: dict[str, Any]) -> dict[str, Any]:
        """
        Execute comprehensive market health analysis.
        
        Args:
            analysis_input: Dictionary containing:
                - market_data: DataFrame with market data
                - current_price: Current market price
                
        Returns:
            Dictionary with health analysis results
        """
        if not self.is_initialized:
            self.logger.error("MarketHealthAnalyzer not initialized")
            return {}
        
        market_data = analysis_input.get("market_data")
        current_price = analysis_input.get("current_price", 0)
        
        if market_data is None or market_data.empty:
            self.logger.warning("No market data provided for health analysis")
            return {}
        
        self.logger.info("Executing market health analysis...")
        
        # Calculate various health metrics
        volume_health = self._analyze_volume_health(market_data)
        volatility_health = self._analyze_volatility_health(market_data)
        liquidity_health = self._analyze_liquidity_health(market_data)
        microstructure_quality = self._analyze_microstructure_quality(market_data)
        
        # Aggregate health score
        health_score = self._calculate_overall_health_score({
            "volume": volume_health,
            "volatility": volatility_health,
            "liquidity": liquidity_health,
            "microstructure": microstructure_quality,
        })
        
        self.analysis_results = {
            "health_score": health_score,
            "volume_health": volume_health,
            "volatility_health": volatility_health,
            "liquidity_health": liquidity_health,
            "microstructure_quality": microstructure_quality,
            "current_price": current_price,
            "status": self._determine_market_status(health_score),
        }
        
        self.logger.info(f"Market health analysis completed. Overall score: {health_score:.2f}")
        
        return self.analysis_results
    
    def _analyze_volume_health(self, market_data: pd.DataFrame) -> dict[str, Any]:
        """Analyze volume patterns and anomalies."""
        if "volume" not in market_data.columns:
            return {"score": 0.5, "status": "unknown"}
        
        volume = market_data["volume"].fillna(0)
        
        # Calculate volume metrics
        recent_volume = volume.tail(20).mean()
        historical_volume = volume.mean()
        volume_std = volume.std()
        
        # Detect volume anomalies
        volume_z_score = abs((recent_volume - historical_volume) / (volume_std + 1e-8))
        is_anomalous = volume_z_score > self.volume_anomaly_threshold
        
        # Calculate volume consistency
        volume_cv = volume_std / (historical_volume + 1e-8)  # Coefficient of variation
        consistency_score = 1 / (1 + volume_cv)
        
        # Overall volume health score
        health_score = consistency_score * (1 - min(volume_z_score / 5, 1))
        
        return {
            "score": health_score,
            "recent_avg": recent_volume,
            "historical_avg": historical_volume,
            "z_score": volume_z_score,
            "is_anomalous": is_anomalous,
            "consistency": consistency_score,
            "status": "healthy" if health_score > 0.7 else "degraded" if health_score > 0.4 else "unhealthy"
        }
    
    def _analyze_volatility_health(self, market_data: pd.DataFrame) -> dict[str, Any]:
        """Analyze price volatility and stability."""
        if "close" not in market_data.columns:
            return {"score": 0.5, "status": "unknown"}
        
        close_prices = market_data["close"].fillna(method='ffill')
        returns = close_prices.pct_change().dropna()
        
        # Calculate volatility metrics for different periods
        volatility_metrics = {}
        for period in self.lookback_periods:
            if len(returns) >= period:
                period_vol = returns.tail(period).std() * np.sqrt(252)  # Annualized
                volatility_metrics[f"vol_{period}"] = period_vol
        
        # Recent vs historical volatility
        recent_vol = returns.tail(20).std() * np.sqrt(252)
        historical_vol = returns.std() * np.sqrt(252)
        vol_ratio = recent_vol / (historical_vol + 1e-8)
        
        # Volatility regime
        is_high_vol = vol_ratio > self.volatility_threshold
        
        # Calculate stability score
        stability_score = 1 / (1 + abs(vol_ratio - 1))
        
        return {
            "score": stability_score,
            "recent_volatility": recent_vol,
            "historical_volatility": historical_vol,
            "volatility_ratio": vol_ratio,
            "is_high_volatility": is_high_vol,
            "period_volatilities": volatility_metrics,
            "status": "stable" if stability_score > 0.7 else "unstable" if stability_score < 0.3 else "moderate"
        }
    
    def _analyze_liquidity_health(self, market_data: pd.DataFrame) -> dict[str, Any]:
        """Analyze market liquidity metrics."""
        liquidity_score = 0.7  # Default moderate liquidity
        
        # If we have bid-ask spread data
        if "bid" in market_data.columns and "ask" in market_data.columns:
            spreads = (market_data["ask"] - market_data["bid"]) / market_data["bid"]
            avg_spread = spreads.mean()
            spread_stability = 1 / (1 + spreads.std())
            liquidity_score = spread_stability * (1 - min(avg_spread * 100, 1))
        
        # Volume-based liquidity proxy
        elif "volume" in market_data.columns and "close" in market_data.columns:
            # Amihud illiquidity ratio
            returns = market_data["close"].pct_change().abs()
            dollar_volume = market_data["volume"] * market_data["close"]
            illiquidity = (returns / (dollar_volume + 1)).mean()
            liquidity_score = 1 / (1 + illiquidity * 1e6)
        
        return {
            "score": liquidity_score,
            "status": "liquid" if liquidity_score > 0.7 else "illiquid" if liquidity_score < 0.3 else "moderate"
        }
    
    def _analyze_microstructure_quality(self, market_data: pd.DataFrame) -> dict[str, Any]:
        """Analyze market microstructure quality."""
        quality_score = 0.7  # Default moderate quality
        
        if "close" in market_data.columns:
            close_prices = market_data["close"]
            
            # Price efficiency: autocorrelation test
            if len(close_prices) > 20:
                returns = close_prices.pct_change().dropna()
                autocorr = returns.autocorr(lag=1)
                efficiency_score = 1 - abs(autocorr)
                
                # Price continuity: gap analysis
                price_gaps = close_prices.diff().abs()
                avg_gap = price_gaps.mean()
                gap_ratio = avg_gap / close_prices.mean()
                continuity_score = 1 / (1 + gap_ratio * 100)
                
                quality_score = (efficiency_score + continuity_score) / 2
        
        return {
            "score": quality_score,
            "status": "good" if quality_score > 0.7 else "poor" if quality_score < 0.3 else "fair"
        }
    
    def _calculate_overall_health_score(self, component_scores: dict[str, dict]) -> float:
        """Calculate weighted overall health score."""
        weights = {
            "volume": 0.25,
            "volatility": 0.35,
            "liquidity": 0.25,
            "microstructure": 0.15
        }
        
        total_score = 0
        total_weight = 0
        
        for component, weight in weights.items():
            if component in component_scores:
                score = component_scores[component].get("score", 0.5)
                total_score += score * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.5
    
    def _determine_market_status(self, health_score: float) -> str:
        """Determine overall market status based on health score."""
        if health_score >= 0.8:
            return "excellent"
        elif health_score >= 0.6:
            return "good"
        elif health_score >= 0.4:
            return "fair"
        elif health_score >= 0.2:
            return "poor"
        else:
            return "critical"
    
    def get_analysis_results(self) -> dict[str, Any]:
        """Get the latest analysis results."""
        return self.analysis_results
    
    async def stop(self) -> None:
        """Stop the analyzer and clean up resources."""
        self.logger.info("Stopping MarketHealthAnalyzer")
        self.is_initialized = False
        self.analysis_results = {}


async def setup_market_health_analyzer(config: dict[str, Any]) -> MarketHealthAnalyzer:
    """
    Setup and initialize MarketHealthAnalyzer.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized MarketHealthAnalyzer instance
    """
    analyzer = MarketHealthAnalyzer(config)
    success = await analyzer.initialize()
    return analyzer if success else None