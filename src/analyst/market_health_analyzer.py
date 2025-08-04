# src/analyst/market_health_analyzer.py
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np

from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger


class MarketHealthAnalyzer:
    """
    Simplified Market Health Analyzer that focuses on essential volatility, market health metrics,
    and simple liquidity and stress analysis.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize market health analyzer with enhanced type safety.

        Args:
            config: Configuration dictionary
        """
        self.config: dict[str, Any] = config
        self.logger = system_logger.getChild("MarketHealthAnalyzer")

        # Market health analyzer state
        self.is_analyzing: bool = False
        self.analysis_results: dict[str, Any] = {}

        # Configuration
        self.market_health_config: dict[str, Any] = self.config.get(
            "market_health_analyzer",
            {},
        )
        self.analysis_interval: int = self.market_health_config.get(
            "analysis_interval",
            3600,
        )
        self.enable_volatility_analysis: bool = self.market_health_config.get(
            "enable_volatility_analysis",
            True,
        )
        self.enable_market_health_metrics: bool = self.market_health_config.get(
            "enable_market_health_metrics",
            True,
        )
        self.enable_liquidity_analysis: bool = self.market_health_config.get(
            "enable_liquidity_analysis",
            True,
        )
        self.enable_stress_analysis: bool = self.market_health_config.get(
            "enable_stress_analysis",
            True,
        )

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid market health analyzer configuration"),
            AttributeError: (
                False,
                "Missing required market health analyzer parameters",
            ),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="market health analyzer initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize market health analyzer with enhanced error handling.

        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Market Health Analyzer...")

            # Load market health analyzer configuration
            await self._load_market_health_configuration()

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error("Invalid configuration for market health analyzer")
                return False

            # Initialize market health analyzer modules
            await self._initialize_market_health_modules()

            self.logger.info(
                "✅ Market Health Analyzer initialization completed successfully",
            )
            return True

        except Exception as e:
            self.logger.error(f"❌ Market Health Analyzer initialization failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="market health configuration loading",
    )
    async def _load_market_health_configuration(self) -> None:
        """Load market health analyzer configuration."""
        self.logger.info("Loading market health analyzer configuration...")
        
        # Additional configuration can be loaded here
        self.logger.info("Market health configuration loaded successfully")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="configuration validation",
    )
    def _validate_configuration(self) -> bool:
        """Validate market health analyzer configuration."""
        try:
            if self.analysis_interval <= 0:
                self.logger.error("analysis_interval must be positive")
                return False

            self.logger.info("Market health analyzer configuration validation passed")
            return True

        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="market health modules initialization",
    )
    async def _initialize_market_health_modules(self) -> None:
        """Initialize market health analyzer modules."""
        self.logger.info("Initializing market health modules...")

        if self.enable_volatility_analysis:
            await self._initialize_volatility_analysis()

        if self.enable_market_health_metrics:
            await self._initialize_market_health_metrics()

        if self.enable_liquidity_analysis:
            await self._initialize_liquidity_analysis()

        if self.enable_stress_analysis:
            await self._initialize_stress_analysis()

        self.logger.info("Market health modules initialized successfully")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="volatility analysis initialization",
    )
    async def _initialize_volatility_analysis(self) -> None:
        """Initialize volatility analysis module."""
        self.logger.info("Initializing volatility analysis...")
        # Volatility analysis initialization logic here
        self.logger.info("Volatility analysis initialized successfully")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="market health metrics initialization",
    )
    async def _initialize_market_health_metrics(self) -> None:
        """Initialize market health metrics module."""
        self.logger.info("Initializing market health metrics...")
        # Market health metrics initialization logic here
        self.logger.info("Market health metrics initialized successfully")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="liquidity analysis initialization",
    )
    async def _initialize_liquidity_analysis(self) -> None:
        """Initialize liquidity analysis module."""
        self.logger.info("Initializing liquidity analysis...")
        # Liquidity analysis initialization logic here
        self.logger.info("Liquidity analysis initialized successfully")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="stress analysis initialization",
    )
    async def _initialize_stress_analysis(self) -> None:
        """Initialize stress analysis module."""
        self.logger.info("Initializing stress analysis...")
        # Stress analysis initialization logic here
        self.logger.info("Stress analysis initialized successfully")

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid analysis parameters"),
            AttributeError: (False, "Missing analysis components"),
            KeyError: (False, "Missing required analysis data"),
        },
        default_return=False,
        context="market health analysis execution",
    )
    async def execute_market_health_analysis(
        self,
        analysis_input: dict[str, Any],
    ) -> bool:
        """
        Execute market health analysis with enhanced error handling.

        Args:
            analysis_input: Input data for analysis

        Returns:
            bool: True if analysis successful, False otherwise
        """
        try:
            if not self._validate_analysis_inputs(analysis_input):
                self.logger.error("Invalid analysis inputs")
                return False

            self.is_analyzing = True
            self.logger.info("Starting market health analysis...")

            # Perform volatility analysis
            if self.enable_volatility_analysis:
                volatility_results = await self._perform_volatility_analysis(analysis_input)
                self.analysis_results["volatility_analysis"] = volatility_results

            # Perform market health metrics
            if self.enable_market_health_metrics:
                health_metrics = await self._perform_market_health_metrics(analysis_input)
                self.analysis_results["market_health_metrics"] = health_metrics

            # Perform liquidity analysis with advanced features
            if self.enable_liquidity_analysis:
                liquidity_results = await self._perform_liquidity_analysis(analysis_input)
                self.analysis_results["liquidity_analysis"] = liquidity_results

            # Perform stress analysis
            if self.enable_stress_analysis:
                stress_results = await self._perform_stress_analysis(analysis_input)
                self.analysis_results["stress_analysis"] = stress_results

            # Calculate overall market health score
            overall_health = await self._calculate_overall_market_health()
            self.analysis_results["overall_market_health"] = overall_health

            # Add timestamp
            self.analysis_results["timestamp"] = datetime.now().isoformat()

            self.is_analyzing = False
            self.logger.info("✅ Market health analysis completed successfully")
            return True

        except Exception as e:
            self.logger.error(f"❌ Error in market health analysis: {e}")
            self.is_analyzing = False
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="analysis inputs validation",
    )
    def _validate_analysis_inputs(self, analysis_input: dict[str, Any]) -> bool:
        """Validate analysis input data."""
        try:
            required_keys = ["market_data", "current_price"]
            for key in required_keys:
                if key not in analysis_input:
                    self.logger.error(f"Missing required analysis input: {key}")
                    return False

            market_data = analysis_input.get("market_data")
            if not isinstance(market_data, pd.DataFrame) or market_data.empty:
                self.logger.error("Invalid market data provided")
                return False

            current_price = analysis_input.get("current_price")
            if not isinstance(current_price, (int, float)) or current_price <= 0:
                self.logger.error("Invalid current price provided")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Analysis inputs validation failed: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="volatility analysis",
    )
    async def _perform_volatility_analysis(
        self,
        analysis_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform volatility analysis.

        Args:
            analysis_input: Input data for analysis

        Returns:
            dict: Volatility analysis results
        """
        try:
            market_data = analysis_input.get("market_data")
            current_price = analysis_input.get("current_price")

            # Calculate basic volatility metrics
            volatility_results = {
                "current_volatility": self._calculate_current_volatility(market_data),
                "volatility_regime": self._classify_volatility_regime(market_data),
                "volatility_forecast": self._forecast_volatility(market_data),
                "volatility_health": self._assess_volatility_health(market_data),
                "timestamp": datetime.now().isoformat(),
            }

            self.logger.info("Volatility analysis completed successfully")
            return volatility_results

        except Exception as e:
            self.logger.error(f"Error performing volatility analysis: {e}")
            return None

    def _calculate_current_volatility(self, market_data: pd.DataFrame) -> float:
        """Calculate current volatility using standard deviation of returns."""
        try:
            if "close" not in market_data.columns:
                return 0.0

            # Calculate returns
            returns = market_data["close"].pct_change().dropna()
            
            # Calculate rolling volatility (20-period)
            volatility = returns.rolling(window=20).std().iloc[-1]
            
            return float(volatility) if not pd.isna(volatility) else 0.0

        except Exception as e:
            self.logger.error(f"Error calculating current volatility: {e}")
            return 0.0

    def _classify_volatility_regime(self, market_data: pd.DataFrame) -> str:
        """Classify current volatility regime."""
        try:
            current_vol = self._calculate_current_volatility(market_data)
            
            if current_vol <= 0.02:  # 2% daily volatility
                return "LOW"
            elif current_vol <= 0.04:  # 4% daily volatility
                return "NORMAL"
            elif current_vol <= 0.08:  # 8% daily volatility
                return "HIGH"
            else:
                return "EXTREME"

        except Exception as e:
            self.logger.error(f"Error classifying volatility regime: {e}")
            return "UNKNOWN"

    def _forecast_volatility(self, market_data: pd.DataFrame) -> float:
        """Simple volatility forecasting using recent trend."""
        try:
            if "close" not in market_data.columns:
                return 0.0

            # Calculate recent volatility trend
            returns = market_data["close"].pct_change().dropna()
            recent_vol = returns.rolling(window=10).std().iloc[-1]
            older_vol = returns.rolling(window=10).std().iloc[-10] if len(returns) >= 20 else recent_vol
            
            # Simple trend-based forecast
            vol_trend = recent_vol - older_vol
            forecast_vol = recent_vol + (vol_trend * 0.5)  # Extrapolate trend
            
            return max(0.0, float(forecast_vol)) if not pd.isna(forecast_vol) else recent_vol

        except Exception as e:
            self.logger.error(f"Error forecasting volatility: {e}")
            return 0.0

    def _assess_volatility_health(self, market_data: pd.DataFrame) -> str:
        """Assess overall volatility health."""
        try:
            current_vol = self._calculate_current_volatility(market_data)
            regime = self._classify_volatility_regime(market_data)
            
            if regime == "LOW":
                return "HEALTHY"
            elif regime == "NORMAL":
                return "NORMAL"
            elif regime == "HIGH":
                return "CAUTION"
            else:
                return "DANGEROUS"

        except Exception as e:
            self.logger.error(f"Error assessing volatility health: {e}")
            return "UNKNOWN"

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="market health metrics",
    )
    async def _perform_market_health_metrics(
        self,
        analysis_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform market health metrics analysis.

        Args:
            analysis_input: Input data for analysis

        Returns:
            dict: Market health metrics results
        """
        try:
            market_data = analysis_input.get("market_data")
            current_price = analysis_input.get("current_price")

            # Calculate market health metrics
            health_results = {
                "price_trend": self._calculate_price_trend(market_data),
                "volume_health": self._assess_volume_health(market_data),
                "market_strength": self._calculate_market_strength(market_data),
                "overall_health": self._calculate_overall_health(market_data),
                "timestamp": datetime.now().isoformat(),
            }

            self.logger.info("Market health metrics analysis completed successfully")
            return health_results

        except Exception as e:
            self.logger.error(f"Error performing market health metrics: {e}")
            return None

    def _calculate_price_trend(self, market_data: pd.DataFrame) -> str:
        """Calculate current price trend."""
        try:
            if "close" not in market_data.columns:
                return "UNKNOWN"

            # Calculate short-term trend (last 20 periods)
            recent_prices = market_data["close"].tail(20)
            if len(recent_prices) < 10:
                return "UNKNOWN"

            # Simple trend calculation
            start_price = recent_prices.iloc[0]
            end_price = recent_prices.iloc[-1]
            trend_pct = (end_price - start_price) / start_price

            if trend_pct > 0.02:  # 2% increase
                return "BULLISH"
            elif trend_pct < -0.02:  # 2% decrease
                return "BEARISH"
            else:
                return "SIDEWAYS"

        except Exception as e:
            self.logger.error(f"Error calculating price trend: {e}")
            return "UNKNOWN"

    def _assess_volume_health(self, market_data: pd.DataFrame) -> str:
        """Assess volume health."""
        try:
            if "volume" not in market_data.columns:
                return "UNKNOWN"

            # Calculate average volume
            avg_volume = market_data["volume"].mean()
            current_volume = market_data["volume"].iloc[-1]

            if current_volume > avg_volume * 1.5:
                return "HIGH"
            elif current_volume < avg_volume * 0.5:
                return "LOW"
            else:
                return "NORMAL"

        except Exception as e:
            self.logger.error(f"Error assessing volume health: {e}")
            return "UNKNOWN"

    def _calculate_market_strength(self, market_data: pd.DataFrame) -> float:
        """Calculate market strength score (0-100)."""
        try:
            if "close" not in market_data.columns:
                return 50.0

            # Simple market strength calculation
            returns = market_data["close"].pct_change().dropna()
            
            # Positive returns ratio
            positive_ratio = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0.5
            
            # Average return magnitude
            avg_return = abs(returns).mean() if len(returns) > 0 else 0.0
            
            # Combine factors
            strength = (positive_ratio * 60) + (min(avg_return * 1000, 40))
            
            return max(0.0, min(100.0, float(strength)))

        except Exception as e:
            self.logger.error(f"Error calculating market strength: {e}")
            return 50.0

    def _calculate_overall_health(self, market_data: pd.DataFrame) -> str:
        """Calculate overall market health."""
        try:
            trend = self._calculate_price_trend(market_data)
            volume_health = self._assess_volume_health(market_data)
            strength = self._calculate_market_strength(market_data)

            # Combine factors for overall health
            if trend == "BULLISH" and volume_health == "HIGH" and strength > 70:
                return "EXCELLENT"
            elif trend in ["BULLISH", "SIDEWAYS"] and strength > 50:
                return "GOOD"
            elif strength > 30:
                return "FAIR"
            else:
                return "POOR"

        except Exception as e:
            self.logger.error(f"Error calculating overall health: {e}")
            return "UNKNOWN"

    async def _perform_liquidity_analysis(self, analysis_input: dict[str, Any]) -> dict[str, Any]:
        """Perform comprehensive liquidity analysis using advanced features."""
        try:
            market_data = analysis_input.get("market_data")
            current_price = analysis_input.get("current_price")
            
            # Initialize advanced feature engineering for liquidity analysis
            from src.analyst.advanced_feature_engineering import AdvancedFeatureEngineering
            
            # Create feature engineering instance
            feature_engineering = AdvancedFeatureEngineering(self.config)
            await feature_engineering.initialize()
            
            # Prepare data for liquidity analysis
            price_data = market_data[['open', 'high', 'low', 'close']].copy()
            volume_data = market_data[['volume']].copy()
            
            # Get order flow data if available
            order_flow_data = analysis_input.get("order_flow_data")
            
            # Engineer liquidity features
            liquidity_features = await feature_engineering.engineer_features(
                price_data=price_data,
                volume_data=volume_data,
                order_flow_data=order_flow_data
            )
            
            # Extract liquidity-specific features
            liquidity_metrics = {
                "volume_liquidity": liquidity_features.get("volume_liquidity", 1.0),
                "price_impact": liquidity_features.get("price_impact", 0.0),
                "spread_liquidity": liquidity_features.get("spread_liquidity", 0.0),
                "liquidity_regime": liquidity_features.get("liquidity_regime", "medium"),
                "liquidity_percentile": liquidity_features.get("liquidity_percentile", 0.5),
                "kyle_lambda": liquidity_features.get("kyle_lambda", 0.0),
                "amihud_illiquidity": liquidity_features.get("amihud_illiquidity", 0.0),
                "order_flow_imbalance": liquidity_features.get("order_flow_imbalance", 0.0),
                "large_order_ratio": liquidity_features.get("large_order_ratio", 0.0),
                "vwap": liquidity_features.get("vwap", current_price),
                "volume_roc": liquidity_features.get("volume_roc", 0.0),
                "volume_ma_ratio": liquidity_features.get("volume_ma_ratio", 1.0),
            }
            
            # Calculate liquidity stress indicators
            liquidity_stress = self._calculate_liquidity_stress(liquidity_metrics)
            liquidity_metrics["liquidity_stress"] = liquidity_stress
            
            # Determine liquidity health status
            liquidity_health = self._determine_liquidity_health(liquidity_metrics)
            liquidity_metrics["liquidity_health"] = liquidity_health
            
            self.logger.info(f"Liquidity analysis completed - Health: {liquidity_health}")
            return liquidity_metrics
            
        except Exception as e:
            self.logger.error(f"Error performing liquidity analysis: {e}")
            return {}

    def _calculate_liquidity_stress(self, liquidity_metrics: dict[str, Any]) -> float:
        """Calculate liquidity stress score."""
        try:
            stress_factors = []
            
            # Volume-based stress
            volume_liquidity = liquidity_metrics.get("volume_liquidity", 1.0)
            if volume_liquidity < 0.5:
                stress_factors.append(0.8)  # High stress
            elif volume_liquidity < 0.8:
                stress_factors.append(0.4)  # Medium stress
            else:
                stress_factors.append(0.1)  # Low stress
            
            # Price impact stress
            price_impact = liquidity_metrics.get("price_impact", 0.0)
            if price_impact > 0.001:  # High price impact
                stress_factors.append(0.9)
            elif price_impact > 0.0005:  # Medium price impact
                stress_factors.append(0.5)
            else:
                stress_factors.append(0.1)
            
            # Amihud illiquidity stress
            amihud_illiquidity = liquidity_metrics.get("amihud_illiquidity", 0.0)
            if amihud_illiquidity > 0.01:  # High illiquidity
                stress_factors.append(0.9)
            elif amihud_illiquidity > 0.005:  # Medium illiquidity
                stress_factors.append(0.5)
            else:
                stress_factors.append(0.1)
            
            # Order flow imbalance stress
            order_flow_imbalance = abs(liquidity_metrics.get("order_flow_imbalance", 0.0))
            if order_flow_imbalance > 0.3:  # High imbalance
                stress_factors.append(0.8)
            elif order_flow_imbalance > 0.1:  # Medium imbalance
                stress_factors.append(0.4)
            else:
                stress_factors.append(0.1)
            
            # Calculate average stress score
            if stress_factors:
                return sum(stress_factors) / len(stress_factors)
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity stress: {e}")
            return 0.0

    def _determine_liquidity_health(self, liquidity_metrics: dict[str, Any]) -> str:
        """Determine overall liquidity health status."""
        try:
            liquidity_stress = liquidity_metrics.get("liquidity_stress", 0.0)
            liquidity_regime = liquidity_metrics.get("liquidity_regime", "medium")
            
            if liquidity_stress > 0.7 or liquidity_regime == "low":
                return "poor"
            elif liquidity_stress > 0.4 or liquidity_regime == "medium":
                return "fair"
            else:
                return "good"
                
        except Exception as e:
            self.logger.error(f"Error determining liquidity health: {e}")
            return "unknown"

    async def _calculate_overall_market_health(self) -> dict[str, Any]:
        """Calculate overall market health score incorporating liquidity factors."""
        try:
            volatility_analysis = self.analysis_results.get("volatility_analysis", {})
            health_metrics = self.analysis_results.get("market_health_metrics", {})
            liquidity_analysis = self.analysis_results.get("liquidity_analysis", {})
            stress_analysis = self.analysis_results.get("stress_analysis", {})
            
            # Extract key metrics
            volatility_score = volatility_analysis.get("volatility_score", 0.5)
            liquidity_stress = liquidity_analysis.get("liquidity_stress", 0.5)
            liquidity_health = liquidity_analysis.get("liquidity_health", "fair")
            
            # Calculate weighted health score
            weights = {
                "volatility": 0.3,
                "liquidity": 0.4,  # Higher weight for liquidity
                "stress": 0.3
            }
            
            # Normalize scores
            volatility_normalized = 1.0 - min(volatility_score, 1.0)  # Lower volatility = better
            liquidity_normalized = 1.0 - liquidity_stress  # Lower stress = better
            stress_normalized = 1.0 - stress_analysis.get("stress_score", 0.5)
            
            # Calculate weighted average
            overall_score = (
                weights["volatility"] * volatility_normalized +
                weights["liquidity"] * liquidity_normalized +
                weights["stress"] * stress_normalized
            )
            
            # Determine health status
            if overall_score > 0.7:
                health_status = "excellent"
            elif overall_score > 0.5:
                health_status = "good"
            elif overall_score > 0.3:
                health_status = "fair"
            else:
                health_status = "poor"
            
            return {
                "overall_score": overall_score,
                "health_status": health_status,
                "volatility_contribution": volatility_normalized,
                "liquidity_contribution": liquidity_normalized,
                "stress_contribution": stress_normalized,
                "liquidity_health": liquidity_health,
                "liquidity_stress": liquidity_stress,
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating overall market health: {e}")
            return {"overall_score": 0.5, "health_status": "unknown"}

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="liquidity analysis",
    )
    async def _perform_liquidity_analysis(
        self,
        analysis_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform simple liquidity analysis.

        Args:
            analysis_input: Input data for analysis

        Returns:
            dict: Liquidity analysis results
        """
        try:
            market_data = analysis_input.get("market_data")
            current_price = analysis_input.get("current_price")

            # Calculate simple liquidity metrics
            liquidity_results = {
                "spread_estimate": self._estimate_spread(market_data),
                "volume_liquidity": self._assess_volume_liquidity(market_data),
                "price_impact": self._estimate_price_impact(market_data),
                "liquidity_score": self._calculate_liquidity_score(market_data),
                "liquidity_regime": self._classify_liquidity_regime(market_data),
                "timestamp": datetime.now().isoformat(),
            }

            self.logger.info("Liquidity analysis completed successfully")
            return liquidity_results

        except Exception as e:
            self.logger.error(f"Error performing liquidity analysis: {e}")
            return None

    def _estimate_spread(self, market_data: pd.DataFrame) -> float:
        """Estimate bid-ask spread using high-low range."""
        try:
            if "high" not in market_data.columns or "low" not in market_data.columns:
                return 0.0

            # Use recent high-low range as spread estimate
            recent_data = market_data.tail(20)
            avg_spread = ((recent_data["high"] - recent_data["low"]) / recent_data["close"]).mean()
            
            return float(avg_spread) if not pd.isna(avg_spread) else 0.0

        except Exception as e:
            self.logger.error(f"Error estimating spread: {e}")
            return 0.0

    def _assess_volume_liquidity(self, market_data: pd.DataFrame) -> str:
        """Assess liquidity based on volume."""
        try:
            if "volume" not in market_data.columns:
                return "UNKNOWN"

            # Calculate volume-based liquidity
            recent_volume = market_data["volume"].tail(20).mean()
            historical_volume = market_data["volume"].mean()
            
            volume_ratio = recent_volume / historical_volume if historical_volume > 0 else 1.0
            
            if volume_ratio > 1.2:
                return "HIGH"
            elif volume_ratio < 0.8:
                return "LOW"
            else:
                return "NORMAL"

        except Exception as e:
            self.logger.error(f"Error assessing volume liquidity: {e}")
            return "UNKNOWN"

    def _estimate_price_impact(self, market_data: pd.DataFrame) -> float:
        """Estimate price impact using volatility and volume."""
        try:
            if "close" not in market_data.columns or "volume" not in market_data.columns:
                return 0.0

            # Simple price impact estimation
            returns = market_data["close"].pct_change().dropna()
            volatility = returns.rolling(window=20).std().iloc[-1]
            volume_ratio = market_data["volume"].iloc[-1] / market_data["volume"].rolling(window=20).mean().iloc[-1]
            
            # Price impact = volatility / volume_ratio
            price_impact = volatility / volume_ratio if volume_ratio > 0 else volatility
            
            return float(price_impact) if not pd.isna(price_impact) else 0.0

        except Exception as e:
            self.logger.error(f"Error estimating price impact: {e}")
            return 0.0

    def _calculate_liquidity_score(self, market_data: pd.DataFrame) -> float:
        """Calculate overall liquidity score (0-100)."""
        try:
            spread = self._estimate_spread(market_data)
            volume_liquidity = self._assess_volume_liquidity(market_data)
            price_impact = self._estimate_price_impact(market_data)
            
            # Convert to numerical scores
            volume_score = {"HIGH": 80, "NORMAL": 60, "LOW": 30, "UNKNOWN": 50}.get(volume_liquidity, 50)
            
            # Calculate composite score
            spread_score = max(0, 100 - (spread * 1000))  # Lower spread = higher score
            impact_score = max(0, 100 - (price_impact * 1000))  # Lower impact = higher score
            
            liquidity_score = (volume_score + spread_score + impact_score) / 3
            
            return max(0.0, min(100.0, float(liquidity_score)))

        except Exception as e:
            self.logger.error(f"Error calculating liquidity score: {e}")
            return 50.0

    def _classify_liquidity_regime(self, market_data: pd.DataFrame) -> str:
        """Classify liquidity regime."""
        try:
            liquidity_score = self._calculate_liquidity_score(market_data)
            
            if liquidity_score >= 80:
                return "HIGH"
            elif liquidity_score >= 60:
                return "NORMAL"
            elif liquidity_score >= 40:
                return "LOW"
            else:
                return "POOR"

        except Exception as e:
            self.logger.error(f"Error classifying liquidity regime: {e}")
            return "UNKNOWN"

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="stress analysis",
    )
    async def _perform_stress_analysis(
        self,
        analysis_input: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Perform simple stress analysis.

        Args:
            analysis_input: Input data for analysis

        Returns:
            dict: Stress analysis results
        """
        try:
            market_data = analysis_input.get("market_data")
            current_price = analysis_input.get("current_price")

            # Calculate simple stress metrics
            stress_results = {
                "volatility_stress": self._calculate_volatility_stress(market_data),
                "drawdown_stress": self._calculate_drawdown_stress(market_data),
                "volume_stress": self._calculate_volume_stress(market_data),
                "stress_score": self._calculate_stress_score(market_data),
                "stress_regime": self._classify_stress_regime(market_data),
                "timestamp": datetime.now().isoformat(),
            }

            self.logger.info("Stress analysis completed successfully")
            return stress_results

        except Exception as e:
            self.logger.error(f"Error performing stress analysis: {e}")
            return None

    def _calculate_volatility_stress(self, market_data: pd.DataFrame) -> float:
        """Calculate volatility-based stress."""
        try:
            if "close" not in market_data.columns:
                return 0.0

            returns = market_data["close"].pct_change().dropna()
            current_vol = returns.rolling(window=20).std().iloc[-1]
            historical_vol = returns.rolling(window=100).std().iloc[-1]
            
            # Stress = current volatility / historical volatility
            vol_stress = current_vol / historical_vol if historical_vol > 0 else 1.0
            
            return float(vol_stress) if not pd.isna(vol_stress) else 1.0

        except Exception as e:
            self.logger.error(f"Error calculating volatility stress: {e}")
            return 1.0

    def _calculate_drawdown_stress(self, market_data: pd.DataFrame) -> float:
        """Calculate drawdown-based stress."""
        try:
            if "close" not in market_data.columns:
                return 0.0

            # Calculate current drawdown
            rolling_max = market_data["close"].rolling(window=20).max()
            current_drawdown = (market_data["close"].iloc[-1] - rolling_max.iloc[-1]) / rolling_max.iloc[-1]
            
            # Convert to stress score (0-1, higher = more stress)
            drawdown_stress = abs(current_drawdown) if not pd.isna(current_drawdown) else 0.0
            
            return float(drawdown_stress)

        except Exception as e:
            self.logger.error(f"Error calculating drawdown stress: {e}")
            return 0.0

    def _calculate_volume_stress(self, market_data: pd.DataFrame) -> float:
        """Calculate volume-based stress."""
        try:
            if "volume" not in market_data.columns:
                return 0.0

            # Calculate volume stress (low volume = high stress)
            recent_volume = market_data["volume"].tail(20).mean()
            historical_volume = market_data["volume"].mean()
            
            volume_stress = 1.0 - (recent_volume / historical_volume) if historical_volume > 0 else 0.0
            
            return max(0.0, min(1.0, float(volume_stress)))

        except Exception as e:
            self.logger.error(f"Error calculating volume stress: {e}")
            return 0.0

    def _calculate_stress_score(self, market_data: pd.DataFrame) -> float:
        """Calculate overall stress score (0-100)."""
        try:
            vol_stress = self._calculate_volatility_stress(market_data)
            drawdown_stress = self._calculate_drawdown_stress(market_data)
            volume_stress = self._calculate_volume_stress(market_data)
            
            # Combine stress factors
            stress_score = (vol_stress * 40) + (drawdown_stress * 40) + (volume_stress * 20)
            
            return max(0.0, min(100.0, float(stress_score)))

        except Exception as e:
            self.logger.error(f"Error calculating stress score: {e}")
            return 50.0

    def _classify_stress_regime(self, market_data: pd.DataFrame) -> str:
        """Classify stress regime."""
        try:
            stress_score = self._calculate_stress_score(market_data)
            
            if stress_score <= 20:
                return "LOW"
            elif stress_score <= 40:
                return "NORMAL"
            elif stress_score <= 60:
                return "HIGH"
            else:
                return "EXTREME"

        except Exception as e:
            self.logger.error(f"Error classifying stress regime: {e}")
            return "UNKNOWN"

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="analysis results storage",
    )
    async def _store_analysis_results(self) -> None:
        """Store analysis results."""
        try:
            self.logger.info("Storing market health analysis results...")
            # Results are already stored in self.analysis_results
            self.logger.info("Analysis results stored successfully")
        except Exception as e:
            self.logger.error(f"Error storing analysis results: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="analysis results getting",
    )
    def get_analysis_results(self, analysis_type: str | None = None) -> dict[str, Any]:
        """
        Get analysis results.

        Args:
            analysis_type: Type of analysis results to retrieve

        Returns:
            dict: Analysis results
        """
        try:
            if analysis_type is None:
                return self.analysis_results
            else:
                return self.analysis_results.get(analysis_type, {})

        except Exception as e:
            self.logger.error(f"Error getting analysis results: {e}")
            return {}

    def get_analysis_status(self) -> dict[str, Any]:
        """Get analysis status."""
        return {
            "is_analyzing": self.is_analyzing,
            "last_analysis": self.analysis_results.get("timestamp"),
            "analysis_count": len(self.analysis_results),
        }

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="market health analyzer cleanup",
    )
    async def stop(self) -> None:
        """Clean up market health analyzer resources."""
        try:
            self.logger.info("Stopping Market Health Analyzer...")
            self.is_analyzing = False
            self.analysis_results = {}
            self.logger.info("✅ Market Health Analyzer stopped successfully")
        except Exception as e:
            self.logger.error(f"❌ Error stopping Market Health Analyzer: {e}")


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="market health analyzer setup",
)
async def setup_market_health_analyzer(
    config: dict[str, Any] | None = None,
) -> MarketHealthAnalyzer | None:
    """
    Setup and initialize Market Health Analyzer.

    Args:
        config: Configuration dictionary

    Returns:
        MarketHealthAnalyzer: Initialized market health analyzer or None if failed
    """
    try:
        if config is None:
            config = {}

        health_analyzer = MarketHealthAnalyzer(config)
        
        if await health_analyzer.initialize():
            system_logger.info("✅ Market Health Analyzer setup completed successfully")
            return health_analyzer
        else:
            system_logger.error("❌ Market Health Analyzer setup failed")
            return None

    except Exception as e:
        system_logger.error(f"❌ Error setting up Market Health Analyzer: {e}")
        return None
