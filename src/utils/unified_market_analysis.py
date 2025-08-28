#!/usr/bin/env python3
"""
Unified Market Analysis Module

This module provides a unified interface that integrates centralized S/R logic,
enhanced HMM regime management, and optimized feature engineering into a
cohesive market analysis system.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
import asyncio

from src.utils.logger import system_logger
from src.utils.centralized_decorators import (
    handle_errors,
    validate_data_structure,
    monitor_feature_engineering,
    memory_efficient
)
from src.utils.centralized_sr_logic import CentralizedSRAnalyzer
from src.utils.enhanced_hmm_regime_manager import EnhancedHMMRegimeManager, RegimeState, RegimeTransition
from src.utils.optimized_feature_engineering import OptimizedFeatureEngineering, FeatureCategory


@dataclass
class MarketAnalysisResult:
    """Comprehensive market analysis result."""
    timestamp: datetime
    symbol: str
    exchange: str
    timeframe: str
    
    # Support/Resistance analysis
    sr_analysis: Dict[str, Any]
    
    # Regime analysis
    regime_analysis: Dict[str, Any]
    
    # Feature engineering
    features: pd.DataFrame
    
    # Combined insights
    market_insights: Dict[str, Any]
    
    # Quality metrics
    quality_metrics: Dict[str, Any]


class UnifiedMarketAnalysis:
    """
    Unified market analysis system that integrates S/R logic, regime management,
    and feature engineering into a cohesive analysis framework.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("UnifiedMarketAnalysis")
        
        # Initialize components
        self.sr_analyzer = CentralizedSRAnalyzer(config)
        self.regime_manager = EnhancedHMMRegimeManager(config)
        self.feature_engine = OptimizedFeatureEngineering(config)
        
        # Analysis configuration
        self.enable_sr_analysis = config.get("enable_sr_analysis", True)
        self.enable_regime_analysis = config.get("enable_regime_analysis", True)
        self.enable_feature_engineering = config.get("enable_feature_engineering", True)
        
        # Cache for analysis results
        self._analysis_cache: Dict[str, MarketAnalysisResult] = {}
        self._cache_enabled = config.get("analysis_cache_enabled", True)
        
        # Performance tracking
        self.analysis_timings: Dict[str, float] = {}
        
        self.logger.info("🚀 Unified Market Analysis initialized successfully")
    
    @handle_errors(
        exceptions=(Exception,),
        default_return={"success": False, "error": "Market analysis failed"},
        context="unified_market_analysis"
    )
    @validate_data_structure(required_columns=["open", "high", "low", "close", "volume"])
    @monitor_feature_engineering
    @memory_efficient
    async def analyze_market(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN",
        exchange: str = "UNKNOWN",
        timeframe: str = "1m",
        include_sr: bool = True,
        include_regime: bool = True,
        include_features: bool = True
    ) -> Dict[str, Any]:
        """
        Perform comprehensive market analysis integrating all components.
        
        Args:
            df: OHLCV DataFrame
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            include_sr: Include support/resistance analysis
            include_regime: Include regime analysis
            include_features: Include feature engineering
            
        Returns:
            Comprehensive market analysis results
        """
        if df.empty:
            return {"success": False, "error": "Empty DataFrame provided"}
        
        start_time = datetime.now()
        self.logger.info(f"🎯 Starting unified market analysis for {symbol} on {exchange} ({timeframe})")
        
        try:
            # Generate cache key
            cache_key = self._generate_analysis_cache_key(df, symbol, exchange, timeframe)
            
            # Check cache
            if self._cache_enabled and cache_key in self._analysis_cache:
                cached_result = self._analysis_cache[cache_key]
                self.logger.info("📋 Using cached analysis results")
                return self._format_analysis_result(cached_result)
            
            # Initialize results
            analysis_results = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "timestamp": start_time,
                "success": True
            }
            
            # 1. Support/Resistance Analysis
            if include_sr and self.enable_sr_analysis:
                self.logger.info("📊 Performing S/R analysis...")
                sr_start = datetime.now()
                sr_analysis = await self._perform_sr_analysis(df)
                sr_elapsed = (datetime.now() - sr_start).total_seconds()
                analysis_results["sr_analysis"] = sr_analysis
                analysis_results["sr_analysis_time"] = sr_elapsed
                self.analysis_timings["sr_analysis"] = sr_elapsed
            
            # 2. Regime Analysis
            if include_regime and self.enable_regime_analysis:
                self.logger.info("🧠 Performing regime analysis...")
                regime_start = datetime.now()
                regime_analysis = await self._perform_regime_analysis(df)
                regime_elapsed = (datetime.now() - regime_start).total_seconds()
                analysis_results["regime_analysis"] = regime_analysis
                analysis_results["regime_analysis_time"] = regime_elapsed
                self.analysis_timings["regime_analysis"] = regime_elapsed
            
            # 3. Feature Engineering
            if include_features and self.enable_feature_engineering:
                self.logger.info("🔧 Performing feature engineering...")
                feature_start = datetime.now()
                feature_analysis = await self._perform_feature_engineering(df)
                feature_elapsed = (datetime.now() - feature_start).total_seconds()
                analysis_results["feature_analysis"] = feature_analysis
                analysis_results["feature_analysis_time"] = feature_elapsed
                self.analysis_timings["feature_engineering"] = feature_elapsed
            
            # 4. Generate Market Insights
            self.logger.info("💡 Generating market insights...")
            insights_start = datetime.now()
            market_insights = await self._generate_market_insights(
                analysis_results, df
            )
            insights_elapsed = (datetime.now() - insights_start).total_seconds()
            analysis_results["market_insights"] = market_insights
            analysis_results["insights_generation_time"] = insights_elapsed
            self.analysis_timings["insights_generation"] = insights_elapsed
            
            # 5. Calculate Quality Metrics
            quality_metrics = self._calculate_quality_metrics(analysis_results, df)
            analysis_results["quality_metrics"] = quality_metrics
            
            # 6. Create comprehensive result
            total_elapsed = (datetime.now() - start_time).total_seconds()
            analysis_results["total_analysis_time"] = total_elapsed
            
            # Create MarketAnalysisResult object
            market_result = MarketAnalysisResult(
                timestamp=start_time,
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                sr_analysis=analysis_results.get("sr_analysis", {}),
                regime_analysis=analysis_results.get("regime_analysis", {}),
                features=analysis_results.get("feature_analysis", {}).get("features", pd.DataFrame()),
                market_insights=market_insights,
                quality_metrics=quality_metrics
            )
            
            # Cache result
            if self._cache_enabled:
                self._analysis_cache[cache_key] = market_result
            
            self.logger.info(f"✅ Unified market analysis completed in {total_elapsed:.2f} seconds")
            
            return self._format_analysis_result(market_result)
            
        except Exception as e:
            self.logger.error(f"❌ Error in unified market analysis: {e}")
            return {"success": False, "error": str(e)}
    
    async def _perform_sr_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform support/resistance analysis."""
        try:
            current_price = df['close'].iloc[-1]
            sr_result = self.sr_analyzer.analyze_sr_levels(df, current_price)
            
            if sr_result.get("error"):
                return {"success": False, "error": sr_result["error"]}
            
            # Enhance S/R analysis with additional insights
            enhanced_sr = {
                "success": True,
                "supports": sr_result.get("supports", []),
                "resistances": sr_result.get("resistances", []),
                "current_price": current_price,
                "nearest_support": self._find_nearest_level(sr_result.get("supports", []), current_price, "support"),
                "nearest_resistance": self._find_nearest_level(sr_result.get("resistances", []), current_price, "resistance"),
                "sr_breakout_potential": self._calculate_breakout_potential(sr_result, current_price),
                "sr_strength_distribution": self._analyze_sr_strength_distribution(sr_result),
                "analysis_metadata": sr_result.get("analysis_metadata", {})
            }
            
            return enhanced_sr
            
        except Exception as e:
            self.logger.error(f"❌ Error in S/R analysis: {e}")
            return {"success": False, "error": str(e)}
    
    async def _perform_regime_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform regime analysis."""
        try:
            # Train regime models if needed
            if not self.regime_manager.hmm_model:
                training_result = await self.regime_manager.train_regime_models(df)
                if not training_result.get("success"):
                    return {"success": False, "error": training_result.get("error")}
            
            # Predict regime changes
            prediction_result = await self.regime_manager.predict_regime_changes(df)
            
            if not prediction_result.get("success"):
                return {"success": False, "error": prediction_result.get("error")}
            
            # Get regime summary
            regime_summary = self.regime_manager.get_regime_summary()
            
            # Enhance regime analysis
            enhanced_regime = {
                "success": True,
                "current_regime": prediction_result.get("current_regime"),
                "regime_changes": prediction_result.get("regime_changes", []),
                "transition_probabilities": prediction_result.get("transition_probabilities", []),
                "hmm_states": prediction_result.get("hmm_states", []),
                "cluster_labels": prediction_result.get("cluster_labels", []),
                "regime_summary": regime_summary,
                "regime_stability": self._calculate_regime_stability(prediction_result),
                "regime_transition_risk": self._calculate_transition_risk(prediction_result)
            }
            
            return enhanced_regime
            
        except Exception as e:
            self.logger.error(f"❌ Error in regime analysis: {e}")
            return {"success": False, "error": str(e)}
    
    async def _perform_feature_engineering(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform feature engineering."""
        try:
            # Generate comprehensive features
            features = await self.feature_engine.generate_features(
                df,
                include_sr_analysis=True,
                include_regime_analysis=True
            )
            
            if features.empty:
                return {"success": False, "error": "No features generated"}
            
            # Get feature summary
            feature_summary = self.feature_engine.get_feature_summary()
            
            # Analyze feature quality
            feature_quality = self._analyze_feature_quality(features)
            
            enhanced_features = {
                "success": True,
                "features": features,
                "feature_summary": feature_summary,
                "feature_quality": feature_quality,
                "feature_categories": self._categorize_features(features),
                "feature_importance": self._estimate_feature_importance(features)
            }
            
            return enhanced_features
            
        except Exception as e:
            self.logger.error(f"❌ Error in feature engineering: {e}")
            return {"success": False, "error": str(e)}
    
    async def _generate_market_insights(
        self, 
        analysis_results: Dict[str, Any], 
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate comprehensive market insights from all analyses."""
        try:
            insights = {
                "market_condition": "UNKNOWN",
                "trading_opportunities": [],
                "risk_factors": [],
                "key_levels": [],
                "regime_insights": [],
                "feature_insights": [],
                "overall_sentiment": "NEUTRAL",
                "confidence_score": 0.0
            }
            
            # Analyze S/R insights
            sr_analysis = analysis_results.get("sr_analysis", {})
            if sr_analysis.get("success"):
                sr_insights = self._extract_sr_insights(sr_analysis, df)
                insights.update(sr_insights)
            
            # Analyze regime insights
            regime_analysis = analysis_results.get("regime_analysis", {})
            if regime_analysis.get("success"):
                regime_insights = self._extract_regime_insights(regime_analysis, df)
                insights.update(regime_insights)
            
            # Analyze feature insights
            feature_analysis = analysis_results.get("feature_analysis", {})
            if feature_analysis.get("success"):
                feature_insights = self._extract_feature_insights(feature_analysis, df)
                insights.update(feature_insights)
            
            # Generate overall market condition
            insights["market_condition"] = self._determine_market_condition(insights)
            insights["overall_sentiment"] = self._determine_sentiment(insights)
            insights["confidence_score"] = self._calculate_confidence_score(insights)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"❌ Error generating market insights: {e}")
            return {"error": str(e)}
    
    def _find_nearest_level(self, levels: List[Dict], current_price: float, level_type: str) -> Optional[Dict]:
        """Find the nearest support or resistance level."""
        if not levels:
            return None
        
        if level_type == "support":
            valid_levels = [level for level in levels if level['price'] < current_price]
            if valid_levels:
                return max(valid_levels, key=lambda x: x['price'])
        else:  # resistance
            valid_levels = [level for level in levels if level['price'] > current_price]
            if valid_levels:
                return min(valid_levels, key=lambda x: x['price'])
        
        return None
    
    def _calculate_breakout_potential(self, sr_result: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """Calculate breakout potential based on S/R levels."""
        try:
            supports = sr_result.get("supports", [])
            resistances = sr_result.get("resistances", [])
            
            # Find nearest levels
            nearest_support = self._find_nearest_level(supports, current_price, "support")
            nearest_resistance = self._find_nearest_level(resistances, current_price, "resistance")
            
            # Calculate distances
            support_distance = (current_price - nearest_support['price']) / current_price if nearest_support else 1.0
            resistance_distance = (nearest_resistance['price'] - current_price) / current_price if nearest_resistance else 1.0
            
            # Determine breakout potential
            if support_distance < 0.01:  # Very close to support
                breakout_potential = "support_breakdown_risk"
                confidence = nearest_support.get('confidence', 0.5)
            elif resistance_distance < 0.01:  # Very close to resistance
                breakout_potential = "resistance_breakout_potential"
                confidence = nearest_resistance.get('confidence', 0.5)
            else:
                breakout_potential = "no_immediate_breakout"
                confidence = 0.0
            
            return {
                "breakout_type": breakout_potential,
                "confidence": confidence,
                "support_distance": support_distance,
                "resistance_distance": resistance_distance,
                "nearest_support": nearest_support,
                "nearest_resistance": nearest_resistance
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating breakout potential: {e}")
            return {"breakout_type": "unknown", "confidence": 0.0}
    
    def _analyze_sr_strength_distribution(self, sr_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the distribution of S/R strengths."""
        try:
            all_levels = sr_result.get("supports", []) + sr_result.get("resistances", [])
            
            if not all_levels:
                return {"error": "No S/R levels found"}
            
            strengths = [level.get('strength', 0) for level in all_levels]
            
            return {
                "mean_strength": np.mean(strengths),
                "median_strength": np.median(strengths),
                "std_strength": np.std(strengths),
                "strong_levels": len([s for s in strengths if s > 0.7]),
                "weak_levels": len([s for s in strengths if s < 0.3]),
                "total_levels": len(all_levels)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing S/R strength distribution: {e}")
            return {"error": str(e)}
    
    def _calculate_regime_stability(self, prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate regime stability metrics."""
        try:
            cluster_labels = prediction_result.get("cluster_labels", [])
            
            if len(cluster_labels) < 2:
                return {"stability_score": 0.0, "regime_changes": 0}
            
            # Count regime changes
            regime_changes = sum(1 for i in range(1, len(cluster_labels)) if cluster_labels[i] != cluster_labels[i-1])
            
            # Calculate stability score (higher = more stable)
            stability_score = 1.0 - (regime_changes / (len(cluster_labels) - 1))
            
            return {
                "stability_score": stability_score,
                "regime_changes": regime_changes,
                "total_periods": len(cluster_labels),
                "stability_level": "high" if stability_score > 0.8 else "medium" if stability_score > 0.5 else "low"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating regime stability: {e}")
            return {"stability_score": 0.0, "error": str(e)}
    
    def _calculate_transition_risk(self, prediction_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate regime transition risk."""
        try:
            transition_probs = prediction_result.get("transition_probabilities", [])
            
            if not transition_probs:
                return {"transition_risk": 0.0, "risk_level": "low"}
            
            # Use recent transition probabilities
            recent_probs = transition_probs[-10:] if len(transition_probs) >= 10 else transition_probs
            avg_transition_prob = np.mean(recent_probs)
            
            # Determine risk level
            if avg_transition_prob > 0.7:
                risk_level = "high"
            elif avg_transition_prob > 0.4:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            return {
                "transition_risk": avg_transition_prob,
                "risk_level": risk_level,
                "recent_transitions": len([p for p in recent_probs if p > 0.5])
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating transition risk: {e}")
            return {"transition_risk": 0.0, "error": str(e)}
    
    def _analyze_feature_quality(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the quality of generated features."""
        try:
            quality_metrics = {
                "total_features": len(features.columns),
                "feature_completeness": 1.0 - (features.isna().sum().sum() / (len(features) * len(features.columns))),
                "feature_variance": features.var().mean(),
                "feature_correlation_mean": features.corr().abs().mean().mean(),
                "low_variance_features": len(features.columns[features.var() < 0.01]),
                "high_correlation_pairs": len(features.corr().abs()[features.corr().abs() > 0.95].stack()) // 2
            }
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing feature quality: {e}")
            return {"error": str(e)}
    
    def _categorize_features(self, features: pd.DataFrame) -> Dict[str, List[str]]:
        """Categorize features by type."""
        try:
            categories = {
                "price": [],
                "volume": [],
                "volatility": [],
                "momentum": [],
                "technical": [],
                "sr": [],
                "regime": [],
                "interaction": [],
                "wavelet": [],
                "statistical": []
            }
            
            for col in features.columns:
                col_lower = col.lower()
                if any(term in col_lower for term in ["price", "close", "high", "low", "open"]):
                    categories["price"].append(col)
                elif "volume" in col_lower:
                    categories["volume"].append(col)
                elif "volatility" in col_lower or "vol" in col_lower:
                    categories["volatility"].append(col)
                elif any(term in col_lower for term in ["rsi", "macd", "stochastic", "momentum"]):
                    categories["momentum"].append(col)
                elif any(term in col_lower for term in ["bb_", "atr", "adx", "cci", "technical"]):
                    categories["technical"].append(col)
                elif "sr_" in col_lower or "support" in col_lower or "resistance" in col_lower:
                    categories["sr"].append(col)
                elif "regime" in col_lower or "hmm" in col_lower or "cluster" in col_lower:
                    categories["regime"].append(col)
                elif "interaction" in col_lower or "trend" in col_lower:
                    categories["interaction"].append(col)
                elif "wavelet" in col_lower:
                    categories["wavelet"].append(col)
                elif any(term in col_lower for term in ["zscore", "skewness", "kurtosis", "percentile"]):
                    categories["statistical"].append(col)
                else:
                    categories["technical"].append(col)
            
            return categories
            
        except Exception as e:
            self.logger.error(f"❌ Error categorizing features: {e}")
            return {}
    
    def _estimate_feature_importance(self, features: pd.DataFrame) -> Dict[str, float]:
        """Estimate feature importance using variance and correlation."""
        try:
            # Simple importance estimation based on variance and correlation
            variance = features.var()
            correlation_mean = features.corr().abs().mean()
            
            # Combine variance and inverse correlation for importance
            importance = variance * (1 - correlation_mean)
            importance = importance / importance.sum()  # Normalize
            
            return importance.to_dict()
            
        except Exception as e:
            self.logger.error(f"❌ Error estimating feature importance: {e}")
            return {}
    
    def _extract_sr_insights(self, sr_analysis: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Extract insights from S/R analysis."""
        try:
            insights = {}
            
            # Key levels
            supports = sr_analysis.get("supports", [])
            resistances = sr_analysis.get("resistances", [])
            
            insights["key_levels"] = {
                "strong_supports": [s for s in supports if s.get('strength', 0) > 0.7],
                "strong_resistances": [r for r in resistances if r.get('strength', 0) > 0.7],
                "nearest_support": sr_analysis.get("nearest_support"),
                "nearest_resistance": sr_analysis.get("nearest_resistance")
            }
            
            # Trading opportunities
            breakout_potential = sr_analysis.get("sr_breakout_potential", {})
            if breakout_potential.get("breakout_type") != "no_immediate_breakout":
                insights["trading_opportunities"].append({
                    "type": breakout_potential.get("breakout_type"),
                    "confidence": breakout_potential.get("confidence", 0),
                    "description": f"Potential {breakout_potential.get('breakout_type')} near key level"
                })
            
            return insights
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting S/R insights: {e}")
            return {}
    
    def _extract_regime_insights(self, regime_analysis: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Extract insights from regime analysis."""
        try:
            insights = {}
            
            current_regime = regime_analysis.get("current_regime")
            if current_regime:
                insights["regime_insights"] = [{
                    "type": "current_regime",
                    "regime_id": current_regime.regime_id,
                    "regime_type": current_regime.regime_type.value,
                    "confidence": current_regime.confidence,
                    "duration": current_regime.duration
                }]
            
            # Regime stability
            stability = regime_analysis.get("regime_stability", {})
            if stability.get("stability_level") == "low":
                insights["risk_factors"].append({
                    "type": "regime_instability",
                    "description": "Market regime is unstable",
                    "severity": "medium"
                })
            
            # Transition risk
            transition_risk = regime_analysis.get("regime_transition_risk", {})
            if transition_risk.get("risk_level") == "high":
                insights["risk_factors"].append({
                    "type": "regime_transition",
                    "description": "High probability of regime change",
                    "severity": "high"
                })
            
            return insights
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting regime insights: {e}")
            return {}
    
    def _extract_feature_insights(self, feature_analysis: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Extract insights from feature analysis."""
        try:
            insights = {}
            
            features = feature_analysis.get("features", pd.DataFrame())
            if not features.empty:
                # Analyze feature patterns
                feature_quality = feature_analysis.get("feature_quality", {})
                
                if feature_quality.get("low_variance_features", 0) > len(features.columns) * 0.1:
                    insights["risk_factors"].append({
                        "type": "low_feature_quality",
                        "description": "Many features have low variance",
                        "severity": "low"
                    })
                
                if feature_quality.get("high_correlation_pairs", 0) > len(features.columns) * 0.2:
                    insights["risk_factors"].append({
                        "type": "feature_redundancy",
                        "description": "Many features are highly correlated",
                        "severity": "medium"
                    })
            
            return insights
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting feature insights: {e}")
            return {}
    
    def _determine_market_condition(self, insights: Dict[str, Any]) -> str:
        """Determine overall market condition."""
        try:
            # Analyze regime insights
            regime_insights = insights.get("regime_insights", [])
            for insight in regime_insights:
                if insight.get("type") == "current_regime":
                    regime_type = insight.get("regime_type", "")
                    if "bull" in regime_type:
                        return "BULLISH"
                    elif "bear" in regime_type:
                        return "BEARISH"
                    elif "volatile" in regime_type:
                        return "VOLATILE"
                    elif "sideways" in regime_type:
                        return "SIDEWAYS"
            
            # Analyze S/R insights
            key_levels = insights.get("key_levels", {})
            if key_levels.get("nearest_support") and key_levels.get("nearest_resistance"):
                return "RANGING"
            
            return "NEUTRAL"
            
        except Exception as e:
            self.logger.error(f"❌ Error determining market condition: {e}")
            return "UNKNOWN"
    
    def _determine_sentiment(self, insights: Dict[str, Any]) -> str:
        """Determine market sentiment."""
        try:
            # Count positive and negative factors
            positive_factors = 0
            negative_factors = 0
            
            # Analyze trading opportunities
            opportunities = insights.get("trading_opportunities", [])
            for opp in opportunities:
                if "breakout" in opp.get("type", ""):
                    positive_factors += 1
            
            # Analyze risk factors
            risk_factors = insights.get("risk_factors", [])
            for risk in risk_factors:
                if risk.get("severity") in ["high", "medium"]:
                    negative_factors += 1
            
            # Determine sentiment
            if positive_factors > negative_factors:
                return "BULLISH"
            elif negative_factors > positive_factors:
                return "BEARISH"
            else:
                return "NEUTRAL"
                
        except Exception as e:
            self.logger.error(f"❌ Error determining sentiment: {e}")
            return "NEUTRAL"
    
    def _calculate_confidence_score(self, insights: Dict[str, Any]) -> float:
        """Calculate overall confidence score."""
        try:
            confidence_factors = []
            
            # S/R confidence
            key_levels = insights.get("key_levels", {})
            if key_levels.get("nearest_support"):
                confidence_factors.append(key_levels["nearest_support"].get("confidence", 0))
            if key_levels.get("nearest_resistance"):
                confidence_factors.append(key_levels["nearest_resistance"].get("confidence", 0))
            
            # Regime confidence
            regime_insights = insights.get("regime_insights", [])
            for insight in regime_insights:
                if insight.get("type") == "current_regime":
                    confidence_factors.append(insight.get("confidence", 0))
            
            # Feature quality confidence
            feature_quality = insights.get("feature_quality", {})
            if feature_quality:
                completeness = feature_quality.get("feature_completeness", 0)
                confidence_factors.append(completeness)
            
            return np.mean(confidence_factors) if confidence_factors else 0.0
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating confidence score: {e}")
            return 0.0
    
    def _calculate_quality_metrics(self, analysis_results: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate overall quality metrics."""
        try:
            quality_metrics = {
                "data_quality": {
                    "completeness": 1.0 - (df.isna().sum().sum() / (len(df) * len(df.columns))),
                    "consistency": self._check_data_consistency(df),
                    "freshness": self._check_data_freshness(df)
                },
                "analysis_quality": {
                    "sr_analysis_success": analysis_results.get("sr_analysis", {}).get("success", False),
                    "regime_analysis_success": analysis_results.get("regime_analysis", {}).get("success", False),
                    "feature_analysis_success": analysis_results.get("feature_analysis", {}).get("success", False)
                },
                "performance_metrics": {
                    "total_time": analysis_results.get("total_analysis_time", 0),
                    "component_times": self.analysis_timings
                }
            }
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating quality metrics: {e}")
            return {"error": str(e)}
    
    def _check_data_consistency(self, df: pd.DataFrame) -> float:
        """Check data consistency."""
        try:
            # Check for price consistency
            price_checks = []
            for i in range(1, len(df)):
                high = df['high'].iloc[i]
                low = df['low'].iloc[i]
                open_price = df['open'].iloc[i]
                close = df['close'].iloc[i]
                
                # Check if high >= low
                price_checks.append(high >= low)
                # Check if high >= open and high >= close
                price_checks.append(high >= open_price and high >= close)
                # Check if low <= open and low <= close
                price_checks.append(low <= open_price and low <= close)
            
            return np.mean(price_checks) if price_checks else 0.0
            
        except Exception:
            return 0.0
    
    def _check_data_freshness(self, df: pd.DataFrame) -> float:
        """Check data freshness."""
        try:
            # This would typically check timestamps
            # For now, return a default value
            return 1.0
        except Exception:
            return 0.0
    
    def _generate_analysis_cache_key(self, df: pd.DataFrame, symbol: str, exchange: str, timeframe: str) -> str:
        """Generate cache key for analysis."""
        try:
            key_data = f"{symbol}_{exchange}_{timeframe}_{df.shape}_{df['close'].iloc[-1]:.6f}"
            return str(hash(key_data))
        except Exception:
            return str(hash(f"{symbol}_{exchange}_{timeframe}"))
    
    def _format_analysis_result(self, result: MarketAnalysisResult) -> Dict[str, Any]:
        """Format analysis result for output."""
        try:
            return {
                "success": True,
                "timestamp": result.timestamp.isoformat(),
                "symbol": result.symbol,
                "exchange": result.exchange,
                "timeframe": result.timeframe,
                "sr_analysis": result.sr_analysis,
                "regime_analysis": result.regime_analysis,
                "features_shape": result.features.shape if not result.features.empty else (0, 0),
                "market_insights": result.market_insights,
                "quality_metrics": result.quality_metrics
            }
        except Exception as e:
            self.logger.error(f"❌ Error formatting analysis result: {e}")
            return {"success": False, "error": str(e)}
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of analysis system."""
        try:
            return {
                "components": {
                    "sr_analyzer": self.sr_analyzer is not None,
                    "regime_manager": self.regime_manager is not None,
                    "feature_engine": self.feature_engine is not None
                },
                "cache_stats": {
                    "cache_enabled": self._cache_enabled,
                    "cache_size": len(self._analysis_cache)
                },
                "performance": {
                    "average_times": {k: np.mean(v) for k, v in self.analysis_timings.items()},
                    "total_analyses": len(self.analysis_timings)
                }
            }
        except Exception as e:
            self.logger.error(f"❌ Error getting analysis summary: {e}")
            return {"error": str(e)}
    
    def clear_cache(self) -> None:
        """Clear analysis cache."""
        self._analysis_cache.clear()
        self.logger.info("Analysis cache cleared")