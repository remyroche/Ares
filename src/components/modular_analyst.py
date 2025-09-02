# src/components/modular_analyst.py

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import asyncio
import json
import traceback

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.warning_symbols import error, failed, initialization_error, invalid, missing


class ModularAnalyst:
    """
    Enhanced modular analyst with comprehensive error handling and type safety.
    
    This class provides financial analysis capabilities including technical analysis,
    fundamental analysis, sentiment analysis, and risk assessment.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize the ModularAnalyst with configuration.
        
        Args:
            config: Configuration dictionary containing analyst settings
        """
        self.config: Dict[str, Any] = config
        self.logger = system_logger.getChild("ModularAnalyst")
        
        # Analysis state
        self.is_analyzing: bool = False
        self.analysis_results: Dict[str, Any] = {}
        self.analysis_history: List[Dict[str, Any]] = []
        
        # Configuration
        self.analyst_config: Dict[str, Any] = self.config.get("modular_analyst", {})
        self.analysis_interval: int = self.analyst_config.get("analysis_interval", 60)
        self.max_analysis_history: int = self.analyst_config.get("max_analysis_history", 100)
        self.enable_technical_analysis: bool = self.analyst_config.get("enable_technical_analysis", True)
        self.enable_fundamental_analysis: bool = self.analyst_config.get("enable_fundamental_analysis", True)
        self.enable_sentiment_analysis: bool = self.analyst_config.get("enable_sentiment_analysis", False)
        self.enable_risk_analysis: bool = self.analyst_config.get("enable_risk_analysis", True)
        
        # Analysis modules
        self.technical_analyzer = None
        self.fundamental_analyzer = None
        self.sentiment_analyzer = None
        self.risk_analyzer = None
        
        self.logger.info("ModularAnalyst initialized with configuration")

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid modular analyst configuration"),
            AttributeError: (False, "Missing required analyst parameters"),
            KeyError: (False, "Missing configuration keys"),
        },
        default_return=False,
        context="modular analyst initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize the analyst and all its modules.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing Modular Analyst...")
            
            # Load analyst configuration
            await self._load_analyst_configuration()
            
            # Validate configuration
            if not self._validate_configuration():
                self.logger.error(invalid("Invalid configuration for modular analyst"))
                return False
            
            # Initialize analysis modules
            await self._initialize_analysis_modules()
            
            self.logger.info("✅ Modular Analyst initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(failed(f"❌ Modular Analyst initialization failed: {e}"))
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="analyst configuration loading",
    )
    async def _load_analyst_configuration(self) -> None:
        """
        Load and validate analyst configuration.
        """
        try:
            # Set default analyst parameters
            self.analyst_config.setdefault("analysis_interval", 60)
            self.analyst_config.setdefault("max_analysis_history", 100)
            self.analyst_config.setdefault("enable_technical_analysis", True)
            self.analyst_config.setdefault("enable_fundamental_analysis", True)
            self.analyst_config.setdefault("enable_sentiment_analysis", False)
            self.analyst_config.setdefault("enable_risk_analysis", True)
            
            # Update configuration
            self.analysis_interval = self.analyst_config["analysis_interval"]
            self.max_analysis_history = self.analyst_config["max_analysis_history"]
            self.enable_technical_analysis = self.analyst_config["enable_technical_analysis"]
            self.enable_fundamental_analysis = self.analyst_config["enable_fundamental_analysis"]
            self.enable_sentiment_analysis = self.analyst_config["enable_sentiment_analysis"]
            self.enable_risk_analysis = self.analyst_config["enable_risk_analysis"]
            
            self.logger.info("Analyst configuration loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading analyst configuration: {e}")
            raise

    def _validate_configuration(self) -> bool:
        """
        Validate the analyst configuration.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            required_keys = ["analysis_interval", "max_analysis_history"]
            for key in required_keys:
                if key not in self.analyst_config:
                    self.logger.error(missing(f"Missing required configuration key: {key}"))
                    return False
            
            if self.analysis_interval <= 0:
                self.logger.error(invalid("Analysis interval must be positive"))
                return False
                
            if self.max_analysis_history <= 0:
                self.logger.error(invalid("Max analysis history must be positive"))
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating configuration: {e}")
            return False

    async def _initialize_analysis_modules(self) -> None:
        """
        Initialize all analysis modules based on configuration.
        """
        try:
            if self.enable_technical_analysis:
                self.technical_analyzer = TechnicalAnalyzer(self.analyst_config)
                self.logger.info("Technical analyzer initialized")
            
            if self.enable_fundamental_analysis:
                self.fundamental_analyzer = FundamentalAnalyzer(self.analyst_config)
                self.logger.info("Fundamental analyzer initialized")
            
            if self.enable_sentiment_analysis:
                self.sentiment_analyzer = SentimentAnalyzer(self.analyst_config)
                self.logger.info("Sentiment analyzer initialized")
            
            if self.enable_risk_analysis:
                self.risk_analyzer = RiskAnalyzer(self.analyst_config)
                self.logger.info("Risk analyzer initialized")
                
        except Exception as e:
            self.logger.error(f"Error initializing analysis modules: {e}")
            raise

    @handle_errors(
        exceptions=(ValueError, AttributeError, RuntimeError),
        default_return=None,
        context="financial analysis",
    )
    async def analyze_market(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Perform comprehensive market analysis.
        
        Args:
            market_data: Market data to analyze
            
        Returns:
            Dict containing analysis results or None if analysis fails
        """
        try:
            if self.is_analyzing:
                self.logger.warning("Analysis already in progress")
                return None
            
            self.is_analyzing = True
            self.logger.info("Starting market analysis...")
            
            analysis_result = {
                "timestamp": datetime.now().isoformat(),
                "market_data": market_data,
                "technical_analysis": None,
                "fundamental_analysis": None,
                "sentiment_analysis": None,
                "risk_assessment": None,
                "overall_score": 0.0,
                "recommendations": []
            }
            
            # Perform technical analysis
            if self.technical_analyzer and self.enable_technical_analysis:
                try:
                    analysis_result["technical_analysis"] = await self.technical_analyzer.analyze(market_data)
                except Exception as e:
                    self.logger.error(f"Technical analysis failed: {e}")
            
            # Perform fundamental analysis
            if self.fundamental_analyzer and self.enable_fundamental_analysis:
                try:
                    analysis_result["fundamental_analysis"] = await self.fundamental_analyzer.analyze(market_data)
                except Exception as e:
                    self.logger.error(f"Fundamental analysis failed: {e}")
            
            # Perform sentiment analysis
            if self.sentiment_analyzer and self.enable_sentiment_analysis:
                try:
                    analysis_result["sentiment_analysis"] = await self.sentiment_analyzer.analyze(market_data)
                except Exception as e:
                    self.logger.error(f"Sentiment analysis failed: {e}")
            
            # Perform risk assessment
            if self.risk_analyzer and self.enable_risk_analysis:
                try:
                    analysis_result["risk_assessment"] = await self.risk_analyzer.assess_risk(market_data)
                except Exception as e:
                    self.logger.error(f"Risk assessment failed: {e}")
            
            # Calculate overall score
            analysis_result["overall_score"] = self._calculate_overall_score(analysis_result)
            
            # Generate recommendations
            analysis_result["recommendations"] = self._generate_recommendations(analysis_result)
            
            # Store results
            self.analysis_results = analysis_result
            self._add_to_history(analysis_result)
            
            self.logger.info(f"Market analysis completed. Overall score: {analysis_result['overall_score']:.2f}")
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Market analysis failed: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            return None
            
        finally:
            self.is_analyzing = False

    def _calculate_overall_score(self, analysis_result: Dict[str, Any]) -> float:
        """
        Calculate overall analysis score based on individual analysis results.
        
        Args:
            analysis_result: Analysis results dictionary
            
        Returns:
            float: Overall score between 0.0 and 1.0
        """
        try:
            scores = []
            weights = []
            
            # Technical analysis score
            if analysis_result["technical_analysis"]:
                tech_score = analysis_result["technical_analysis"].get("score", 0.0)
                scores.append(tech_score)
                weights.append(0.3)
            
            # Fundamental analysis score
            if analysis_result["fundamental_analysis"]:
                fund_score = analysis_result["fundamental_analysis"].get("score", 0.0)
                scores.append(fund_score)
                weights.append(0.4)
            
            # Sentiment analysis score
            if analysis_result["sentiment_analysis"]:
                sent_score = analysis_result["sentiment_analysis"].get("score", 0.0)
                scores.append(sent_score)
                weights.append(0.2)
            
            # Risk assessment score (inverted - lower risk = higher score)
            if analysis_result["risk_assessment"]:
                risk_score = 1.0 - analysis_result["risk_assessment"].get("risk_level", 0.5)
                scores.append(risk_score)
                weights.append(0.1)
            
            if not scores:
                return 0.0
            
            # Calculate weighted average
            total_weight = sum(weights)
            weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
            
            return weighted_sum / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating overall score: {e}")
            return 0.0

    def _generate_recommendations(self, analysis_result: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on analysis results.
        
        Args:
            analysis_result: Analysis results dictionary
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        try:
            overall_score = analysis_result.get("overall_score", 0.0)
            
            if overall_score >= 0.8:
                recommendations.append("Strong buy recommendation - market conditions are favorable")
            elif overall_score >= 0.6:
                recommendations.append("Buy recommendation - market shows positive signals")
            elif overall_score >= 0.4:
                recommendations.append("Hold position - market is neutral, monitor for changes")
            elif overall_score >= 0.2:
                recommendations.append("Consider selling - market shows negative signals")
            else:
                recommendations.append("Strong sell recommendation - market conditions are unfavorable")
            
            # Add specific recommendations based on individual analysis
            if analysis_result.get("technical_analysis"):
                tech_analysis = analysis_result["technical_analysis"]
                if tech_analysis.get("trend", "").lower() == "bullish":
                    recommendations.append("Technical indicators show bullish trend")
                elif tech_analysis.get("trend", "").lower() == "bearish":
                    recommendations.append("Technical indicators show bearish trend")
            
            if analysis_result.get("risk_assessment"):
                risk_level = analysis_result["risk_assessment"].get("risk_level", 0.5)
                if risk_level > 0.7:
                    recommendations.append("High risk detected - implement risk management strategies")
                elif risk_level < 0.3:
                    recommendations.append("Low risk environment - consider increasing position sizes")
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            recommendations.append("Unable to generate specific recommendations due to analysis errors")
        
        return recommendations

    def _add_to_history(self, analysis_result: Dict[str, Any]) -> None:
        """
        Add analysis result to history, maintaining maximum history size.
        
        Args:
            analysis_result: Analysis result to add
        """
        try:
            self.analysis_history.append(analysis_result)
            
            # Maintain maximum history size
            if len(self.analysis_history) > self.max_analysis_history:
                self.analysis_history.pop(0)
                
        except Exception as e:
            self.logger.error(f"Error adding to history: {e}")

    def get_analysis_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get analysis history.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of analysis results
        """
        try:
            if limit is None:
                return self.analysis_history.copy()
            else:
                return self.analysis_history[-limit:].copy()
        except Exception as e:
            self.logger.error(f"Error retrieving analysis history: {e}")
            return []

    def get_latest_analysis(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent analysis result.
        
        Returns:
            Latest analysis result or None if no analysis performed
        """
        try:
            if self.analysis_history:
                return self.analysis_history[-1].copy()
            return None
        except Exception as e:
            self.logger.error(f"Error retrieving latest analysis: {e}")
            return None

    def clear_history(self) -> None:
        """Clear analysis history."""
        try:
            self.analysis_history.clear()
            self.logger.info("Analysis history cleared")
        except Exception as e:
            self.logger.error(f"Error clearing history: {e}")

    def get_status(self) -> Dict[str, Any]:
        """
        Get current analyst status.
        
        Returns:
            Dictionary containing current status information
        """
        try:
            return {
                "is_analyzing": self.is_analyzing,
                "analysis_interval": self.analysis_interval,
                "history_size": len(self.analysis_history),
                "max_history_size": self.max_analysis_history,
                "enabled_modules": {
                    "technical_analysis": self.enable_technical_analysis,
                    "fundamental_analysis": self.enable_fundamental_analysis,
                    "sentiment_analysis": self.enable_sentiment_analysis,
                    "risk_analysis": self.enable_risk_analysis
                },
                "last_analysis": self.analysis_history[-1]["timestamp"] if self.analysis_history else None
            }
        except Exception as e:
            self.logger.error(f"Error getting status: {e}")
            return {}


# Placeholder classes for analysis modules
class TechnicalAnalyzer:
    """Placeholder for technical analysis module."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    async def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder technical analysis."""
        return {
            "score": 0.7,
            "trend": "neutral",
            "indicators": {},
            "signals": []
        }


class FundamentalAnalyzer:
    """Placeholder for fundamental analysis module."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    async def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder fundamental analysis."""
        return {
            "score": 0.6,
            "valuation": "fair",
            "metrics": {},
            "outlook": "stable"
        }


class SentimentAnalyzer:
    """Placeholder for sentiment analysis module."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    async def analyze(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder sentiment analysis."""
        return {
            "score": 0.5,
            "sentiment": "neutral",
            "confidence": 0.8,
            "sources": []
        }


class RiskAnalyzer:
    """Placeholder for risk analysis module."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    async def assess_risk(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Placeholder risk assessment."""
        return {
            "risk_level": 0.4,
            "risk_factors": [],
            "mitigation_strategies": [],
            "confidence": 0.7
        }
