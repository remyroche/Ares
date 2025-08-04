# src/training/enhanced_optimization_orchestrator.py

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import warnings

from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.logger import system_logger
from src.training.multi_objective_optimizer import MultiObjectiveOptimizer
from src.training.bayesian_optimizer import AdvancedBayesianOptimizer
from src.training.adaptive_optimizer import AdaptiveOptimizer
from src.config import CONFIG


class EnhancedOptimizationOrchestrator:
    """
    Orchestrates multiple advanced hyperparameter optimization techniques.
    
    Combines:
    - Multi-objective optimization
    - Bayesian optimization with advanced sampling
    - Adaptive optimization based on market regimes
    - Performance tracking and analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("EnhancedOptimizationOrchestrator")
        
        # Initialize optimizers
        self.multi_objective_optimizer = None
        self.bayesian_optimizer = None
        self.adaptive_optimizer = None
        
        # Optimization results tracking
        self.optimization_history = []
        self.performance_metrics = {}
        self.best_parameters = {}
        
        # Initialize optimizers based on configuration
        self._initialize_optimizers()

    def _initialize_optimizers(self):
        """Initialize optimization components based on configuration."""
        
        opt_config = self.config.get("hyperparameter_optimization", {})
        
        # Initialize multi-objective optimizer
        if opt_config.get("multi_objective", {}).get("enabled", False):
            self.multi_objective_optimizer = MultiObjectiveOptimizer(opt_config)
            self.logger.info("Multi-objective optimizer initialized")
        
        # Initialize Bayesian optimizer
        if opt_config.get("bayesian_optimization", {}).get("enabled", False):
            self.bayesian_optimizer = AdvancedBayesianOptimizer(opt_config)
            self.logger.info("Bayesian optimizer initialized")
        
        # Initialize adaptive optimizer
        if opt_config.get("adaptive_optimization", {}).get("enabled", False):
            self.adaptive_optimizer = AdaptiveOptimizer(opt_config)
            self.logger.info("Adaptive optimizer initialized")

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="enhanced optimization orchestration"
    )
    async def run_comprehensive_optimization(self, market_data: pd.DataFrame, 
                                          optimization_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Run comprehensive hyperparameter optimization using multiple techniques.
        
        Args:
            market_data: Historical market data for optimization
            optimization_type: Type of optimization to run
                - "comprehensive": Run all optimizers
                - "multi_objective": Only multi-objective optimization
                - "bayesian": Only Bayesian optimization
                - "adaptive": Only adaptive optimization
                - "quick": Quick optimization with reduced trials
        """
        
        self.logger.info(f"Starting {optimization_type} optimization...")
        
        results = {
            "optimization_type": optimization_type,
            "timestamp": datetime.now(),
            "results": {},
            "summary": {}
        }
        
        try:
            # Run different optimization strategies based on type
            if optimization_type == "comprehensive":
                results["results"] = await self._run_comprehensive_optimization(market_data)
            elif optimization_type == "multi_objective":
                results["results"] = await self._run_multi_objective_optimization(market_data)
            elif optimization_type == "bayesian":
                results["results"] = await self._run_bayesian_optimization(market_data)
            elif optimization_type == "adaptive":
                results["results"] = await self._run_adaptive_optimization(market_data)
            elif optimization_type == "quick":
                results["results"] = await self._run_quick_optimization(market_data)
            else:
                raise ValueError(f"Unknown optimization type: {optimization_type}")
            
            # Analyze and summarize results
            results["summary"] = self._analyze_optimization_results(results["results"])
            
            # Update optimization history
            self.optimization_history.append(results)
            
            self.logger.info(f"{optimization_type} optimization completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in {optimization_type} optimization: {e}")
            results["error"] = str(e)
        
        return results

    async def _run_comprehensive_optimization(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Run all optimization techniques and combine results."""
        
        results = {}
        
        # Run multi-objective optimization
        if self.multi_objective_optimizer:
            self.logger.info("Running multi-objective optimization...")
            try:
                mo_results = await self._run_multi_objective_optimization(market_data)
                results["multi_objective"] = mo_results
            except Exception as e:
                self.logger.warning(f"Multi-objective optimization failed: {e}")
        
        # Run Bayesian optimization
        if self.bayesian_optimizer:
            self.logger.info("Running Bayesian optimization...")
            try:
                bayes_results = await self._run_bayesian_optimization(market_data)
                results["bayesian"] = bayes_results
            except Exception as e:
                self.logger.warning(f"Bayesian optimization failed: {e}")
        
        # Run adaptive optimization
        if self.adaptive_optimizer:
            self.logger.info("Running adaptive optimization...")
            try:
                adaptive_results = await self._run_adaptive_optimization(market_data)
                results["adaptive"] = adaptive_results
            except Exception as e:
                self.logger.warning(f"Adaptive optimization failed: {e}")
        
        # Combine and rank results
        results["combined"] = self._combine_optimization_results(results)
        
        return results

    async def _run_multi_objective_optimization(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Run multi-objective optimization."""
        
        if not self.multi_objective_optimizer:
            raise ValueError("Multi-objective optimizer not initialized")
        
        # Run optimization
        results = self.multi_objective_optimizer.run_optimization(n_trials=300)
        
        return {
            "best_params": results["best_params"],
            "pareto_front": results["pareto_front"],
            "optimization_metrics": self._extract_optimization_metrics(results)
        }

    async def _run_bayesian_optimization(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Run Bayesian optimization."""
        
        if not self.bayesian_optimizer:
            raise ValueError("Bayesian optimizer not initialized")
        
        # Run optimization
        results = self.bayesian_optimizer.run_optimization()
        
        return {
            "best_params": results["best_params"],
            "best_score": results["best_score"],
            "parameter_importance": results["parameter_importance"],
            "convergence_metrics": results["convergence_metrics"]
        }

    async def _run_adaptive_optimization(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Run adaptive optimization based on market regimes."""
        
        if not self.adaptive_optimizer:
            raise ValueError("Adaptive optimizer not initialized")
        
        # Detect market regime
        regime = self.adaptive_optimizer.detect_market_regime(market_data)
        
        # Optimize for detected regime
        results = self.adaptive_optimizer.optimize_for_regime(regime, market_data)
        
        return {
            "detected_regime": regime.name,
            "regime_confidence": regime.confidence,
            "best_params": results["best_params"],
            "best_score": results["best_score"],
            "regime_insights": self.adaptive_optimizer.get_regime_insights()
        }

    async def _run_quick_optimization(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Run quick optimization with reduced trials."""
        
        results = {}
        
        # Quick Bayesian optimization
        if self.bayesian_optimizer:
            quick_config = self.config.copy()
            quick_config["hyperparameter_optimization"]["bayesian_optimization"]["max_trials"] = 50
            
            quick_bayes = AdvancedBayesianOptimizer(quick_config)
            bayes_results = quick_bayes.run_optimization()
            results["bayesian"] = bayes_results
        
        # Quick adaptive optimization
        if self.adaptive_optimizer:
            regime = self.adaptive_optimizer.detect_market_regime(market_data)
            adaptive_results = self.adaptive_optimizer.optimize_for_regime(regime, market_data)
            results["adaptive"] = adaptive_results
        
        return results

    def _combine_optimization_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Combine results from different optimization techniques."""
        
        combined_results = {
            "best_parameters": {},
            "performance_comparison": {},
            "recommended_approach": None
        }
        
        # Extract best parameters from each method
        if "multi_objective" in results:
            combined_results["best_parameters"]["multi_objective"] = results["multi_objective"]["best_params"]
        
        if "bayesian" in results:
            combined_results["best_parameters"]["bayesian"] = results["bayesian"]["best_params"]
        
        if "adaptive" in results:
            combined_results["best_parameters"]["adaptive"] = results["adaptive"]["best_params"]
        
        # Compare performance
        performance_scores = {}
        for method, result in results.items():
            if method in ["multi_objective", "bayesian", "adaptive"]:
                if "best_score" in result:
                    performance_scores[method] = result["best_score"]
                elif "optimization_metrics" in result:
                    # Use weighted score for multi-objective
                    metrics = result["optimization_metrics"]
                    performance_scores[method] = metrics.get("weighted_score", 0)
        
        # Determine recommended approach
        if performance_scores:
            best_method = max(performance_scores, key=performance_scores.get)
            combined_results["recommended_approach"] = best_method
            combined_results["performance_comparison"] = performance_scores
        
        return combined_results

    def _extract_optimization_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from optimization results."""
        
        metrics = {}
        
        if "pareto_front" in results:
            pareto_front = results["pareto_front"]
            if pareto_front:
                # Calculate metrics from Pareto front
                scores = [solution.get("weighted_score", 0) for solution in pareto_front]
                metrics["best_score"] = max(scores) if scores else 0
                metrics["avg_score"] = np.mean(scores) if scores else 0
                metrics["pareto_front_size"] = len(pareto_front)
        
        return metrics

    def _analyze_optimization_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze and summarize optimization results."""
        
        summary = {
            "total_optimizations": len(results),
            "successful_optimizations": 0,
            "best_overall_score": -np.inf,
            "recommended_parameters": {},
            "optimization_insights": {}
        }
        
        # Analyze each optimization result
        for method, result in results.items():
            if method in ["multi_objective", "bayesian", "adaptive"]:
                if result and "best_params" in result:
                    summary["successful_optimizations"] += 1
                    
                    # Track best score
                    if "best_score" in result:
                        score = result["best_score"]
                        if score > summary["best_overall_score"]:
                            summary["best_overall_score"] = score
                            summary["recommended_parameters"] = result["best_params"]
                    
                    # Collect insights
                    summary["optimization_insights"][method] = {
                        "parameters_found": len(result["best_params"]),
                        "optimization_quality": self._assess_optimization_quality(result)
                    }
        
        return summary

    def _assess_optimization_quality(self, result: Dict[str, Any]) -> str:
        """Assess the quality of optimization results."""
        
        if "best_score" in result:
            score = result["best_score"]
            if score > 0.8:
                return "excellent"
            elif score > 0.6:
                return "good"
            elif score > 0.4:
                return "fair"
            else:
                return "poor"
        
        return "unknown"

    def get_optimization_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get optimization history."""
        
        if limit:
            return self.optimization_history[-limit:]
        
        return self.optimization_history

    def get_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        
        if not self.optimization_history:
            return {"message": "No optimization history available"}
        
        trends = {
            "score_trend": [],
            "parameter_stability": {},
            "optimization_frequency": {}
        }
        
        # Analyze score trends
        for result in self.optimization_history:
            if "summary" in result and "best_overall_score" in result["summary"]:
                trends["score_trend"].append({
                    "timestamp": result["timestamp"],
                    "score": result["summary"]["best_overall_score"]
                })
        
        return trends

    @handle_errors(
        exceptions=(Exception,),
        default_return=None,
        context="scheduled optimization"
    )
    async def run_scheduled_optimization(self, schedule_type: str = "daily") -> Dict[str, Any]:
        """Run optimization based on schedule."""
        
        schedule_config = self.config.get("hyperparameter_optimization", {}).get("optimization_schedules", {})
        schedule = schedule_config.get(schedule_type, {})
        
        if not schedule.get("enabled", False):
            return {"message": f"{schedule_type} optimization not enabled"}
        
        # Determine optimization type based on schedule
        focus = schedule.get("focus", "comprehensive")
        max_trials = schedule.get("max_trials", 100)
        
        # Load market data (this would integrate with your data loading)
        market_data = self._load_market_data_for_optimization()
        
        # Run optimization
        results = await self.run_comprehensive_optimization(
            market_data=market_data,
            optimization_type=focus
        )
        
        return results

    def _load_market_data_for_optimization(self) -> pd.DataFrame:
        """Load market data for optimization (placeholder)."""
        
        # This would integrate with your existing data loading infrastructure
        # For now, returning mock data
        dates = pd.date_range(start="2024-01-01", end="2024-12-31", freq="1H")
        
        mock_data = pd.DataFrame({
            "timestamp": dates,
            "open": np.random.normal(100, 10, len(dates)),
            "high": np.random.normal(105, 10, len(dates)),
            "low": np.random.normal(95, 10, len(dates)),
            "close": np.random.normal(100, 10, len(dates)),
            "volume": np.random.normal(1000, 200, len(dates))
        })
        
        return mock_data 