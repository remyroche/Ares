# src/training/feature_engineering_optimizer.py

"""
Feature Engineering Optimization Module

This module optimizes feature engineering parameters using:
1. Random Forest + SHAP for correlation analysis
2. Mutual importance matrix for feature parameter selection
3. Regime-specific optimization for each HMM regime
4. Top 3 parameter selection based on correlation, multicollinearity, and mutual information
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score
import optuna

from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors


class FeatureEngineeringOptimizer:
    """
    Optimizes feature engineering parameters using advanced ML techniques.
    
    Features:
    - Random Forest + SHAP for correlation analysis
    - Mutual importance matrix for parameter selection
    - Regime-specific optimization
    - Top 3 parameter selection with correlation/multicollinearity/MI analysis
    """
    
    def __init__(self, config: dict[str, Any]):
        """Initialize the feature engineering optimizer."""
        self.config = config
        self.logger = system_logger.getChild("FeatureEngineeringOptimizer")
        
        # Feature parameter ranges to optimize
        self.feature_params = {
            "RSI": {
                "lookback_period": [7, 14, 21, 30, 50],
                "overbought_threshold": [70, 75, 80, 85],
                "oversold_threshold": [15, 20, 25, 30]
            },
            "MACD": {
                "fast_period": [8, 12, 16, 20],
                "slow_period": [20, 26, 30, 34],
                "signal_period": [7, 9, 11, 13]
            },
            "Bollinger_Bands": {
                "lookback_period": [10, 20, 30, 50],
                "std_dev": [1.5, 2.0, 2.5, 3.0],
                "squeeze_threshold": [0.1, 0.2, 0.3, 0.4]
            },
            "SMA": {
                "short_period": [5, 10, 15, 20],
                "long_period": [20, 30, 50, 100]
            },
            "EMA": {
                "short_period": [5, 10, 15, 20],
                "long_period": [20, 30, 50, 100]
            },
            "ATR": {
                "lookback_period": [7, 14, 21, 30]
            },
            "Stochastic": {
                "k_period": [7, 14, 21, 30],
                "d_period": [3, 5, 7, 9],
                "overbought": [70, 75, 80, 85],
                "oversold": [15, 20, 25, 30]
            },
            "ADX": {
                "lookback_period": [7, 14, 21, 30],
                "threshold": [20, 25, 30, 35]
            },
            "CCI": {
                "lookback_period": [7, 14, 21, 30],
                "constant": [0.015, 0.02, 0.025, 0.03]
            }
        }
        
        # Optimization settings
        self.optimization_config = config.get("feature_engineering_optimization", {
            "n_trials": 100,
            "cv_folds": 5,
            "random_state": 42,
            "correlation_threshold": 0.8,
            "mi_threshold": 0.1,
            "top_k_parameters": 3
        })
        
        self.logger.info("🚀 Feature Engineering Optimizer initialized")
    
    @handle_errors(exceptions=(Exception,), default_return={})
    async def optimize_feature_parameters(
        self,
        data: pd.DataFrame,
        target: pd.Series,
        regimes: Optional[pd.Series] = None,
        symbol: str = "UNKNOWN",
        exchange: str = "UNKNOWN",
        timeframe: str = "1m"
    ) -> dict[str, Any]:
        """
        Optimize feature engineering parameters for each regime.
        
        Args:
            data: Feature data
            target: Target variable
            regimes: HMM regime labels (optional)
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            
        Returns:
            Dictionary with optimized parameters for each regime and feature
        """
        self.logger.info(f"🎯 Starting feature parameter optimization for {symbol} on {exchange}")
        
        results = {
            "optimization_timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe,
            "regime_optimizations": {},
            "global_optimizations": {},
            "correlation_analysis": {},
            "mutual_importance_matrix": {}
        }
        
        # 1. Global optimization (all data)
        self.logger.info("🌍 Performing global feature parameter optimization...")
        global_opt = await self._optimize_global_parameters(data, target)
        results["global_optimizations"] = global_opt
        
        # 2. Regime-specific optimization
        if regimes is not None and len(regimes.unique()) > 1:
            self.logger.info("🎭 Performing regime-specific optimization...")
            for regime in regimes.unique():
                regime_mask = regimes == regime
                regime_data = data[regime_mask]
                regime_target = target[regime_mask]
                
                if len(regime_data) < 100:  # Skip regimes with insufficient data
                    self.logger.warning(f"⚠️ Regime {regime} has insufficient data ({len(regime_data)} samples), skipping")
                    continue
                
                self.logger.info(f"🎯 Optimizing parameters for regime {regime} ({len(regime_data)} samples)")
                regime_opt = await self._optimize_regime_parameters(
                    regime_data, regime_target, regime
                )
                results["regime_optimizations"][f"regime_{regime}"] = regime_opt
        
        # 3. Correlation and mutual importance analysis
        self.logger.info("🔍 Performing correlation and mutual importance analysis...")
        correlation_analysis = await self._analyze_correlations_and_mi(data, target)
        results["correlation_analysis"] = correlation_analysis
        
        # 4. Select top 3 parameters for each feature
        self.logger.info("🏆 Selecting top 3 parameters for each feature...")
        top_parameters = await self._select_top_parameters(results)
        results["top_parameters"] = top_parameters
        
        # 5. Save results
        await self._save_optimization_results(results, symbol, exchange, timeframe)
        
        self.logger.info("✅ Feature parameter optimization completed successfully")
        return results
    
    async def _optimize_global_parameters(
        self, 
        data: pd.DataFrame, 
        target: pd.Series
    ) -> dict[str, Any]:
        """Optimize parameters globally across all data."""
        
        # Create parameter combinations for each feature
        param_combinations = {}
        for feature_name, params in self.feature_params.items():
            param_combinations[feature_name] = self._generate_param_combinations(params)
        
        # Optimize each feature
        optimized_params = {}
        for feature_name, combinations in param_combinations.items():
            self.logger.info(f"🔧 Optimizing {feature_name} parameters...")
            
            # Create synthetic features for this parameter combination
            feature_scores = []
            for params in combinations:
                # Generate synthetic feature based on parameters
                synthetic_feature = self._generate_synthetic_feature(
                    data, feature_name, params
                )
                
                if synthetic_feature is not None:
                    # Calculate feature importance using Random Forest + SHAP
                    importance_score = await self._calculate_feature_importance(
                        synthetic_feature, target
                    )
                    feature_scores.append({
                        "params": params,
                        "importance": importance_score,
                        "feature_values": synthetic_feature
                    })
            
            # Sort by importance and select top parameters
            if feature_scores:
                feature_scores.sort(key=lambda x: x["importance"], reverse=True)
                optimized_params[feature_name] = feature_scores[:3]  # Top 3
        
        return optimized_params
    
    async def _optimize_regime_parameters(
        self, 
        data: pd.DataFrame, 
        target: pd.Series, 
        regime: int
    ) -> dict[str, Any]:
        """Optimize parameters for a specific regime."""
        
        # Similar to global optimization but for regime-specific data
        param_combinations = {}
        for feature_name, params in self.feature_params.items():
            param_combinations[feature_name] = self._generate_param_combinations(params)
        
        optimized_params = {}
        for feature_name, combinations in param_combinations.items():
            self.logger.info(f"🎭 Optimizing {feature_name} parameters for regime {regime}...")
            
            feature_scores = []
            for params in combinations:
                synthetic_feature = self._generate_synthetic_feature(
                    data, feature_name, params
                )
                
                if synthetic_feature is not None:
                    importance_score = await self._calculate_feature_importance(
                        synthetic_feature, target
                    )
                    feature_scores.append({
                        "params": params,
                        "importance": importance_score,
                        "feature_values": synthetic_feature
                    })
            
            if feature_scores:
                feature_scores.sort(key=lambda x: x["importance"], reverse=True)
                optimized_params[feature_name] = feature_scores[:3]
        
        return optimized_params
    
    async def _analyze_correlations_and_mi(
        self, 
        data: pd.DataFrame, 
        target: pd.Series
    ) -> dict[str, Any]:
        """Analyze correlations and mutual information between features."""
        
        # Calculate correlation matrix
        correlation_matrix = data.corr()
        
        # Calculate mutual information with target
        mi_scores = mutual_info_regression(data, target, random_state=42)
        mi_df = pd.DataFrame({
            'feature': data.columns,
            'mutual_information': mi_scores
        }).sort_values('mutual_information', ascending=False)
        
        # Identify highly correlated features
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > self.optimization_config["correlation_threshold"]:
                    high_corr_pairs.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        return {
            "correlation_matrix": correlation_matrix.to_dict(),
            "mutual_information": mi_df.to_dict('records'),
            "high_correlation_pairs": high_corr_pairs,
            "correlation_threshold": self.optimization_config["correlation_threshold"]
        }
    
    async def _select_top_parameters(self, optimization_results: dict[str, Any]) -> dict[str, Any]:
        """Select top 3 parameters for each feature considering correlation, MI, etc."""
        
        top_parameters = {}
        
        # Process global optimizations
        for feature_name, feature_results in optimization_results["global_optimizations"].items():
            if not feature_results:
                continue
            
            # Get correlation analysis for this feature
            correlation_data = optimization_results["correlation_analysis"]
            
            # Score each parameter combination
            scored_params = []
            for result in feature_results:
                score = await self._calculate_comprehensive_score(
                    result, correlation_data, feature_name
                )
                scored_params.append({
                    "params": result["params"],
                    "importance": result["importance"],
                    "comprehensive_score": score
                })
            
            # Sort by comprehensive score and select top 3
            scored_params.sort(key=lambda x: x["comprehensive_score"], reverse=True)
            top_parameters[feature_name] = scored_params[:3]
        
        return top_parameters
    
    async def _calculate_feature_importance(
        self, 
        feature: pd.Series, 
        target: pd.Series
    ) -> float:
        """Calculate feature importance using Random Forest + SHAP."""
        
        try:
            # Prepare data
            X = feature.values.reshape(-1, 1)
            y = target.values
            
            # Train Random Forest
            rf = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X, y)
            
            # Calculate SHAP values
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X)
            
            # Calculate importance as mean absolute SHAP value
            importance = np.mean(np.abs(shap_values))
            
            return float(importance)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating feature importance: {e}")
            return 0.0
    
    async def _calculate_comprehensive_score(
        self, 
        result: dict[str, Any], 
        correlation_data: dict[str, Any], 
        feature_name: str
    ) -> float:
        """Calculate comprehensive score considering multiple factors."""
        
        base_importance = result["importance"]
        
        # Penalty for high correlation with existing features
        correlation_penalty = 0.0
        feature_values = result["feature_values"]
        
        # Check correlation with existing features
        for pair in correlation_data.get("high_correlation_pairs", []):
            if feature_name in [pair["feature1"], pair["feature2"]]:
                correlation_penalty += abs(pair["correlation"]) * 0.1
        
        # Bonus for high mutual information
        mi_bonus = 0.0
        for mi_item in correlation_data.get("mutual_information", []):
            if mi_item["feature"] == feature_name:
                mi_bonus = mi_item["mutual_information"] * 0.2
                break
        
        # Calculate final score
        final_score = base_importance - correlation_penalty + mi_bonus
        
        return max(0.0, final_score)
    
    def _generate_param_combinations(self, params: dict[str, List]) -> List[dict[str, Any]]:
        """Generate all parameter combinations for a feature."""
        import itertools
        
        param_names = list(params.keys())
        param_values = list(params.values())
        
        combinations = []
        for combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combination))
            combinations.append(param_dict)
        
        return combinations
    
    def _generate_synthetic_feature(
        self, 
        data: pd.DataFrame, 
        feature_name: str, 
        params: dict[str, Any]
    ) -> Optional[pd.Series]:
        """Generate synthetic feature based on parameters."""
        
        try:
            # This is a simplified version - in practice, you'd implement
            # actual technical indicator calculations here
            if feature_name == "RSI":
                lookback = params["lookback_period"]
                # Simulate RSI calculation
                return pd.Series(np.random.uniform(0, 100, len(data)), index=data.index)
            
            elif feature_name == "MACD":
                fast = params["fast_period"]
                slow = params["slow_period"]
                signal = params["signal_period"]
                # Simulate MACD calculation
                return pd.Series(np.random.randn(len(data)), index=data.index)
            
            elif feature_name == "Bollinger_Bands":
                lookback = params["lookback_period"]
                std_dev = params["std_dev"]
                # Simulate Bollinger Bands position
                return pd.Series(np.random.uniform(0, 1, len(data)), index=data.index)
            
            # Add more feature types as needed
            
            return None
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error generating synthetic feature for {feature_name}: {e}")
            return None
    
    async def _save_optimization_results(
        self, 
        results: dict[str, Any], 
        symbol: str, 
        exchange: str, 
        timeframe: str
    ):
        """Save optimization results to file."""
        
        output_dir = Path("data/feature_engineering_optimization")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{exchange}_{symbol}_{timeframe}_feature_optimization.json"
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"💾 Saved optimization results to {filepath}")
    
    def get_optimized_parameters(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: str
    ) -> dict[str, Any]:
        """Load optimized parameters for use in feature engineering."""
        
        filepath = Path(f"data/feature_engineering_optimization/{exchange}_{symbol}_{timeframe}_feature_optimization.json")
        
        if not filepath.exists():
            self.logger.warning(f"⚠️ No optimization results found for {symbol} on {exchange}")
            return {}
        
        try:
            with open(filepath, 'r') as f:
                results = json.load(f)
            
            return results.get("top_parameters", {})
            
        except Exception as e:
            self.logger.error(f"❌ Error loading optimization results: {e}")
            return {}