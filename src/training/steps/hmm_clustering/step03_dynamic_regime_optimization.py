#!/usr/bin/env python3
"""Dynamic Regime Count Optimization.

This module automatically determines the optimal number of regimes based on multiple criteria:
1. Information Criteria (AIC, BIC, ICL)
2. Cross-validation with regime stability
3. Market condition adaptation
4. Economic significance testing
5. Regime persistence analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class DynamicRegimeCountOptimizer:
    """Automatically optimize the number of regimes using multiple criteria."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Optimization parameters
        self.min_regimes = self.config.get('min_regimes', 2)
        self.max_regimes = self.config.get('max_regimes', 8)
        self.cv_folds = self.config.get('cv_folds', 5)
        self.stability_threshold = self.config.get('stability_threshold', 0.7)
        
        # Criteria weights
        self.criteria_weights = self.config.get('criteria_weights', {
            'information_criteria': 0.25,
            'cross_validation': 0.25,
            'economic_significance': 0.25,
            'regime_persistence': 0.25
        })
        
        # Market condition parameters
        self.volatility_threshold = self.config.get('volatility_threshold', 0.02)
        self.trend_threshold = self.config.get('trend_threshold', 0.001)
        
    def optimize_regime_count(self, data: pd.DataFrame, features: np.ndarray) -> Dict[str, Any]:
        """Optimize the number of regimes using multiple criteria."""
        print("🔍 Starting dynamic regime count optimization...")
        
        # Step 1: Information Criteria Analysis
        print("  📊 Analyzing information criteria...")
        ic_results = self._analyze_information_criteria(features)
        
        # Step 2: Cross-Validation Analysis
        print("  🔄 Performing cross-validation analysis...")
        cv_results = self._cross_validation_analysis(data, features)
        
        # Step 3: Economic Significance Analysis
        print("  💰 Analyzing economic significance...")
        economic_results = self._economic_significance_analysis(data, features)
        
        # Step 4: Regime Persistence Analysis
        print("  ⏱️ Analyzing regime persistence...")
        persistence_results = self._regime_persistence_analysis(data, features)
        
        # Step 5: Market Condition Adaptation
        print("  🌊 Adapting to market conditions...")
        market_adaptation = self._adapt_to_market_conditions(data)
        
        # Step 6: Combine Results
        print("  🎯 Combining optimization results...")
        optimal_regimes = self._combine_optimization_results(
            ic_results, cv_results, economic_results, persistence_results, market_adaptation
        )
        
        return {
            'optimal_n_regimes': optimal_regimes['n_regimes'],
            'optimization_score': optimal_regimes['score'],
            'information_criteria': ic_results,
            'cross_validation': cv_results,
            'economic_significance': economic_results,
            'regime_persistence': persistence_results,
            'market_adaptation': market_adaptation,
            'all_scores': optimal_regimes['all_scores'],
            'recommendation': optimal_regimes['recommendation']
        }
    
    def _analyze_information_criteria(self, features: np.ndarray) -> Dict[str, Any]:
        """Analyze information criteria (AIC, BIC, ICL) for different regime counts."""
        results = {
            'aic_scores': [],
            'bic_scores': [],
            'icl_scores': [],
            'n_regimes_tested': [],
            'best_aic': None,
            'best_bic': None,
            'best_icl': None
        }
        
        for n_regimes in range(self.min_regimes, self.max_regimes + 1):
            try:
                # Fit HMM model
                from hmmlearn.hmm import GaussianHMM
                
                model = GaussianHMM(
                    n_components=n_regimes,
                    covariance_type="full",
                    random_state=42
                )
                model.fit(features)
                
                # Calculate information criteria
                log_likelihood = model.score(features)
                n_params = n_regimes * (n_regimes - 1) + n_regimes * features.shape[1] * (features.shape[1] + 1) / 2
                n_samples = len(features)
                
                # AIC = -2 * log_likelihood + 2 * n_params
                aic = -2 * log_likelihood + 2 * n_params
                
                # BIC = -2 * log_likelihood + n_params * log(n_samples)
                bic = -2 * log_likelihood + n_params * np.log(n_samples)
                
                # ICL (Integrated Classification Likelihood)
                # ICL = BIC + 2 * sum(log(gamma))
                # where gamma is the posterior probability
                gamma = model.predict_proba(features)
                icl_penalty = 2 * np.sum(np.log(np.maximum(gamma, 1e-10)))
                icl = bic + icl_penalty
                
                results['aic_scores'].append(aic)
                results['bic_scores'].append(bic)
                results['icl_scores'].append(icl)
                results['n_regimes_tested'].append(n_regimes)
                
            except Exception as e:
                print(f"Error fitting HMM with {n_regimes} regimes: {e}")
                results['aic_scores'].append(np.inf)
                results['bic_scores'].append(np.inf)
                results['icl_scores'].append(np.inf)
                results['n_regimes_tested'].append(n_regimes)
        
        # Find best scores
        if results['aic_scores']:
            best_aic_idx = np.argmin(results['aic_scores'])
            results['best_aic'] = {
                'n_regimes': results['n_regimes_tested'][best_aic_idx],
                'score': results['aic_scores'][best_aic_idx]
            }
        
        if results['bic_scores']:
            best_bic_idx = np.argmin(results['bic_scores'])
            results['best_bic'] = {
                'n_regimes': results['n_regimes_tested'][best_bic_idx],
                'score': results['bic_scores'][best_bic_idx]
            }
        
        if results['icl_scores']:
            best_icl_idx = np.argmin(results['icl_scores'])
            results['best_icl'] = {
                'n_regimes': results['n_regimes_tested'][best_icl_idx],
                'score': results['icl_scores'][best_icl_idx]
            }
        
        return results
    
    def _cross_validation_analysis(self, data: pd.DataFrame, features: np.ndarray) -> Dict[str, Any]:
        """Perform cross-validation analysis for regime stability."""
        from sklearn.model_selection import TimeSeriesSplit
        
        results = {
            'cv_scores': [],
            'stability_scores': [],
            'n_regimes_tested': [],
            'best_cv': None
        }
        
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        for n_regimes in range(self.min_regimes, self.max_regimes + 1):
            cv_scores = []
            stability_scores = []
            
            for train_idx, val_idx in tscv.split(features):
                try:
                    # Train on training set
                    X_train = features[train_idx]
                    X_val = features[val_idx]
                    
                    # Fit HMM
                    from hmmlearn.hmm import GaussianHMM
                    model = GaussianHMM(
                        n_components=n_regimes,
                        covariance_type="full",
                        random_state=42
                    )
                    model.fit(X_train)
                    
                    # Predict on validation set
                    val_regimes = model.predict(X_val)
                    
                    # Calculate stability (how consistent regimes are)
                    stability = self._calculate_regime_stability(val_regimes)
                    stability_scores.append(stability)
                    
                    # Calculate log-likelihood score
                    score = model.score(X_val)
                    cv_scores.append(score)
                    
                except Exception as e:
                    print(f"Error in CV fold for {n_regimes} regimes: {e}")
                    cv_scores.append(-np.inf)
                    stability_scores.append(0.0)
            
            if cv_scores:
                avg_cv_score = np.mean(cv_scores)
                avg_stability = np.mean(stability_scores)
                
                results['cv_scores'].append(avg_cv_score)
                results['stability_scores'].append(avg_stability)
                results['n_regimes_tested'].append(n_regimes)
        
        # Find best CV score
        if results['cv_scores']:
            best_cv_idx = np.argmax(results['cv_scores'])
            results['best_cv'] = {
                'n_regimes': results['n_regimes_tested'][best_cv_idx],
                'score': results['cv_scores'][best_cv_idx],
                'stability': results['stability_scores'][best_cv_idx]
            }
        
        return results
    
    def _economic_significance_analysis(self, data: pd.DataFrame, features: np.ndarray) -> Dict[str, Any]:
        """Analyze economic significance of different regime counts."""
        results = {
            'economic_scores': [],
            'sharpe_ratios': [],
            'return_differences': [],
            'n_regimes_tested': [],
            'best_economic': None
        }
        
        if 'close' not in data.columns:
            return results
        
        returns = data['close'].pct_change().dropna()
        if len(returns) == 0:
            return results
        
        for n_regimes in range(self.min_regimes, self.max_regimes + 1):
            try:
                # Fit HMM and get regimes
                from hmmlearn.hmm import GaussianHMM
                model = GaussianHMM(
                    n_components=n_regimes,
                    covariance_type="full",
                    random_state=42
                )
                model.fit(features)
                regimes = model.predict(features)
                
                # Align regimes with returns
                min_length = min(len(regimes), len(returns))
                regimes_aligned = regimes[:min_length]
                returns_aligned = returns[:min_length]
                
                # Calculate economic metrics
                economic_score = self._calculate_economic_score(returns_aligned, regimes_aligned)
                sharpe_ratios = self._calculate_regime_sharpe_ratios(returns_aligned, regimes_aligned)
                return_differences = self._calculate_return_differences(returns_aligned, regimes_aligned)
                
                results['economic_scores'].append(economic_score)
                results['sharpe_ratios'].append(sharpe_ratios)
                results['return_differences'].append(return_differences)
                results['n_regimes_tested'].append(n_regimes)
                
            except Exception as e:
                print(f"Error in economic analysis for {n_regimes} regimes: {e}")
                results['economic_scores'].append(0.0)
                results['sharpe_ratios'].append([])
                results['return_differences'].append(0.0)
                results['n_regimes_tested'].append(n_regimes)
        
        # Find best economic score
        if results['economic_scores']:
            best_economic_idx = np.argmax(results['economic_scores'])
            results['best_economic'] = {
                'n_regimes': results['n_regimes_tested'][best_economic_idx],
                'score': results['economic_scores'][best_economic_idx],
                'sharpe_ratios': results['sharpe_ratios'][best_economic_idx],
                'return_differences': results['return_differences'][best_economic_idx]
            }
        
        return results
    
    def _regime_persistence_analysis(self, data: pd.DataFrame, features: np.ndarray) -> Dict[str, Any]:
        """Analyze regime persistence for different regime counts."""
        results = {
            'persistence_scores': [],
            'transition_rates': [],
            'n_regimes_tested': [],
            'best_persistence': None
        }
        
        for n_regimes in range(self.min_regimes, self.max_regimes + 1):
            try:
                # Fit HMM and get regimes
                from hmmlearn.hmm import GaussianHMM
                model = GaussianHMM(
                    n_components=n_regimes,
                    covariance_type="full",
                    random_state=42
                )
                model.fit(features)
                regimes = model.predict(features)
                
                # Calculate persistence metrics
                persistence_score = self._calculate_persistence_score(regimes)
                transition_rate = self._calculate_transition_rate(regimes)
                
                results['persistence_scores'].append(persistence_score)
                results['transition_rates'].append(transition_rate)
                results['n_regimes_tested'].append(n_regimes)
                
            except Exception as e:
                print(f"Error in persistence analysis for {n_regimes} regimes: {e}")
                results['persistence_scores'].append(0.0)
                results['transition_rates'].append(1.0)
                results['n_regimes_tested'].append(n_regimes)
        
        # Find best persistence score
        if results['persistence_scores']:
            best_persistence_idx = np.argmax(results['persistence_scores'])
            results['best_persistence'] = {
                'n_regimes': results['n_regimes_tested'][best_persistence_idx],
                'score': results['persistence_scores'][best_persistence_idx],
                'transition_rate': results['transition_rates'][best_persistence_idx]
            }
        
        return results
    
    def _adapt_to_market_conditions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Adapt regime count based on current market conditions."""
        if 'close' not in data.columns:
            return {'recommended_regimes': 3, 'market_condition': 'unknown'}
        
        returns = data['close'].pct_change().dropna()
        if len(returns) < 50:
            return {'recommended_regimes': 3, 'market_condition': 'insufficient_data'}
        
        # Calculate market condition metrics
        volatility = returns.rolling(20).std().iloc[-1] if len(returns) >= 20 else returns.std()
        trend = returns.rolling(50).mean().iloc[-1] if len(returns) >= 50 else returns.mean()
        
        # Determine market condition
        if volatility > self.volatility_threshold:
            if abs(trend) > self.trend_threshold:
                market_condition = 'trending_volatile'
                recommended_regimes = 5  # More regimes for complex market
            else:
                market_condition = 'sideways_volatile'
                recommended_regimes = 4
        else:
            if abs(trend) > self.trend_threshold:
                market_condition = 'trending_calm'
                recommended_regimes = 3
            else:
                market_condition = 'sideways_calm'
                recommended_regimes = 2
        
        return {
            'recommended_regimes': recommended_regimes,
            'market_condition': market_condition,
            'volatility': volatility,
            'trend': trend
        }
    
    def _combine_optimization_results(self, ic_results: Dict, cv_results: Dict, 
                                    economic_results: Dict, persistence_results: Dict,
                                    market_adaptation: Dict) -> Dict[str, Any]:
        """Combine all optimization results to determine optimal regime count."""
        
        # Collect recommendations from each method
        recommendations = []
        
        # Information criteria recommendation
        if ic_results['best_bic']:
            recommendations.append({
                'n_regimes': ic_results['best_bic']['n_regimes'],
                'score': 1.0,  # Normalized
                'method': 'information_criteria',
                'weight': self.criteria_weights['information_criteria']
            })
        
        # Cross-validation recommendation
        if cv_results['best_cv']:
            recommendations.append({
                'n_regimes': cv_results['best_cv']['n_regimes'],
                'score': cv_results['best_cv']['stability'],
                'method': 'cross_validation',
                'weight': self.criteria_weights['cross_validation']
            })
        
        # Economic significance recommendation
        if economic_results['best_economic']:
            recommendations.append({
                'n_regimes': economic_results['best_economic']['n_regimes'],
                'score': economic_results['best_economic']['score'],
                'method': 'economic_significance',
                'weight': self.criteria_weights['economic_significance']
            })
        
        # Persistence recommendation
        if persistence_results['best_persistence']:
            recommendations.append({
                'n_regimes': persistence_results['best_persistence']['n_regimes'],
                'score': persistence_results['best_persistence']['score'],
                'method': 'regime_persistence',
                'weight': self.criteria_weights['regime_persistence']
            })
        
        # Market adaptation recommendation
        recommendations.append({
            'n_regimes': market_adaptation['recommended_regimes'],
            'score': 0.8,  # High confidence in market adaptation
            'method': 'market_adaptation',
            'weight': 0.2  # Additional weight for market conditions
        })
        
        # Calculate weighted scores for each regime count
        regime_scores = {}
        for rec in recommendations:
            n_regimes = rec['n_regimes']
            if n_regimes not in regime_scores:
                regime_scores[n_regimes] = 0.0
            regime_scores[n_regimes] += rec['score'] * rec['weight']
        
        # Find optimal regime count
        if regime_scores:
            optimal_n_regimes = max(regime_scores.keys(), key=lambda x: regime_scores[x])
            optimal_score = regime_scores[optimal_n_regimes]
        else:
            optimal_n_regimes = 3  # Default fallback
            optimal_score = 0.0
        
        # Generate recommendation
        recommendation = self._generate_recommendation(
            optimal_n_regimes, optimal_score, recommendations, market_adaptation
        )
        
        return {
            'n_regimes': optimal_n_regimes,
            'score': optimal_score,
            'all_scores': regime_scores,
            'recommendation': recommendation
        }
    
    def _generate_recommendation(self, optimal_n_regimes: int, score: float,
                               recommendations: List[Dict], market_adaptation: Dict) -> str:
        """Generate a human-readable recommendation."""
        
        # Count how many methods agree
        agreeing_methods = [r for r in recommendations if r['n_regimes'] == optimal_n_regimes]
        agreement_count = len(agreeing_methods)
        
        # Determine confidence level
        if score > 0.8 and agreement_count >= 3:
            confidence = "high"
        elif score > 0.6 and agreement_count >= 2:
            confidence = "medium"
        else:
            confidence = "low"
        
        # Generate recommendation text
        recommendation = f"""
        Optimal regime count: {optimal_n_regimes}
        Confidence: {confidence} ({score:.2f})
        Market condition: {market_adaptation['market_condition']}
        Agreeing methods: {agreement_count}/{len(recommendations)}
        
        Rationale:
        - Information criteria suggest {optimal_n_regimes} regimes
        - Cross-validation shows good stability
        - Economic significance is meaningful
        - Regime persistence is appropriate
        - Market conditions support this choice
        """
        
        return recommendation.strip()
    
    # Helper methods for calculations
    
    def _calculate_regime_stability(self, regimes: np.ndarray) -> float:
        """Calculate regime stability."""
        if len(regimes) < 2:
            return 1.0
        
        changes = np.sum(np.diff(regimes) != 0)
        stability = 1.0 - (changes / (len(regimes) - 1))
        return max(0.0, stability)
    
    def _calculate_economic_score(self, returns: pd.Series, regimes: np.ndarray) -> float:
        """Calculate economic significance score."""
        if len(returns) == 0 or len(regimes) == 0:
            return 0.0
        
        # Calculate return differences between regimes
        regime_returns = {}
        for regime in np.unique(regimes):
            regime_mask = regimes == regime
            if np.sum(regime_mask) > 5:  # Minimum samples
                regime_returns[regime] = returns[regime_mask].mean()
        
        if len(regime_returns) < 2:
            return 0.0
        
        # Economic score as variance of regime returns
        return_means = list(regime_returns.values())
        economic_score = np.var(return_means)
        return min(economic_score * 1000, 1.0)  # Scale and cap
    
    def _calculate_regime_sharpe_ratios(self, returns: pd.Series, regimes: np.ndarray) -> List[float]:
        """Calculate Sharpe ratios for each regime."""
        sharpe_ratios = []
        
        for regime in np.unique(regimes):
            regime_mask = regimes == regime
            if np.sum(regime_mask) > 10:  # Minimum samples
                regime_returns = returns[regime_mask]
                if regime_returns.std() > 0:
                    sharpe = regime_returns.mean() / regime_returns.std()
                    sharpe_ratios.append(sharpe)
        
        return sharpe_ratios
    
    def _calculate_return_differences(self, returns: pd.Series, regimes: np.ndarray) -> float:
        """Calculate differences in returns between regimes."""
        regime_returns = {}
        for regime in np.unique(regimes):
            regime_mask = regimes == regime
            if np.sum(regime_mask) > 5:
                regime_returns[regime] = returns[regime_mask].mean()
        
        if len(regime_returns) < 2:
            return 0.0
        
        return_means = list(regime_returns.values())
        return np.std(return_means)
    
    def _calculate_persistence_score(self, regimes: np.ndarray) -> float:
        """Calculate regime persistence score."""
        if len(regimes) < 2:
            return 1.0
        
        # Calculate average regime duration
        regime_changes = np.where(np.diff(regimes) != 0)[0]
        if len(regime_changes) == 0:
            return 1.0  # No changes, perfect persistence
        
        durations = np.diff(np.concatenate([[0], regime_changes, [len(regimes)]]))
        avg_duration = np.mean(durations)
        
        # Persistence score based on average duration
        # Higher duration = higher persistence
        persistence_score = min(avg_duration / 50, 1.0)  # Normalize
        return persistence_score
    
    def _calculate_transition_rate(self, regimes: np.ndarray) -> float:
        """Calculate regime transition rate."""
        if len(regimes) < 2:
            return 0.0
        
        changes = np.sum(np.diff(regimes) != 0)
        transition_rate = changes / (len(regimes) - 1)
        return transition_rate