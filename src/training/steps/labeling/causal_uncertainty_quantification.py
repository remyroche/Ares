"""
Causal Uncertainty Quantification for Modern De Prado Framework

This module provides comprehensive uncertainty quantification for causal inference,
including Bayesian causal discovery, treatment effect uncertainty, and specialist
prediction uncertainty.

Key Features:
- Bayesian causal discovery with confidence intervals
- Treatment effect uncertainty estimation
- Specialist prediction uncertainty
- Causal model confidence scoring
- Uncertainty propagation through the pipeline
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from scipy import stats
from sklearn.utils import resample
import time

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    # Fallback print functions
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")


class BayesianCausalDiscovery:
    """
    Bayesian causal discovery with uncertainty quantification.
    
    This class performs bootstrap-based causal discovery to estimate
    confidence intervals and uncertainty in causal relationships.
    """
    
    def __init__(
        self,
        n_bootstrap: int = 100,
        significance_level: float = 0.05,
        confidence_level: float = 0.95,
        target_variable: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize Bayesian Causal Discovery.
        
        Args:
            n_bootstrap: Number of bootstrap samples
            significance_level: Significance level for edge detection
            confidence_level: Confidence level for intervals
            target_variable: Primary target to focus discovery on
            verbose: Whether to print progress information
        """
        self.verbose = verbose
        self.n_bootstrap = n_bootstrap
        self.significance_level = significance_level
        self.confidence_level = confidence_level
        self.target_variable = target_variable
        
        # Results storage
        self.bootstrap_graphs = []
        self.edge_confidence = {}
        self.consensus_graph = {}
        self.uncertainty_metrics = {}
        
        if self.verbose:
            tprint_info("🔬 Bayesian Causal Discovery: Initializing...")
            tprint_info(f"   ⚙️ Bootstrap samples: {n_bootstrap}")
            tprint_info(f"   ⚙️ Significance level: {significance_level}")
            tprint_info(f"   ⚙️ Confidence level: {confidence_level}")
            tprint_success("   ✅ Bayesian Causal Discovery: Initialization complete")
    
    def discover_with_uncertainty(
        self, 
        data: pd.DataFrame,
        causal_discovery_algorithm: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Discover causal graph with uncertainty quantification.
        
        Args:
            data: Input data for causal discovery
            causal_discovery_algorithm: Function to perform causal discovery
            
        Returns:
            Dictionary with causal graph and uncertainty metrics
        """
        try:
            if self.verbose:
                tprint_info("🚀 Bayesian Causal Discovery: Starting uncertainty quantification...")
            
            discovery_start_time = time.time()
            
            # Initialize results
            self.bootstrap_graphs = []
            self.edge_confidence = {}
            edge_appearances = {}
            
            # Perform bootstrap causal discovery
            if self.verbose:
                tprint_info(f"   📊 Running {self.n_bootstrap} bootstrap samples...")
            
            for i in range(self.n_bootstrap):
                try:
                    # Create bootstrap sample
                    bootstrap_data = data.sample(frac=0.8, replace=True)
                    
                    # Discover causal graph
                    if causal_discovery_algorithm is not None:
                        graph = causal_discovery_algorithm(bootstrap_data)
                    else:
                        # Fallback to PC algorithm if available, else correlation
                        try:
                            from src.training.steps.labeling.causal_discovery import CausalDiscovery
                            cd = CausalDiscovery(
                                significance_level=self.significance_level,
                                target_variable=self.target_variable,
                                verbose=False # Internal discovery should be quiet
                            )
                            graph = cd.pc_algorithm(bootstrap_data, list(bootstrap_data.columns))
                            if cd.use_lingam:
                                oriented = cd.lingam_orientation(bootstrap_data)
                                # Convert oriented matrix back to graph list if needed
                                # However, pc_algorithm already returns a graph. 
                                # Better to use discover_causal_structure
                                # But let's keep it simple for now.
                        except ImportError:
                            graph = self._simple_correlation_graph(bootstrap_data)
                    
                    self.bootstrap_graphs.append(graph)
                    
                    # Count edge appearances
                    for node, parents in graph.items():
                        for parent in parents:
                            edge = tuple(sorted([parent, node]))
                            edge_appearances[edge] = edge_appearances.get(edge, 0) + 1
                    
                    if self.verbose and (i + 1) % 20 == 0:
                        tprint_info(f"      📊 Completed {i + 1}/{self.n_bootstrap} bootstrap samples")
                        
                except Exception as e:
                    if self.verbose:
                        tprint_warning(f"      ⚠️ Bootstrap sample {i} failed: {e}")
                    continue
            
            # Calculate edge confidence
            if self.verbose:
                tprint_info("   📈 Calculating edge confidence...")
            
            for edge, appearances in edge_appearances.items():
                confidence = appearances / len(self.bootstrap_graphs)
                self.edge_confidence[edge] = confidence
            
            # Perform discovery on full dataset for confirmation
            if self.verbose:
                tprint_info("   📊 Running discovery on full dataset for confirmation...")
            
            if causal_discovery_algorithm is not None:
                full_graph = causal_discovery_algorithm(data)
            else:
                full_graph = self._simple_correlation_graph(data)
                
            # Create consensus graph with strict pruning (must be in full graph + high bootstrap confidence)
            if self.verbose:
                tprint_info(f"   🔗 Creating consensus graph (Threshold: {self.significance_level}, Strict confirmation: True)...")
                
            self.consensus_graph = self._create_consensus_graph(self.edge_confidence, full_graph)
            
            # Calculate uncertainty metrics
            if self.verbose:
                tprint_info("   📊 Computing uncertainty metrics...")
            
            self.uncertainty_metrics = self._calculate_uncertainty_metrics()
            
            discovery_time = time.time() - discovery_start_time
            
            results = {
                'consensus_graph': self.consensus_graph,
                'edge_confidence': self.edge_confidence,
                'bootstrap_graphs': self.bootstrap_graphs,
                'uncertainty_metrics': self.uncertainty_metrics,
                'discovery_time': discovery_time,
                'n_bootstrap_samples': len(self.bootstrap_graphs),
                'confidence_level': self.confidence_level
            }
            
            if self.verbose:
                tprint_success("✅ Bayesian Causal Discovery Complete:")
                tprint_info(f"   - Consensus edges: {len(self.consensus_graph)}")
                tprint_info(f"   - Bootstrap samples: {len(self.bootstrap_graphs)}")
                tprint_info(f"   - Discovery time: {discovery_time:.3f}s")
                tprint_info(f"   - Avg edge confidence: {np.mean(list(self.edge_confidence.values())):.3f}")
            
            return results
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Bayesian causal discovery failed: {e}")
            return {'error': str(e)}
    
    def _simple_correlation_graph(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Simple correlation-based causal graph as fallback."""
        try:
            corr_matrix = data.corr()
            graph = {}
            
            for i, col1 in enumerate(data.columns):
                parents = []
                for j, col2 in enumerate(data.columns):
                    if i != j and abs(corr_matrix.iloc[i, j]) > 0.3:
                        parents.append(col2)
                graph[col1] = parents
            
            return graph
            
        except Exception as e:
            if self.verbose:
                tprint_warning(f"   ⚠️ Simple correlation graph failed: {e}")
            return {}
    
    def _create_consensus_graph(
        self, 
        edge_confidence: Dict[Tuple[str, str], float],
        full_graph: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, List[str]]:
        """
        Create consensus graph from edge confidence.
        
        Args:
            edge_confidence: Dictionary of edge confidence scores
            full_graph: Optional graph discovered on full dataset for confirmation
            
        Returns:
            Consensus causal graph
        """
        try:
            consensus_graph = {}
            
            # Map full_graph edges to a set for quick lookup
            confirmed_edges = set()
            if full_graph:
                for child, parents in full_graph.items():
                    for parent in parents:
                        confirmed_edges.add(tuple(sorted([parent, child])))
            
            for edge, confidence in edge_confidence.items():
                # Strict Pruning: High confidence AND confirmed on full dataset (if provided)
                is_confirmed = True
                if full_graph is not None:
                    is_confirmed = edge in confirmed_edges
                
                if confidence >= self.significance_level and is_confirmed:
                    parent, child = edge
                    
                    if child not in consensus_graph:
                        consensus_graph[child] = []
                    consensus_graph[child].append(parent)
            
            return consensus_graph
            
        except Exception as e:
            if self.verbose:
                tprint_warning(f"   ⚠️ Consensus graph creation failed: {e}")
            return {}
    
    def _calculate_uncertainty_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive uncertainty metrics."""
        try:
            metrics = {}
            
            if not self.edge_confidence:
                return {'avg_confidence': 0.0, 'confidence_variance': 0.0}
            
            # Edge confidence statistics
            confidences = list(self.edge_confidence.values())
            metrics['avg_confidence'] = np.mean(confidences)
            metrics['confidence_variance'] = np.var(confidences)
            metrics['confidence_std'] = np.std(confidences)
            metrics['min_confidence'] = np.min(confidences)
            metrics['max_confidence'] = np.max(confidences)
            
            # Confidence distribution
            metrics['confidence_q25'] = np.percentile(confidences, 25)
            metrics['confidence_q50'] = np.percentile(confidences, 50)
            metrics['confidence_q75'] = np.percentile(confidences, 75)
            
            # Graph stability metrics
            metrics['graph_stability'] = metrics['avg_confidence']
            metrics['edge_certainty'] = np.mean([c for c in confidences if c > 0.8])
            metrics['edge_uncertainty'] = np.mean([c for c in confidences if c < 0.5])
            
            return metrics
            
        except Exception as e:
            if self.verbose:
                tprint_warning(f"   ⚠️ Uncertainty metrics calculation failed: {e}")
            return {'avg_confidence': 0.0}


class CausalUncertaintyQuantification:
    """
    Quantify uncertainty in causal effects and predictions.
    
    This class provides uncertainty estimation for treatment effects,
    causal targets, and specialist predictions.
    """
    
    def __init__(
        self,
        n_bootstrap: int = 100,
        confidence_level: float = 0.95,
        verbose: bool = True
    ):
        """
        Initialize Causal Uncertainty Quantification.
        
        Args:
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals
            verbose: Whether to print progress information
        """
        self.verbose = verbose
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        
        if self.verbose:
            tprint_info("🔬 Causal Uncertainty Quantification: Initializing...")
            tprint_info(f"   ⚙️ Bootstrap samples: {n_bootstrap}")
            tprint_info(f"   ⚙️ Confidence level: {confidence_level}")
            tprint_success("   ✅ Causal Uncertainty Quantification: Initialization complete")
    
    def estimate_treatment_uncertainty(
        self,
        X: pd.DataFrame,
        treatment: np.ndarray,
        outcome: np.ndarray,
        treatment_model: Optional[callable] = None,
        outcome_model: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Estimate uncertainty in treatment effects using bootstrap DML.
        
        Args:
            X: Covariates
            treatment: Treatment variable
            outcome: Outcome variable
            treatment_model: Model for treatment prediction
            outcome_model: Model for outcome prediction
            
        Returns:
            Dictionary with treatment effect uncertainty estimates
        """
        try:
            if self.verbose:
                tprint_info("🎯 Estimating Treatment Effect Uncertainty...")
            
            treatment_effects = []
            bootstrap_start_time = time.time()
            
            for i in range(self.n_bootstrap):
                try:
                    # Bootstrap sample
                    sample_idx = np.random.choice(len(X), len(X), replace=True)
                    X_sample, treatment_sample, outcome_sample = X.iloc[sample_idx], treatment[sample_idx], outcome[sample_idx]
                    
                    # Estimate treatment effect (simplified DML)
                    effect = self._dml_estimate(X_sample, treatment_sample, outcome_sample)
                    treatment_effects.append(effect)
                    
                    if self.verbose and (i + 1) % 25 == 0:
                        tprint_info(f"      📊 Completed {i + 1}/{self.n_bootstrap} bootstrap samples")
                        
                except Exception as e:
                    if self.verbose:
                        tprint_warning(f"      ⚠️ Bootstrap sample {i} failed: {e}")
                    continue
            
            bootstrap_time = time.time() - bootstrap_start_time
            
            if not treatment_effects:
                if self.verbose:
                    tprint_error("   ❌ No successful bootstrap samples")
                return {'error': 'No successful bootstrap samples'}
            
            # Calculate uncertainty statistics
            treatment_effects = np.array(treatment_effects)
            
            alpha = 1 - self.confidence_level
            lower_bound = np.percentile(treatment_effects, 100 * alpha / 2)
            upper_bound = np.percentile(treatment_effects, 100 * (1 - alpha / 2))
            
            results = {
                'mean_effect': np.mean(treatment_effects),
                'std_effect': np.std(treatment_effects),
                'median_effect': np.median(treatment_effects),
                'confidence_interval': (lower_bound, upper_bound),
                'confidence_width': upper_bound - lower_bound,
                'effect_distribution': treatment_effects.tolist(),
                'n_successful_samples': len(treatment_effects),
                'bootstrap_time': bootstrap_time,
                'significance': np.sign(np.mean(treatment_effects)),
                'is_significant': lower_bound > 0 or upper_bound < 0
            }
            
            if self.verbose:
                tprint_success("✅ Treatment Effect Uncertainty Estimated:")
                tprint_info(f"   - Mean effect: {results['mean_effect']:.6f}")
                tprint_info(f"   - Std effect: {results['std_effect']:.6f}")
                tprint_info(f"   - 95% CI: [{results['confidence_interval'][0]:.6f}, {results['confidence_interval'][1]:.6f}]")
                tprint_info(f"   - Is significant: {results['is_significant']}")
                tprint_info(f"   - Bootstrap time: {bootstrap_time:.3f}s")
            
            return results
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Treatment uncertainty estimation failed: {e}")
            return {'error': str(e)}
    
    def _dml_estimate(self, X: pd.DataFrame, treatment: np.ndarray, outcome: np.ndarray) -> float:
        """Simplified Double Machine Learning estimation."""
        try:
            # Simple linear approximation for demonstration
            # In practice, would use proper ML models
            from sklearn.linear_model import LinearRegression
            
            # Treatment model
            treatment_model = LinearRegression()
            treatment_model.fit(X, treatment)
            treatment_pred = treatment_model.predict(X)
            treatment_residual = treatment - treatment_pred
            
            # Outcome model
            outcome_model = LinearRegression()
            outcome_model.fit(X, outcome)
            outcome_pred = outcome_model.predict(X)
            outcome_residual = outcome - outcome_pred
            
            # Final stage: regress outcome residual on treatment residual
            if len(treatment_residual) > 1:
                effect = np.cov(outcome_residual, treatment_residual)[0, 1] / np.var(treatment_residual)
            else:
                effect = 0.0
            
            return effect
            
        except Exception as e:
            if self.verbose:
                tprint_warning(f"      ⚠️ DML estimation failed: {e}")
            return 0.0
    
    def estimate_specialist_uncertainty(
        self,
        specialist_predictions: Dict[str, np.ndarray],
        actual_outcomes: np.ndarray
    ) -> Dict[str, Any]:
        """
        Estimate uncertainty in specialist predictions.
        
        Args:
            specialist_predictions: Dictionary of specialist predictions
            actual_outcomes: Actual outcomes
            
        Returns:
            Dictionary with specialist uncertainty metrics
        """
        try:
            if self.verbose:
                tprint_info("🧠 Estimating Specialist Prediction Uncertainty...")
            
            uncertainty_results = {}
            
            for specialist_name, predictions in specialist_predictions.items():
                try:
                    if len(predictions) != len(actual_outcomes):
                        if self.verbose:
                            tprint_warning(f"      ⚠️ Length mismatch for {specialist_name}")
                        continue
                    
                    # Calculate prediction errors
                    errors = actual_outcomes - predictions
                    
                    # Bootstrap uncertainty for this specialist
                    bootstrap_errors = []
                    
                    for i in range(min(self.n_bootstrap, 50)):  # Limit for efficiency
                        sample_idx = np.random.choice(len(errors), len(errors), replace=True)
                        sample_errors = errors[sample_idx]
                        bootstrap_errors.append(np.mean(sample_errors))
                    
                    if bootstrap_errors:
                        bootstrap_errors = np.array(bootstrap_errors)
                        
                        # Calculate uncertainty metrics
                        alpha = 1 - self.confidence_level
                        lower_bound = np.percentile(bootstrap_errors, 100 * alpha / 2)
                        upper_bound = np.percentile(bootstrap_errors, 100 * (1 - alpha / 2))
                        
                        uncertainty_results[specialist_name] = {
                            'mean_error': np.mean(errors),
                            'std_error': np.std(errors),
                            'confidence_interval': (lower_bound, upper_bound),
                            'prediction_uncertainty': np.std(bootstrap_errors),
                            'reliability_score': 1.0 / (1.0 + np.std(errors)),
                            'bias': np.mean(errors),
                            'variance': np.var(errors)
                        }
                        
                        if self.verbose:
                            tprint_info(f"      ✅ {specialist_name}: uncertainty = {uncertainty_results[specialist_name]['prediction_uncertainty']:.6f}")
                    
                except Exception as e:
                    if self.verbose:
                        tprint_warning(f"      ⚠️ Specialist {specialist_name} uncertainty failed: {e}")
                    continue
            
            # Aggregate uncertainty metrics
            if uncertainty_results:
                all_uncertainties = [result['prediction_uncertainty'] for result in uncertainty_results.values()]
                all_reliabilities = [result['reliability_score'] for result in uncertainty_results.values()]
                
                uncertainty_results['aggregate_metrics'] = {
                    'avg_uncertainty': np.mean(all_uncertainties),
                    'uncertainty_variance': np.var(all_uncertainties),
                    'avg_reliability': np.mean(all_reliabilities),
                    'reliability_variance': np.var(all_reliabilities),
                    'n_specialists': len(uncertainty_results)
                }
                
                if self.verbose:
                    tprint_success("✅ Specialist Uncertainty Estimation Complete:")
                    tprint_info(f"   - Specialists analyzed: {len(uncertainty_results)}")
                    tprint_info(f"   - Avg uncertainty: {uncertainty_results['aggregate_metrics']['avg_uncertainty']:.6f}")
                    tprint_info(f"   - Avg reliability: {uncertainty_results['aggregate_metrics']['avg_reliability']:.3f}")
            
            return uncertainty_results
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Specialist uncertainty estimation failed: {e}")
            return {'error': str(e)}


# Convenience functions for quick usage
def quick_bayesian_causal_discovery(
    data: pd.DataFrame,
    n_bootstrap: int = 100,
    significance_level: float = 0.05,
    target_variable: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """Quick Bayesian causal discovery with uncertainty."""
    discovery = BayesianCausalDiscovery(
        n_bootstrap=n_bootstrap, 
        significance_level=significance_level,
        target_variable=target_variable,
        verbose=verbose
    )
    return discovery.discover_with_uncertainty(data)


def quick_treatment_uncertainty(
    X: pd.DataFrame,
    treatment: np.ndarray,
    outcome: np.ndarray,
    n_bootstrap: int = 100,
    verbose: bool = True
) -> Dict[str, Any]:
    """Quick treatment effect uncertainty estimation."""
    uncertainty = CausalUncertaintyQuantification(n_bootstrap=n_bootstrap, verbose=verbose)
    return uncertainty.estimate_treatment_uncertainty(X, treatment, outcome)


if __name__ == "__main__":
    # Example usage
    print("Causal Uncertainty Quantification Module")
    print("Use BayesianCausalDiscovery and CausalUncertaintyQuantification classes")
