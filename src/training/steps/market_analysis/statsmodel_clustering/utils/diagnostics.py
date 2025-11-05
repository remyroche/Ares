"""
Diagnostics utilities for Statsmodels Clustering

This module provides diagnostic functions for statsmodels regime switching models.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import utilities
try:
    from src.utils.tprint import (
        tprint_info, tprint_success, tprint_warning, tprint_error
    )
except ImportError:
    def tprint_info(msg): print(f'ℹ️  {msg}')
    def tprint_success(msg): print(f'✅ {msg}')
    def tprint_warning(msg): print(f'⚠️  {msg}')
    def tprint_error(msg): print(f'❌ {msg}')


class DiagnosticsConfig:
    """Configuration for model diagnostics."""
    
    def __init__(self, 
                 create_plots: bool = False,
                 save_plots: bool = False,
                 plot_dir: str = "plots",
                 confidence_level: float = 0.95):
        self.create_plots = create_plots
        self.save_plots = save_plots
        self.plot_dir = plot_dir
        self.confidence_level = confidence_level


class DiagnosticsResult:
    """Result container for diagnostic operations."""
    
    def __init__(self,
                 success: bool = True,
                 diagnostics: Optional[Dict[str, Any]] = None,
                 plots: Optional[Dict[str, str]] = None,
                 error_message: Optional[str] = None):
        self.success = success
        self.diagnostics = diagnostics or {}
        self.plots = plots or {}
        self.error_message = error_message


class ModelDiagnostics:
    """Comprehensive model analysis for regime switching models."""
    
    def __init__(self, config: Optional[DiagnosticsConfig] = None):
        self.config = config or DiagnosticsConfig()
        tprint_info("🔧 Initialized Model Diagnostics")
    
    def analyze_model_fit(self, model: Any, data: pd.DataFrame) -> DiagnosticsResult:
        """Analyze model fit quality."""
        tprint_info("🔍 Analyzing model fit quality")
        
        try:
            diagnostics = {}
            
            # Basic fit statistics
            tprint_info("📊 Extracting basic fit statistics")
            if hasattr(model, 'llf'):
                diagnostics['log_likelihood'] = model.llf
                tprint_info(f"📈 Log likelihood: {model.llf:.4f}")
            if hasattr(model, 'aic'):
                diagnostics['aic'] = model.aic
                tprint_info(f"📈 AIC: {model.aic:.4f}")
            if hasattr(model, 'bic'):
                diagnostics['bic'] = model.bic
                tprint_info(f"📈 BIC: {model.bic:.4f}")
            
            # Parameter analysis
            tprint_info("📊 Analyzing model parameters")
            if hasattr(model, 'params'):
                diagnostics['parameter_stats'] = {
                    'count': len(model.params),
                    'mean': np.mean(model.params),
                    'std': np.std(model.params),
                    'min': np.min(model.params),
                    'max': np.max(model.params)
                }
                tprint_info(f"📊 Parameter count: {len(model.params)}")
            
            # Residual analysis
            tprint_info("📊 Analyzing model residuals")
            if hasattr(model, 'resid'):
                residuals = model.resid
                diagnostics['residual_stats'] = {
                    'mean': np.mean(residuals),
                    'std': np.std(residuals),
                    'skewness': self._calculate_skewness(residuals),
                    'kurtosis': self._calculate_kurtosis(residuals)
                }
                tprint_info(f"📊 Residual count: {len(residuals)}")
            
            tprint_success("✅ Model fit analysis complete")
            return DiagnosticsResult(success=True, diagnostics=diagnostics)
        except Exception as e:
            tprint_error(f"❌ Model fit analysis failed: {e}")
            return DiagnosticsResult(success=False, error_message=str(e))
    
    def analyze_regime_stability(self, model: Any, data: pd.DataFrame) -> DiagnosticsResult:
        """Analyze regime stability and transitions."""
        tprint_info("🔍 Analyzing regime stability and transitions")
        
        try:
            diagnostics = {}
            
            # Transition matrix analysis
            tprint_info("📊 Analyzing transition matrix")
            if hasattr(model, 'regime_transition'):
                transition_matrix = model.regime_transition
                diagnostics['transition_analysis'] = self._analyze_transition_matrix(transition_matrix)
                tprint_info(f"📊 Transition matrix shape: {transition_matrix.shape}")
            
            # Regime persistence
            tprint_info("📈 Analyzing regime persistence")
            if hasattr(model, 'smoothed_marginal_probabilities'):
                probs = model.smoothed_marginal_probabilities
                diagnostics['regime_persistence'] = self._analyze_regime_persistence(probs)
                tprint_info(f"📊 Probability matrix shape: {probs.shape}")
            
            # Regime characteristics
            tprint_info("📊 Analyzing regime characteristics")
            if hasattr(model, 'params'):
                diagnostics['regime_characteristics'] = self._analyze_regime_characteristics(model.params)
                tprint_info(f"📊 Parameter count: {len(model.params)}")
            
            tprint_success("✅ Regime stability analysis complete")
            return DiagnosticsResult(success=True, diagnostics=diagnostics)
        except Exception as e:
            tprint_error(f"❌ Regime stability analysis failed: {e}")
            return DiagnosticsResult(success=False, error_message=str(e))
    
    def create_diagnostics_report(self, model: Any, data: pd.DataFrame) -> DiagnosticsResult:
        """Create comprehensive diagnostics report."""
        tprint_info("🔍 Creating comprehensive diagnostics report")
        
        try:
            # Combine all analyses
            tprint_info("📊 Analyzing model fit")
            fit_analysis = self.analyze_model_fit(model, data)
            
            tprint_info("📈 Analyzing regime stability")
            stability_analysis = self.analyze_regime_stability(model, data)
            
            if not fit_analysis.success or not stability_analysis.success:
                tprint_error("❌ Failed to generate diagnostics report")
                return DiagnosticsResult(
                    success=False,
                    error_message="Failed to generate diagnostics report"
                )
            
            combined_diagnostics = {
                'model_fit': fit_analysis.diagnostics,
                'regime_stability': stability_analysis.diagnostics,
                'timestamp': pd.Timestamp.now(),
                'data_shape': data.shape
            }
            
            # Create plots if requested
            plots = {}
            if self.config.create_plots:
                tprint_info("📈 Creating diagnostic plots")
                plots = self._create_diagnostic_plots(model, data, combined_diagnostics)
            
            tprint_success("✅ Comprehensive diagnostics report created")
            return DiagnosticsResult(
                success=True,
                diagnostics=combined_diagnostics,
                plots=plots
            )
        except Exception as e:
            tprint_error(f"❌ Diagnostics report creation failed: {e}")
            return DiagnosticsResult(success=False, error_message=str(e))
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        from scipy.stats import skew
        return skew(data)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        from scipy.stats import kurtosis
        return kurtosis(data)
    
    def _analyze_transition_matrix(self, transition_matrix: np.ndarray) -> Dict[str, Any]:
        """Analyze transition matrix properties."""
        tprint_info("📊 Analyzing transition matrix properties")
        n_regimes = transition_matrix.shape[0]
        tprint_info(f"📈 Number of regimes: {n_regimes}")
        
        # Stationary distribution
        tprint_info("🔄 Calculating stationary distribution")
        eigenvals, eigenvecs = np.linalg.eig(transition_matrix.T)
        stationary_idx = np.argmax(np.real(eigenvals))
        stationary_dist = np.real(eigenvecs[:, stationary_idx])
        stationary_dist = stationary_dist / np.sum(stationary_dist)
        
        # Expected time in each regime
        tprint_info("⏱️ Calculating expected regime duration")
        expected_time = 1.0 / (1.0 - np.diag(transition_matrix))
        
        # Transition entropy
        tprint_info("📊 Calculating transition entropy")
        transition_entropy = self._calculate_transition_entropy(transition_matrix)
        
        result = {
            'n_regimes': n_regimes,
            'stationary_distribution': stationary_dist.tolist(),
            'expected_regime_duration': expected_time.tolist(),
            'transition_entropy': transition_entropy
        }
        
        tprint_success("✅ Transition matrix analysis complete")
        return result
    
    def _analyze_regime_persistence(self, probabilities: np.ndarray) -> Dict[str, Any]:
        """Analyze regime persistence from probabilities."""
        tprint_info("📈 Analyzing regime persistence from probabilities")
        
        # Most likely regime at each time point
        most_likely = np.argmax(probabilities, axis=1)
        
        # Calculate regime durations
        tprint_info("⏱️ Calculating regime durations")
        regime_changes = np.diff(most_likely) != 0
        regime_durations = []
        current_duration = 1
        
        for i in range(1, len(most_likely)):
            if regime_changes[i]:
                regime_durations.append(current_duration)
                current_duration = 1
            else:
                current_duration += 1
        
        regime_durations.append(current_duration)  # Add last regime duration
        
        result = {
            'mean_regime_duration': np.mean(regime_durations),
            'std_regime_duration': np.std(regime_durations),
            'min_regime_duration': np.min(regime_durations),
            'max_regime_duration': np.max(regime_durations),
            'total_regime_changes': np.sum(regime_changes)
        }
        
        tprint_info(f"📊 Total regime changes: {result['total_regime_changes']}")
        tprint_info(f"📊 Mean regime duration: {result['mean_regime_duration']:.2f}")
        tprint_success("✅ Regime persistence analysis complete")
        return result
    
    def _analyze_regime_characteristics(self, params: np.ndarray) -> Dict[str, Any]:
        """Analyze characteristics of each regime."""
        tprint_info("📊 Analyzing regime characteristics")
        
        # This is a simplified analysis - in practice would depend on model structure
        n_regimes = len(params) // 2  # Assuming mean and variance for each regime
        tprint_info(f"📈 Number of regimes: {n_regimes}")
        
        characteristics = {}
        for i in range(n_regimes):
            mean_val = params[2*i] if 2*i < len(params) else None
            var_val = params[2*i+1] if 2*i+1 < len(params) else None
            
            characteristics[f'regime_{i}'] = {
                'mean': mean_val,
                'variance': var_val
            }
            
            if mean_val is not None:
                tprint_info(f"📊 Regime {i} mean: {mean_val:.4f}")
            if var_val is not None:
                tprint_info(f"📊 Regime {i} variance: {var_val:.4f}")
        
        tprint_success("✅ Regime characteristics analysis complete")
        return characteristics
    
    def _calculate_transition_entropy(self, transition_matrix: np.ndarray) -> float:
        """Calculate entropy of transition matrix."""
        tprint_info("📊 Calculating transition entropy")
        
        entropy = 0
        for i in range(transition_matrix.shape[0]):
            row = transition_matrix[i, :]
            row = row[row > 0]  # Remove zero probabilities
            if len(row) > 0:
                entropy -= np.sum(row * np.log(row + 1e-10))
        
        entropy_val = float(entropy)
        tprint_info(f"📈 Transition entropy: {entropy_val:.4f}")
        tprint_success("✅ Transition entropy calculation complete")
        return entropy_val
    
    def _create_diagnostic_plots(self, model: Any, data: pd.DataFrame, diagnostics: Dict[str, Any]) -> Dict[str, str]:
        """Create diagnostic plots."""
        tprint_info("📈 Creating diagnostic plots")
        plots = {}
        
        try:
            # Residual plot
            if hasattr(model, 'resid'):
                tprint_info("📊 Creating residual plot")
                plt.figure(figsize=(10, 6))
                plt.plot(model.resid)
                plt.title('Model Residuals')
                plt.xlabel('Time')
                plt.ylabel('Residual')
                
                if self.config.save_plots:
                    plot_path = f"{self.config.plot_dir}/residuals.png"
                    plt.savefig(plot_path)
                    plots['residuals'] = plot_path
                    tprint_info(f"💾 Saved residual plot to {plot_path}")
                else:
                    plots['residuals'] = 'Residual plot generated'
                    tprint_info("📊 Residual plot generated")
                plt.close()
            
            # Regime probabilities plot
            if hasattr(model, 'smoothed_marginal_probabilities'):
                tprint_info("📊 Creating regime probabilities plot")
                plt.figure(figsize=(12, 8))
                probs = model.smoothed_marginal_probabilities
                for i in range(probs.shape[1]):
                    plt.plot(probs[:, i], label=f'Regime {i}')
                plt.title('Regime Probabilities')
                plt.xlabel('Time')
                plt.ylabel('Probability')
                plt.legend()
                
                if self.config.save_plots:
                    plot_path = f"{self.config.plot_dir}/regime_probabilities.png"
                    plt.savefig(plot_path)
                    plots['regime_probabilities'] = plot_path
                    tprint_info(f"💾 Saved regime probabilities plot to {plot_path}")
                else:
                    plots['regime_probabilities'] = 'Regime probabilities plot generated'
                    tprint_info("📊 Regime probabilities plot generated")
                plt.close()
                
        except Exception as e:
            tprint_error(f"❌ Failed to create plots: {e}")
            plots['error'] = f"Failed to create plots: {str(e)}"
        
        tprint_success("✅ Diagnostic plots creation complete")
        return plots


def analyze_model_fit(model: Any, data: pd.DataFrame, config: Optional[DiagnosticsConfig] = None) -> DiagnosticsResult:
    """Convenience function to analyze model fit."""
    tprint_info("🏭 Convenience function: analyzing model fit")
    diagnostics = ModelDiagnostics(config)
    result = diagnostics.analyze_model_fit(model, data)
    tprint_success("✅ Model fit analysis complete")
    return result


def analyze_regime_stability(model: Any, data: pd.DataFrame, config: Optional[DiagnosticsConfig] = None) -> DiagnosticsResult:
    """Convenience function to analyze regime stability."""
    tprint_info("🏭 Convenience function: analyzing regime stability")
    diagnostics = ModelDiagnostics(config)
    result = diagnostics.analyze_regime_stability(model, data)
    tprint_success("✅ Regime stability analysis complete")
    return result


def create_diagnostics_report(model: Any, data: pd.DataFrame, config: Optional[DiagnosticsConfig] = None) -> DiagnosticsResult:
    """Convenience function to create diagnostics report."""
    tprint_info("🏭 Convenience function: creating diagnostics report")
    diagnostics = ModelDiagnostics(config)
    result = diagnostics.create_diagnostics_report(model, data)
    tprint_success("✅ Diagnostics report creation complete")
    return result