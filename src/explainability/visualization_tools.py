from src.utils.tprint import tprint

from typing import Dict, List, Optional, Union, Any, Tuple
from ..utils.logger import system_logger
"""Visualization tools for model explanations and decision traces.

This module provides comprehensive visualization capabilities for SHAP/LIME
explanations and trade decision traces.
"""
from datetime import datetime
from pathlib import Path
from .explainability.base_explainer import ExplanationResult, TradeDecisionTrace
import logging
import numpy as np
import time

try:
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    tprint('Warning: matplotlib not available, visualization features disabled')
try:
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    tprint('Warning: plotly not available, interactive visualization features disabled')
try:
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

class ExplanationVisualizer:
    """Visualizer for model explanations."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize explanation visualizer."""
        self.config = config
        self.logger = system_logger.getChild('ExplanationVisualizer')
        self.viz_config = config.get('explainability', {}).get('visualization', {})
        self.output_path = Path(self.viz_config.get('output_path', 'data/visualizations'))
        self.output_path.mkdir(parents = True, exist_ok = True)
        self.style_config = self.viz_config.get('style', {})
        self.colors = self.style_config.get('colors', {'positive': '#2E8B57', 'negative': '#DC143C', 'neutral': '#4682B4', 'background': '#F8F8FF', 'text': '#2F4F4F'})
        self.figure_size = self.style_config.get('figure_size', (12, 8))
        self.dpi = self.style_config.get('dpi', 300)

    def visualize_shap_values(self, explanation: ExplanationResult, max_features: int = 20, save_path: Optional[str]=None) -> Optional[str]:
        """Visualize SHAP values."""
        try:
            if not MATPLOTLIB_AVAILABLE:
                self.logger.warning('⚠️ matplotlib not available for SHAP visualization')
                return None
            if explanation.shap_values is None:
                self.logger.warning('⚠️ No SHAP values available for visualization')
                return None
            fig, ax = plt.subplots(figsize = self.figure_size, dpi = self.dpi)
            if isinstance(explanation.shap_values, dict):
                self._create_multi_model_shap_plot(explanation, max_features, save_path)
                return save_path
            shap_values = explanation.shap_values
            feature_names = explanation.feature_names
            feature_importance = list(zip(feature_names, shap_values))
            feature_importance.sort(key = lambda x: abs(x[1]), reverse = True)
            feature_importance = feature_importance[:max_features]
            features = [item[0] for item in feature_importance]
            values = [item[1] for item in feature_importance]
            colors = [self.colors['positive'] if v > 0 else self.colors['negative'] for v in values]
            bars = ax.barh(range(len(features)), values, color = colors, alpha = 0.7)
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features, fontsize = 10)
            ax.set_xlabel('SHAP Value', fontsize = 12, fontweight='bold')
            ax.set_title(f'SHAP Values - {explanation.model_name}', fontsize = 14, fontweight='bold')
            ax.grid(True, alpha = 0.3)
            ax.axvline(x = 0, color='black', linestyle='-', alpha = 0.5)
            for i, (bar, value) in enumerate(zip(bars, values)):
                ax.text(value + (0.01 if value > 0 else -0.01), i, f'{value:.3f}', va='center', ha='left' if value > 0 else 'right', fontsize = 9)
            ax.invert_yaxis()
            plt.tight_layout()
            if save_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = self.output_path / f'shap_values_{explanation.model_name}_{timestamp}.png'
            plt.savefig(save_path, dpi = self.dpi, bbox_inches='tight')
            plt.close()
            self.logger.info(f'📊 SHAP visualization saved to {save_path}')
            return str(save_path)
        except Exception as e:
            self.logger.error(f'❌ Failed to visualize SHAP values: {e}')
            return None

    def _create_multi_model_shap_plot(self, explanation: ExplanationResult, max_features: int, save_path: Optional[str]) -> None:
        """Create multi-model SHAP plot."""
        try:
            shap_dict = explanation.shap_values
            n_models = len(shap_dict)
            fig, axes = plt.subplots(1, n_models, figsize=(self.figure_size[0] * n_models, self.figure_size[1]))
            if n_models == 1:
                axes = [axes]
            for i, (model_name, shap_values) in enumerate(shap_dict.items()):
                ax = axes[i]
                feature_importance = list(zip(explanation.feature_names, shap_values))
                feature_importance.sort(key = lambda x: abs(x[1]), reverse = True)
                feature_importance = feature_importance[:max_features]
                features = [item[0] for item in feature_importance]
                values = [item[1] for item in feature_importance]
                colors = [self.colors['positive'] if v > 0 else self.colors['negative'] for v in values]
                bars = ax.barh(range(len(features)), values, color = colors, alpha = 0.7)
                ax.set_yticks(range(len(features)))
                ax.set_yticklabels(features, fontsize = 9)
                ax.set_xlabel('SHAP Value', fontsize = 10)
                ax.set_title(f'{model_name}', fontsize = 12, fontweight='bold')
                ax.grid(True, alpha = 0.3)
                ax.axvline(x = 0, color='black', linestyle='-', alpha = 0.5)
                ax.invert_yaxis()
            plt.suptitle(f'Multi-Model SHAP Values - {explanation.model_name}', fontsize = 16, fontweight='bold')
            plt.tight_layout()
            if save_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = self.output_path / f'multi_model_shap_{explanation.model_name}_{timestamp}.png'
            plt.savefig(save_path, dpi = self.dpi, bbox_inches='tight')
            plt.close()
        except Exception as e:
            self.logger.error(f'❌ Failed to create multi-model SHAP plot: {e}')

    def visualize_lime_explanation(self, explanation: ExplanationResult, max_features: int = 15, save_path: Optional[str]=None) -> Optional[str]:
        """Visualize LIME explanation."""
        try:
            if not MATPLOTLIB_AVAILABLE:
                self.logger.warning('⚠️ matplotlib not available for LIME visualization')
                return None
            if explanation.lime_explanation is None:
                self.logger.warning('⚠️ No LIME explanation available for visualization')
                return None
            fig, ax = plt.subplots(figsize = self.figure_size, dpi = self.dpi)
            lime_data = explanation.lime_explanation.get('feature_importance', [])
            if not lime_data:
                self.logger.warning('⚠️ No feature importance data in LIME explanation')
                return None
            lime_data.sort(key = lambda x: abs(x[1]), reverse = True)
            lime_data = lime_data[:max_features]
            features = [item[0] for item in lime_data]
            values = [item[1] for item in lime_data]
            colors = [self.colors['positive'] if v > 0 else self.colors['negative'] for v in values]
            bars = ax.barh(range(len(features)), values, color = colors, alpha = 0.7)
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features, fontsize = 10)
            ax.set_xlabel('LIME Importance', fontsize = 12, fontweight='bold')
            ax.set_title(f'LIME Explanation - {explanation.model_name}', fontsize = 14, fontweight='bold')
            ax.grid(True, alpha = 0.3)
            ax.axvline(x = 0, color='black', linestyle='-', alpha = 0.5)
            for i, (bar, value) in enumerate(zip(bars, values)):
                ax.text(value + (0.01 if value > 0 else -0.01), i, f'{value:.3f}', va='center', ha='left' if value > 0 else 'right', fontsize = 9)
            ax.invert_yaxis()
            if 'prediction' in explanation.lime_explanation:
                pred_text = f"Prediction: {explanation.lime_explanation['prediction']:.3f}"
                ax.text(0.02, 0.98, pred_text, transform = ax.transAxes, bbox = dict(boxstyle='round,pad = 0.3', facecolor = self.colors['background']), fontsize = 10, verticalalignment='top')
            plt.tight_layout()
            if save_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = self.output_path / f'lime_explanation_{explanation.model_name}_{timestamp}.png'
            plt.savefig(save_path, dpi = self.dpi, bbox_inches='tight')
            plt.close()
            self.logger.info(f'📊 LIME visualization saved to {save_path}')
            return str(save_path)
        except Exception as e:
            self.logger.error(f'❌ Failed to visualize LIME explanation: {e}')
            return None

    def visualize_feature_importance(self, explanation: ExplanationResult, max_features: int = 20, save_path: Optional[str]=None) -> Optional[str]:
        """Visualize feature importance."""
        try:
            if not MATPLOTLIB_AVAILABLE:
                self.logger.warning('⚠️ matplotlib not available for feature importance visualization')
                return None
            if explanation.feature_importance is None:
                self.logger.warning('⚠️ No feature importance available for visualization')
                return None
            fig, ax = plt.subplots(figsize = self.figure_size, dpi = self.dpi)
            feature_importance = list(explanation.feature_importance.items())
            feature_importance.sort(key = lambda x: abs(x[1]), reverse = True)
            feature_importance = feature_importance[:max_features]
            features = [item[0] for item in feature_importance]
            values = [item[1] for item in feature_importance]
            colors = [self.colors['positive'] if v > 0 else self.colors['negative'] for v in values]
            bars = ax.barh(range(len(features)), values, color = colors, alpha = 0.7)
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features, fontsize = 10)
            ax.set_xlabel('Feature Importance', fontsize = 12, fontweight='bold')
            ax.set_title(f'Feature Importance - {explanation.model_name}', fontsize = 14, fontweight='bold')
            ax.grid(True, alpha = 0.3)
            ax.axvline(x = 0, color='black', linestyle='-', alpha = 0.5)
            for i, (bar, value) in enumerate(zip(bars, values)):
                ax.text(value + (0.01 if value > 0 else -0.01), i, f'{value:.3f}', va='center', ha='left' if value > 0 else 'right', fontsize = 9)
            ax.invert_yaxis()
            plt.tight_layout()
            if save_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = self.output_path / f'feature_importance_{explanation.model_name}_{timestamp}.png'
            plt.savefig(save_path, dpi = self.dpi, bbox_inches='tight')
            plt.close()
            self.logger.info(f'📊 Feature importance visualization saved to {save_path}')
            return str(save_path)
        except Exception as e:
            self.logger.error(f'❌ Failed to visualize feature importance: {e}')
            return None

    def create_explanation_dashboard(self, explanation: ExplanationResult, save_path: Optional[str]=None) -> Optional[str]:
        """Create comprehensive explanation dashboard."""
        try:
            if not MATPLOTLIB_AVAILABLE:
                self.logger.warning('⚠️ matplotlib not available for dashboard creation')
                return None
            fig = plt.figure(figsize=(16, 12), dpi = self.dpi)
            gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1])
            if explanation.shap_values is not None:
                ax1 = fig.add_subplot(gs[0, 0])
                self._plot_shap_subplot(ax1, explanation)
            if explanation.lime_explanation is not None:
                ax2 = fig.add_subplot(gs[0, 1])
                self._plot_lime_subplot(ax2, explanation)
            if explanation.feature_importance is not None:
                ax3 = fig.add_subplot(gs[1, 0])
                self._plot_feature_importance_subplot(ax3, explanation)
            ax4 = fig.add_subplot(gs[1, 1])
            self._plot_prediction_summary_subplot(ax4, explanation)
            ax5 = fig.add_subplot(gs[2, 0])
            self._plot_feature_values_subplot(ax5, explanation)
            ax6 = fig.add_subplot(gs[2, 1])
            self._plot_metadata_subplot(ax6, explanation)
            fig.suptitle(f'Explanation Dashboard - {explanation.model_name}', fontsize = 16, fontweight='bold')
            plt.tight_layout()
            if save_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = self.output_path / f'explanation_dashboard_{explanation.model_name}_{timestamp}.png'
            plt.savefig(save_path, dpi = self.dpi, bbox_inches='tight')
            plt.close()
            self.logger.info(f'📊 Explanation dashboard saved to {save_path}')
            return str(save_path)
        except Exception as e:
            self.logger.error(f'❌ Failed to create explanation dashboard: {e}')
            return None

    def _plot_shap_subplot(self, ax: Any, explanation: ExplanationResult) -> None:
        """Plot SHAP values subplot."""
        try:
            if explanation.shap_values is None:
                ax.text(0.5, 0.5, 'No SHAP values available', ha='center', va='center', transform = ax.transAxes)
                return
            shap_values = explanation.shap_values
            feature_names = explanation.feature_names
            feature_importance = list(zip(feature_names, shap_values))
            feature_importance.sort(key = lambda x: abs(x[1]), reverse = True)
            feature_importance = feature_importance[:10]
            features = [item[0] for item in feature_importance]
            values = [item[1] for item in feature_importance]
            colors = [self.colors['positive'] if v > 0 else self.colors['negative'] for v in values]
            bars = ax.barh(range(len(features)), values, color = colors, alpha = 0.7)
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features, fontsize = 8)
            ax.set_xlabel('SHAP Value', fontsize = 10)
            ax.set_title('SHAP Values (Top 10)', fontsize = 12, fontweight='bold')
            ax.grid(True, alpha = 0.3)
            ax.axvline(x = 0, color='black', linestyle='-', alpha = 0.5)
            ax.invert_yaxis()
        except Exception as e:
            ax.text(0.5, 0.5, f'SHAP plot error: {e}', ha='center', va='center', transform = ax.transAxes)

    def _plot_lime_subplot(self, ax: Any, explanation: ExplanationResult) -> None:
        """Plot LIME explanation subplot."""
        try:
            if explanation.lime_explanation is None:
                ax.text(0.5, 0.5, 'No LIME explanation available', ha='center', va='center', transform = ax.transAxes)
                return
            lime_data = explanation.lime_explanation.get('feature_importance', [])
            if not lime_data:
                ax.text(0.5, 0.5, 'No LIME feature importance', ha='center', va='center', transform = ax.transAxes)
                return
            lime_data.sort(key = lambda x: abs(x[1]), reverse = True)
            lime_data = lime_data[:10]
            features = [item[0] for item in lime_data]
            values = [item[1] for item in lime_data]
            colors = [self.colors['positive'] if v > 0 else self.colors['negative'] for v in values]
            bars = ax.barh(range(len(features)), values, color = colors, alpha = 0.7)
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features, fontsize = 8)
            ax.set_xlabel('LIME Importance', fontsize = 10)
            ax.set_title('LIME Explanation (Top 10)', fontsize = 12, fontweight='bold')
            ax.grid(True, alpha = 0.3)
            ax.axvline(x = 0, color='black', linestyle='-', alpha = 0.5)
            ax.invert_yaxis()
        except Exception as e:
            ax.text(0.5, 0.5, f'LIME plot error: {e}', ha='center', va='center', transform = ax.transAxes)

    def _plot_feature_importance_subplot(self, ax: Any, explanation: ExplanationResult) -> None:
        """Plot feature importance subplot."""
        try:
            if explanation.feature_importance is None:
                ax.text(0.5, 0.5, 'No feature importance available', ha='center', va='center', transform = ax.transAxes)
                return
            feature_importance = list(explanation.feature_importance.items())
            feature_importance.sort(key = lambda x: abs(x[1]), reverse = True)
            feature_importance = feature_importance[:10]
            features = [item[0] for item in feature_importance]
            values = [item[1] for item in feature_importance]
            colors = [self.colors['positive'] if v > 0 else self.colors['negative'] for v in values]
            bars = ax.barh(range(len(features)), values, color = colors, alpha = 0.7)
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features, fontsize = 8)
            ax.set_xlabel('Importance', fontsize = 10)
            ax.set_title('Feature Importance (Top 10)', fontsize = 12, fontweight='bold')
            ax.grid(True, alpha = 0.3)
            ax.axvline(x = 0, color='black', linestyle='-', alpha = 0.5)
            ax.invert_yaxis()
        except Exception as e:
            ax.text(0.5, 0.5, f'Feature importance plot error: {e}', ha='center', va='center', transform = ax.transAxes)

    def _plot_prediction_summary_subplot(self, ax: Any, explanation: ExplanationResult) -> None:
        """Plot prediction summary subplot."""
        try:
            ax.axis('off')
            summary_text = f"\nModel: {explanation.model_name}\nPrediction: {explanation.prediction}\nConfidence: {explanation.confidence:.3f}\nTimestamp: {explanation.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\nFeatures Used: {len(explanation.feature_names)}\nSHAP Available: {('Yes' if explanation.shap_values is not None else 'No')}\nLIME Available: {('Yes' if explanation.lime_explanation is not None else 'No')}\nFeature Importance: {('Yes' if explanation.feature_importance is not None else 'No')}\n            ".strip()
            ax.text(0.05, 0.95, summary_text, transform = ax.transAxes, fontsize = 10, verticalalignment='top', fontfamily='monospace', bbox = dict(boxstyle='round,pad = 0.5', facecolor = self.colors['background']))
        except Exception as e:
            ax.text(0.5, 0.5, f'Summary plot error: {e}', ha='center', va='center', transform = ax.transAxes)

    def _plot_feature_values_subplot(self, ax: Any, explanation: ExplanationResult) -> None:
        """Plot feature values subplot."""
        try:
            if explanation.feature_values is None or len(explanation.feature_values) == 0:
                ax.text(0.5, 0.5, 'No feature values available', ha='center', va='center', transform = ax.transAxes)
                return
            feature_data = list(zip(explanation.feature_names, explanation.feature_values))
            feature_data.sort(key = lambda x: abs(x[1]), reverse = True)
            feature_data = feature_data[:10]
            features = [item[0] for item in feature_data]
            values = [item[1] for item in feature_data]
            colors = [self.colors['positive'] if v > 0 else self.colors['negative'] for v in values]
            bars = ax.barh(range(len(features)), values, color = colors, alpha = 0.7)
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features, fontsize = 8)
            ax.set_xlabel('Feature Value', fontsize = 10)
            ax.set_title('Feature Values (Top 10)', fontsize = 12, fontweight='bold')
            ax.grid(True, alpha = 0.3)
            ax.axvline(x = 0, color='black', linestyle='-', alpha = 0.5)
            ax.invert_yaxis()
        except Exception as e:
            ax.text(0.5, 0.5, f'Feature values plot error: {e}', ha='center', va='center', transform = ax.transAxes)

    def _plot_metadata_subplot(self, ax: Any, explanation: ExplanationResult) -> None:
        """Plot metadata subplot."""
        try:
            ax.axis('off')
            metadata_text = 'Model Metadata:\n\n'
            if explanation.metadata:
                for key, value in explanation.metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        metadata_text += f'{key}: {value}\n'
                    elif isinstance(value, list):
                        metadata_text += f'{key}: {len(value)} items\n'
                    elif isinstance(value, dict):
                        metadata_text += f'{key}: {len(value)} keys\n'
            if not explanation.metadata:
                metadata_text += 'No metadata available'
            ax.text(0.05, 0.95, metadata_text, transform = ax.transAxes, fontsize = 9, verticalalignment='top', fontfamily='monospace', bbox = dict(boxstyle='round,pad = 0.5', facecolor = self.colors['background']))
        except Exception as e:
            ax.text(0.5, 0.5, f'Metadata plot error: {e}', ha='center', va='center', transform = ax.transAxes)

class DecisionTraceVisualizer:
    """Visualizer for trade decision traces."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize decision trace visualizer."""
        self.config = config
        self.logger = system_logger.getChild('DecisionTraceVisualizer')
        self.viz_config = config.get('explainability', {}).get('visualization', {})
        self.output_path = Path(self.viz_config.get('output_path', 'data/visualizations'))
        self.output_path.mkdir(parents = True, exist_ok = True)
        self.style_config = self.viz_config.get('style', {})
        self.colors = self.style_config.get('colors', {'tactician': '#FF6B6B', 'hmm': '#4ECDC4', 'sr': '#45B7D1', 'analyst': '#96CEB4', 'risk': '#FF8C94', 'opportunity': '#98D8C8', 'neutral': '#F7DC6F'})
        self.figure_size = self.style_config.get('figure_size', (14, 10))
        self.dpi = self.style_config.get('dpi', 300)

    def visualize_decision_trace(self, trace: TradeDecisionTrace, save_path: Optional[str]=None) -> Optional[str]:
        """Visualize complete decision trace."""
        try:
            if not MATPLOTLIB_AVAILABLE:
                self.logger.warning('⚠️ matplotlib not available for decision trace visualization')
                return None
            fig = plt.figure(figsize = self.figure_size, dpi = self.dpi)
            gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1])
            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_decision_summary_subplot(ax1, trace)
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_model_contributions_subplot(ax2, trace)
            ax3 = fig.add_subplot(gs[1, 0])
            self._plot_top_factors_subplot(ax3, trace)
            ax4 = fig.add_subplot(gs[1, 1])
            self._plot_risk_opportunity_subplot(ax4, trace)
            ax5 = fig.add_subplot(gs[2, 0])
            self._plot_market_conditions_subplot(ax5, trace)
            ax6 = fig.add_subplot(gs[2, 1])
            self._plot_decision_timeline_subplot(ax6, trace)
            fig.suptitle(f'Decision Trace - {trace.decision_id}', fontsize = 16, fontweight='bold')
            plt.tight_layout()
            if save_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                save_path = self.output_path / f'decision_trace_{trace.decision_id}_{timestamp}.png'
            plt.savefig(save_path, dpi = self.dpi, bbox_inches='tight')
            plt.close()
            self.logger.info(f'📊 Decision trace visualization saved to {save_path}')
            return str(save_path)
        except Exception as e:
            self.logger.error(f'❌ Failed to visualize decision trace: {e}')
            return None

    def _plot_decision_summary_subplot(self, ax: Any, trace: TradeDecisionTrace) -> None:
        """Plot decision summary subplot."""
        try:
            ax.axis('off')
            summary_text = f"\nDecision ID: {trace.decision_id}\nType: {trace.decision_type}\nFinal Decision: {trace.final_decision}\nConfidence: {trace.confidence:.3f}\nTimestamp: {trace.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\nModels Used:\n• Tactician: {('Yes' if trace.tactician_explanation else 'No')}\n• HMM: {('Yes' if trace.hmm_explanation else 'No')}\n• SR: {('Yes' if trace.sr_explanation else 'No')}\n• Analyst: {('Yes' if trace.analyst_explanation else 'No')}\n\nFactors:\n• Top Contributing: {len(trace.top_contributing_factors)}\n• Risk Factors: {len(trace.risk_factors)}\n• Opportunity Factors: {len(trace.opportunity_factors)}\n            ".strip()
            ax.text(0.05, 0.95, summary_text, transform = ax.transAxes, fontsize = 10, verticalalignment='top', fontfamily='monospace', bbox = dict(boxstyle='round,pad = 0.5', facecolor='#F8F8FF'))
        except Exception as e:
            ax.text(0.5, 0.5, f'Summary plot error: {e}', ha='center', va='center', transform = ax.transAxes)

    def _plot_model_contributions_subplot(self, ax: Any, trace: TradeDecisionTrace) -> None:
        """Plot model contributions subplot."""
        try:
            models = ['Tactician', 'HMM', 'SR', 'Analyst']
            explanations = [trace.tactician_explanation, trace.hmm_explanation, trace.sr_explanation, trace.analyst_explanation]
            model_counts = [1 if exp else 0 for exp in explanations]
            colors = [self.colors[model.lower()] for model in models]
            bars = ax.bar(models, model_counts, color = colors, alpha = 0.7)
            ax.set_ylabel('Has Explanation', fontsize = 10)
            ax.set_title('Model Contributions', fontsize = 12, fontweight='bold')
            ax.set_ylim(0, 1.2)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['No', 'Yes'])
            for bar, count in zip(bars, model_counts):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, 'Yes' if count else 'No', ha='center', va='bottom', fontweight='bold')
        except Exception as e:
            ax.text(0.5, 0.5, f'Model contributions plot error: {e}', ha='center', va='center', transform = ax.transAxes)

    def _plot_top_factors_subplot(self, ax: Any, trace: TradeDecisionTrace) -> None:
        """Plot top contributing factors subplot."""
        try:
            if not trace.top_contributing_factors:
                ax.text(0.5, 0.5, 'No contributing factors available', ha='center', va='center', transform = ax.transAxes)
                return
            factors = trace.top_contributing_factors[:10]
            features = [factor.get('feature', 'Unknown') for factor in factors]
            importance = [factor.get('importance', 0) for factor in factors]
            models = [factor.get('model', 'Unknown') for factor in factors]
            colors = [self.colors.get(model.lower(), self.colors['neutral']) for model in models]
            bars = ax.barh(range(len(features)), importance, color = colors, alpha = 0.7)
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features, fontsize = 8)
            ax.set_xlabel('Importance', fontsize = 10)
            ax.set_title('Top Contributing Factors', fontsize = 12, fontweight='bold')
            ax.grid(True, alpha = 0.3)
            ax.invert_yaxis()
        except Exception as e:
            ax.text(0.5, 0.5, f'Top factors plot error: {e}', ha='center', va='center', transform = ax.transAxes)

    def _plot_risk_opportunity_subplot(self, ax: Any, trace: TradeDecisionTrace) -> None:
        """Plot risk vs opportunity factors subplot."""
        try:
            risk_count = len(trace.risk_factors)
            opportunity_count = len(trace.opportunity_factors)
            categories = ['Risk Factors', 'Opportunity Factors']
            counts = [risk_count, opportunity_count]
            colors = [self.colors['risk'], self.colors['opportunity']]
            bars = ax.bar(categories, counts, color = colors, alpha = 0.7)
            ax.set_ylabel('Number of Factors', fontsize = 10)
            ax.set_title('Risk vs Opportunity Factors', fontsize = 12, fontweight='bold')
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, str(count), ha='center', va='bottom', fontweight='bold')
        except Exception as e:
            ax.text(0.5, 0.5, f'Risk/opportunity plot error: {e}', ha='center', va='center', transform = ax.transAxes)

    def _plot_market_conditions_subplot(self, ax: Any, trace: TradeDecisionTrace) -> None:
        """Plot market conditions subplot."""
        try:
            ax.axis('off')
            if not trace.market_conditions:
                ax.text(0.5, 0.5, 'No market conditions available', ha='center', va='center', transform = ax.transAxes)
                return
            conditions_text = 'Market Conditions:\n\n'
            for key, value in trace.market_conditions.items():
                if isinstance(value, (int, float)):
                    conditions_text += f'{key}: {value:.3f}\n'
                else:
                    conditions_text += f'{key}: {value}\n'
            ax.text(0.05, 0.95, conditions_text, transform = ax.transAxes, fontsize = 9, verticalalignment='top', fontfamily='monospace', bbox = dict(boxstyle='round,pad = 0.5', facecolor='#F8F8FF'))
        except Exception as e:
            ax.text(0.5, 0.5, f'Market conditions plot error: {e}', ha='center', va='center', transform = ax.transAxes)

    def _plot_decision_timeline_subplot(self, ax: Any, trace: TradeDecisionTrace) -> None:
        """Plot decision timeline subplot."""
        try:
            ax.axis('off')
            timeline_text = f"\nDecision Timeline:\n\nStart: {trace.timestamp.strftime('%H:%M:%S')}\nType: {trace.decision_type}\nStatus: {('Completed' if trace.final_decision is not None else 'In Progress')}\n\nExplanation Sources:\n• Tactician: {('Available' if trace.tactician_explanation else 'Not Available')}\n• HMM: {('Available' if trace.hmm_explanation else 'Not Available')}\n• SR: {('Available' if trace.sr_explanation else 'Not Available')}\n• Analyst: {('Available' if trace.analyst_explanation else 'Not Available')}\n            ".strip()
            ax.text(0.05, 0.95, timeline_text, transform = ax.transAxes, fontsize = 9, verticalalignment='top', fontfamily='monospace', bbox = dict(boxstyle='round,pad = 0.5', facecolor='#F8F8FF'))
        except Exception as e:
            ax.text(0.5, 0.5, f'Timeline plot error: {e}', ha='center', va='center', transform = ax.transAxes)
