"""S/R Performance Monitor Module.

This module monitors S/R prediction performance in real-time and provides
detailed analytics for continuous improvement.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Deque
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import json
import os

from src.core.decorators import handles_errors, traced
from src.utils.logger import system_logger
from src.core.decorators.errors import handles_errors


@dataclass
class SRPrediction:
    """S/R prediction record."""
    timestamp: datetime
    level_price: float
    level_type: str  # 'support' or 'resistance'
    level_strength: float
    predicted_outcome: str  # 'breakout', 'rebounce', 'consolidation'
    outcome_probabilities: Dict[str, float]
    context: Dict[str, Any]
    method_used: str  # Which method/ensemble was used


@dataclass
class SROutcome:
    """Actual S/R interaction outcome."""
    timestamp: datetime
    level_price: float
    actual_outcome: str
    price_movement: float  # Percentage move
    time_to_outcome: int  # Bars until outcome
    volume_at_interaction: float
    profit_loss: float  # If traded


@dataclass
class PerformanceMetrics:
    """S/R performance metrics."""
    # Accuracy metrics
    overall_accuracy: float
    breakout_accuracy: float
    rebounce_accuracy: float
    consolidation_accuracy: float
    
    # Quality metrics
    avg_level_strength: float
    strong_level_accuracy: float  # Accuracy for high-strength levels
    weak_level_accuracy: float
    
    # Financial metrics
    total_pnl: float
    sharpe_ratio: float
    profit_factor: float
    win_rate: float
    
    # Timing metrics
    avg_time_to_outcome: float
    early_prediction_rate: float  # Predictions made well in advance
    
    # Context metrics
    accuracy_by_volatility: Dict[str, float]
    accuracy_by_session: Dict[str, float]
    accuracy_by_method: Dict[str, float]


class SRPerformanceMonitor:
    """Monitors S/R prediction performance in real-time."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("SRPerformanceMonitor")
        
        # Performance tracking
        self.predictions: Deque[SRPrediction] = deque(maxlen=10000)
        self.outcomes: Deque[SROutcome] = deque(maxlen=10000)
        self.matched_results: Deque[Tuple[SRPrediction, SROutcome]] = deque(maxlen=5000)
        
        # Rolling metrics
        self.rolling_window = config.get("performance_rolling_window", 500)
        self.alert_threshold = config.get("performance_alert_threshold", 0.45)
        
        # Performance by category
        self.performance_by_category = defaultdict(lambda: {
            'correct': 0, 'total': 0, 'pnl': 0.0
        })
        
        # Alert configuration
        self.alert_callback = None
        self.last_alert_time = None
        self.alert_cooldown = timedelta(hours=1)
        
        # Auto-save configuration
        self.auto_save_interval = config.get("performance_save_interval", 3600)  # 1 hour
        self.last_save_time = datetime.now()
        
    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="track SR prediction"
    )
    async def track_prediction(
        self,
        level_price: float,
        level_type: str,
        level_strength: float,
        predicted_outcome: str,
        outcome_probabilities: Dict[str, float],
        context: Dict[str, Any],
        method_used: str = "unknown"
    ) -> None:
        """Track a new S/R prediction."""
        
        prediction = SRPrediction(
            timestamp=datetime.now(),
            level_price=level_price,
            level_type=level_type,
            level_strength=level_strength,
            predicted_outcome=predicted_outcome,
            outcome_probabilities=outcome_probabilities,
            context=context,
            method_used=method_used
        )
        
        self.predictions.append(prediction)
        
        # Check if we need to save
        if (datetime.now() - self.last_save_time).total_seconds() > self.auto_save_interval:
            await self.save_performance_data()
    
    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="track SR outcome"
    )
    async def track_outcome(
        self,
        level_price: float,
        actual_outcome: str,
        price_movement: float,
        time_to_outcome: int,
        volume_at_interaction: float,
        profit_loss: float = 0.0
    ) -> None:
        """Track actual S/R interaction outcome."""
        
        outcome = SROutcome(
            timestamp=datetime.now(),
            level_price=level_price,
            actual_outcome=actual_outcome,
            price_movement=price_movement,
            time_to_outcome=time_to_outcome,
            volume_at_interaction=volume_at_interaction,
            profit_loss=profit_loss
        )
        
        self.outcomes.append(outcome)
        
        # Try to match with prediction
        matched_prediction = self._match_prediction_to_outcome(outcome)
        
        if matched_prediction:
            self.matched_results.append((matched_prediction, outcome))
            
            # Update category performance
            self._update_category_performance(matched_prediction, outcome)
            
            # Check performance and alert if needed
            await self._check_performance_alerts()
    
    def _match_prediction_to_outcome(self, outcome: SROutcome) -> Optional[SRPrediction]:
        """Match outcome to a recent prediction."""
        
        # Look for predictions near this price level
        for prediction in reversed(self.predictions):
            # Time constraint - outcome should be after prediction
            if prediction.timestamp > outcome.timestamp:
                continue
            
            # Price proximity constraint
            price_diff = abs(prediction.level_price - outcome.level_price) / outcome.level_price
            if price_diff < 0.001:  # Within 0.1%
                
                # Time window constraint - outcome within reasonable time
                time_diff = (outcome.timestamp - prediction.timestamp).total_seconds() / 3600
                if time_diff < 24:  # Within 24 hours
                    return prediction
        
        return None
    
    def _update_category_performance(
        self,
        prediction: SRPrediction,
        outcome: SROutcome
    ) -> None:
        """Update performance tracking by category."""
        
        # Overall performance
        is_correct = prediction.predicted_outcome == outcome.actual_outcome
        self.performance_by_category['overall']['total'] += 1
        if is_correct:
            self.performance_by_category['overall']['correct'] += 1
        self.performance_by_category['overall']['pnl'] += outcome.profit_loss
        
        # By outcome type
        outcome_key = f'outcome_{outcome.actual_outcome}'
        self.performance_by_category[outcome_key]['total'] += 1
        if is_correct:
            self.performance_by_category[outcome_key]['correct'] += 1
        
        # By level strength
        strength_key = 'strong' if prediction.level_strength > 0.7 else 'weak'
        self.performance_by_category[f'strength_{strength_key}']['total'] += 1
        if is_correct:
            self.performance_by_category[f'strength_{strength_key}']['correct'] += 1
        
        # By method
        method_key = f'method_{prediction.method_used}'
        self.performance_by_category[method_key]['total'] += 1
        if is_correct:
            self.performance_by_category[method_key]['correct'] += 1
        
        # By context (if available)
        if 'volatility_regime' in prediction.context:
            vol_key = f'volatility_{prediction.context["volatility_regime"]}'
            self.performance_by_category[vol_key]['total'] += 1
            if is_correct:
                self.performance_by_category[vol_key]['correct'] += 1
    
    async def _check_performance_alerts(self) -> None:
        """Check if performance has degraded and send alerts."""
        
        # Calculate rolling accuracy
        recent_matches = list(self.matched_results)[-self.rolling_window:]
        
        if len(recent_matches) < 50:  # Need minimum samples
            return
        
        correct = sum(1 for pred, outcome in recent_matches 
                     if pred.predicted_outcome == outcome.actual_outcome)
        accuracy = correct / len(recent_matches)
        
        # Check if below threshold
        if accuracy < self.alert_threshold:
            # Check cooldown
            if self.last_alert_time is None or \
               datetime.now() - self.last_alert_time > self.alert_cooldown:
                
                await self._send_performance_alert(accuracy)
                self.last_alert_time = datetime.now()
    
    async def _send_performance_alert(self, accuracy: float) -> None:
        """Send performance degradation alert."""
        
        self.logger.warning(
            f"⚠️ S/R Performance Alert: Accuracy dropped to {accuracy:.1%} "
            f"(threshold: {self.alert_threshold:.1%})"
        )
        
        if self.alert_callback:
            try:
                await self.alert_callback({
                    'type': 'sr_performance_degradation',
                    'accuracy': accuracy,
                    'threshold': self.alert_threshold,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                self.logger.error(f"Error sending alert: {e}")
    
    @traced(span_name="SRMonitor.calculate_metrics")
    def calculate_performance_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        
        if not self.matched_results:
            return self._get_empty_metrics()
        
        # Convert to lists for easier processing
        matches = list(self.matched_results)
        
        # Overall accuracy
        correct = sum(1 for pred, outcome in matches 
                     if pred.predicted_outcome == outcome.actual_outcome)
        overall_accuracy = correct / len(matches)
        
        # Accuracy by outcome type
        outcome_accuracy = {}
        for outcome_type in ['breakout', 'rebounce', 'consolidation']:
            relevant_matches = [(p, o) for p, o in matches if o.actual_outcome == outcome_type]
            if relevant_matches:
                correct = sum(1 for p, o in relevant_matches if p.predicted_outcome == o.actual_outcome)
                outcome_accuracy[outcome_type] = correct / len(relevant_matches)
            else:
                outcome_accuracy[outcome_type] = 0.0
        
        # Level strength analysis
        strong_matches = [(p, o) for p, o in matches if p.level_strength > 0.7]
        weak_matches = [(p, o) for p, o in matches if p.level_strength <= 0.7]
        
        strong_accuracy = (sum(1 for p, o in strong_matches if p.predicted_outcome == o.actual_outcome) / 
                          len(strong_matches)) if strong_matches else 0.0
        
        weak_accuracy = (sum(1 for p, o in weak_matches if p.predicted_outcome == o.actual_outcome) / 
                        len(weak_matches)) if weak_matches else 0.0
        
        # Financial metrics
        pnls = [o.profit_loss for _, o in matches]
        total_pnl = sum(pnls)
        
        returns = pd.Series(pnls)
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        
        wins = [pnl for pnl in pnls if pnl > 0]
        losses = [abs(pnl) for pnl in pnls if pnl < 0]
        
        profit_factor = sum(wins) / sum(losses) if losses else float('inf')
        win_rate = len(wins) / len(pnls) if pnls else 0.0
        
        # Timing metrics
        times_to_outcome = [o.time_to_outcome for _, o in matches]
        avg_time = np.mean(times_to_outcome) if times_to_outcome else 0.0
        
        # Early predictions (outcome within 5 bars)
        early_predictions = sum(1 for t in times_to_outcome if t <= 5)
        early_rate = early_predictions / len(times_to_outcome) if times_to_outcome else 0.0
        
        # Context-based accuracy
        accuracy_by_volatility = {}
        accuracy_by_method = {}
        
        # Calculate from category performance
        for key, stats in self.performance_by_category.items():
            if stats['total'] > 0:
                accuracy = stats['correct'] / stats['total']
                
                if key.startswith('volatility_'):
                    vol_regime = key.replace('volatility_', '')
                    accuracy_by_volatility[vol_regime] = accuracy
                elif key.startswith('method_'):
                    method = key.replace('method_', '')
                    accuracy_by_method[method] = accuracy
        
        return PerformanceMetrics(
            overall_accuracy=overall_accuracy,
            breakout_accuracy=outcome_accuracy['breakout'],
            rebounce_accuracy=outcome_accuracy['rebounce'],
            consolidation_accuracy=outcome_accuracy['consolidation'],
            avg_level_strength=np.mean([p.level_strength for p, _ in matches]),
            strong_level_accuracy=strong_accuracy,
            weak_level_accuracy=weak_accuracy,
            total_pnl=total_pnl,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            win_rate=win_rate,
            avg_time_to_outcome=avg_time,
            early_prediction_rate=early_rate,
            accuracy_by_volatility=accuracy_by_volatility,
            accuracy_by_session={},  # TODO: Implement session tracking
            accuracy_by_method=accuracy_by_method
        )
    
    def _get_empty_metrics(self) -> PerformanceMetrics:
        """Get empty metrics when no data available."""
        return PerformanceMetrics(
            overall_accuracy=0.0,
            breakout_accuracy=0.0,
            rebounce_accuracy=0.0,
            consolidation_accuracy=0.0,
            avg_level_strength=0.0,
            strong_level_accuracy=0.0,
            weak_level_accuracy=0.0,
            total_pnl=0.0,
            sharpe_ratio=0.0,
            profit_factor=0.0,
            win_rate=0.0,
            avg_time_to_outcome=0.0,
            early_prediction_rate=0.0,
            accuracy_by_volatility={},
            accuracy_by_session={},
            accuracy_by_method={}
        )
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate detailed performance report."""
        
        metrics = self.calculate_performance_metrics()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_metrics': asdict(metrics),
            'sample_size': len(self.matched_results),
            'prediction_count': len(self.predictions),
            'outcome_count': len(self.outcomes),
            'match_rate': len(self.matched_results) / len(self.predictions) if self.predictions else 0,
            
            # Detailed breakdowns
            'accuracy_trend': self._calculate_accuracy_trend(),
            'performance_by_hour': self._analyze_performance_by_hour(),
            'confidence_calibration': self._analyze_confidence_calibration(),
            'optimization_recommendations': self._generate_recommendations(metrics)
        }
        
        return report
    
    def _calculate_accuracy_trend(self) -> List[Dict[str, Any]]:
        """Calculate accuracy trend over time."""
        
        if not self.matched_results:
            return []
        
        # Group by day
        daily_performance = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for pred, outcome in self.matched_results:
            day = pred.timestamp.date()
            daily_performance[day]['total'] += 1
            if pred.predicted_outcome == outcome.actual_outcome:
                daily_performance[day]['correct'] += 1
        
        # Calculate daily accuracy
        trend = []
        for day in sorted(daily_performance.keys()):
            stats = daily_performance[day]
            trend.append({
                'date': day.isoformat(),
                'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0,
                'sample_size': stats['total']
            })
        
        return trend
    
    def _analyze_performance_by_hour(self) -> Dict[int, float]:
        """Analyze performance by hour of day."""
        
        hourly_performance = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for pred, outcome in self.matched_results:
            hour = pred.timestamp.hour
            hourly_performance[hour]['total'] += 1
            if pred.predicted_outcome == outcome.actual_outcome:
                hourly_performance[hour]['correct'] += 1
        
        # Calculate hourly accuracy
        return {
            hour: stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            for hour, stats in hourly_performance.items()
        }
    
    def _analyze_confidence_calibration(self) -> Dict[str, Any]:
        """Analyze if confidence scores are well-calibrated."""
        
        # Group predictions by confidence buckets
        confidence_buckets = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for pred, outcome in self.matched_results:
            # Get confidence for predicted outcome
            confidence = pred.outcome_probabilities.get(pred.predicted_outcome, 0.5)
            bucket = int(confidence * 10) / 10  # Round to nearest 0.1
            
            confidence_buckets[bucket]['total'] += 1
            if pred.predicted_outcome == outcome.actual_outcome:
                confidence_buckets[bucket]['correct'] += 1
        
        # Calculate calibration
        calibration = {}
        for bucket in sorted(confidence_buckets.keys()):
            stats = confidence_buckets[bucket]
            if stats['total'] > 0:
                actual_accuracy = stats['correct'] / stats['total']
                calibration[f"{bucket:.1f}"] = {
                    'expected': bucket,
                    'actual': actual_accuracy,
                    'samples': stats['total']
                }
        
        # Calculate calibration error
        calibration_errors = [
            abs(data['expected'] - data['actual']) 
            for data in calibration.values()
        ]
        avg_calibration_error = np.mean(calibration_errors) if calibration_errors else 0
        
        return {
            'buckets': calibration,
            'average_calibration_error': avg_calibration_error,
            'is_well_calibrated': avg_calibration_error < 0.1
        }
    
    def _generate_recommendations(self, metrics: PerformanceMetrics) -> List[str]:
        """Generate recommendations based on performance."""
        
        recommendations = []
        
        # Overall accuracy
        if metrics.overall_accuracy < 0.5:
            recommendations.append(
                "Overall accuracy is below 50%. Consider re-optimizing S/R parameters."
            )
        
        # Outcome-specific issues
        if metrics.breakout_accuracy < 0.4:
            recommendations.append(
                "Breakout predictions are underperforming. "
                "Consider adjusting momentum and volatility weights."
            )
        
        if metrics.rebounce_accuracy < 0.4:
            recommendations.append(
                "Rebounce predictions are weak. "
                "Review bounce ratio thresholds and touch requirements."
            )
        
        # Level strength correlation
        if metrics.strong_level_accuracy < metrics.weak_level_accuracy:
            recommendations.append(
                "Strong levels performing worse than weak levels. "
                "Strength calculation may need recalibration."
            )
        
        # Financial performance
        if metrics.sharpe_ratio < 0.5:
            recommendations.append(
                "Low Sharpe ratio indicates poor risk-adjusted returns. "
                "Consider more selective entry criteria."
            )
        
        # Method-specific issues
        worst_method = None
        worst_accuracy = 1.0
        
        for method, accuracy in metrics.accuracy_by_method.items():
            if accuracy < worst_accuracy:
                worst_accuracy = accuracy
                worst_method = method
        
        if worst_method and worst_accuracy < 0.4:
            recommendations.append(
                f"Method '{worst_method}' is underperforming ({worst_accuracy:.1%}). "
                "Consider adjusting its weight in the ensemble."
            )
        
        # Volatility regime performance
        for regime, accuracy in metrics.accuracy_by_volatility.items():
            if accuracy < 0.45:
                recommendations.append(
                    f"Poor performance in {regime} volatility ({accuracy:.1%}). "
                    "Review context-aware adjustments for this regime."
                )
        
        return recommendations
    
    async def save_performance_data(self) -> None:
        """Save performance data to file."""
        
        try:
            data_file = os.path.join(
                self.config.get("model_save_path", "models"),
                "sr_performance_data.json"
            )
            
            # Prepare data for saving
            save_data = {
                'timestamp': datetime.now().isoformat(),
                'predictions': [
                    {
                        'timestamp': p.timestamp.isoformat(),
                        'level_price': p.level_price,
                        'level_type': p.level_type,
                        'level_strength': p.level_strength,
                        'predicted_outcome': p.predicted_outcome,
                        'outcome_probabilities': p.outcome_probabilities,
                        'method_used': p.method_used
                    }
                    for p in list(self.predictions)[-1000:]  # Save last 1000
                ],
                'performance_by_category': dict(self.performance_by_category),
                'current_metrics': asdict(self.calculate_performance_metrics())
            }
            
            with open(data_file, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            self.last_save_time = datetime.now()
            self.logger.info(f"Saved performance data to {data_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving performance data: {e}")
    
    async def load_performance_data(self) -> None:
        """Load historical performance data."""
        
        try:
            data_file = os.path.join(
                self.config.get("model_save_path", "models"),
                "sr_performance_data.json"
            )
            
            if os.path.exists(data_file):
                with open(data_file, 'r') as f:
                    data = json.load(f)
                
                # Restore performance by category
                if 'performance_by_category' in data:
                    for key, stats in data['performance_by_category'].items():
                        self.performance_by_category[key] = stats
                
                self.logger.info(f"Loaded performance data from {data_file}")
                
        except Exception as e:
            self.logger.error(f"Error loading performance data: {e}")
    
    def set_alert_callback(self, callback) -> None:
        """Set callback for performance alerts."""
        self.alert_callback = callback


# Factory function
async def create_sr_performance_monitor(config: Dict[str, Any]) -> SRPerformanceMonitor:
    """Create and initialize S/R performance monitor."""
    monitor = SRPerformanceMonitor(config)
    await monitor.load_performance_data()
    return monitor