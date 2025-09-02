"""
Trend Reporter

Tracks code quality metrics over time and provides historical analysis.
Generates trend reports showing improvements or regressions in code quality.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import statistics
from collections import defaultdict

from ..core.config import ReportingConfig


@dataclass
class TrendPoint:
    """Container for a single trend data point."""
    timestamp: str
    metrics: Dict[str, float]
    file_count: int
    total_issues: int
    quality_score: float


@dataclass
class TrendAnalysis:
    """Container for trend analysis results."""
    metric_trends: Dict[str, Dict[str, float]]
    overall_trend: str  # 'improving', 'stable', 'declining'
    period_summary: Dict[str, Any]
    recommendations: List[str]


@dataclass
class TrendReport:
    """Container for comprehensive trend report."""
    project_name: str
    analysis_period: str
    data_points: List[TrendPoint]
    analysis: TrendAnalysis
    generated_at: str


class TrendReporter:
    """
    Analyzes code quality trends over time.
    
    Features:
    - Historical metric tracking
    - Trend analysis and forecasting
    - Performance comparisons
    - Improvement recommendations
    """
    
    def __init__(self, config: Optional[ReportingConfig] = None):
        """
        Initialize the trend reporter.
        
        Args:
            config: Reporting configuration
        """
        self.config = config or ReportingConfig()
        self.history_file = Path(self.config.report_dir) / "quality_history.json"
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        
    def add_data_point(self, metrics: Dict[str, Any], project_name: str = "default") -> None:
        """
        Add a new data point to the quality history.
        
        Args:
            metrics: Current quality metrics
            project_name: Name of the project
        """
        history = self._load_history()
        
        if project_name not in history:
            history[project_name] = []
        
        # Create trend point
        trend_point = TrendPoint(
            timestamp=datetime.now().isoformat(),
            metrics=self._extract_metrics(metrics),
            file_count=metrics.get('total_files', 0),
            total_issues=metrics.get('total_issues', 0),
            quality_score=metrics.get('quality_score', 0.0)
        )
        
        history[project_name].append(asdict(trend_point))
        
        # Keep only last 100 data points per project
        if len(history[project_name]) > 100:
            history[project_name] = history[project_name][-100:]
        
        self._save_history(history)
    
    def generate_trend_report(self, project_name: str = "default", 
                            days: int = 30) -> TrendReport:
        """
        Generate trend report for a project.
        
        Args:
            project_name: Name of the project
            days: Number of days to analyze
            
        Returns:
            TrendReport object
        """
        history = self._load_history()
        
        if project_name not in history:
            raise ValueError(f"No history found for project: {project_name}")
        
        # Filter data points by date
        cutoff_date = datetime.now() - timedelta(days=days)
        filtered_points = []
        
        for point_data in history[project_name]:
            point_date = datetime.fromisoformat(point_data['timestamp'])
            if point_date >= cutoff_date:
                filtered_points.append(TrendPoint(**point_data))
        
        if not filtered_points:
            raise ValueError(f"No data points found in the last {days} days")
        
        # Sort by timestamp
        filtered_points.sort(key=lambda x: x.timestamp)
        
        # Analyze trends
        analysis = self._analyze_trends(filtered_points)
        
        return TrendReport(
            project_name=project_name,
            analysis_period=f"Last {days} days",
            data_points=filtered_points,
            analysis=analysis,
            generated_at=datetime.now().isoformat()
        )
    
    def _load_history(self) -> Dict[str, List]:
        """Load quality history from file."""
        if not self.history_file.exists():
            return {}
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    
    def _save_history(self, history: Dict[str, List]) -> None:
        """Save quality history to file."""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save history: {e}")
    
    def _extract_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract relevant metrics for trend analysis."""
        metrics = {}
        
        # Extract numeric metrics
        for key, value in data.items():
            if isinstance(value, (int, float)):
                metrics[key] = float(value)
            elif isinstance(value, dict):
                # Handle nested metrics
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, (int, float)):
                        metrics[f"{key}_{nested_key}"] = float(nested_value)
        
        return metrics
    
    def _analyze_trends(self, data_points: List[TrendPoint]) -> TrendAnalysis:
        """Analyze trends in the data points."""
        if len(data_points) < 2:
            return TrendAnalysis(
                metric_trends={},
                overall_trend='stable',
                period_summary={},
                recommendations=['Need more data points for trend analysis']
            )
        
        # Analyze each metric
        metric_trends = {}
        for metric_name in data_points[0].metrics.keys():
            values = [point.metrics.get(metric_name, 0) for point in data_points]
            trend_info = self._calculate_trend(values)
            metric_trends[metric_name] = trend_info
        
        # Calculate overall trend
        overall_trend = self._calculate_overall_trend(data_points)
        
        # Generate period summary
        period_summary = self._generate_period_summary(data_points)
        
        # Generate recommendations
        recommendations = self._generate_trend_recommendations(metric_trends, overall_trend)
        
        return TrendAnalysis(
            metric_trends=metric_trends,
            overall_trend=overall_trend,
            period_summary=period_summary,
            recommendations=recommendations
        )
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, float]:
        """Calculate trend for a single metric."""
        if len(values) < 2:
            return {'slope': 0.0, 'change_rate': 0.0, 'trend': 'stable'}
        
        # Calculate linear regression slope
        n = len(values)
        x_values = list(range(n))
        
        # Simple linear regression
        sum_x = sum(x_values)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(x_values, values))
        sum_x2 = sum(x * x for x in x_values)
        
        if n * sum_x2 - sum_x * sum_x == 0:
            slope = 0.0
        else:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Calculate change rate
        if values[0] != 0:
            change_rate = ((values[-1] - values[0]) / values[0]) * 100
        else:
            change_rate = 0.0
        
        # Determine trend direction
        if abs(slope) < 0.01:
            trend = 'stable'
        elif slope > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'
        
        return {
            'slope': slope,
            'change_rate': change_rate,
            'trend': trend,
            'start_value': values[0],
            'end_value': values[-1],
            'min_value': min(values),
            'max_value': max(values),
            'mean_value': statistics.mean(values)
        }
    
    def _calculate_overall_trend(self, data_points: List[TrendPoint]) -> str:
        """Calculate overall trend based on quality score."""
        quality_scores = [point.quality_score for point in data_points]
        trend_info = self._calculate_trend(quality_scores)
        
        if trend_info['slope'] > 0.1:
            return 'improving'
        elif trend_info['slope'] < -0.1:
            return 'declining'
        else:
            return 'stable'
    
    def _generate_period_summary(self, data_points: List[TrendPoint]) -> Dict[str, Any]:
        """Generate summary statistics for the analysis period."""
        if not data_points:
            return {}
        
        # Calculate summary statistics
        quality_scores = [point.quality_score for point in data_points]
        issue_counts = [point.total_issues for point in data_points]
        file_counts = [point.file_count for point in data_points]
        
        summary = {
            'data_points': len(data_points),
            'start_date': data_points[0].timestamp,
            'end_date': data_points[-1].timestamp,
            'quality_score': {
                'start': quality_scores[0],
                'end': quality_scores[-1],
                'min': min(quality_scores),
                'max': max(quality_scores),
                'mean': statistics.mean(quality_scores),
                'improvement': quality_scores[-1] - quality_scores[0]
            },
            'total_issues': {
                'start': issue_counts[0],
                'end': issue_counts[-1],
                'min': min(issue_counts),
                'max': max(issue_counts),
                'mean': statistics.mean(issue_counts),
                'change': issue_counts[-1] - issue_counts[0]
            },
            'file_count': {
                'start': file_counts[0],
                'end': file_counts[-1],
                'min': min(file_counts),
                'max': max(file_counts),
                'mean': statistics.mean(file_counts),
                'growth': file_counts[-1] - file_counts[0]
            }
        }
        
        return summary
    
    def _generate_trend_recommendations(self, metric_trends: Dict[str, Dict], 
                                      overall_trend: str) -> List[str]:
        """Generate recommendations based on trend analysis."""
        recommendations = []
        
        # Overall trend recommendations
        if overall_trend == 'improving':
            recommendations.append("🎉 Code quality is improving! Keep up the good work.")
        elif overall_trend == 'declining':
            recommendations.append("⚠️ Code quality is declining. Review recent changes and consider additional testing.")
        else:
            recommendations.append("📊 Code quality is stable. Consider setting improvement goals.")
        
        # Metric-specific recommendations
        for metric_name, trend_info in metric_trends.items():
            if trend_info['trend'] == 'increasing' and 'error' in metric_name.lower():
                recommendations.append(f"⚠️ {metric_name} is increasing. Review error handling practices.")
            elif trend_info['trend'] == 'decreasing' and 'quality' in metric_name.lower():
                recommendations.append(f"🔴 {metric_name} is decreasing. Investigate recent changes.")
            elif trend_info['trend'] == 'stable' and 'complexity' in metric_name.lower():
                recommendations.append(f"📈 {metric_name} is stable. Consider refactoring to reduce complexity.")
        
        # General recommendations
        if len(metric_trends) > 5:
            recommendations.append("📊 Many metrics tracked. Consider focusing on the most critical ones.")
        
        return recommendations
    
    def compare_periods(self, project_name: str, period1_days: int, 
                       period2_days: int) -> Dict[str, Any]:
        """
        Compare two time periods.
        
        Args:
            project_name: Name of the project
            period1_days: Days for first period
            period2_days: Days for second period
            
        Returns:
            Comparison results
        """
        try:
            report1 = self.generate_trend_report(project_name, period1_days)
            report2 = self.generate_trend_report(project_name, period2_days)
            
            comparison = {
                'period1': {
                    'days': period1_days,
                    'summary': report1.analysis.period_summary
                },
                'period2': {
                    'days': period2_days,
                    'summary': report2.analysis.period_summary
                },
                'changes': {}
            }
            
            # Calculate changes between periods
            for metric in ['quality_score', 'total_issues', 'file_count']:
                if metric in report1.analysis.period_summary and metric in report2.analysis.period_summary:
                    p1_data = report1.analysis.period_summary[metric]
                    p2_data = report2.analysis.period_summary[metric]
                    
                    comparison['changes'][metric] = {
                        'mean_change': p2_data['mean'] - p1_data['mean'],
                        'improvement_rate': ((p2_data['end'] - p1_data['end']) / p1_data['end'] * 100) if p1_data['end'] != 0 else 0
                    }
            
            return comparison
            
        except Exception as e:
            return {'error': str(e)}
    
    def export_trend_data(self, project_name: str, format: str = 'json', 
                          output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Export trend data in various formats.
        
        Args:
            project_name: Name of the project
            format: Export format ('json', 'csv', 'text')
            output_path: Optional path to save the export
            
        Returns:
            Exported data content
        """
        history = self._load_history()
        
        if project_name not in history:
            raise ValueError(f"No history found for project: {project_name}")
        
        if format.lower() == 'json':
            content = json.dumps(history[project_name], indent=2)
        elif format.lower() == 'csv':
            content = self._export_to_csv(history[project_name])
        elif format.lower() == 'text':
            content = self._export_to_text(history[project_name], project_name)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Save to file if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        return content
    
    def _export_to_csv(self, data: List[Dict]) -> str:
        """Export trend data to CSV format."""
        if not data:
            return ""
        
        import csv
        from io import StringIO
        
        output = StringIO()
        writer = csv.writer(output)
        
        # Header
        if data:
            headers = ['timestamp'] + list(data[0]['metrics'].keys()) + ['file_count', 'total_issues', 'quality_score']
            writer.writerow(headers)
            
            # Data rows
            for point in data:
                row = [point['timestamp']]
                row.extend(point['metrics'].values())
                row.extend([point['file_count'], point['total_issues'], point['quality_score']])
                writer.writerow(row)
        
        return output.getvalue()
    
    def _export_to_text(self, data: List[Dict], project_name: str) -> str:
        """Export trend data to text format."""
        if not data:
            return f"No trend data available for {project_name}"
        
        lines = []
        lines.append(f"TREND DATA REPORT - {project_name.upper()}")
        lines.append("=" * 50)
        lines.append(f"Data Points: {len(data)}")
        lines.append(f"Period: {data[0]['timestamp']} to {data[-1]['timestamp']}")
        lines.append("")
        
        # Summary statistics
        quality_scores = [point['quality_score'] for point in data]
        issue_counts = [point['total_issues'] for point in data]
        
        lines.append("SUMMARY STATISTICS")
        lines.append("-" * 20)
        lines.append(f"Quality Score: {min(quality_scores):.2f} - {max(quality_scores):.2f} (avg: {statistics.mean(quality_scores):.2f})")
        lines.append(f"Total Issues: {min(issue_counts)} - {max(issue_counts)} (avg: {statistics.mean(issue_counts):.1f})")
        lines.append("")
        
        # Recent trends
        lines.append("RECENT TRENDS (Last 5 points)")
        lines.append("-" * 30)
        for i, point in enumerate(data[-5:]):
            lines.append(f"{i+1}. {point['timestamp'][:10]}: Score={point['quality_score']:.2f}, Issues={point['total_issues']}")
        
        return '\n'.join(lines)
    
    def get_project_list(self) -> List[str]:
        """Get list of available projects."""
        history = self._load_history()
        return list(history.keys())
    
    def clear_history(self, project_name: Optional[str] = None) -> None:
        """
        Clear quality history.
        
        Args:
            project_name: Specific project to clear, or None for all
        """
        if project_name:
            history = self._load_history()
            if project_name in history:
                del history[project_name]
                self._save_history(history)
        else:
            # Clear all history
            if self.history_file.exists():
                self.history_file.unlink()