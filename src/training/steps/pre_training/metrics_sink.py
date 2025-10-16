"""
Metrics sink for pre-training pipeline.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class MetricsSinkConfig:
    """Configuration for metrics sink."""
    name: str
    enabled: bool = True
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class MetricsSink(ABC):
    """Base class for metrics sinks."""
    
    def __init__(self, config: MetricsSinkConfig):
        self.config = config
        self.name = config.name
        self.enabled = config.enabled
    
    @abstractmethod
    def record_metric(self, name: str, value: Any, **kwargs):
        """Record a metric."""
        pass
    
    @abstractmethod
    def record_metrics(self, metrics: Dict[str, Any], **kwargs):
        """Record multiple metrics."""
        pass
