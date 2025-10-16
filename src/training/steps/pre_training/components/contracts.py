"""
Contracts for pre-training pipeline components.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ComponentContract:
    """Contract for component behavior."""
    
    # Contract identification
    contract_id: str
    contract_version: str = "1.0.0"
    
    # Required methods
    required_methods: List[str] = None
    
    # Required attributes
    required_attributes: List[str] = None
    
    # Validation rules
    validation_rules: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.required_methods is None:
            self.required_methods = []
        if self.required_attributes is None:
            self.required_attributes = []
        if self.validation_rules is None:
            self.validation_rules = {}
    
    def validate_component(self, component: Any) -> bool:
        """Validate that a component satisfies this contract."""
        try:
            # Check required methods
            for method_name in self.required_methods:
                if not hasattr(component, method_name):
                    logger.error(f"Component missing required method: {method_name}")
                    return False
                if not callable(getattr(component, method_name)):
                    logger.error(f"Component attribute {method_name} is not callable")
                    return False
            
            # Check required attributes
            for attr_name in self.required_attributes:
                if not hasattr(component, attr_name):
                    logger.error(f"Component missing required attribute: {attr_name}")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating component contract: {e}")
            return False

class ComponentContractValidator:
    """Validator for component contracts."""
    
    def __init__(self):
        self.contracts: Dict[str, ComponentContract] = {}
    
    def register_contract(self, contract: ComponentContract) -> None:
        """Register a component contract."""
        self.contracts[contract.contract_id] = contract
        logger.info(f"Registered contract: {contract.contract_id}")
    
    def validate_component(self, component: Any, contract_id: str) -> bool:
        """Validate a component against a specific contract."""
        if contract_id not in self.contracts:
            logger.error(f"Contract not found: {contract_id}")
            return False
        
        contract = self.contracts[contract_id]
        return contract.validate_component(component)
    
    def get_contract(self, contract_id: str) -> Optional[ComponentContract]:
        """Get a contract by ID."""
        return self.contracts.get(contract_id)
    
    def list_contracts(self) -> List[str]:
        """List all registered contract IDs."""
        return list(self.contracts.keys())

# Standard contracts
STANDARD_CONTRACTS = {
    "base_component": ComponentContract(
        contract_id="base_component",
        required_methods=["process", "validate"],
        required_attributes=["config", "logger"]
    ),
    "pre_training_component": ComponentContract(
        contract_id="pre_training_component",
        required_methods=["process", "validate", "get_status"],
        required_attributes=["component_type", "config", "logger"]
    )
}

def get_standard_contract(contract_id: str) -> Optional[ComponentContract]:
    """Get a standard contract by ID."""
    return STANDARD_CONTRACTS.get(contract_id)

def validate_component_contract(component: Any, contract_id: str) -> bool:
    """Validate a component against a standard contract."""
    contract = get_standard_contract(contract_id)
    if contract is None:
        logger.error(f"Standard contract not found: {contract_id}")
        return False
    
    return contract.validate_component(component)

@dataclass
class GenericArtifacts:
    """Generic artifacts for component results."""
    
    # Basic artifacts
    data: Any = None
    metadata: Dict[str, Any] = None
    status: str = "pending"  # pending, completed, failed
    
    # Results
    results: Dict[str, Any] = None
    errors: List[str] = None
    
    # Timestamps
    created_at: str = None
    updated_at: str = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.metadata is None:
            self.metadata = {}
        if self.results is None:
            self.results = {}
        if self.errors is None:
            self.errors = []
        if self.created_at is None:
            from datetime import datetime
            self.created_at = datetime.now().isoformat()
        if self.updated_at is None:
            from datetime import datetime
            self.updated_at = datetime.now().isoformat()
    
    def add_result(self, key: str, value: Any) -> None:
        """Add a result."""
        self.results[key] = value
        from datetime import datetime
        self.updated_at = datetime.now().isoformat()
    
    def add_error(self, error: str) -> None:
        """Add an error."""
        self.errors.append(error)
        self.status = "failed"
        from datetime import datetime
        self.updated_at = datetime.now().isoformat()
    
    def set_status(self, status: str) -> None:
        """Set the status."""
        self.status = status
        from datetime import datetime
        self.updated_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert artifacts to dictionary."""
        return {
            'data': self.data,
            'metadata': self.metadata,
            'status': self.status,
            'results': self.results,
            'errors': self.errors,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GenericArtifacts':
        """Create artifacts from dictionary."""
        return cls(**data)

@dataclass
class MultiHorizonArtifacts:
    """Artifacts for multi-horizon analysis."""
    
    # Horizon data
    horizons: List[str] = None
    horizon_data: Dict[str, Any] = None
    
    # Analysis results
    analysis_results: Dict[str, Any] = None
    performance_metrics: Dict[str, Any] = None
    
    # Metadata
    metadata: Dict[str, Any] = None
    created_at: str = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.horizons is None:
            self.horizons = []
        if self.horizon_data is None:
            self.horizon_data = {}
        if self.analysis_results is None:
            self.analysis_results = {}
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            from datetime import datetime
            self.created_at = datetime.now().isoformat()
    
    def add_horizon(self, horizon: str, data: Any) -> None:
        """Add data for a specific horizon."""
        self.horizons.append(horizon)
        self.horizon_data[horizon] = data
    
    def get_horizon_data(self, horizon: str) -> Any:
        """Get data for a specific horizon."""
        return self.horizon_data.get(horizon)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert artifacts to dictionary."""
        return {
            'horizons': self.horizons,
            'horizon_data': self.horizon_data,
            'analysis_results': self.analysis_results,
            'performance_metrics': self.performance_metrics,
            'metadata': self.metadata,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MultiHorizonArtifacts':
        """Create artifacts from dictionary."""
        return cls(**data)

@dataclass
class PipelineState:
    """State of the pre-training pipeline."""
    
    # Pipeline identification
    pipeline_id: str
    pipeline_version: str = "1.0.0"
    
    # Current state
    current_step: str = "initialized"
    status: str = "running"  # running, completed, failed, paused
    
    # Progress tracking
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    
    # Data state
    input_data: Any = None
    processed_data: Any = None
    output_data: Any = None
    
    # Metadata
    metadata: Dict[str, Any] = None
    created_at: str = None
    updated_at: str = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            from datetime import datetime
            self.created_at = datetime.now().isoformat()
        if self.updated_at is None:
            from datetime import datetime
            self.updated_at = datetime.now().isoformat()
    
    def update_status(self, status: str) -> None:
        """Update the pipeline status."""
        self.status = status
        from datetime import datetime
        self.updated_at = datetime.now().isoformat()
    
    def advance_step(self, step_name: str) -> None:
        """Advance to the next step."""
        self.current_step = step_name
        self.completed_steps += 1
        from datetime import datetime
        self.updated_at = datetime.now().isoformat()
    
    def get_progress(self) -> float:
        """Get the progress percentage."""
        if self.total_steps == 0:
            return 0.0
        return (self.completed_steps / self.total_steps) * 100
    
    def is_complete(self) -> bool:
        """Check if the pipeline is complete."""
        return self.status == "completed"
    
    def is_failed(self) -> bool:
        """Check if the pipeline has failed."""
        return self.status == "failed"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            'pipeline_id': self.pipeline_id,
            'pipeline_version': self.pipeline_version,
            'current_step': self.current_step,
            'status': self.status,
            'total_steps': self.total_steps,
            'completed_steps': self.completed_steps,
            'failed_steps': self.failed_steps,
            'input_data': self.input_data,
            'processed_data': self.processed_data,
            'output_data': self.output_data,
            'metadata': self.metadata,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineState':
        """Create state from dictionary."""
        return cls(**data)
