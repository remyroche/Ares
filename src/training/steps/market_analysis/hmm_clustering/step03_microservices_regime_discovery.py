#!/usr/bin/env python3
"""Microservices Architecture for Regime Discovery.

This module implements a microservices-based architecture for regime discovery,
breaking the monolithic step into specialized services with async communication.
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
import json
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ServiceRequest:
    """Standard service request format."""
    service_id: str
    request_id: str
    data: Any
    metadata: Dict[str, Any]
    timestamp: datetime
    priority: int = 1  # 1=high, 2=medium, 3=low

@dataclass
class ServiceResponse:
    """Standard service response format."""
    service_id: str
    request_id: str
    data: Any
    metadata: Dict[str, Any]
    timestamp: datetime
    success: bool
    error: Optional[str] = None
    processing_time: float = 0.0

class BaseService(ABC):
    """Base class for all microservices."""
    
    def __init__(self, service_id: str, config: Dict[str, Any] = None):
        self.service_id = service_id
        self.config = config or {}
        self.is_initialized = False
        self.request_count = 0
        self.total_processing_time = 0.0
        
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the service."""
        pass
    
    @abstractmethod
    async def process_request(self, request: ServiceRequest) -> ServiceResponse:
        """Process a service request."""
        pass
    
    async def handle_request(self, request: ServiceRequest) -> ServiceResponse:
        """Handle incoming request with error handling and metrics."""
        start_time = time.time()
        self.request_count += 1
        
        try:
            if not self.is_initialized:
                await self.initialize()
            
            response = await self.process_request(request)
            response.processing_time = time.time() - start_time
            self.total_processing_time += response.processing_time
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            return ServiceResponse(
                service_id=self.service_id,
                request_id=request.request_id,
                data=None,
                metadata={'error': str(e)},
                timestamp=datetime.now(),
                success=False,
                error=str(e),
                processing_time=processing_time
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics."""
        return {
            'service_id': self.service_id,
            'request_count': self.request_count,
            'total_processing_time': self.total_processing_time,
            'avg_processing_time': self.total_processing_time / max(self.request_count, 1),
            'is_initialized': self.is_initialized
        }

class OptimizationService(BaseService):
    """Bayesian parameter optimization service."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("optimization_service", config)
        self.optimizer = None
        
    async def initialize(self) -> None:
        """Initialize the optimization service."""
        from .step03_optimized_bayesian_optimization import OptimizedBayesianParameterOptimization
        self.optimizer = OptimizedBayesianParameterOptimization(self.config)
        await self.optimizer.initialize()
        self.is_initialized = True
        
    async def process_request(self, request: ServiceRequest) -> ServiceResponse:
        """Process optimization request."""
        data = request.data
        features = data.get('features')
        market_data = data.get('market_data')
        
        # Run optimization
        optimization_results = await self.optimizer.optimize_parameters(market_data, features)
        
        return ServiceResponse(
            service_id=self.service_id,
            request_id=request.request_id,
            data=optimization_results,
            metadata={'optimization_completed': True},
            timestamp=datetime.now(),
            success=True
        )

class FeatureService(BaseService):
    """Feature engineering service."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("feature_service", config)
        self.feature_engineer = None
        
    async def initialize(self) -> None:
        """Initialize the feature service."""
        from .step03_regime_discovery_features import RegimeDiscoveryFeatureEngineer
        self.feature_engineer = RegimeDiscoveryFeatureEngineer(self.config)
        self.is_initialized = True
        
    async def process_request(self, request: ServiceRequest) -> ServiceResponse:
        """Process feature engineering request."""
        data = request.data
        market_data = data.get('market_data')
        existing_regimes = data.get('existing_regimes')
        
        # Engineer features
        features = self.feature_engineer.create_regime_discovery_features(market_data, existing_regimes)
        
        return ServiceResponse(
            service_id=self.service_id,
            request_id=request.request_id,
            data={'features': features},
            metadata={'feature_count': len(features.columns)},
            timestamp=datetime.now(),
            success=True
        )

class ClusteringService(BaseService):
    """Ensemble clustering service."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("clustering_service", config)
        self.ensemble_detector = None
        
    async def initialize(self) -> None:
        """Initialize the clustering service."""
        from .step03_ensemble_clustering import EnsembleClusteringRegimeDetector
        self.ensemble_detector = EnsembleClusteringRegimeDetector(self.config)
        self.is_initialized = True
        
    async def process_request(self, request: ServiceRequest) -> ServiceResponse:
        """Process clustering request."""
        data = request.data
        features = data.get('features')
        optimized_params = data.get('optimized_params')
        
        # Perform ensemble clustering
        consensus_regimes, ensemble_results = self.ensemble_detector.ensemble_regime_detection(
            features, optimized_params
        )
        
        return ServiceResponse(
            service_id=self.service_id,
            request_id=request.request_id,
            data={
                'regimes': consensus_regimes,
                'ensemble_results': ensemble_results
            },
            metadata={'n_regimes': len(np.unique(consensus_regimes))},
            timestamp=datetime.now(),
            success=True
        )

class ValidationService(BaseService):
    """Economic significance validation service."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("validation_service", config)
        self.validator = None
        
    async def initialize(self) -> None:
        """Initialize the validation service."""
        from .step03_economic_significance_validator import EconomicSignificanceValidator
        self.validator = EconomicSignificanceValidator(self.config)
        self.is_initialized = True
        
    async def process_request(self, request: ServiceRequest) -> ServiceResponse:
        """Process validation request."""
        data = request.data
        market_data = data.get('market_data')
        regimes = data.get('regimes')
        
        # Run economic significance validation
        validation_results = self.validator.run_step(market_data, regimes)
        
        return ServiceResponse(
            service_id=self.service_id,
            request_id=request.request_id,
            data=validation_results,
            metadata={'validation_passed': validation_results.get('overall_significant', False)},
            timestamp=datetime.now(),
            success=True
        )

class TransitionService(BaseService):
    """ML transition detection service."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("transition_service", config)
        self.transition_detector = None
        
    async def initialize(self) -> None:
        """Initialize the transition service."""
        from .step03_enhanced_ml_transition_detector import EnhancedMLRegimeTransitionDetector
        self.transition_detector = EnhancedMLRegimeTransitionDetector(self.config)
        self.is_initialized = True
        
    async def process_request(self, request: ServiceRequest) -> ServiceResponse:
        """Process transition detection request."""
        data = request.data
        market_data = data.get('market_data')
        regimes = data.get('regimes')
        
        # Run ML transition detection
        transition_results = await self.transition_detector.detect_transitions(market_data, regimes)
        
        return ServiceResponse(
            service_id=self.service_id,
            request_id=request.request_id,
            data=transition_results,
            metadata={'transitions_detected': len(transition_results.get('transitions', []))},
            timestamp=datetime.now(),
            success=True
        )

class PersistenceService(BaseService):
    """Regime persistence and forecasting service."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("persistence_service", config)
        self.forecaster = None
        
    async def initialize(self) -> None:
        """Initialize the persistence service."""
        from .step03_regime_persistence_forecasting import RegimePersistenceForecaster
        self.forecaster = RegimePersistenceForecaster(self.config)
        self.is_initialized = True
        
    async def process_request(self, request: ServiceRequest) -> ServiceResponse:
        """Process persistence and forecasting request."""
        data = request.data
        market_data = data.get('market_data')
        regimes = data.get('regimes')
        
        # Build persistence models
        persistence_models = self.forecaster.build_persistence_models(market_data, regimes)
        
        return ServiceResponse(
            service_id=self.service_id,
            request_id=request.request_id,
            data=persistence_models,
            metadata={'models_built': len(persistence_models.get('persistence_models', {}))},
            timestamp=datetime.now(),
            success=True
        )

class ServiceOrchestrator:
    """Orchestrates microservices for regime discovery."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.services = {}
        self.service_registry = {}
        self.request_queue = asyncio.Queue()
        self.response_cache = {}
        
    async def initialize_services(self) -> None:
        """Initialize all microservices."""
        print("🚀 Initializing microservices architecture...")
        
        # Initialize services
        service_configs = {
            'optimization': OptimizationService(self.config),
            'features': FeatureService(self.config),
            'clustering': ClusteringService(self.config),
            'validation': ValidationService(self.config),
            'transitions': TransitionService(self.config),
            'persistence': PersistenceService(self.config)
        }
        
        # Initialize each service
        for service_name, service in service_configs.items():
            print(f"  🔧 Initializing {service_name} service...")
            await service.initialize()
            self.services[service_name] = service
            self.service_registry[service_name] = {
                'service': service,
                'status': 'active',
                'last_used': datetime.now()
            }
        
        print("✅ All microservices initialized successfully")
    
    async def discover_regimes_microservices(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Discover regimes using microservices architecture."""
        print("🔍 Starting microservices-based regime discovery...")
        
        # Step 1: Feature Engineering
        print("  📊 Engineering features...")
        feature_request = ServiceRequest(
            service_id="feature_service",
            request_id=f"features_{int(time.time())}",
            data={'market_data': data},
            metadata={'step': 'feature_engineering'},
            timestamp=datetime.now()
        )
        
        feature_response = await self.services['features'].handle_request(feature_request)
        if not feature_response.success:
            raise Exception(f"Feature engineering failed: {feature_response.error}")
        
        features = feature_response.data['features']
        
        # Step 2: Parameter Optimization (parallel with feature engineering)
        print("  🎯 Optimizing parameters...")
        optimization_request = ServiceRequest(
            service_id="optimization_service",
            request_id=f"optimization_{int(time.time())}",
            data={'market_data': data, 'features': features},
            metadata={'step': 'parameter_optimization'},
            timestamp=datetime.now()
        )
        
        optimization_response = await self.services['optimization'].handle_request(optimization_request)
        if not optimization_response.success:
            print(f"⚠️ Optimization failed: {optimization_response.error}, using defaults")
            optimized_params = None
        else:
            optimized_params = optimization_response.data
        
        # Step 3: Ensemble Clustering
        print("  🔄 Performing ensemble clustering...")
        clustering_request = ServiceRequest(
            service_id="clustering_service",
            request_id=f"clustering_{int(time.time())}",
            data={'features': features, 'optimized_params': optimized_params},
            metadata={'step': 'ensemble_clustering'},
            timestamp=datetime.now()
        )
        
        clustering_response = await self.services['clustering'].handle_request(clustering_request)
        if not clustering_response.success:
            raise Exception(f"Clustering failed: {clustering_response.error}")
        
        regimes = clustering_response.data['regimes']
        ensemble_results = clustering_response.data['ensemble_results']
        
        # Step 4: Economic Validation (parallel with other services)
        print("  ✅ Validating economic significance...")
        validation_request = ServiceRequest(
            service_id="validation_service",
            request_id=f"validation_{int(time.time())}",
            data={'market_data': data, 'regimes': regimes},
            metadata={'step': 'economic_validation'},
            timestamp=datetime.now()
        )
        
        validation_response = await self.services['validation'].handle_request(validation_request)
        if not validation_response.success:
            print(f"⚠️ Validation failed: {validation_response.error}")
            validation_results = {'overall_significant': False}
        else:
            validation_results = validation_response.data
        
        # Step 5: Transition Detection (parallel)
        print("  🔄 Detecting regime transitions...")
        transition_request = ServiceRequest(
            service_id="transition_service",
            request_id=f"transitions_{int(time.time())}",
            data={'market_data': data, 'regimes': regimes},
            metadata={'step': 'transition_detection'},
            timestamp=datetime.now()
        )
        
        transition_response = await self.services['transitions'].handle_request(transition_request)
        if not transition_response.success:
            print(f"⚠️ Transition detection failed: {transition_response.error}")
            transition_results = {'transitions': []}
        else:
            transition_results = transition_response.data
        
        # Step 6: Persistence and Forecasting (parallel)
        print("  🔮 Building persistence models...")
        persistence_request = ServiceRequest(
            service_id="persistence_service",
            request_id=f"persistence_{int(time.time())}",
            data={'market_data': data, 'regimes': regimes},
            metadata={'step': 'persistence_forecasting'},
            timestamp=datetime.now()
        )
        
        persistence_response = await self.services['persistence'].handle_request(persistence_request)
        if not persistence_response.success:
            print(f"⚠️ Persistence modeling failed: {persistence_response.error}")
            persistence_results = {}
        else:
            persistence_results = persistence_response.data
        
        # Compile results
        results = {
            'regimes': regimes,
            'features': features,
            'optimized_parameters': optimized_params,
            'ensemble_results': ensemble_results,
            'validation_results': validation_results,
            'transition_results': transition_results,
            'persistence_results': persistence_results,
            'microservices_metrics': self._get_services_metrics(),
            'processing_summary': self._generate_processing_summary()
        }
        
        return results
    
    async def discover_regimes_parallel(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Discover regimes using parallel microservices execution."""
        print("🚀 Starting parallel microservices regime discovery...")
        
        # Create all requests
        requests = {
            'features': ServiceRequest(
                service_id="feature_service",
                request_id=f"features_{int(time.time())}",
                data={'market_data': data},
                metadata={'step': 'feature_engineering'},
                timestamp=datetime.now()
            ),
            'optimization': ServiceRequest(
                service_id="optimization_service",
                request_id=f"optimization_{int(time.time())}",
                data={'market_data': data},
                metadata={'step': 'parameter_optimization'},
                timestamp=datetime.now()
            )
        }
        
        # Execute feature engineering and optimization in parallel
        feature_task = asyncio.create_task(
            self.services['features'].handle_request(requests['features'])
        )
        optimization_task = asyncio.create_task(
            self.services['optimization'].handle_request(requests['optimization'])
        )
        
        # Wait for both to complete
        feature_response, optimization_response = await asyncio.gather(
            feature_task, optimization_task, return_exceptions=True
        )
        
        # Handle responses
        if isinstance(feature_response, Exception):
            raise Exception(f"Feature engineering failed: {feature_response}")
        if not feature_response.success:
            raise Exception(f"Feature engineering failed: {feature_response.error}")
        
        features = feature_response.data['features']
        optimized_params = optimization_response.data if optimization_response.success else None
        
        # Create clustering request
        clustering_request = ServiceRequest(
            service_id="clustering_service",
            request_id=f"clustering_{int(time.time())}",
            data={'features': features, 'optimized_params': optimized_params},
            metadata={'step': 'ensemble_clustering'},
            timestamp=datetime.now()
        )
        
        # Execute clustering
        clustering_response = await self.services['clustering'].handle_request(clustering_request)
        if not clustering_response.success:
            raise Exception(f"Clustering failed: {clustering_response.error}")
        
        regimes = clustering_response.data['regimes']
        
        # Execute remaining services in parallel
        parallel_requests = {
            'validation': ServiceRequest(
                service_id="validation_service",
                request_id=f"validation_{int(time.time())}",
                data={'market_data': data, 'regimes': regimes},
                metadata={'step': 'economic_validation'},
                timestamp=datetime.now()
            ),
            'transitions': ServiceRequest(
                service_id="transition_service",
                request_id=f"transitions_{int(time.time())}",
                data={'market_data': data, 'regimes': regimes},
                metadata={'step': 'transition_detection'},
                timestamp=datetime.now()
            ),
            'persistence': ServiceRequest(
                service_id="persistence_service",
                request_id=f"persistence_{int(time.time())}",
                data={'market_data': data, 'regimes': regimes},
                metadata={'step': 'persistence_forecasting'},
                timestamp=datetime.now()
            )
        }
        
        # Execute in parallel
        parallel_tasks = [
            asyncio.create_task(self.services['validation'].handle_request(parallel_requests['validation'])),
            asyncio.create_task(self.services['transitions'].handle_request(parallel_requests['transitions'])),
            asyncio.create_task(self.services['persistence'].handle_request(parallel_requests['persistence']))
        ]
        
        parallel_responses = await asyncio.gather(*parallel_tasks, return_exceptions=True)
        
        # Process parallel responses
        validation_response = parallel_responses[0] if not isinstance(parallel_responses[0], Exception) else None
        transition_response = parallel_responses[1] if not isinstance(parallel_responses[1], Exception) else None
        persistence_response = parallel_responses[2] if not isinstance(parallel_responses[2], Exception) else None
        
        # Compile results
        results = {
            'regimes': regimes,
            'features': features,
            'optimized_parameters': optimized_params,
            'ensemble_results': clustering_response.data['ensemble_results'],
            'validation_results': validation_response.data if validation_response and validation_response.success else {'overall_significant': False},
            'transition_results': transition_response.data if transition_response and transition_response.success else {'transitions': []},
            'persistence_results': persistence_response.data if persistence_response and persistence_response.success else {},
            'microservices_metrics': self._get_services_metrics(),
            'processing_summary': self._generate_processing_summary()
        }
        
        return results
    
    def _get_services_metrics(self) -> Dict[str, Any]:
        """Get metrics from all services."""
        metrics = {}
        for service_name, service_info in self.service_registry.items():
            service = service_info['service']
            metrics[service_name] = service.get_metrics()
        return metrics
    
    def _generate_processing_summary(self) -> Dict[str, Any]:
        """Generate processing summary."""
        total_requests = sum(service.get_metrics()['request_count'] for service in self.services.values())
        total_time = sum(service.get_metrics()['total_processing_time'] for service in self.services.values())
        
        return {
            'total_requests': total_requests,
            'total_processing_time': total_time,
            'avg_processing_time': total_time / max(total_requests, 1),
            'services_used': list(self.services.keys()),
            'architecture': 'microservices'
        }
    
    async def shutdown_services(self) -> None:
        """Shutdown all services gracefully."""
        print("🛑 Shutting down microservices...")
        for service_name, service in self.services.items():
            print(f"  🔧 Shutting down {service_name} service...")
            # Add any cleanup logic here
        print("✅ All services shut down successfully")

class MicroservicesRegimeDiscovery:
    """Main interface for microservices-based regime discovery."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.orchestrator = ServiceOrchestrator(config)
        
    async def discover_regimes(self, data: pd.DataFrame, use_parallel: bool = True) -> Dict[str, Any]:
        """Discover regimes using microservices architecture."""
        # Initialize services
        await self.orchestrator.initialize_services()
        
        try:
            if use_parallel:
                results = await self.orchestrator.discover_regimes_parallel(data)
            else:
                results = await self.orchestrator.discover_regimes_microservices(data)
            
            return results
            
        finally:
            # Shutdown services
            await self.orchestrator.shutdown_services()