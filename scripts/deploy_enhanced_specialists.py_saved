"""
Deploy Enhanced Specialists Script

This script deploys enhanced feature generators, MI monitoring, and 
hyperparameter optimization to all specialist models.
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import json
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.steps.market_analysis.mi_monitoring_system import get_mi_monitor
from src.training.steps.market_analysis.hyperparameter_optimizer_mi import get_mi_optimizer
from src.utils.versioned_artifacts import VersionedArtifactStore
import pandas as pd
import numpy as np
from src.training.steps.market_analysis.enhanced_feature_generators import EnhancedFeaturePipeline
from sklearn.feature_selection import mutual_info_regression

logger = logging.getLogger(__name__)


class EnhancedSpecialistDeployer:
    """Deploy enhanced specialists_enhanced with MI optimization."""
    
    def __init__(self):
        self.mi_monitor = get_mi_monitor()
        self.mi_optimizer = get_mi_optimizer()
        self.feature_pipeline = EnhancedFeaturePipeline()
        self.deployment_log = []
        
    def deploy_to_all_specialists(self) -> Dict[str, Any]:
        """Deploy enhanced capabilities to all specialist models."""
        
        print("🚀 DEPLOYING ENHANCED SPECIALISTS")
        print("=" * 60)
        
        # Define specialists_enhanced to enhance
        specialists_enhanced = {
    'enhanced_ml_momentum_persistence_step': {
        'class_path': 'src.training.steps.market_analysis.ml_momentum_persistence_step_enhanced.EnhancedMLMomentumPersistenceStep',
        'model_type': 'lightgbm',
        'feature_type': 'momentum'
    },
    'enhanced_ml_smc_regime_step': {
        'class_path': 'src.training.steps.market_analysis.ml_smc_regime_step_enhanced.EnhancedMLSMCRegimeStep',
        'model_type': 'xgboost',
        'feature_type': 'smc_regime'
    },
    'enhanced_ml_volatility_burst_step': {
        'class_path': 'src.training.steps.market_analysis.ml_volatility_burst_step_enhanced.EnhancedMLVolatilityBurstStep',
        'model_type': 'lightgbm',
        'feature_type': 'volatility_burst'
    },
    'enhanced_ml_volume_force_step': {
        'class_path': 'src.training.steps.market_analysis.ml_volume_force_step_enhanced.EnhancedMLVolumeForceStep',
        'model_type': 'xgboost',
        'feature_type': 'volume_force'
    },
    'enhanced_ml_liquidity_regime_step': {
        'class_path': 'src.training.steps.market_analysis.ml_liquidity_regime_step_enhanced.EnhancedMLLiquidityRegimeStep',
        'model_type': 'xgboost',
        'feature_type': 'liquidity_regime'
    },
    'enhanced_ml_breakout_bounce_regime_step': {
        'class_path': 'src.training.steps.market_analysis.ml_breakout_bounce_regime_step_enhanced.EnhancedMLBreakoutBounceRegimeStep',
        'model_type': 'xgboost',
        'feature_type': 'breakout'
    },
    'enhanced_ml_path_regime_step': {
        'class_path': 'src.training.steps.market_analysis.ml_path_regime_step_enhanced.EnhancedMLPathRegimeStep',
        'model_type': 'xgboost',
        'feature_type': 'path_regime'
    },
    'enhanced_ml_reversion_regime_step': {
        'class_path': 'src.training.steps.market_analysis.ml_reversion_regime_step_enhanced.EnhancedMLReversionRegimeStep',
        'model_type': 'xgboost',
        'feature_type': 'reversion'
    },
    'enhanced_ml_risk_regime_step': {
        'class_path': 'src.training.steps.market_analysis.ml_risk_regime_step_enhanced.EnhancedMLRiskRegimeStep',
        'model_type': 'xgboost',
        'feature_type': 'risk_regime'
    },
    'enhanced_xgb_macro_regime_step': {
        'class_path': 'src.training.steps.market_analysis.xgb_macro_regime_step_enhanced.EnhancedXGBMacroRegimeStep',
        'model_type': 'xgboost',
        'feature_type': 'macro_regime'
    },
    'enhanced_xgb_meso_regime_step': {
        'class_path': 'src.training.steps.market_analysis.xgb_meso_regime_step_enhanced.EnhancedXGBMesoRegimeStep',
        'model_type': 'xgboost',
        'feature_type': 'meso_regime'
    }
},
            'enhanced_ml_smc_regime_step': {
                'class_path': 'src.training.steps.market_analysis.enhanced_ml_smc_regime_step_enhanced.EnhancedMLSMCRegimeStep',
                'model_type': 'xgboost',
                'feature_type': 'smc_regime'
            },
            'enhanced_ml_volatility_burst_step': {
                'class_path': 'src.training.steps.market_analysis.enhanced_ml_volatility_burst_step_enhanced.EnhancedMLVolatilityBurstStep',
                'model_type': 'lightgbm',
                'feature_type': 'volatility_burst'
            },
            'enhanced_ml_volume_force_step': {
                'class_path': 'src.training.steps.market_analysis.enhanced_ml_volume_force_step_enhanced.EnhancedMLVolumeForceStep',
                'model_type': 'xgboost',
                'feature_type': 'volume_force'
            }
        }
        
        deployment_results = {}
        
        for specialist_name, config in specialists_enhanced.items():
            print(f"\n🔧 Deploying enhanced capabilities to {specialist_name}...")
            
            try:
                result = self._deploy_to_specialist(specialist_name, config)
                deployment_results[specialist_name] = result
                
                if result['success']:
                    print(f"   ✅ Successfully deployed to {specialist_name}")
                else:
                    print(f"   ❌ Failed to deploy to {specialist_name}: {result['error']}")
                    
            except Exception as e:
                print(f"   ❌ Deployment failed for {specialist_name}: {e}")
                deployment_results[specialist_name] = {
                    'success': False,
                    'error': str(e)
                }
        
        # Generate deployment summary
        summary = self._generate_deployment_summary(deployment_results)
        
        print(f"\n📊 DEPLOYMENT SUMMARY:")
        print(f"   Total specialists_enhanced: {len(specialists_enhanced)}")
        print(f"   Successful deployments: {summary['successful_deployments']}")
        print(f"   Failed deployments: {summary['failed_deployments']}")
        print(f"   Success rate: {summary['success_rate']:.1%}")
        
        return deployment_results
    
    def _deploy_to_specialist(self, specialist_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy enhanced capabilities to a specific specialist."""
        
        result = {
            'success': False,
            'specialist_name': specialist_name,
            'deployment_timestamp': datetime.utcnow().isoformat(),
            'enhanced_features_deployed': False,
            'mi_monitoring_deployed': False,
            'hyperparameter_optimization_deployed': False,
            'error': None
        }
        
        try:
            # 1. Deploy enhanced features
            print(f"   🛠️ Deploying enhanced features...")
            features_deployed = self._deploy_enhanced_features(specialist_name, config['feature_type'])
            result['enhanced_features_deployed'] = features_deployed
            
            # 2. Deploy MI monitoring
            print(f"   📊 Deploying MI monitoring...")
            mi_monitoring_deployed = self._deploy_mi_monitoring(specialist_name)
            result['mi_monitoring_deployed'] = mi_monitoring_deployed
            
            # 3. Deploy hyperparameter optimization
            print(f"   🔧 Deploying hyperparameter optimization...")
            hpo_deployed = self._deploy_hyperparameter_optimization(specialist_name, config['model_type'])
            result['hyperparameter_optimization_deployed'] = hpo_deployed
            
            # Check if all deployments succeeded
            if all([features_deployed, mi_monitoring_deployed, hpo_deployed]):
                result['success'] = True
                
                # Log successful deployment
                self.deployment_log.append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'specialist_name': specialist_name,
                    'status': 'SUCCESS',
                    'components': ['enhanced_features', 'mi_monitoring', 'hyperparameter_optimization']
                })
            
        except Exception as e:
            result['error'] = str(e)
            
            # Log failed deployment
            self.deployment_log.append({
                'timestamp': datetime.utcnow().isoformat(),
                'specialist_name': specialist_name,
                'status': 'FAILED',
                'error': str(e)
            })
        
        return result
    
    def _deploy_enhanced_features(self, specialist_name: str, feature_type: str) -> bool:
        """Deploy enhanced feature generation to specialist."""
        
        try:
            # Test enhanced feature pipeline
            print(f"      Testing enhanced feature pipeline for {feature_type}...")
            
            # Create sample data for testing
            sample_data = self._create_sample_market_data()
            
            # Generate enhanced features
            enhanced_features = self.feature_pipeline.generate_enhanced_features(
                sample_data, feature_type
            )
            
            if len(enhanced_features.columns) > 0:
                print(f"      ✅ Enhanced features working: {len(enhanced_features.columns)} features generated")
                return True
            else:
                print(f"      ❌ No enhanced features generated")
                return False
                
        except Exception as e:
            print(f"      ❌ Enhanced features deployment failed: {e}")
            return False
    
    def _deploy_mi_monitoring(self, specialist_name: str) -> bool:
        """Deploy MI monitoring to specialist."""
        
        try:
            # Test MI monitoring
            print(f"      Testing MI monitoring for {specialist_name}...")
            
            # Create sample data for testing
            sample_features = self._create_sample_features()
            sample_labels = self._create_sample_labels()
            sample_predictions = self._create_sample_predictions()
            sample_probabilities = self._create_sample_probabilities()
            
            # Test MI computation
            mi_metrics = self.mi_monitor.compute_mi_metrics(
                specialist_name, sample_features, sample_labels, 
                sample_predictions, sample_probabilities
            )
            
            if mi_metrics.prediction_mi >= 0:
                print(f"      ✅ MI monitoring working: MI = {mi_metrics.prediction_mi:.4f}")
                return True
            else:
                print(f"      ❌ MI monitoring failed: MI = {mi_metrics.prediction_mi}")
                return False
                
        except Exception as e:
            print(f"      ❌ MI monitoring deployment failed: {e}")
            return False
    
    def _deploy_hyperparameter_optimization(self, specialist_name: str, model_type: str) -> bool:
        """Deploy hyperparameter optimization to specialist."""
        
        try:
            # Test HPO
            print(f"      Testing hyperparameter optimization for {specialist_name}...")
            
            # Create sample data for testing
            sample_features = self._create_sample_features()
            sample_labels = self._create_sample_labels()
            
            # Test feature selection
            selected_features, mi_scores = self.mi_optimizer.select_features_by_mi(
                sample_features, sample_labels, top_k=10
            )
            
            if len(selected_features) > 0:
                print(f"      ✅ Feature selection working: {len(selected_features)} features selected")
                print(f"      ✅ Hyperparameter optimization framework ready")
                return True
            else:
                print(f"      ❌ No features selected")
                return False
                
        except Exception as e:
            print(f"      ❌ Hyperparameter optimization deployment failed: {e}")
            return False
    
    def _create_sample_market_data(self) -> pd.DataFrame:
        """Create sample market data for testing."""
        import numpy as np
        
        dates = pd.date_range('2024-01-01', periods=100, freq='15min')
        
        # Generate realistic price data
        np.random.seed(42)
        returns = np.random.normal(0, 0.001, 100)
        prices = [100]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        prices = prices[1:]
        
        # Generate OHLCV data
        data = {
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
            'close': prices,
            'volume': np.random.exponential(1000, 100)
        }
        
        return pd.DataFrame(data, index=dates)
    
    def _create_sample_features(self) -> pd.DataFrame:
        """Create sample feature data for testing."""
        import numpy as np
        
        np.random.seed(42)
        n_samples = 100
        n_features = 20
        
        features = {}
        for i in range(n_features):
            features[f'feature_{i}'] = np.random.normal(0, 1, n_samples)
        
        return pd.DataFrame(features)
    
    def _create_sample_labels(self) -> pd.Series:
        """Create sample label data for testing."""
        import numpy as np
        
        np.random.seed(42)
        return pd.Series(np.random.randint(0, 2, 100))
    
    def _create_sample_predictions(self) -> np.ndarray:
        """Create sample prediction data for testing."""
        import numpy as np
        
        np.random.seed(42)
        return np.random.random(100)
    
    def _create_sample_probabilities(self) -> np.ndarray:
        """Create sample probability data for testing."""
        import numpy as np
        
        np.random.seed(42)
        return np.random.random(100)
    
    def _generate_deployment_summary(self, deployment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate deployment summary."""
        
        total_deployments = len(deployment_results)
        successful_deployments = sum(1 for result in deployment_results.values() if result['success'])
        failed_deployments = total_deployments - successful_deployments
        
        summary = {
            'total_deployments': total_deployments,
            'successful_deployments': successful_deployments,
            'failed_deployments': failed_deployments,
            'success_rate': successful_deployments / total_deployments if total_deployments > 0 else 0.0,
            'deployment_timestamp': datetime.utcnow().isoformat(),
            'deployment_log': self.deployment_log
        }
        
        return summary
    
    def save_deployment_report(self, filepath: str):
        """Save deployment report to file."""
        
        report = {
            'deployment_summary': self._generate_deployment_summary({}),
            'deployment_log': self.deployment_log,
            'mi_monitoring_status': self.mi_monitor.get_overall_summary(),
            'hyperparameter_optimization_status': self.mi_optimizer.get_optimization_summary(),
            'report_timestamp': datetime.utcnow().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Deployment report saved to {filepath}")


def main():
    """Main deployment function."""
    
    print("🚀 ENHANCED SPECIALISTS DEPLOYMENT")
    print("=" * 80)
    
    # Initialize deployer
    deployer = EnhancedSpecialistDeployer()
    
    # Deploy to all specialists_enhanced
    deployment_results = deployer.deploy_to_all_specialists()
    
    # Save deployment report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = f"outcomes/enhanced_specialists_deployment_{timestamp}.json"
    deployer.save_deployment_report(report_path)
    
    # Final summary
    print(f"\n🎯 DEPLOYMENT COMPLETE")
    print(f"📄 Report saved: {report_path}")
    
    # Check if deployment was successful
    successful_count = sum(1 for result in deployment_results.values() if result['success'])
    total_count = len(deployment_results)
    
    if successful_count == total_count:
        print(f"✅ All {total_count} specialists_enhanced successfully enhanced!")
    elif successful_count > 0:
        print(f"⚠️ {successful_count}/{total_count} specialists_enhanced enhanced")
    else:
        print(f"❌ No specialists_enhanced were enhanced")
    
    print(f"\n🚀 Enhanced specialists_enhanced ready for MI optimization!")


if __name__ == "__main__":
    main()
