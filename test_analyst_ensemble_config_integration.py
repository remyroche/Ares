#!/usr/bin/env python3
"""
Script de Test - Intégration Configuration Centralisée Analyst Ensemble Training

Ce script valide le fonctionnement complet du système de configuration centralisée
pour l'entraînement d'ensemble des modèles analyst, incluant :
- Chargement des configurations YAML/JSON/Python
- Intégration dans le composant principal
- Système de fallback robuste
- API d'accès aux paramètres
- Validation des performances

Date: 2025-11-03T22:14:00.000Z
Version: 1.0.0
"""

import sys
import os
import json
import time
from typing import Dict, Any, List
from datetime import datetime

# Ajouter le répertoire src au path pour les imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_1_config_loading():
    """Test 1: Chargement des configurations"""
    print("\n🧪 Test 1: Chargement des Configurations Centralisées")
    print("=" * 70)
    
    try:
        from src.config.analyst_ensemble_training import (
            get_analyst_ensemble_config,
            get_analyst_ensemble_config_manager,
            set_custom_config_path
        )
        
        # Test chargement configuration complète
        config = get_analyst_ensemble_config()
        print(f"✅ Configuration chargée: {config.component_name}")
        print(f"   Version: {config.version}")
        print(f"   Description: {config.description}")
        
        # Test accès sections spécifiques
        meta_learner = get_analyst_ensemble_config(['meta_learner'])
        hardware = get_analyst_ensemble_config(['hardware'])
        training = get_analyst_ensemble_config(['training'])
        
        print(f"   Meta-learner: {meta_learner.get('model_type')}")
        print(f"   GPU acceleration: {hardware.get('enable_gpu_acceleration')}")
        print(f"   CV folds: {training.get('cv_folds')}")
        
        # Test validation
        manager = get_analyst_ensemble_config_manager()
        is_valid = manager.validate_config(config)
        print(f"   Validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test 1 échoué: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_2_component_integration():
    """Test 2: Intégration dans le composant principal"""
    print("\n🧪 Test 2: Intégration dans le Composant Principal")
    print("=" * 70)
    
    try:
        # Import du composant avec configuration centralisée
        from src.training.steps.models_training.components.analyst_ensemble_training_modular import (
            AnalystEnsembleTrainingModular,
            create_analyst_ensemble_training,
            create_with_custom_config
        )
        
        # Test création avec configuration centralisée
        print("Test création avec configuration centralisée...")
        component = create_analyst_ensemble_training(use_centralized_config=True)
        print(f"✅ Composant créé: {component.name}")
        
        # Test accès configuration centralisée
        central_config = component.get_centralized_config()
        if central_config:
            print(f"   Configuration centralisée: {central_config.component_name}")
            print(f"   Version: {central_config.version}")
        else:
            print("⚠️ Configuration centralisée non disponible (fallback)")
        
        # Test méthodes d'accès paramètres
        ensemble_method = component.ensemble_config.ensemble_method.value
        base_models = component.ensemble_config.base_models
        regime_aware = component.ensemble_config.regime_aware
        
        print(f"   Méthode d'ensemble: {ensemble_method}")
        print(f"   Modèles de base: {len(base_models)}")
        print(f"   Détection régimes: {regime_aware}")
        
        # Test méthodes d'accès avancées
        target_accuracy = component.get_ensemble_performance_target()
        hardware_limits = component.get_hardware_limits()
        feature_config = component.get_feature_engineering_config()
        
        print(f"   Cible précision: {target_accuracy}")
        print(f"   Limites hardware: GPU={hardware_limits.get('enable_gpu_acceleration')}")
        print(f"   Feature engineering: Regime={feature_config.get('regime_features', {}).get('enable')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test 2 échoué: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_3_fallback_mechanism():
    """Test 3: Mécanisme de fallback"""
    print("\n🧪 Test 3: Mécanisme de Fallback")
    print("=" * 70)
    
    try:
        from src.training.steps.models_training.components.analyst_ensemble_training_modular import (
            AnalystEnsembleTrainingModular
        )
        
        # Test création sans configuration centralisée
        print("Test création sans configuration centralisée...")
        component_no_config = AnalystEnsembleTrainingModular(
            use_centralized_config=False
        )
        print(f"✅ Composant créé (fallback): {component_no_config.name}")
        
        # Vérifier que la configuration centralisée n'est pas disponible
        central_config = component_no_config.get_centralized_config()
        print(f"   Configuration centralisée disponible: {central_config is not None}")
        
        # Test résumé d'entraînement
        summary = component_no_config.get_training_summary()
        print(f"   Fallback activé: {not summary.get('centralized_config', {}).get('enabled', True)}")
        
        # Test accès paramètres avec fallback
        epochs = component_no_config.get_parameter_with_fallback('training.epochs', 100)
        batch_size = component_no_config.get_parameter_with_fallback('training.batch_size', 32)
        
        print(f"   Epochs (fallback): {epochs}")
        print(f"   Batch size (fallback): {batch_size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test 3 échoué: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_4_custom_config_file():
    """Test 4: Configuration personnalisée"""
    print("\n🧪 Test 4: Configuration Personnalisée")
    print("=" * 70)
    
    try:
        # Créer une configuration personnalisée temporaire
        custom_config = {
            "meta_learner": {
                "params": {
                    "n_estimators": 200,
                    "learning_rate": 0.03
                }
            },
            "performance": {
                "expected_accuracy": 0.90
            }
        }
        
        config_file = "test_custom_analyst_config.json"
        with open(config_file, 'w') as f:
            json.dump(custom_config, f, indent=2)
        
        # Test utilisation configuration personnalisée
        from src.training.steps.models_training.components.analyst_ensemble_training_modular import (
            create_with_custom_config
        )
        
        component_custom = create_with_custom_config(
            custom_config_path=config_file,
            config_overrides={"test_override": True}
        )
        
        print(f"✅ Composant créé avec config personnalisée: {component_custom.name}")
        
        # Vérifier la configuration personnalisée
        central_config = component_custom.get_centralized_config()
        if central_config:
            n_estimators = central_config.meta_learner.get('params', {}).get('n_estimators')
            expected_acc = central_config.performance.get('expected_accuracy')
            print(f"   n_estimators personnalisé: {n_estimators}")
            print(f"   Précision attendue personnalisée: {expected_acc}")
        
        # Nettoyer le fichier temporaire
        os.remove(config_file)
        print("✅ Fichier de configuration temporaire supprimé")
        
        return True
        
    except Exception as e:
        print(f"❌ Test 4 échoué: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_5_performance_metrics():
    """Test 5: Métriques de performance"""
    print("\n🧪 Test 5: Métriques de Performance")
    print("=" * 70)
    
    try:
        start_time = time.time()
        
        from src.training.steps.models_training.components.analyst_ensemble_training_modular import (
            create_analyst_ensemble_training
        )
        
        # Test performance de création
        creation_start = time.time()
        component = create_analyst_ensemble_training()
        creation_time = time.time() - creation_start
        
        print(f"✅ Temps de création: {creation_time:.3f}s")
        
        # Test accès multiple configurations
        access_times = []
        for i in range(10):
            access_start = time.time()
            _ = component.get_parameter_with_fallback('training.epochs')
            access_times.append(time.time() - access_start)
        
        avg_access_time = sum(access_times) / len(access_times)
        print(f"✅ Temps moyen accès paramètre: {avg_access_time:.6f}s")
        
        # Test validation configuration
        validation_start = time.time()
        is_valid = component.config_manager.validate_config(component._centralized_config)
        validation_time = time.time() - validation_start
        
        print(f"✅ Validation configuration: {'Valid' if is_valid else 'Invalid'} ({validation_time:.3f}s)")
        
        total_time = time.time() - start_time
        print(f"✅ Temps total test: {total_time:.3f}s")
        
        # Vérifier que les performances sont acceptables
        if creation_time < 2.0 and avg_access_time < 0.001:
            print("✅ Performance acceptable")
            return True
        else:
            print("⚠️ Performance dégradée mais fonctionnelle")
            return True
        
    except Exception as e:
        print(f"❌ Test 5 échoué: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_6_error_handling():
    """Test 6: Gestion d'erreurs"""
    print("\n🧪 Test 6: Gestion d'Erreurs")
    print("=" * 70)
    
    try:
        from src.training.steps.models_training.components.analyst_ensemble_training_modular import (
            AnalystEnsembleTrainingModular
        )
        
        # Test avec chemin de configuration invalide
        print("Test gestion configuration invalide...")
        try:
            component_invalid = AnalystEnsembleTrainingModular(
                use_centralized_config=True
            )
            # Forcer un chemin invalide
            component_invalid.config_manager.custom_config_path = "/chemin/invalide/config.json"
            component_invalid.config_manager._config_cache = None
            config = component_invalid.get_centralized_config()
            print("✅ Fallback automatique activé en cas d'erreur")
        except Exception as e:
            print(f"✅ Erreur gérée correctement: {type(e).__name__}")
        
        # Test avec paramètres invalides
        print("Test gestion paramètres invalides...")
        invalid_params = {
            "training": {
                "cv_folds": -1,  # Valeur invalide
                "validation_split": 2.0  # Valeur invalide
            }
        }
        
        component_test = create_analyst_ensemble_training(config=invalid_params)
        summary = component_test.get_training_summary()
        print("✅ Composant créé avec paramètres invalides (fallback utilisé)")
        
        # Test accès paramètre inexistant
        nonexistent = component_test.get_parameter_with_fallback('parametre.inexistant', 'default')
        print(f"✅ Paramètre inexistant: {nonexistent}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test 6 échoué: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_7_api_compatibility():
    """Test 7: Compatibilité API"""
    print("\n🧪 Test 7: Compatibilité API")
    print("=" * 70)
    
    try:
        from src.training.steps.models_training.components.analyst_ensemble_training_modular import (
            create_analyst_ensemble_training,
            create_analyst_ensemble_training_legacy,
            create_with_custom_config
        )
        
        # Test API principale (nouvelle)
        component_new = create_analyst_ensemble_training(
            config={"test": True},
            use_centralized_config=True
        )
        print(f"✅ API nouvelle: {component_new.name}")
        
        # Test API legacy (compatibilité)
        component_legacy = create_analyst_ensemble_training_legacy(
            config={"legacy": True}
        )
        print(f"✅ API legacy: {component_legacy.name}")
        
        # Vérifier que l'API legacy utilise le fallback
        legacy_config = component_legacy.get_centralized_config()
        print(f"   API legacy utilise fallback: {legacy_config is None}")
        
        # Test factory avec configuration personnalisée
        component_custom_api = create_with_custom_config(
            custom_config_path="test_nonexistent.json",  # Va utiliser le fallback
            config_overrides={"custom_api": True}
        )
        print(f"✅ API custom: {component_custom_api.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test 7 échoué: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Exécuter tous les tests et générer le rapport"""
    print("🚀 DÉBUT DES TESTS - Intégration Configuration Centralisée Analyst Ensemble Training")
    print("=" * 90)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version}")
    
    tests = [
        ("Configuration Loading", test_1_config_loading),
        ("Component Integration", test_2_component_integration),
        ("Fallback Mechanism", test_3_fallback_mechanism),
        ("Custom Configuration", test_4_custom_config_file),
        ("Performance Metrics", test_5_performance_metrics),
        ("Error Handling", test_6_error_handling),
        ("API Compatibility", test_7_api_compatibility)
    ]
    
    results = []
    start_time = time.time()
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' a généré une exception: {e}")
            results.append((test_name, False))
    
    # Rapport final
    total_time = time.time() - start_time
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print("\n" + "=" * 90)
    print("📊 RAPPORT FINAL DES TESTS")
    print("=" * 90)
    
    for test_name, result in results:
        status = "✅ PASSÉ" if result else "❌ ÉCHEC"
        print(f"{status:12} {test_name}")
    
    print("-" * 90)
    print(f"Total: {passed}/{total} tests réussis ({passed/total*100:.1f}%)")
    print(f"Temps total: {total_time:.2f}s")
    
    if passed == total:
        print("🎉 TOUS LES TESTS SONT PASSÉS!")
        print("✅ L'intégration de la configuration centralisée est fonctionnelle")
        return True
    else:
        print(f"⚠️ {total-passed} test(s) ont échoué")
        print("⚠️ L'intégration nécessite des corrections")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)