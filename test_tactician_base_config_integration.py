#!/usr/bin/env python3
"""
Script de Test - Configuration Centralisée Tactician Base Training

Ce script teste complètement le système de configuration centralisée pour
l'entraînement des modèles tactician de base, validant tous les aspects
du système implémenté.

Version: 1.0.0
Date: 2025-11-03T22:31:00.000Z
"""

import os
import sys
import time
import json
import traceback
from typing import Dict, Any, List, Optional

# Ajouter le répertoire du projet au path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Configuration de test
TEST_CONFIG = {
    'symbol': 'ETHUSDT',
    'exchange': 'binance',
    'timeframe': '15m',
    'direction': 'both'
}

class TacticianBaseConfigTestSuite:
    """Suite de tests pour la configuration centralisée tactician_base_training."""
    
    def __init__(self):
        self.results = []
        self.start_time = time.time()
        
    def log_test(self, test_name: str, success: bool, message: str = "", execution_time: float = 0):
        """Enregistrer le résultat d'un test."""
        result = {
            'test_name': test_name,
            'success': success,
            'message': message,
            'execution_time': execution_time,
            'timestamp': time.time()
        }
        self.results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} [{test_name}] {message} ({execution_time:.3f}s)")
        
        return success
    
    def test_1_config_loading(self) -> bool:
        """Test 1: Chargement de la configuration de base."""
        test_name = "test_1_config_loading"
        start_time = time.time()
        
        try:
            # Import du gestionnaire de configuration
            from src.config.tactician_base_training import get_tactician_base_training_config
            
            # Chargement de la configuration
            config = get_tactician_base_training_config()
            
            if config is None:
                return self.log_test(test_name, False, "Configuration non chargée", time.time() - start_time)
            
            if not hasattr(config, 'tactician_config'):
                return self.log_test(test_name, False, "Configuration tactician manquante", time.time() - start_time)
            
            execution_time = time.time() - start_time
            message = f"Configuration chargée: {config.tactician_config.model_name}, {len(config.tactician_config.base_models)} modèles"
            return self.log_test(test_name, True, message, execution_time)
            
        except Exception as e:
            return self.log_test(test_name, False, f"Erreur: {e}", time.time() - start_time)
    
    def test_2_config_manager(self) -> bool:
        """Test 2: Gestionnaire de configuration."""
        test_name = "test_2_config_manager"
        start_time = time.time()
        
        try:
            from src.config.tactician_base_training import get_tactician_base_training_config_manager
            
            # Création du gestionnaire
            manager = get_tactician_base_training_config_manager()
            
            if manager is None:
                return self.log_test(test_name, False, "Gestionnaire non créé", time.time() - start_time)
            
            # Test de chargement via manager
            config = manager.load_config()
            
            if config is None:
                return self.log_test(test_name, False, "Chargement via manager échoué", time.time() - start_time)
            
            # Test d'accès aux sections
            model_name = manager.get_config_section(['tactician_config', 'model_name'])
            if model_name is None:
                return self.log_test(test_name, False, "Accès aux sections échoué", time.time() - start_time)
            
            execution_time = time.time() - start_time
            message = f"Gestionnaire opérationnel: {model_name}"
            return self.log_test(test_name, True, message, execution_time)
            
        except Exception as e:
            return self.log_test(test_name, False, f"Erreur: {e}", time.time() - start_time)
    
    def test_3_tactician_base_training_step_integration(self) -> bool:
        """Test 3: Intégration avec TacticianBaseTrainingStep."""
        test_name = "test_3_tactician_base_training_step_integration"
        start_time = time.time()
        
        try:
            from src.training.steps.model_training.tactician_base_training_step import create_tactician_base_training_step
            
            # Création de l'étape
            step = create_tactician_base_training_step()
            
            if step is None:
                return self.log_test(test_name, False, "Étape non créée", time.time() - start_time)
            
            # Test d'accès aux configurations
            config_source = step.get_training_summary()['config_source']
            centralized_enabled = step.get_training_summary()['centralized_config']['enabled']
            
            # Test d'accès aux paramètres
            target = step.get_parameter_with_fallback('tactician.target', 'unknown')
            accuracy = step.get_parameter_with_fallback('performance.expected_accuracy', 0.0)
            
            execution_time = time.time() - start_time
            message = f"Intégration réussie: {config_source}, target={target}, accuracy={accuracy}"
            return self.log_test(test_name, True, message, execution_time)
            
        except Exception as e:
            return self.log_test(test_name, False, f"Erreur: {e}", time.time() - start_time)
    
    def test_4_fallback_configuration(self) -> bool:
        """Test 4: Configuration de fallback."""
        test_name = "test_4_fallback_configuration"
        start_time = time.time()
        
        try:
            from src.config.tactician_base_training.config_manager import TacticianBaseTrainingConfigManager
            
            # Forcer l'échec de chargement
            manager = TacticianBaseTrainingConfigManager(custom_config_path="/non/existent/file.yaml")
            config = manager.load_config()
            
            if config is None:
                return self.log_test(test_name, False, "Configuration de fallback non créée", time.time() - start_time)
            
            # Vérifier que la configuration fallback est valide
            if not config.validate():
                return self.log_test(test_name, False, "Configuration fallback invalide", time.time() - start_time)
            
            execution_time = time.time() - start_time
            message = f"Fallback opérationnel: {config.tactician_config.model_name}"
            return self.log_test(test_name, True, message, execution_time)
            
        except Exception as e:
            return self.log_test(test_name, False, f"Erreur: {e}", time.time() - start_time)
    
    def test_5_multi_format_support(self) -> bool:
        """Test 5: Support multi-format (YAML/JSON/Python)."""
        test_name = "test_5_multi_format_support"
        start_time = time.time()
        
        try:
            from src.config.tactician_base_training import get_tactician_base_training_config_manager
            
            manager = get_tactician_base_training_config_manager()
            
            # Test YAML
            yaml_path = os.path.join(os.path.dirname(__file__), "src", "config", "tactician_base_training", "default_config.yaml")
            if os.path.exists(yaml_path):
                config_yaml = manager._load_config_from_path(yaml_path)
                if config_yaml is None:
                    return self.log_test(test_name, False, "Échec chargement YAML", time.time() - start_time)
            
            # Test JSON
            json_path = os.path.join(os.path.dirname(__file__), "src", "config", "tactician_base_training", "default_config.json")
            if os.path.exists(json_path):
                config_json = manager._load_config_from_path(json_path)
                if config_json is None:
                    return self.log_test(test_name, False, "Échec chargement JSON", time.time() - start_time)
            
            # Test Python
            python_path = os.path.join(os.path.dirname(__file__), "src", "config", "tactician_base_training", "default_config.py")
            if os.path.exists(python_path):
                config_python = manager._load_config_from_path(python_path)
                if config_python is None:
                    return self.log_test(test_name, False, "Échec chargement Python", time.time() - start_time)
            
            execution_time = time.time() - start_time
            message = "Support multi-format validé"
            return self.log_test(test_name, True, message, execution_time)
            
        except Exception as e:
            return self.log_test(test_name, False, f"Erreur: {e}", time.time() - start_time)
    
    def test_6_config_sections_access(self) -> bool:
        """Test 6: Accès aux sections de configuration."""
        test_name = "test_6_config_sections_access"
        start_time = time.time()
        
        try:
            from src.config.tactician_base_training import get_tactician_base_training_config_manager
            
            manager = get_tactician_base_training_config_manager()
            
            # Test d'accès à toutes les sections principales
            sections_to_test = [
                ['tactician_config', 'model_name'],
                ['feature_engineering', 'primary_features', 'artifact_name'],
                ['training', 'cv_folds'],
                ['performance', 'expected_accuracy'],
                ['hardware', 'enable_gpu_acceleration']
            ]
            
            for section_path in sections_to_test:
                value = manager.get_config_section(section_path)
                if value is None:
                    return self.log_test(test_name, False, f"Section non accessible: {'.'.join(section_path)}", time.time() - start_time)
            
            execution_time = time.time() - start_time
            message = f"Accès sections validé ({len(sections_to_test)} sections)"
            return self.log_test(test_name, True, message, execution_time)
            
        except Exception as e:
            return self.log_test(test_name, False, f"Erreur: {e}", time.time() - start_time)
    
    def test_7_performance_loading(self) -> bool:
        """Test 7: Performance de chargement."""
        test_name = "test_7_performance_loading"
        start_time = time.time()
        
        try:
            from src.config.tactician_base_training import get_tactician_base_training_config
            
            # Mesurer le temps de chargement
            config_start = time.time()
            config = get_tactician_base_training_config()
            config_time = time.time() - config_start
            
            if config is None:
                return self.log_test(test_name, False, "Configuration non chargée", time.time() - start_time)
            
            # Vérifier que le chargement est rapide (< 2 secondes)
            if config_time > 2.0:
                return self.log_test(test_name, False, f"Temps de chargement trop lent: {config_time:.3f}s", time.time() - start_time)
            
            # Mesurer le temps d'accès aux paramètres
            access_start = time.time()
            for _ in range(100):
                _ = config.tactician_config.model_name
                _ = config.performance.expected_accuracy
            access_time = (time.time() - access_start) / 100
            
            execution_time = time.time() - start_time
            message = f"Performance OK: chargement {config_time:.3f}s, accès {access_time:.6f}s"
            return self.log_test(test_name, True, message, execution_time)
            
        except Exception as e:
            return self.log_test(test_name, False, f"Erreur: {e}", time.time() - start_time)
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Exécuter tous les tests."""
        print("🧪 Suite de Tests - Configuration Centralisée Tactician Base Training")
        print("=" * 80)
        print(f"🚀 Démarrage des tests à {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Liste des tests à exécuter
        tests = [
            self.test_1_config_loading,
            self.test_2_config_manager,
            self.test_3_tactician_base_training_step_integration,
            self.test_4_fallback_configuration,
            self.test_5_multi_format_support,
            self.test_6_config_sections_access,
            self.test_7_performance_loading
        ]
        
        # Exécution des tests
        passed = 0
        failed = 0
        
        for test_func in tests:
            try:
                if test_func():
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                failed += 1
                print(f"❌ EXCEPTION in {test_func.__name__}: {e}")
        
        # Résumé
        total_time = time.time() - self.start_time
        success_rate = (passed / (passed + failed)) * 100 if (passed + failed) > 0 else 0
        
        print("\n" + "=" * 80)
        print("📊 RÉSUMÉ DES TESTS")
        print("=" * 80)
        print(f"✅ Tests réussis: {passed}")
        print(f"❌ Tests échoués: {failed}")
        print(f"📈 Taux de réussite: {success_rate:.1f}%")
        print(f"⏱️ Temps total: {total_time:.3f}s")
        print()
        
        # Rapport détaillé
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_tests': len(tests),
            'passed': passed,
            'failed': failed,
            'success_rate': success_rate,
            'total_execution_time': total_time,
            'test_results': self.results
        }
        
        return report
    
    def export_report(self, report: Dict[str, Any], filename: str = "test_tactician_base_config_report.json"):
        """Exporter le rapport de test."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"📄 Rapport exporté: {filename}")
            return True
        except Exception as e:
            print(f"⚠️ Erreur d'export: {e}")
            return False


def main():
    """Fonction principale."""
    print("Configuration du système de test...")
    
    try:
        # Créer et exécuter la suite de tests
        test_suite = TacticianBaseConfigTestSuite()
        report = test_suite.run_all_tests()
        
        # Exporter le rapport
        report_filename = f"test_tactician_base_config_report_{int(time.time())}.json"
        test_suite.export_report(report, report_filename)
        
        # Déterminer le code de sortie
        if report['failed'] == 0:
            print("\n🎉 TOUS LES TESTS SONT PASSÉS!")
            print("✅ Le système de configuration centralisée est prêt pour la production")
            return 0
        else:
            print(f"\n⚠️ {report['failed']} TEST(S) ÉCHOUE(S)")
            print("❌ Le système nécessite des corrections avant la production")
            return 1
            
    except Exception as e:
        print(f"\n💥 ERREUR CRITIQUE: {e}")
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())