#!/usr/bin/env python3
"""
Script de Validation de Migration des Assertions Standardisées ARES

Ce script valide que les migrations ont été correctement appliquées
et génère des rapports de qualité.
"""

import os
import re
import sys
import ast
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
import logging
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MigrationValidator:
    """Classe pour valider les migrations d'assertions."""
    
    def __init__(self):
        self.validation_stats = {
            'files_validated': 0,
            'files_with_issues': 0,
            'total_issues': 0,
            'issues_by_type': {},
            'imports_validated': 0,
            'imports_missing': 0
        }
        
        # Patterns de validation
        self.validation_patterns = {
            # Vérifier que les imports standardisés sont présents
            'standardized_imports': {
                'pattern': r'from\s+tests\.utils\s+import\s+\([^)]+\)',
                'description': 'Import des assertions standardisées',
                'severity': 'error'
            },
            
            # Vérifier l'absence d'anciennes assertions manuelles
            'manual_success_assertions': {
                'patterns': [
                    r'assert\s+result\[[\'"]success[\'"]\]\s+is\s+True',
                    r'assert\s+result\.get\([\'"]success[\'"]\)\s+is\s+True',
                    r'assert\s+result\[[\'"]success[\'"]\]\s*==\s*True'
                ],
                'description': 'Assertions de succès manuelles',
                'severity': 'warning'
            },
            
            'manual_error_assertions': {
                'patterns': [
                    r'assert\s+result\[[\'"]success[\'"]\]\s+is\s+False',
                    r'assert\s+result\.get\([\'"]success[\'"]\)\s+is\s+False',
                    r'assert\s+result\[[\'"]success[\'"]\]\s*==\s*False'
                ],
                'description': 'Assertions d\'erreur manuelles',
                'severity': 'warning'
            },
            
            # Vérifier les comparaisons de prix sans tolérance
            'price_comparisons': {
                'patterns': [
                    r'assert\s+\w+\[[\'"]price[\'"]\]\s*==\s*[\d.]+',
                    r'assert\s+\w+\[[\'"]price[\'"]\]\s*!=\s*[\d.]+'
                ],
                'description': 'Comparaisons de prix sans tolérance',
                'severity': 'warning'
            },
            
            # Vérifier les comparaisons de flottants sans tolérance
            'float_comparisons': {
                'patterns': [
                    r'assert\s+abs\([^)]+\s*-\s*[^)]+\)\s*[<<=]\s*[\d.]+',
                    r'assert\s+[^=<>!]+\s*[=!<>]+\s*[\d.]+\s*#[\s#]*tolérance'
                ],
                'description': 'Comparaisons de flottants sans tolérance explicite',
                'severity': 'info'
            }
        }
    
    def validate_file(self, file_path: Path) -> Dict:
        """Valide un fichier de test migré."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            validation_result = {
                'file_path': str(file_path),
                'issues': [],
                'imports_found': [],
                'standardized_assertions': [],
                'manual_assertions': [],
                'score': 100  # Score de qualité (0-100)
            }
            
            # Validation des imports
            import_issues = self._validate_imports(content)
            validation_result['issues'].extend(import_issues)
            validation_result['imports_found'] = self._find_imports(content)
            
            # Validation des patterns
            for pattern_name, pattern_info in self.validation_patterns.items():
                if 'patterns' in pattern_info:
                    # Multiple patterns pour ce type
                    for pattern in pattern_info['patterns']:
                        matches = re.finditer(pattern, content, re.MULTILINE)
                        for match in matches:
                            issue = {
                                'type': pattern_name,
                                'description': pattern_info['description'],
                                'severity': pattern_info['severity'],
                                'line_number': content[:match.start()].count('\n') + 1,
                                'match_text': match.group(0),
                                'suggestion': self._get_suggestion(pattern_name, match)
                            }
                            validation_result['issues'].append(issue)
                            
                            # Catégoriser l'assertion
                            if pattern_name in ['manual_success_assertions', 'manual_error_assertions']:
                                validation_result['manual_assertions'].append(issue)
                            else:
                                validation_result['standardized_assertions'].append(issue)
                else:
                    # Pattern unique
                    matches = re.finditer(pattern_info['pattern'], content, re.MULTILINE)
                    for match in matches:
                        if pattern_name == 'standardized_imports':
                            # C'est une bonne chose, pas une issue
                            validation_result['imports_found'].append(match.group(0))
                        else:
                            issue = {
                                'type': pattern_name,
                                'description': pattern_info['description'],
                                'severity': pattern_info['severity'],
                                'line_number': content[:match.start()].count('\n') + 1,
                                'match_text': match.group(0)
                            }
                            validation_result['issues'].append(issue)
            
            # Calculer le score de qualité
            validation_result['score'] = self._calculate_quality_score(validation_result)
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Erreur lors de la validation du fichier {file_path}: {e}")
            return {
                'file_path': str(file_path),
                'error': str(e),
                'score': 0
            }
    
    def _validate_imports(self, content: str) -> List[Dict]:
        """Valide que les imports nécessaires sont présents."""
        issues = []
        
        # Vérifier l'import standardisé
        import_pattern = r'from\s+tests\.utils\s+import\s+\([^)]+\)'
        if not re.search(import_pattern, content):
            issues.append({
                'type': 'missing_import',
                'description': 'Import des assertions standardisées manquant',
                'severity': 'error',
                'suggestion': 'Ajouter: from tests.utils import (assert_success_response, assert_error_response, ...)'
            })
        
        return issues
    
    def _find_imports(self, content: str) -> List[str]:
        """Trouve tous les imports d'assertions standardisées."""
        imports = []
        
        # Pattern pour capturer les imports
        import_pattern = r'from\s+tests\.utils\s+import\s+\(([^)]+)\)'
        match = re.search(import_pattern, content)
        
        if match:
            import_list = match.group(1)
            # Nettoyer et séparer les imports
            imports = [imp.strip() for imp in import_list.split(',')]
            imports = [imp for imp in imports if imp]  # Filtrer les vides
        
        return imports
    
    def _get_suggestion(self, pattern_name: str, match: re.Match) -> str:
        """Génère une suggestion pour un pattern problématique."""
        suggestions = {
            'manual_success_assertions': 'Remplacer par: assert_success_response(result)',
            'manual_error_assertions': 'Remplacer par: assert_error_response(result)',
            'price_comparisons': 'Remplacer par: assert_price_equals(actual, expected)',
            'float_comparisons': 'Remplacer par: assert_float_equals(actual, expected, tolerance=X.X)'
        }
        
        return suggestions.get(pattern_name, 'Consulter le guide de migration')
    
    def _calculate_quality_score(self, validation_result: Dict) -> int:
        """Calcule un score de qualité pour le fichier."""
        if 'error' in validation_result:
            return 0
        
        base_score = 100
        
        # Pénalités par type de problème
        penalties = {
            'error': 20,
            'warning': 10,
            'info': 5
        }
        
        total_penalty = 0
        for issue in validation_result['issues']:
            severity = issue.get('severity', 'info')
            total_penalty += penalties.get(severity, 5)
        
        # Bonus pour les imports standardisés
        if validation_result['imports_found']:
            base_score += 5
        
        # Bonus pour l'absence d'assertions manuelles
        if not validation_result['manual_assertions']:
            base_score += 10
        
        return max(0, base_score - total_penalty)
    
    def generate_quality_report(self, validations: List[Dict]) -> str:
        """Génère un rapport de qualité des migrations."""
        report = []
        report.append("# Rapport de Qualité de Migration des Assertions\n")
        report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Statistiques générales
        total_files = len(validations)
        files_with_issues = sum(1 for v in validations if v.get('issues') and not v.get('error'))
        total_issues = sum(len(v.get('issues', [])) for v in validations)
        
        # Calculer le score moyen
        scores = [v.get('score', 0) for v in validations if not v.get('error')]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        report.append("## Statistiques de Qualité\n")
        report.append(f"- Fichiers validés: {total_files}")
        report.append(f"- Fichiers avec problèmes: {files_with_issues}")
        report.append(f"- Total des problèmes: {total_issues}")
        report.append(f"- Score moyen de qualité: {avg_score:.1f}/100\n")
        
        # Répartition par sévérité
        severity_counts = {'error': 0, 'warning': 0, 'info': 0}
        for validation in validations:
            for issue in validation.get('issues', []):
                severity = issue.get('severity', 'info')
                severity_counts[severity] += 1
        
        if any(severity_counts.values()):
            report.append("## Répartition des Problèmes par Sévérité\n")
            if severity_counts['error'] > 0:
                report.append(f"- ❌ Erreurs: {severity_counts['error']}")
            if severity_counts['warning'] > 0:
                report.append(f"- ⚠️  Avertissements: {severity_counts['warning']}")
            if severity_counts['info'] > 0:
                report.append(f"- ℹ️  Informations: {severity_counts['info']}")
            report.append("")
        
        # Détail par fichier
        report.append("## Détail par Fichier\n")
        
        # Trier par score (du moins bon au meilleur)
        sorted_validations = sorted(validations, key=lambda x: x.get('score', 0))
        
        for validation in sorted_validations:
            if 'error' in validation:
                report.append(f"### ❌ {Path(validation['file_path']).name}")
                report.append(f"Erreur de validation: {validation['error']}\n")
                continue
            
            file_path = Path(validation['file_path']).name
            score = validation['score']
            issues = validation['issues']
            
            # Icône de qualité
            if score >= 90:
                status_icon = "🟢"
                status_text = "Excellent"
            elif score >= 80:
                status_icon = "🟡"
                status_text = "Bon"
            elif score >= 70:
                status_icon = "🟠"
                status_text = "Moyen"
            else:
                status_icon = "🔴"
                status_text = "À améliorer"
            
            report.append(f"### {status_icon} {file_path} - Score: {score}/100 ({status_text})")
            
            if not issues:
                report.append("✅ Aucun problème détecté\n")
                continue
            
            # Grouper les problèmes par sévérité
            issues_by_severity = {'error': [], 'warning': [], 'info': []}
            for issue in issues:
                severity = issue.get('severity', 'info')
                issues_by_severity[severity].append(issue)
            
            # Afficher les problèmes par sévérité
            for severity in ['error', 'warning', 'info']:
                severity_issues = issues_by_severity[severity]
                if not severity_issues:
                    continue
                
                severity_icons = {'error': '❌', 'warning': '⚠️', 'info': 'ℹ️'}
                report.append(f"\n{severity_icons[severity]} **{severity.title()}s** ({len(severity_issues)}):")
                
                for issue in severity_issues:
                    line_num = issue.get('line_number', 'N/A')
                    report.append(f"- Ligne {line_num}: {issue['description']}")
                    if 'match_text' in issue:
                        report.append(f"  Code: `{issue['match_text']}`")
                    if 'suggestion' in issue:
                        report.append(f"  💡 Suggestion: {issue['suggestion']}")
            
            report.append("")
        
        # Imports trouvés
        all_imports = set()
        for validation in validations:
            if 'imports_found' in validation:
                all_imports.update(validation['imports_found'])
        
        if all_imports:
            report.append("## Imports Standardisés Utilisés\n")
            report.append("```python")
            report.append(f"from tests.utils import ({', '.join(sorted(all_imports))})")
            report.append("```\n")
        
        # Recommandations
        report.append("## Recommandations d'Amélioration\n")
        
        if severity_counts['error'] > 0:
            report.append("1. **Corriger les erreurs critiques** avant de merger")
        
        if severity_counts['warning'] > 0:
            report.append("2. **Compléter la migration** des assertions manuelles restantes")
        
        if severity_counts['info'] > 0:
            report.append("3. **Optimiser les comparaisons** numériques avec tolérances")
        
        report.append("4. **Valider les tests** après correction")
        report.append("5. **Documenter les patterns** spécifiques au projet\n")
        
        return '\n'.join(report)
    
    def validate_directory(self, directory: Path, pattern: str = "*.py") -> List[Dict]:
        """Valide tous les fichiers de test dans un répertoire."""
        validations = []
        
        for file_path in directory.rglob(pattern):
            if 'test' in file_path.name.lower() and file_path.suffix == '.py':
                logger.info(f"Validation du fichier: {file_path}")
                
                validation = self.validate_file(file_path)
                validations.append(validation)
                
                self.validation_stats['files_validated'] += 1
                
                if validation.get('issues') and not validation.get('error'):
                    self.validation_stats['files_with_issues'] += 1
                    self.validation_stats['total_issues'] += len(validation['issues'])
                
                # Mettre à jour les statistiques par type
                for issue in validation.get('issues', []):
                    issue_type = issue['type']
                    if issue_type not in self.validation_stats['issues_by_type']:
                        self.validation_stats['issues_by_type'][issue_type] = 0
                    self.validation_stats['issues_by_type'][issue_type] += 1
        
        return validations
    
    def print_summary(self):
        """Affiche un résumé des statistiques de validation."""
        logger.info("\n" + "="*50)
        logger.info("RÉSUMÉ DE VALIDATION")
        logger.info("="*50)
        logger.info(f"Fichiers validés: {self.validation_stats['files_validated']}")
        logger.info(f"Fichiers avec problèmes: {self.validation_stats['files_with_issues']}")
        logger.info(f"Total des problèmes: {self.validation_stats['total_issues']}")
        
        if self.validation_stats['issues_by_type']:
            logger.info("\nProblèmes par type:")
            for issue_type, count in self.validation_stats['issues_by_type'].items():
                logger.info(f"  - {issue_type}: {count}")
        
        # Score moyen de qualité
        if self.validation_stats['files_validated'] > 0:
            quality_score = max(0, 100 - (self.validation_stats['total_issues'] * 2))
            logger.info(f"Score global de qualité: {quality_score}/100")
        
        logger.info("="*50)


def main():
    """Fonction principale du script."""
    parser = argparse.ArgumentParser(
        description="Script de validation de migration des assertions standardisées ARES"
    )
    parser.add_argument(
        'directory',
        nargs='?',
        default='tests',
        help='Répertoire des tests à valider (défaut: tests)'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Générer un rapport détaillé'
    )
    parser.add_argument(
        '--output',
        default='validation_report.md',
        help='Fichier de sortie du rapport (défaut: validation_report.md)'
    )
    parser.add_argument(
        '--threshold',
        type=int,
        default=80,
        help='Seuil de score de qualité acceptable (défaut: 80)'
    )
    
    args = parser.parse_args()
    
    # Configuration
    directory = Path(args.directory)
    
    if not directory.exists():
        logger.error(f"Le répertoire {directory} n'existe pas")
        sys.exit(1)
    
    logger.info(f"Démarrage de la validation")
    logger.info(f"Répertoire cible: {directory.absolute()}")
    
    # Initialisation du validateur
    validator = MigrationValidator()
    
    # Validation des fichiers
    validations = validator.validate_directory(directory)
    
    # Génération du rapport si demandé
    if args.report:
        report = validator.generate_quality_report(validations)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Rapport généré: {args.output}")
        
        # Vérifier le seuil de qualité
        scores = [v.get('score', 0) for v in validations if not v.get('error')]
        if scores:
            avg_score = sum(scores) / len(scores)
            if avg_score < args.threshold:
                logger.warning(f"Score moyen ({avg_score:.1f}) inférieur au seuil ({args.threshold})")
                sys.exit(1)
            else:
                logger.info(f"Score moyen ({avg_score:.1f}) supérieur au seuil ({args.threshold})")
    
    # Affichage du résumé
    validator.print_summary()


if __name__ == "__main__":
    main()