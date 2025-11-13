#!/usr/bin/env python3
"""
Script de Migration Automatisée vers les Assertions Standardisées ARES

Ce script analyse les fichiers de tests existants et suggère/remplace
automatiquement les assertions manuelles par les assertions standardisées.
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import ast
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('migration.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AssertionMigrator:
    """Classe principale pour la migration des assertions."""
    
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.migration_stats = {
            'files_processed': 0,
            'assertions_found': 0,
            'assertions_migrated': 0,
            'patterns_detected': {}
        }
        
        # Patterns de migration
        self.migration_patterns = {
            # Assertions de succès
            'assert_success': {
                'patterns': [
                    r"assert\s+result\['success'\]\s+is\s+True",
                    r"assert\s+result\.get\('success'\)\s+is\s+True",
                    r"assert\s+result\['success'\]\s*==\s*True"
                ],
                'replacement': 'assert_success_response(result, "La réponse devrait indiquer un succès")',
                'import_needed': 'assert_success_response',
                'description': 'Assertion de réponse API succès'
            },
            
            # Assertions d'erreur
            'assert_error': {
                'patterns': [
                    r"assert\s+result\['success'\]\s+is\s+False",
                    r"assert\s+result\.get\('success'\)\s+is\s+False",
                    r"assert\s+result\['success'\]\s*==\s*False"
                ],
                'replacement': 'assert_error_response(result, message="La réponse devrait indiquer une erreur")',
                'import_needed': 'assert_error_response',
                'description': 'Assertion de réponse API erreur'
            },
            
            # Assertions de structure de dictionnaire
            'assert_dict_keys': {
                'patterns': [
                    r"assert\s+['\"]([^'\"]+)['\"]\s+in\s+result",
                    r"assert\s+['\"]([^'\"]+)['\"]\s+in\s+(\w+)",
                    r"assert\s+all\(key\s+in\s+\w+\s+for\s+key\s+in\s+\[([^\]]+)\]\)"
                ],
                'replacement': 'assert_dict_structure(result, required_keys=[\\1])',
                'import_needed': 'assert_dict_structure',
                'description': 'Assertion de structure de dictionnaire'
            },
            
            # Assertions de prix
            'assert_price': {
                'patterns': [
                    r"assert\s+(\w+)\['price'\]\s*==\s*([\d.]+)",
                    r"assert\s+abs\((\w+)\['price'\]\s*-\s*([\d.]+)\)\s*<\s*([\d.]+)"
                ],
                'replacement': 'assert_price_equals(\\1['price'], \\2)',
                'import_needed': 'assert_price_equals',
                'description': 'Assertion de comparaison de prix'
            },
            
            # Assertions de statut d'ordre
            'assert_order_status': {
                'patterns': [
                    r"assert\s+(\w+)\['status'\]\s*==\s*['\"]([^'\"]+)['\"]",
                    r"assert\s+(\w+)\['status'\]\s*==\s*OrderStatus\.(\w+)"
                ],
                'replacement': 'assert_order_status(\\1['status'], '\\2')',
                'import_needed': 'assert_order_status',
                'description': 'Assertion de statut d\'ordre'
            },
            
            # Assertions de statut d'exchange
            'assert_exchange_status': {
                'patterns': [
                    r"assert\s+(\w+)\['status'\]\s*==\s*['\"]([^'\"]+)['\"]",
                    r"assert\s+(\w+)\['status'\]\s*==\s*ExchangeStatus\.(\w+)"
                ],
                'replacement': 'assert_exchange_status(\\1['status'], '\\2')',
                'import_needed': 'assert_exchange_status',
                'description': 'Assertion de statut d\'exchange'
            },
            
            # Assertions de flottants
            'assert_float': {
                'patterns': [
                    r"assert\s+abs\(([^)]+)\s*-\s*([^)]+)\)\s*<\s*([\d.]+)",
                    r"assert\s+abs\(([^)]+)\s*-\s*([^)]+)\)\s*<=\s*([\d.]+)"
                ],
                'replacement': 'assert_float_equals(\\1, \\2, tolerance=\\3)',
                'import_needed': 'assert_float_equals',
                'description': 'Assertion de comparaison de flottants'
            },
            
            # Assertions de performance
            'assert_performance': {
                'patterns': [
                    r"assert\s+['\"]total_return['\"]\s+in\s+(\w+)",
                    r"assert\s+['\"]sharpe_ratio['\"]\s+in\s+(\w+)",
                    r"assert\s+['\"]max_drawdown['\"]\s+in\s+(\w+)"
                ],
                'replacement': 'assert_performance_metrics(\\1)',
                'import_needed': 'assert_performance_metrics',
                'description': 'Assertion de métriques de performance'
            }
        }
    
    def analyze_file(self, file_path: Path) -> Dict:
        """Analyse un fichier de test et identifie les patterns de migration."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            analysis = {
                'file_path': str(file_path),
                'assertions_found': [],
                'imports_needed': set(),
                'suggested_replacements': []
            }
            
            # Analyser chaque pattern
            for pattern_name, pattern_info in self.migration_patterns.items():
                for pattern in pattern_info['patterns']:
                    matches = re.finditer(pattern, content, re.MULTILINE)
                    for match in matches:
                        assertion_info = {
                            'type': pattern_name,
                            'description': pattern_info['description'],
                            'line_number': content[:match.start()].count('\n') + 1,
                            'match_text': match.group(0),
                            'suggested_replacement': pattern_info['replacement'],
                            'import_needed': pattern_info['import_needed']
                        }
                        analysis['assertions_found'].append(assertion_info)
                        analysis['imports_needed'].add(pattern_info['import_needed'])
                        
                        # Générer le remplacement suggéré
                        replacement = self._generate_replacement(match, pattern_info)
                        analysis['suggested_replacements'].append({
                            'line_number': assertion_info['line_number'],
                            'original': match.group(0),
                            'replacement': replacement
                        })
            
            return analysis
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse du fichier {file_path}: {e}")
            return {'file_path': str(file_path), 'error': str(e)}
    
    def _generate_replacement(self, match: re.Match, pattern_info: Dict) -> str:
        """Génère le remplacement approprié pour un match."""
        replacement = pattern_info['replacement']
        
        # Remplacer les groupes de capture
        for i in range(1, len(match.groups()) + 1):
            replacement = replacement.replace(f'\\{i}', match.group(i))
        
        return replacement
    
    def migrate_file(self, file_path: Path) -> bool:
        """Migration effective d'un fichier de test."""
        if self.dry_run:
            logger.info(f"DRY RUN: Analyse de {file_path}")
            return True
        
        try:
            analysis = self.analyze_file(file_path)
            
            if 'error' in analysis:
                logger.error(f"Erreur dans l'analyse de {file_path}: {analysis['error']}")
                return False
            
            if not analysis['assertions_found']:
                logger.info(f"Aucune assertion à migrer dans {file_path}")
                return True
            
            # Lire le contenu original
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Ajouter les imports nécessaires
            new_content = self._add_imports(original_content, analysis['imports_needed'])
            
            # Appliquer les remplacements
            # Trier par ligne_number en ordre décroissant pour éviter les décalages
            replacements = sorted(analysis['suggested_replacements'], 
                             key=lambda x: x['line_number'], reverse=True)
            
            for repl in replacements:
                lines = new_content.split('\n')
                line_idx = repl['line_number'] - 1
                
                if 0 <= line_idx < len(lines):
                    lines[line_idx] = repl['replacement']
                    new_content = '\n'.join(lines)
            
            # Écrire le nouveau contenu
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            logger.info(f"Fichier {file_path} migré avec succès")
            self.migration_stats['assertions_migrated'] += len(analysis['assertions_found'])
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la migration du fichier {file_path}: {e}")
            return False
    
    def _add_imports(self, content: str, imports_needed: set) -> str:
        """Ajoute les imports nécessaires au contenu."""
        if not imports_needed:
            return content
        
        # Vérifier si les imports existent déjà
        existing_imports = set()
        import_pattern = r"from\s+tests\.utils\s+import\s+\(([^)]+)\)"
        import_match = re.search(import_pattern, content)
        
        if import_match:
            existing_imports = set(imp.strip() for imp in import_match.group(1).split(','))
        
        # Identifier les nouveaux imports nécessaires
        new_imports = imports_needed - existing_imports
        
        if not new_imports:
            return content
        
        # Construire la nouvelle ligne d'import
        if import_match:
            # Mettre à jour l'import existant
            all_imports = existing_imports.union(new_imports)
            import_line = f"from tests.utils import ({', '.join(sorted(all_imports))})"
            content = re.sub(import_pattern, import_line, content)
        else:
            # Ajouter un nouvel import après les imports existants
            import_line = f"from tests.utils import ({', '.join(sorted(new_imports))})"
            
            # Trouver où insérer l'import (après les imports de pytest)
            lines = content.split('\n')
            insert_idx = 0
            
            for i, line in enumerate(lines):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    insert_idx = i + 1
                elif line.strip() == '' and insert_idx > 0:
                    break
            
            lines.insert(insert_idx, import_line)
            content = '\n'.join(lines)
        
        return content
    
    def generate_report(self, analyses: List[Dict]) -> str:
        """Génère un rapport détaillé des analyses."""
        report = []
        report.append("# Rapport de Migration des Assertions Standardisées\n")
        report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Statistiques générales
        total_assertions = sum(len(analysis.get('assertions_found', [])) for analysis in analyses)
        total_files = len(analyses)
        
        report.append("## Statistiques Générales\n")
        report.append(f"- Fichiers analysés: {total_files}")
        report.append(f"- Assertions trouvées: {total_assertions}")
        report.append(f"- Types de patterns détectés: {len(self.migration_stats['patterns_detected'])}\n")
        
        # Détail par fichier
        report.append("## Détail par Fichier\n")
        
        for analysis in analyses:
            if 'error' in analysis:
                report.append(f"### ❌ {analysis['file_path']}")
                report.append(f"Erreur: {analysis['error']}\n")
                continue
            
            file_path = Path(analysis['file_path']).name
            assertions = analysis['assertions_found']
            
            if not assertions:
                report.append(f"### ✅ {file_path}")
                report.append("Aucune assertion à migrer\n")
                continue
            
            report.append(f"### 🔄 {file_path}")
            report.append(f"Assertions trouvées: {len(assertions)}")
            
            # Grouper par type
            by_type = {}
            for assertion in assertions:
                assertion_type = assertion['type']
                if assertion_type not in by_type:
                    by_type[assertion_type] = []
                by_type[assertion_type].append(assertion)
            
            for assertion_type, type_assertions in by_type.items():
                report.append(f"\n**{self.migration_patterns[assertion_type]['description']}** ({len(type_assertions)}):")
                
                for assertion in type_assertions:
                    report.append(f"- Ligne {assertion['line_number']}: `{assertion['match_text']}`")
                    report.append(f"  → Suggestion: `{assertion['suggested_replacement']}`")
            
            report.append("")
        
        # Imports nécessaires
        all_imports = set()
        for analysis in analyses:
            if 'imports_needed' in analysis:
                all_imports.update(analysis['imports_needed'])
        
        if all_imports:
            report.append("## Imports Nécessaires\n")
            report.append("```python")
            report.append(f"from tests.utils import ({', '.join(sorted(all_imports))})")
            report.append("```\n")
        
        # Recommandations
        report.append("## Recommandations\n")
        report.append("1. **Revérifier manuellement** les remplacements automatiques")
        report.append("2. **Tester les modifications** avant de commiter")
        report.append("3. **Adapter les messages** d'erreur au contexte métier")
        report.append("4. **Valider la couverture** des tests après migration\n")
        
        return '\n'.join(report)
    
    def process_directory(self, directory: Path, pattern: str = "*.py") -> List[Dict]:
        """Traite tous les fichiers de test dans un répertoire."""
        analyses = []
        
        # Parcourir récursivement les fichiers
        for file_path in directory.rglob(pattern):
            if 'test' in file_path.name.lower() and file_path.suffix == '.py':
                logger.info(f"Analyse du fichier: {file_path}")
                
                analysis = self.analyze_file(file_path)
                analyses.append(analysis)
                
                self.migration_stats['files_processed'] += 1
                self.migration_stats['assertions_found'] += len(analysis.get('assertions_found', []))
                
                # Mettre à jour les patterns détectés
                for assertion in analysis.get('assertions_found', []):
                    assertion_type = assertion['type']
                    if assertion_type not in self.migration_stats['patterns_detected']:
                        self.migration_stats['patterns_detected'][assertion_type] = 0
                    self.migration_stats['patterns_detected'][assertion_type] += 1
        
        return analyses
    
    def print_summary(self):
        """Affiche un résumé des statistiques de migration."""
        logger.info("\n" + "="*50)
        logger.info("RÉSUMÉ DE LA MIGRATION")
        logger.info("="*50)
        logger.info(f"Fichiers traités: {self.migration_stats['files_processed']}")
        logger.info(f"Assertions trouvées: {self.migration_stats['assertions_found']}")
        logger.info(f"Assertions migrées: {self.migration_stats['assertions_migrated']}")
        
        if self.migration_stats['patterns_detected']:
            logger.info("\nPatterns détectés:")
            for pattern, count in self.migration_stats['patterns_detected'].items():
                pattern_desc = self.migration_patterns[pattern]['description']
                logger.info(f"  - {pattern_desc}: {count}")
        
        logger.info("="*50)


def main():
    """Fonction principale du script."""
    parser = argparse.ArgumentParser(
        description="Script de migration automatisée vers les assertions standardisées ARES"
    )
    parser.add_argument(
        'directory',
        nargs='?',
        default='tests',
        help='Répertoire des tests à analyser (défaut: tests)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        default=True,
        help='Mode simulation (défaut: True)'
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Appliquer les modifications (désactive le mode dry-run)'
    )
    parser.add_argument(
        '--report',
        action='store_true',
        help='Générer un rapport détaillé'
    )
    parser.add_argument(
        '--output',
        default='migration_report.md',
        help='Fichier de sortie du rapport (défaut: migration_report.md)'
    )
    
    args = parser.parse_args()
    
    # Configuration
    dry_run = not args.apply
    directory = Path(args.directory)
    
    if not directory.exists():
        logger.error(f"Le répertoire {directory} n'existe pas")
        sys.exit(1)
    
    logger.info(f"Démarrage de la migration {'(DRY RUN)' if dry_run else '(APPLICATION)'}")
    logger.info(f"Répertoire cible: {directory.absolute()}")
    
    # Initialisation du migrateur
    migrator = AssertionMigrator(dry_run=dry_run)
    
    # Analyse des fichiers
    analyses = migrator.process_directory(directory)
    
    # Génération du rapport si demandé
    if args.report:
        report = migrator.generate_report(analyses)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Rapport généré: {args.output}")
    
    # Application des modifications si demandé
    if not dry_run:
        logger.info("Application des modifications...")
        success_count = 0
        
        for analysis in analyses:
            if 'error' not in analysis:
                file_path = Path(analysis['file_path'])
                if migrator.migrate_file(file_path):
                    success_count += 1
        
        logger.info(f"Migration réussie pour {success_count}/{len(analyses)} fichiers")
    
    # Affichage du résumé
    migrator.print_summary()


if __name__ == "__main__":
    main()