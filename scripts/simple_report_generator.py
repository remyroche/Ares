#!/usr/bin/env python3
"""
Script simplifié de génération de rapports détaillés pour la validation des migrations
"""

import csv
from pathlib import Path
from datetime import datetime

def generate_html_report():
    """Génère un rapport HTML détaillé basé sur les résultats de validation."""
    
    # Données extraites du rapport de validation
    validation_data = {
        'test_order_manager.py': {
            'score': 0,
            'issues': [
                {'type': 'missing_import', 'severity': 'error', 'description': 'Import des assertions standardisées manquant'},
                {'type': 'manual_success_assertions', 'severity': 'warning', 'description': 'Assertions de succès manuelles', 'count': 9},
                {'type': 'manual_error_assertions', 'severity': 'warning', 'description': 'Assertions d\'erreur manuelles', 'count': 1}
            ]
        },
        'test_regime_economic_relevance.py': {
            'score': 90,
            'issues': [
                {'type': 'missing_import', 'severity': 'error', 'description': 'Import des assertions standardisées manquant'}
            ]
        },
        'test_paper_trading_simulator.py': {
            'score': 90,
            'issues': [
                {'type': 'missing_import', 'severity': 'error', 'description': 'Import des assertions standardisées manquant'}
            ]
        },
        'test_exchange_dispatcher.py': {
            'score': 90,
            'issues': [
                {'type': 'missing_import', 'severity': 'error', 'description': 'Import des assertions standardisées manquant'}
            ]
        }
    }
    
    # Calculer les statistiques
    total_files = len(validation_data)
    files_with_issues = sum(1 for v in validation_data.values() if v['issues'])
    total_issues = sum(len(v['issues']) for v in validation_data.values())
    scores = [v['score'] for v in validation_data.values()]
    avg_score = sum(scores) / len(scores)
    
    # Compter les problèmes par type
    manual_assertions = 0
    missing_imports = 0
    
    for file_data in validation_data.values():
        for issue in file_data['issues']:
            if 'manual' in issue['type']:
                manual_assertions += issue.get('count', 1)
            elif 'missing_import' in issue['type']:
                missing_imports += 1
    
    # Générer le HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rapport de Validation des Migrations d'Assertions</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s ease;
        }}
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        .stat-number {{
            font-size: 2.5em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 10px;
        }}
        .stat-label {{
            color: #666;
            font-size: 1.1em;
        }}
        .score-indicator {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            margin-left: 10px;
        }}
        .score-excellent {{ background: #28a745; color: white; }}
        .score-good {{ background: #ffc107; color: #333; }}
        .score-poor {{ background: #dc3545; color: white; }}
        .file-section {{
            background: white;
            margin-bottom: 25px;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .file-header {{
            padding: 20px;
            font-weight: bold;
            font-size: 1.2em;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .file-content {{
            padding: 0 20px 20px;
        }}
        .issue {{
            background: #f8f9fa;
            border-left: 4px solid #dc3545;
            padding: 15px;
            margin: 10px 0;
            border-radius: 0 5px 5px 0;
        }}
        .issue.warning {{
            border-left-color: #ffc107;
        }}
        .issue.error {{
            border-left-color: #dc3545;
        }}
        .issue-suggestion {{
            background: #d1ecf1;
            border-left: 4px solid #17a2b8;
            padding: 15px;
            margin: 10px 0;
            border-radius: 0 5px 5px 0;
        }}
        .summary-section {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .progress-bar {{
            width: 100%;
            height: 30px;
            background: #e9ecef;
            border-radius: 15px;
            overflow: hidden;
            margin: 20px 0;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #dc3545, #ffc107);
            transition: width 1s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }}
        .status-fail {{ color: #dc3545; font-weight: bold; }}
        .status-success {{ color: #28a745; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Rapport de Validation des Migrations d'Assertions</h1>
        <p>Généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-number">{total_files}</div>
            <div class="stat-label">Fichiers validés</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{files_with_issues}</div>
            <div class="stat-label">Fichiers avec problèmes</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{total_issues}</div>
            <div class="stat-label">Total des problèmes</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{avg_score:.1f}/100</div>
            <div class="stat-label">Score moyen de qualité</div>
        </div>
    </div>

    <div class="summary-section">
        <h2>📊 Résumé de la Migration</h2>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {avg_score}%">
                {avg_score:.1f}%
            </div>
        </div>
        
        <h3>🎯 Objectifs de Migration</h3>
        <ul>
            <li>Score de qualité cible: 85/100 <span class="status-fail">❌ NON ATTEINT</span></li>
            <li>Assertions manuelles restantes: {manual_assertions} <span class="status-fail">⚠️ {manual_assertions} restantes</span></li>
            <li>Imports manquants: {missing_imports} <span class="status-fail">❌ {missing_imports} manquants</span></li>
        </ul>
        
        <h3>📈 Statistiques Détaillées</h3>
        <ul>
            <li>Fichiers avec score excellent (≥90): 3/4</li>
            <li>Fichiers nécessitant des améliorations (<80): 1/4</li>
            <li>Assertions manuelles à migrer: {manual_assertions}</li>
            <li>Imports standardisés à ajouter: {missing_imports}</li>
        </ul>
    </div>

    <h2>📁 Détail par Fichier</h2>
"""

    # Ajouter le détail par fichier
    for file_name, file_data in validation_data.items():
        score = file_data['score']
        issues = file_data['issues']
        
        # Déterminer la classe de score
        if score >= 90:
            score_class = "score-excellent"
            score_text = "Excellent"
        elif score >= 80:
            score_class = "score-good"
            score_text = "Bon"
        else:
            score_class = "score-poor"
            score_text = "À améliorer"
        
        html_content += f"""
    <div class="file-section">
        <div class="file-header">
            <span>📄 {file_name}</span>
            <span class="score-indicator {score_class}">{score}/100 ({score_text})</span>
        </div>
        <div class="file-content">
"""
        
        if not issues:
            html_content += "<p>✅ Aucun problème détecté</p>"
        else:
            for issue in issues:
                severity = issue.get('severity', 'info')
                issue_class = f"issue {severity}"
                count = issue.get('count', 1)
                
                html_content += f"""
            <div class="{issue_class}">
                <strong>{issue['description']}</strong>
                {f' ({count} occurrences)' if count > 1 else ''}
"""
                
                if issue['type'] == 'missing_import':
                    html_content += """
                <div class="issue-suggestion">
                    💡 <strong>Suggestion:</strong> Ajouter: from tests.utils import (assert_success_response, assert_error_response, ...)
                </div>
"""
                elif 'manual' in issue['type']:
                    html_content += """
                <div class="issue-suggestion">
                    💡 <strong>Suggestion:</strong> Remplacer par les assertions standardisées correspondantes
                </div>
"""
                
                html_content += "</div>"
        
        html_content += """
        </div>
    </div>
"""

    # Ajouter les recommandations
    html_content += f"""
    <div class="summary-section">
        <h2>🚀 Recommandations d'Amélioration</h2>
        <ol>
            <li><strong>Corriger les erreurs critiques</strong> - Ajouter les {missing_imports} imports manquants dans tous les fichiers</li>
            <li><strong>Compléter la migration</strong> - Remplacer les {manual_assertions} assertions manuelles par les assertions standardisées</li>
            <li><strong>Améliorer test_order_manager.py</strong> - Ce fichier a le score le plus bas (0/100) et nécessite une attention particulière</li>
            <li><strong>Valider les tests</strong> - Exécuter les tests après corrections pour vérifier le fonctionnement</li>
        </ol>
        
        <h3>📋 Actions Immédiates</h3>
        <ul>
            <li>➕ Ajouter <code>from tests.utils.assertions import (...)</code> dans tous les fichiers</li>
            <li>🔄 Remplacer <code>assert result['success'] is True</code> par <code>assert_success_response(result)</code></li>
            <li>🔄 Remplacer <code>assert result['success'] is False</code> par <code>assert_error_response(result)</code></li>
            <li>✅ Valider la syntaxe Python avec <code>python -m py_compile</code></li>
            <li>🧪 Exécuter les tests unitaires pour vérifier le fonctionnement</li>
        </ul>
        
        <h3>🎯 Prochaines Étapes</h3>
        <ul>
            <li>Corriger les imports manquants (priorité haute)</li>
            <li>Migrer les assertions manuelles dans test_order_manager.py (priorité haute)</li>
            <li>Relancer la validation pour vérifier l'atteinte de l'objectif 85/100</li>
            <li>Générer un rapport final de migration complète</li>
        </ul>
    </div>

    <footer style="text-align: center; margin-top: 50px; padding: 20px; color: #666;">
        <p>Rapport généré par l'outil de validation des migrations ARES</p>
        <p>Score actuel: {avg_score:.1f}/100 (Objectif: 85/100)</p>
    </footer>
</body>
</html>
"""

    # Écrire le fichier HTML
    output_path = "validation_reports/detailed_report.html"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Rapport HTML généré: {output_path}")
    return output_path

def generate_csv_report():
    """Génère un rapport CSV détaillé."""
    
    validation_data = [
        {
            'file_path': 'temp_validation/test_order_manager.py',
            'score': 0,
            'total_issues': 12,
            'error_count': 1,
            'warning_count': 11,
            'manual_assertions': 10,
            'missing_imports': 1
        },
        {
            'file_path': 'temp_validation/test_regime_economic_relevance.py',
            'score': 90,
            'total_issues': 1,
            'error_count': 1,
            'warning_count': 0,
            'manual_assertions': 0,
            'missing_imports': 1
        },
        {
            'file_path': 'temp_validation/test_paper_trading_simulator.py',
            'score': 90,
            'total_issues': 1,
            'error_count': 1,
            'warning_count': 0,
            'manual_assertions': 0,
            'missing_imports': 1
        },
        {
            'file_path': 'temp_validation/test_exchange_dispatcher.py',
            'score': 90,
            'total_issues': 1,
            'error_count': 1,
            'warning_count': 0,
            'manual_assertions': 0,
            'missing_imports': 1
        }
    ]
    
    output_path = "validation_reports/detailed_report.csv"
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'file_path', 'score', 'total_issues', 'error_count', 
            'warning_count', 'manual_assertions', 'missing_imports'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for row in validation_data:
            writer.writerow(row)
    
    print(f"✅ Rapport CSV généré: {output_path}")
    return output_path

def main():
    """Fonction principale."""
    
    # Créer le répertoire de sortie
    Path("validation_reports").mkdir(parents=True, exist_ok=True)
    
    # Générer les rapports
    html_path = generate_html_report()
    csv_path = generate_csv_report()
    
    # Calculer et afficher le résumé
    total_files = 4
    files_with_issues = 4
    total_issues = 15
    avg_score = 67.5
    manual_assertions = 10
    missing_imports = 4
    
    print("\n" + "="*60)
    print("📊 RÉSUMÉ DE VALIDATION DES MIGRATIONS")
    print("="*60)
    print(f"📁 Fichiers validés: {total_files}")
    print(f"⚠️  Fichiers avec problèmes: {files_with_issues}")
    print(f"🔍 Total des problèmes: {total_issues}")
    print(f"📈 Score moyen de qualité: {avg_score:.1f}/100")
    print(f"📝 Assertions manuelles restantes: {manual_assertions}")
    print(f"📦 Imports manquants: {missing_imports}")
    print(f"🎯 Objectif 85/100: {'✅ ATTEINT' if avg_score >= 85 else '❌ NON ATTEINT'}")
    print("="*60)
    
    return {
        'html_path': html_path,
        'csv_path': csv_path,
        'avg_score': avg_score,
        'manual_assertions': manual_assertions,
        'missing_imports': missing_imports,
        'objective_reached': avg_score >= 85
    }

if __name__ == "__main__":
    main()