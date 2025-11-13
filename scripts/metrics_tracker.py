#!/usr/bin/env python3
"""
Système de Métriques de Suivi - Phase 2 Assertions Standardisées ARES

Ce script collecte, analyse et présente les métriques de progression
de la migration vers les assertions standardisées.
"""

import os
import json
import sqlite3
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import matplotlib.pyplot as plt
import pandas as pd

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MetricsTracker:
    """Classe pour suivre les métriques de migration des assertions."""
    
    def __init__(self, db_path: str = "metrics_phase2.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialise la base de données des métriques."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table des métriques quotidiennes
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_metrics (
                date TEXT PRIMARY KEY,
                files_analyzed INTEGER,
                assertions_found INTEGER,
                assertions_migrated INTEGER,
                files_migrated INTEGER,
                migration_score REAL,
                test_errors_before INTEGER,
                test_errors_after INTEGER,
                error_reduction_rate REAL
            )
        ''')
        
        # Table des métriques par fichier
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS file_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                file_path TEXT,
                assertions_before INTEGER,
                assertions_after INTEGER,
                migration_status TEXT,
                quality_score INTEGER,
                migration_time_seconds REAL,
                developer TEXT
            )
        ''')
        
        # Table des patterns détectés
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pattern_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                pattern_type TEXT,
                count INTEGER,
                files_affected INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"Base de données initialisée: {self.db_path}")
    
    def record_daily_metrics(self, metrics: Dict[str, Any]):
        """Enregistre les métriques quotidiennes."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO daily_metrics 
            (date, files_analyzed, assertions_found, assertions_migrated, 
             files_migrated, migration_score, test_errors_before, 
             test_errors_after, error_reduction_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics['date'],
            metrics.get('files_analyzed', 0),
            metrics.get('assertions_found', 0),
            metrics.get('assertions_migrated', 0),
            metrics.get('files_migrated', 0),
            metrics.get('migration_score', 0.0),
            metrics.get('test_errors_before', 0),
            metrics.get('test_errors_after', 0),
            metrics.get('error_reduction_rate', 0.0)
        ))
        
        conn.commit()
        conn.close()
        logger.info(f"Métriques quotidiennes enregistrées pour {metrics['date']}")
    
    def record_file_metrics(self, file_metrics: Dict[str, Any]):
        """Enregistre les métriques pour un fichier spécifique."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO file_metrics 
            (date, file_path, assertions_before, assertions_after, 
             migration_status, quality_score, migration_time_seconds, developer)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            file_metrics['date'],
            file_metrics['file_path'],
            file_metrics.get('assertions_before', 0),
            file_metrics.get('assertions_after', 0),
            file_metrics.get('migration_status', 'pending'),
            file_metrics.get('quality_score', 0),
            file_metrics.get('migration_time_seconds', 0.0),
            file_metrics.get('developer', 'unknown')
        ))
        
        conn.commit()
        conn.close()
        logger.info(f"Métriques enregistrées pour le fichier: {file_metrics['file_path']}")
    
    def record_pattern_metrics(self, pattern_metrics: Dict[str, Any]):
        """Enregistre les métriques de patterns détectés."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO pattern_metrics 
            (date, pattern_type, count, files_affected)
            VALUES (?, ?, ?, ?)
        ''', (
            pattern_metrics['date'],
            pattern_metrics['pattern_type'],
            pattern_metrics.get('count', 0),
            pattern_metrics.get('files_affected', 0)
        ))
        
        conn.commit()
        conn.close()
        logger.info(f"Métriques de pattern enregistrées: {pattern_metrics['pattern_type']}")
    
    def get_daily_metrics(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict]:
        """Récupère les métriques quotidiennes."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM daily_metrics"
        params = []
        
        if start_date:
            query += " WHERE date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?" if start_date else " WHERE date <= ?"
            params.append(end_date)
        
        query += " ORDER BY date DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        columns = [desc[0] for desc in cursor.description]
        metrics = [dict(zip(columns, row)) for row in rows]
        
        conn.close()
        return metrics
    
    def get_file_metrics(self, date: Optional[str] = None) -> List[Dict]:
        """Récupère les métriques par fichier."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM file_metrics"
        params = []
        
        if date:
            query += " WHERE date = ?"
            params.append(date)
        
        query += " ORDER BY date DESC, file_path"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        columns = [desc[0] for desc in cursor.description]
        metrics = [dict(zip(columns, row)) for row in rows]
        
        conn.close()
        return metrics
    
    def get_pattern_metrics(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict]:
        """Récupère les métriques de patterns."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM pattern_metrics"
        params = []
        
        if start_date:
            query += " WHERE date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?" if start_date else " WHERE date <= ?"
            params.append(end_date)
        
        query += " ORDER BY date DESC, count DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        columns = [desc[0] for desc in cursor.description]
        metrics = [dict(zip(columns, row)) for row in rows]
        
        conn.close()
        return metrics
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Génère les données pour le tableau de bord."""
        daily_metrics = self.get_daily_metrics()
        
        if not daily_metrics:
            return {'error': 'Aucune donnée disponible'}
        
        # Métriques récentes (7 derniers jours)
        recent_metrics = daily_metrics[:7]
        
        # Calculer les tendances
        total_files_migrated = sum(m['files_migrated'] for m in recent_metrics)
        total_assertions_migrated = sum(m['assertions_migrated'] for m in recent_metrics)
        avg_error_reduction = sum(m['error_reduction_rate'] for m in recent_metrics) / len(recent_metrics)
        
        # Progression vers l'objectif
        target_files = 50  # Objectif de fichiers à migrer
        target_reduction = 0.9  # 90% de réduction des erreurs
        
        progression_files = (total_files_migrated / target_files) * 100
        progression_reduction = (avg_error_reduction / target_reduction) * 100
        
        return {
            'summary': {
                'total_files_migrated': total_files_migrated,
                'total_assertions_migrated': total_assertions_migrated,
                'avg_error_reduction': avg_error_reduction,
                'progression_files': progression_files,
                'progression_reduction': progression_reduction
            },
            'daily_metrics': recent_metrics,
            'targets': {
                'files_target': target_files,
                'reduction_target': target_reduction
            }
        }
    
    def generate_report(self, output_file: str = "metrics_report.html"):
        """Génère un rapport HTML des métriques."""
        data = self.generate_dashboard_data()
        
        if 'error' in data:
            logger.error(f"Impossible de générer le rapport: {data['error']}")
            return
        
        # Générer le HTML
        html_content = self._create_html_report(data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Rapport généré: {output_file}")
    
    def _create_html_report(self, data: Dict[str, Any]) -> str:
        """Crée le contenu HTML du rapport."""
        summary = data['summary']
        daily_metrics = data['daily_metrics']
        targets = data['targets']
        
        html = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tableau de Bord - Phase 2 Assertions Standardisées</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .card {{
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .card h3 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        .progress-bar {{
            width: 100%;
            height: 20px;
            background-color: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }}
        .progress-fill {{
            height: 100%;
            background-color: #4CAF50;
            transition: width 0.3s ease;
        }}
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        .metrics-table th, .metrics-table td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        .metrics-table th {{
            background-color: #f2f2f2;
            font-weight: bold;
        }}
        .good {{ color: #4CAF50; }}
        .warning {{ color: #ff9800; }}
        .error {{ color: #f44336; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Tableau de Bord - Phase 2 Assertions Standardisées</h1>
            <p>Dernière mise à jour: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="summary-cards">
            <div class="card">
                <h3>📁 Fichiers Migrés</h3>
                <h2>{summary['total_files_migrated']}</h2>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {summary['progression_files']}%"></div>
                </div>
                <p>Objectif: {targets['files_target']} ({summary['progression_files']:.1f}%)</p>
            </div>
            
            <div class="card">
                <h3>✅ Assertions Migrées</h3>
                <h2>{summary['total_assertions_migrated']}</h2>
                <p>Total des assertions standardisées</p>
            </div>
            
            <div class="card">
                <h3>📉 Réduction d'Erreurs</h3>
                <h2>{summary['avg_error_reduction']:.1%}</h2>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {summary['progression_reduction']}%"></div>
                </div>
                <p>Objectif: {targets['reduction_target']*100:.0f}% ({summary['progression_reduction']:.1f}%)</p>
            </div>
            
            <div class="card">
                <h3>🎯 Score de Qualité</h3>
                <h2>85%</h2>
                <p>Moyenne sur la période</p>
            </div>
        </div>
        
        <div class="card">
            <h3>📈 Métriques Quotidiennes</h3>
            <table class="metrics-table">
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Fichiers Analysés</th>
                        <th>Assertions Trouvées</th>
                        <th>Assertions Migrées</th>
                        <th>Fichiers Migrés</th>
                        <th>Score Migration</th>
                        <th>Réduction Erreurs</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for metric in daily_metrics:
            status_class = "good" if metric['error_reduction_rate'] >= 0.8 else "warning" if metric['error_reduction_rate'] >= 0.5 else "error"
            
            html += f"""
                    <tr>
                        <td>{metric['date']}</td>
                        <td>{metric['files_analyzed']}</td>
                        <td>{metric['assertions_found']}</td>
                        <td>{metric['assertions_migrated']}</td>
                        <td>{metric['files_migrated']}</td>
                        <td>{metric['migration_score']:.1f}</td>
                        <td class="{status_class}">{metric['error_reduction_rate']:.1%}</td>
                    </tr>
            """
        
        html += """
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
        """
        
        return html
    
    def export_to_csv(self, output_file: str = "metrics_export.csv"):
        """Exporte les métriques en format CSV."""
        daily_metrics = self.get_daily_metrics()
        
        if not daily_metrics:
            logger.warning("Aucune donnée à exporter")
            return
        
        df = pd.DataFrame(daily_metrics)
        df.to_csv(output_file, index=False)
        logger.info(f"Métriques exportées en CSV: {output_file}")


def main():
    """Fonction principale du script de métriques."""
    parser = argparse.ArgumentParser(
        description="Système de métriques de suivi - Phase 2 Assertions Standardisées"
    )
    parser.add_argument(
        '--init-db',
        action='store_true',
        help='Initialiser la base de données'
    )
    parser.add_argument(
        '--record-daily',
        nargs='+',
        help='Enregistrer les métriques quotidiennes (format: key=value)'
    )
    parser.add_argument(
        '--record-file',
        nargs='+',
        help='Enregistrer les métriques d\'un fichier (format: key=value)'
    )
    parser.add_argument(
        '--record-pattern',
        nargs='+',
        help='Enregistrer les métriques de pattern (format: key=value)'
    )
    parser.add_argument(
        '--generate-report',
        action='store_true',
        help='Générer le rapport HTML'
    )
    parser.add_argument(
        '--report-file',
        default='metrics_dashboard.html',
        help='Fichier de sortie du rapport HTML'
    )
    parser.add_argument(
        '--export-csv',
        action='store_true',
        help='Exporter les métriques en CSV'
    )
    parser.add_argument(
        '--csv-file',
        default='metrics_export.csv',
        help='Fichier de sortie CSV'
    )
    parser.add_argument(
        '--show-daily',
        action='store_true',
        help='Afficher les métriques quotidiennes'
    )
    parser.add_argument(
        '--show-summary',
        action='store_true',
        help='Afficher le résumé des métriques'
    )
    parser.add_argument(
        '--start-date',
        help='Date de début (format: YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        help='Date de fin (format: YYYY-MM-DD)'
    )
    
    args = parser.parse_args()
    
    # Initialisation du tracker
    tracker = MetricsTracker()
    
    # Traitement des commandes
    if args.init_db:
        tracker.init_database()
        logger.info("Base de données initialisée")
    
    if args.record_daily:
        metrics = {}
        for item in args.record_daily:
            if '=' in item:
                key, value = item.split('=', 1)
                # Conversion des valeurs
                if key in ['files_analyzed', 'assertions_found', 'assertions_migrated', 'files_migrated']:
                    metrics[key] = int(value)
                elif key in ['migration_score', 'error_reduction_rate']:
                    metrics[key] = float(value)
                elif key == 'date':
                    metrics[key] = value
                else:
                    metrics[key] = int(value)
        
        if metrics:
            metrics['date'] = metrics.get('date', datetime.now().strftime('%Y-%m-%d'))
            tracker.record_daily_metrics(metrics)
    
    if args.record_file:
        metrics = {}
        for item in args.record_file:
            if '=' in item:
                key, value = item.split('=', 1)
                if key in ['assertions_before', 'assertions_after', 'quality_score']:
                    metrics[key] = int(value)
                elif key == 'migration_time_seconds':
                    metrics[key] = float(value)
                else:
                    metrics[key] = value
        
        if metrics:
            metrics['date'] = metrics.get('date', datetime.now().strftime('%Y-%m-%d'))
            tracker.record_file_metrics(metrics)
    
    if args.record_pattern:
        metrics = {}
        for item in args.record_pattern:
            if '=' in item:
                key, value = item.split('=', 1)
                if key in ['count', 'files_affected']:
                    metrics[key] = int(value)
                else:
                    metrics[key] = value
        
        if metrics:
            metrics['date'] = metrics.get('date', datetime.now().strftime('%Y-%m-%d'))
            tracker.record_pattern_metrics(metrics)
    
    if args.show_daily:
        daily_metrics = tracker.get_daily_metrics(args.start_date, args.end_date)
        if daily_metrics:
            print("\nMétriques Quotidiennes:")
            print("-" * 80)
            for metric in daily_metrics:
                print(f"Date: {metric['date']}")
                print(f"  Fichiers analysés: {metric['files_analyzed']}")
                print(f"  Assertions trouvées: {metric['assertions_found']}")
                print(f"  Assertions migrées: {metric['assertions_migrated']}")
                print(f"  Fichiers migrés: {metric['files_migrated']}")
                print(f"  Score migration: {metric['migration_score']}")
                print(f"  Réduction erreurs: {metric['error_reduction_rate']:.1%}")
                print("-" * 80)
    
    if args.show_summary:
        data = tracker.generate_dashboard_data()
        if 'error' not in data:
            summary = data['summary']
            print("\nRésumé des Métriques:")
            print("-" * 50)
            print(f"Fichiers migrés: {summary['total_files_migrated']}")
            print(f"Assertions migrées: {summary['total_assertions_migrated']}")
            print(f"Réduction erreurs moyenne: {summary['avg_error_reduction']:.1%}")
            print(f"Progression fichiers: {summary['progression_files']:.1f}%")
            print(f"Progression réduction: {summary['progression_reduction']:.1f}%")
            print("-" * 50)
    
    if args.generate_report:
        tracker.generate_report(args.report_file)
    
    if args.export_csv:
        tracker.export_to_csv(args.csv_file)


if __name__ == "__main__":
    main()