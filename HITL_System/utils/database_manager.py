# utils/database_manager.py
import sqlite3
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd

class DatabaseManager:
    """Manage all database operations for HITL system."""
    
    def __init__(self, db_path='database/hitl_system.db'):
        """Initialize database manager."""
        self.db_path = db_path
        
    def get_connection(self):
        """Get database connection with proper settings."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn
    
    # =======================================================================
    # UNCERTAIN CASES MANAGEMENT
    # =======================================================================
    
    def add_uncertain_case(self, text: str, rule_pred: str, rule_conf: float, 
                          fuzzy_pred: str, fuzzy_conf: float, uncertainty_score: float,
                          disagreement: bool = False) -> str:
        """Add new uncertain case to queue."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Calculate priority score (higher = more urgent)
        priority_score = uncertainty_score
        if disagreement:
            priority_score += 0.2  # Boost priority for disagreement cases
        
        cursor.execute('''
            INSERT OR REPLACE INTO uncertain_cases 
            (text_hash, text, rule_prediction, rule_confidence, fuzzy_prediction, 
             fuzzy_confidence, uncertainty_score, disagreement, priority_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (text_hash, text, rule_pred, rule_conf, fuzzy_pred, fuzzy_conf, 
              uncertainty_score, disagreement, priority_score))
        
        conn.commit()
        conn.close()
        return text_hash
    
    def get_uncertain_cases_batch(self, batch_size: int = 15) -> List[Dict]:
        """Get batch of highest priority uncertain cases."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM uncertain_cases 
            WHERE status = 'pending'
            ORDER BY priority_score DESC, created_at ASC
            LIMIT ?
        ''', (batch_size,))
        
        cases = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return cases
    
    def mark_case_annotated(self, text_hash: str):
        """Mark uncertain case as annotated."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE uncertain_cases 
            SET status = 'annotated', updated_at = CURRENT_TIMESTAMP
            WHERE text_hash = ?
        ''', (text_hash,))
        
        conn.commit()
        conn.close()
    
    def get_uncertain_cases_count(self) -> int:
        """Get count of pending uncertain cases."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM uncertain_cases WHERE status = "pending"')
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    # =======================================================================
    # HUMAN ANNOTATIONS MANAGEMENT
    # =======================================================================
    
    def add_human_annotation(self, text_hash: str, text: str, 
                           original_rule_pred: str, original_fuzzy_pred: str,
                           human_label: str, confidence_rating: int,
                           annotation_time: float = 0.0) -> int:
        """Add human annotation to database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO human_annotations 
            (text_hash, text, original_rule_pred, original_fuzzy_pred, 
             human_label, confidence_rating, annotation_time_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (text_hash, text, original_rule_pred, original_fuzzy_pred,
              human_label, confidence_rating, annotation_time))
        
        annotation_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Mark corresponding uncertain case as annotated
        self.mark_case_annotated(text_hash)
        
        return annotation_id
    
    def get_recent_annotations(self, limit: int = 50) -> List[Dict]:
        """Get recent human annotations."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM human_annotations 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (limit,))
        
        annotations = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return annotations
    
    def get_annotations_for_training(self) -> List[Dict]:
        """Get all annotations that can be used for training."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM human_annotations 
            WHERE used_for_training = TRUE
            ORDER BY created_at DESC
        ''')
        
        annotations = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return annotations
    
    def get_annotation_count(self) -> int:
        """Get total count of human annotations."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM human_annotations')
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    # =======================================================================
    # MODEL SNAPSHOTS MANAGEMENT
    # =======================================================================
    
    def save_model_snapshot(self, snapshot_name: str, rules: Dict, examples: Dict,
                          performance_metrics: Dict = None, trigger_reason: str = None) -> int:
        """Save model snapshot to database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_snapshots 
            (snapshot_name, rules_json, examples_json, performance_metrics, trigger_reason)
            VALUES (?, ?, ?, ?, ?)
        ''', (snapshot_name, json.dumps(rules), json.dumps(examples),
              json.dumps(performance_metrics) if performance_metrics else None,
              trigger_reason))
        
        snapshot_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return snapshot_id
    
    def get_latest_model_snapshot(self) -> Optional[Dict]:
        """Get the most recent model snapshot."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM model_snapshots 
            ORDER BY created_at DESC 
            LIMIT 1
        ''')
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            snapshot = dict(row)
            snapshot['rules_json'] = json.loads(snapshot['rules_json'])
            snapshot['examples_json'] = json.loads(snapshot['examples_json'])
            if snapshot['performance_metrics']:
                snapshot['performance_metrics'] = json.loads(snapshot['performance_metrics'])
            return snapshot
        return None
    
    def get_model_snapshots_history(self, limit: int = 20) -> List[Dict]:
        """Get model snapshots history."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, snapshot_name, created_at, trigger_reason,
                   performance_metrics
            FROM model_snapshots 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (limit,))
        
        snapshots = []
        for row in cursor.fetchall():
            snapshot = dict(row)
            if snapshot['performance_metrics']:
                snapshot['performance_metrics'] = json.loads(snapshot['performance_metrics'])
            snapshots.append(snapshot)
        
        conn.close()
        return snapshots
    
    # =======================================================================
    # PERFORMANCE METRICS MANAGEMENT
    # =======================================================================
    
    def save_performance_metrics(self, metric_type: str, classifier_type: str,
                               accuracy: float, precision: float, recall: float,
                               f1_score: float, total_predictions: int,
                               correct_predictions: int, annotation_count: int) -> int:
        """Save performance metrics to database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO performance_metrics 
            (metric_type, classifier_type, accuracy, precision_macro, recall_macro,
             f1_macro, total_predictions, correct_predictions, annotation_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (metric_type, classifier_type, accuracy, precision, recall,
              f1_score, total_predictions, correct_predictions, annotation_count))
        
        metric_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return metric_id
    
    def get_performance_history(self, days: int = 30) -> List[Dict]:
        """Get performance metrics history."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        since_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute('''
            SELECT * FROM performance_metrics 
            WHERE created_at >= ?
            ORDER BY created_at ASC
        ''', (since_date,))
        
        metrics = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return metrics
    
    def get_latest_performance(self, classifier_type: str) -> Optional[Dict]:
        """Get latest performance metrics for classifier."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM performance_metrics 
            WHERE classifier_type = ?
            ORDER BY created_at DESC 
            LIMIT 1
        ''', (classifier_type,))
        
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None
    
    # =======================================================================
    # RULES AND EXAMPLES MANAGEMENT
    # =======================================================================
    
    def get_current_rules(self) -> Dict:
        """Get current active rules."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Get latest version of each rule
        cursor.execute('''
            SELECT DISTINCT rule_label,
                   FIRST_VALUE(rule_expression) OVER (PARTITION BY rule_label ORDER BY created_at DESC) as rule_expression,
                   FIRST_VALUE(weight) OVER (PARTITION BY rule_label ORDER BY created_at DESC) as weight,
                   FIRST_VALUE(description) OVER (PARTITION BY rule_label ORDER BY created_at DESC) as description,
                   FIRST_VALUE(action_type) OVER (PARTITION BY rule_label ORDER BY created_at DESC) as action_type
            FROM rules_history
            WHERE action_type != 'deleted'
        ''')
        
        rules = {}
        for row in cursor.fetchall():
            rules[row['rule_label']] = {
                'rule': row['rule_expression'],
                'weight': row['weight'],
                'description': row['description'],
                'keywords': []
            }
        
        conn.close()
        return rules
    
    def get_current_examples(self) -> Dict:
        """Get current active examples."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT label, example_text FROM examples_history
            WHERE action_type != 'deleted'
            ORDER BY created_at ASC
        ''')
        
        examples = {}
        for row in cursor.fetchall():
            if row['label'] not in examples:
                examples[row['label']] = []
            examples[row['label']].append(row['example_text'])
        
        conn.close()
        return examples
    
    def add_rule(self, rule_label: str, rule_expression: str, weight: float,
                description: str, source: str = 'manual') -> int:
        """Add new rule to database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO rules_history 
            (rule_label, rule_expression, weight, description, action_type, source)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (rule_label, rule_expression, weight, description, 'added', source))
        
        rule_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return rule_id
    
    def add_example(self, label: str, example_text: str, source: str = 'manual') -> int:
        """Add new example to database."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO examples_history 
            (label, example_text, action_type, source)
            VALUES (?, ?, ?, ?)
        ''', (label, example_text, 'added', source))
        
        example_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return example_id
    
    # =======================================================================
    # ACTIVE LEARNING LOG
    # =======================================================================
    
    def log_active_learning_selection(self, text_hash: str, selection_reason: str,
                                    uncertainty_score: float) -> int:
        """Log active learning case selection."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO active_learning_log 
            (text_hash, selection_reason, uncertainty_score)
            VALUES (?, ?, ?)
        ''', (text_hash, selection_reason, uncertainty_score))
        
        log_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return log_id
    
    def update_active_learning_result(self, text_hash: str, was_annotated: bool,
                                    human_agreed: bool = None):
        """Update active learning log with annotation result."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE active_learning_log 
            SET was_annotated = ?, human_agreed = ?
            WHERE text_hash = ? AND was_annotated = FALSE
        ''', (was_annotated, human_agreed, text_hash))
        
        conn.commit()
        conn.close()
    
    # =======================================================================
    # SYSTEM CONFIGURATION
    # =======================================================================
    
    def get_config(self, key: str, default_value: str = None) -> str:
        """Get system configuration value."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT config_value FROM system_config WHERE config_key = ?', (key,))
        row = cursor.fetchone()
        conn.close()
        
        return row['config_value'] if row else default_value
    
    def set_config(self, key: str, value: str):
        """Set system configuration value."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO system_config (config_key, config_value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        ''', (key, value))
        
        conn.commit()
        conn.close()
    
    def get_all_config(self) -> Dict[str, str]:
        """Get all system configuration."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT config_key, config_value FROM system_config')
        config = {row['config_key']: row['config_value'] for row in cursor.fetchall()}
        conn.close()
        return config
    
    # =======================================================================
    # ANALYTICS AND INSIGHTS
    # =======================================================================
    
    def get_annotation_stats(self) -> Dict:
        """Get annotation statistics."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Total annotations
        cursor.execute('SELECT COUNT(*) as total FROM human_annotations')
        total = cursor.fetchone()['total']
        
        # Annotations today
        today = datetime.now().strftime('%Y-%m-%d')
        cursor.execute('SELECT COUNT(*) as today FROM human_annotations WHERE DATE(created_at) = ?', (today,))
        today_count = cursor.fetchone()['today']
        
        # Average confidence
        cursor.execute('SELECT AVG(confidence_rating) as avg_conf FROM human_annotations')
        avg_confidence = cursor.fetchone()['avg_conf'] or 0
        
        # Label distribution
        cursor.execute('''
            SELECT human_label, COUNT(*) as count 
            FROM human_annotations 
            GROUP BY human_label 
            ORDER BY count DESC
        ''')
        label_dist = {row['human_label']: row['count'] for row in cursor.fetchall()}
        
        conn.close()
        return {
            'total_annotations': total,
            'annotations_today': today_count,
            'average_confidence': avg_confidence,
            'label_distribution': label_dist
        }
    
    def get_learning_effectiveness(self) -> Dict:
        """Analyze learning effectiveness."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Agreement rate between human and model predictions
        cursor.execute('''
            SELECT 
                AVG(CASE WHEN human_label = original_rule_pred THEN 1.0 ELSE 0.0 END) as rule_agreement,
                AVG(CASE WHEN human_label = original_fuzzy_pred THEN 1.0 ELSE 0.0 END) as fuzzy_agreement,
                COUNT(*) as total_cases
            FROM human_annotations
        ''')
        
        row = cursor.fetchone()
        
        # Cases where models disagreed but human provided clarity
        cursor.execute('''
            SELECT COUNT(*) as disagreement_resolved
            FROM human_annotations
            WHERE original_rule_pred != original_fuzzy_pred
        ''')
        
        disagreement_count = cursor.fetchone()['disagreement_resolved']
        
        conn.close()
        return {
            'rule_agreement_rate': row['rule_agreement'] or 0,
            'fuzzy_agreement_rate': row['fuzzy_agreement'] or 0,
            'total_evaluated_cases': row['total_cases'],
            'disagreement_cases_resolved': disagreement_count
        }
    
    # =======================================================================
    # CLEANUP AND MAINTENANCE
    # =======================================================================
    
    def cleanup_old_uncertain_cases(self, days: int = 7):
        """Remove old uncertain cases that were never annotated."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute('''
            DELETE FROM uncertain_cases 
            WHERE status = 'pending' AND created_at < ?
        ''', (cutoff_date,))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        return deleted_count
    
    def get_database_stats(self) -> Dict:
        """Get database statistics."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        tables = [
            'uncertain_cases', 'human_annotations', 'model_snapshots',
            'performance_metrics', 'active_learning_log', 'rules_history',
            'examples_history', 'prediction_log'
        ]
        
        stats = {}
        for table in tables:
            cursor.execute(f'SELECT COUNT(*) as count FROM {table}')
            stats[table] = cursor.fetchone()['count']
        
        conn.close()
        return stats