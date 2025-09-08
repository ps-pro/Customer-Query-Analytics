# database/init_db.py
import sqlite3
import pandas as pd
import hashlib
from datetime import datetime
import json
import os
from collections import Counter

class DatabaseInitializer:
    """Initialize SQLite database with proper schema for HITL system."""
    
    def __init__(self, db_path='database/hitl_system.db'):
        """Initialize database connection."""
        self.db_path = db_path
        self.ensure_directory()
        
    def ensure_directory(self):
        """Ensure database directory exists."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
    def create_schema(self):
        """Create all necessary tables for HITL system."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Uncertain cases queue
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS uncertain_cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text_hash TEXT UNIQUE NOT NULL,
                text TEXT NOT NULL,
                rule_prediction TEXT,
                rule_confidence REAL,
                fuzzy_prediction TEXT,
                fuzzy_confidence REAL,
                uncertainty_score REAL,
                disagreement BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'pending',
                priority_score REAL
            )
        ''')
        
        # Human annotations
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS human_annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text_hash TEXT NOT NULL,
                text TEXT NOT NULL,
                original_rule_pred TEXT,
                original_fuzzy_pred TEXT,
                human_label TEXT NOT NULL,
                confidence_rating INTEGER CHECK(confidence_rating >= 1 AND confidence_rating <= 5),
                annotation_time_seconds REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                used_for_training BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Model snapshots
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_name TEXT NOT NULL,
                rules_json TEXT NOT NULL,
                examples_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                performance_metrics TEXT,
                trigger_reason TEXT
            )
        ''')
        
        # Performance metrics tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_type TEXT NOT NULL,
                classifier_type TEXT NOT NULL,
                accuracy REAL,
                precision_macro REAL,
                recall_macro REAL,
                f1_macro REAL,
                total_predictions INTEGER,
                correct_predictions INTEGER,
                annotation_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Active learning log
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS active_learning_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text_hash TEXT NOT NULL,
                selection_reason TEXT,
                uncertainty_score REAL,
                was_annotated BOOLEAN DEFAULT FALSE,
                human_agreed BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Rules history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rules_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_label TEXT NOT NULL,
                rule_expression TEXT NOT NULL,
                weight REAL NOT NULL,
                description TEXT,
                action_type TEXT NOT NULL, -- 'added', 'modified', 'deleted'
                source TEXT DEFAULT 'manual', -- 'manual', 'auto_generated', 'human_feedback'
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Examples history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS examples_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT NOT NULL,
                example_text TEXT NOT NULL,
                action_type TEXT NOT NULL, -- 'added', 'modified', 'deleted'
                source TEXT DEFAULT 'manual', -- 'manual', 'human_feedback'
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Prediction log
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prediction_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text_hash TEXT NOT NULL,
                text TEXT NOT NULL,
                rule_prediction TEXT,
                rule_confidence REAL,
                fuzzy_prediction TEXT,
                fuzzy_confidence REAL,
                final_prediction TEXT,
                prediction_method TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # System configuration
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_key TEXT UNIQUE NOT NULL,
                config_value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_uncertain_cases_status ON uncertain_cases(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_uncertain_cases_priority ON uncertain_cases(priority_score DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_human_annotations_hash ON human_annotations(text_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_prediction_log_hash ON prediction_log(text_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_performance_metrics_type ON performance_metrics(metric_type, classifier_type)')
        
        conn.commit()
        conn.close()
        print("[INFO] Database schema created successfully")
        
    def migrate_csv_data(self, csv_path='data/data.csv'):
        """Migrate existing CSV data to SQLite database."""
        if not os.path.exists(csv_path):
            print(f"[WARNING] CSV file not found at {csv_path}")
            return
            
        print(f"[INFO] Migrating data from {csv_path}")
        df = pd.read_csv(csv_path)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calculate human consensus for each document
        consensus_data = {}
        for doc_id in df['id'].unique():
            doc_annotations = df[df['id'] == doc_id]
            if len(doc_annotations) > 0:
                text = doc_annotations['text'].iloc[0]
                labels = doc_annotations['full_label'].tolist()
                
                label_counts = Counter(labels)
                majority_label = label_counts.most_common(1)[0][0]
                confidence = label_counts.most_common(1)[0][1] / len(labels)
                
                # Create text hash
                text_hash = hashlib.md5(text.encode()).hexdigest()
                
                consensus_data[doc_id] = {
                    'text': text,
                    'text_hash': text_hash,
                    'label': majority_label,
                    'confidence': confidence,
                    'n_annotators': len(labels)
                }
        
        # Insert consensus data as initial human annotations
        for doc_id, data in consensus_data.items():
            cursor.execute('''
                INSERT OR REPLACE INTO human_annotations 
                (text_hash, text, human_label, confidence_rating, used_for_training)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                data['text_hash'],
                data['text'],
                data['label'],
                min(5, max(1, int(data['confidence'] * 5))),  # Convert to 1-5 scale
                True
            ))
        
        # Initialize default system configuration
        default_configs = [
            ('uncertainty_threshold', '0.6'),
            ('batch_size', '15'),
            ('auto_update_frequency', '5'),
            ('similarity_method', 'character'),
            ('enable_auto_updates', 'true'),
            ('min_confidence_for_auto_rule', '0.8')
        ]
        
        for key, value in default_configs:
            cursor.execute('''
                INSERT OR REPLACE INTO system_config (config_key, config_value)
                VALUES (?, ?)
            ''', (key, value))
        
        conn.commit()
        conn.close()
        
        print(f"[INFO] Migrated {len(consensus_data)} consensus annotations to database")
        
    def initialize_default_rules_and_examples(self):
        """Initialize default rules and examples in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Default rules
        default_rules = {
            "Account Management_Password Reset": {
                "rule": "(password AND (reset OR forgot OR change)) OR (login AND (problem OR issue OR trouble))",
                "weight": 1.0,
                "description": "Password reset related queries"
            },
            "Account Management_Update Personal Info": {
                "rule": "(update OR change OR modify) AND (profile OR personal OR info OR information OR details)",
                "weight": 1.0,
                "description": "Profile update requests"
            },
            "Account Management_Close Account": {
                "rule": "(close OR delete OR cancel OR deactivate OR remove) AND account",
                "weight": 1.0,
                "description": "Account closure requests"
            },
            "Technical Issue_Login Issue": {
                "rule": "(login OR signin OR access) AND (issue OR problem OR trouble OR error OR fail)",
                "weight": 1.0,
                "description": "Login related problems"
            },
            "Technical Issue_Feature Bug": {
                "rule": "(bug OR error OR broken OR fail) AND NOT (login OR password)",
                "weight": 1.0,
                "description": "Feature functionality bugs"
            },
            "Technical Issue_Performance Issue": {
                "rule": "(slow OR loading OR performance OR timeout OR lag) OR (takes AND (long OR time))",
                "weight": 1.0,
                "description": "Performance related issues"
            },
            "Billing_Refund Request": {
                "rule": "(refund OR return) AND (money OR payment OR charge)",
                "weight": 1.0,
                "description": "Refund requests"
            },
            "Billing_Unrecognized Charge": {
                "rule": "(charge OR billing OR payment) AND (unknown OR unrecognized OR unauthorized OR wrong)",
                "weight": 1.0,
                "description": "Disputed charges"
            },
            "Billing_Invoice Inquiry": {
                "rule": "(invoice OR bill OR receipt OR statement) AND (question OR inquiry OR need OR want)",
                "weight": 1.0,
                "description": "Invoice related questions"
            }
        }
        
        for label, rule_data in default_rules.items():
            cursor.execute('''
                INSERT INTO rules_history 
                (rule_label, rule_expression, weight, description, action_type, source)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                label,
                rule_data['rule'],
                rule_data['weight'],
                rule_data['description'],
                'added',
                'system_default'
            ))
        
        # Default examples
        default_examples = {
            "Account Management_Password Reset": [
                "I forgot my password and need to reset it",
                "Can you help me change my password please",
                "Password reset link not working",
                "I want to update my login credentials"
            ],
            "Account Management_Update Personal Info": [
                "I need to update my profile information",
                "How do I change my personal details",
                "Update my email address in my account",
                "Modify my contact information"
            ],
            "Account Management_Close Account": [
                "I want to delete my account permanently",
                "How can I close my account",
                "Cancel my subscription and remove account",
                "Deactivate my profile please"
            ],
            "Technical Issue_Login Issue": [
                "I cannot log into my account",
                "Login page is not working",
                "Authentication failed when signing in",
                "Access denied error message"
            ],
            "Technical Issue_Feature Bug": [
                "The search function is broken",
                "Button not working properly",
                "Feature showing error message",
                "Application crashed when using tool"
            ],
            "Technical Issue_Performance Issue": [
                "The application is very slow",
                "Pages take too long to load",
                "Performance is laggy and unresponsive",
                "Timeout errors when processing"
            ],
            "Billing_Refund Request": [
                "I want my money back for this charge",
                "Please process a refund for my payment",
                "Return the funds to my account",
                "I need a refund for incorrect billing"
            ],
            "Billing_Unrecognized Charge": [
                "I see a charge I don't recognize",
                "Unknown payment on my statement",
                "Unauthorized billing on my account",
                "Wrong amount charged to my card"
            ],
            "Billing_Invoice Inquiry": [
                "I have questions about my invoice",
                "Need to see my billing statement",
                "Where can I find my receipt",
                "Invoice shows incorrect information"
            ]
        }
        
        for label, examples in default_examples.items():
            for example in examples:
                cursor.execute('''
                    INSERT INTO examples_history 
                    (label, example_text, action_type, source)
                    VALUES (?, ?, ?, ?)
                ''', (label, example, 'added', 'system_default'))
        
        conn.commit()
        conn.close()
        
        print("[INFO] Default rules and examples initialized")
        
    def full_initialization(self, csv_path='data/data.csv'):
        """Complete database initialization process."""
        print("[INFO] Starting database initialization...")
        
        self.create_schema()
        self.initialize_default_rules_and_examples()
        self.migrate_csv_data(csv_path)
        
        print("[INFO] Database initialization completed successfully!")
        print(f"[INFO] Database location: {os.path.abspath(self.db_path)}")

def main():
    """Initialize database if run directly."""
    initializer = DatabaseInitializer()
    initializer.full_initialization()

if __name__ == '__main__':
    main()