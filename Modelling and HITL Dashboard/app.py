# Dashboard 3: Modelling and HITL Dashboard
# Complete Human-in-the-Loop Baseline Model Demonstration - FIXED VERSION

# ============================================================================
# IMPORTS
# ============================================================================
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback, dash_table, State, ALL, MATCH
import dash_bootstrap_components as dbc
from collections import Counter, defaultdict
import re
import json
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CORE CLASSIFIERS
# ============================================================================

class BaselineRuleClassifier:
    """Rule-based classifier with boolean logic support."""

    def __init__(self):
        """Initialize with default rules."""
        self.rules = self._get_initial_rules()
        print(f"[DEBUG] BaselineRuleClassifier initialized with {len(self.rules)} rule categories")

    def _get_initial_rules(self):
        """Define initial hardcoded rules with boolean logic."""
        return {
            "Account Management_Password Reset": {
                "rule": "(password AND (reset OR forgot OR change)) OR (login AND (problem OR issue OR trouble))",
                "keywords": ["password", "reset", "forgot", "change", "login", "problem", "issue", "trouble"],
                "weight": 1.0,
                "description": "Password reset related queries"
            },
            "Account Management_Update Personal Info": {
                "rule": "(update OR change OR modify) AND (profile OR personal OR info OR information OR details)",
                "keywords": ["update", "change", "modify", "profile", "personal", "info", "information", "details"],
                "weight": 1.0,
                "description": "Profile update requests"
            },
            "Account Management_Close Account": {
                "rule": "(close OR delete OR cancel OR deactivate OR remove) AND account",
                "keywords": ["close", "delete", "cancel", "deactivate", "remove", "account"],
                "weight": 1.0,
                "description": "Account closure requests"
            },
            "Technical Issue_Login Issue": {
                "rule": "(login OR signin OR access) AND (issue OR problem OR trouble OR error OR fail)",
                "keywords": ["login", "signin", "access", "issue", "problem", "trouble", "error", "fail"],
                "weight": 1.0,
                "description": "Login related problems"
            },
            "Technical Issue_Feature Bug": {
                "rule": "(bug OR error OR broken OR fail) AND NOT (login OR password)",
                "keywords": ["bug", "error", "broken", "fail", "not working", "glitch"],
                "weight": 1.0,
                "description": "Feature functionality bugs"
            },
            "Technical Issue_Performance Issue": {
                "rule": "(slow OR loading OR performance OR timeout OR lag) OR (takes AND (long OR time))",
                "keywords": ["slow", "loading", "performance", "timeout", "lag", "takes", "long", "time"],
                "weight": 1.0,
                "description": "Performance related issues"
            },
            "Billing_Refund Request": {
                "rule": "(refund OR return) AND (money OR payment OR charge)",
                "keywords": ["refund", "return", "money", "payment", "charge", "back"],
                "weight": 1.0,
                "description": "Refund requests"
            },
            "Billing_Unrecognized Charge": {
                "rule": "(charge OR billing OR payment) AND (unknown OR unrecognized OR unauthorized OR wrong)",
                "keywords": ["charge", "billing", "payment", "unknown", "unrecognized", "unauthorized", "wrong"],
                "weight": 1.0,
                "description": "Disputed charges"
            },
            "Billing_Invoice Inquiry": {
                "rule": "(invoice OR bill OR receipt OR statement) AND (question OR inquiry OR need OR want)",
                "keywords": ["invoice", "bill", "receipt", "statement", "question", "inquiry", "need", "want"],
                "weight": 1.0,
                "description": "Invoice related questions"
            }
        }

    def _evaluate_boolean_rule(self, rule_expression, text_lower):
        """Evaluate boolean rule expression against text."""
        try:
            expression = rule_expression.replace(" AND ", " and ").replace(" OR ", " or ").replace(" NOT ", " not ")
            keywords = re.findall(r'\b[a-zA-Z]+\b', expression)
            
            eval_context = {}
            for keyword in keywords:
                if keyword.lower() not in ['and', 'or', 'not']:
                    eval_context[keyword] = keyword.lower() in text_lower

            eval_expression = expression
            for keyword, present in eval_context.items():
                eval_expression = eval_expression.replace(keyword, str(present))

            result = eval(eval_expression)
            return result, eval_context

        except Exception as e:
            print(f"[ERROR] Rule evaluation failed for '{rule_expression}': {str(e)}")
            return False, {}

    def predict_single(self, text):
        """Predict label for single text with confidence."""
        text_lower = text.lower()
        predictions = []

        for label, rule_data in self.rules.items():
            rule_expression = rule_data['rule']
            weight = rule_data['weight']

            matches, keyword_matches = self._evaluate_boolean_rule(rule_expression, text_lower)

            if matches:
                matched_keywords = sum(keyword_matches.values())
                total_keywords = len(keyword_matches)
                confidence = (matched_keywords / max(total_keywords, 1)) * weight

                predictions.append({
                    'label': label,
                    'confidence': confidence,
                    'matched_keywords': [k for k, v in keyword_matches.items() if v],
                    'rule_fired': rule_expression
                })

        if predictions:
            best_prediction = max(predictions, key=lambda x: x['confidence'])
            return best_prediction['label'], best_prediction['confidence'], best_prediction
        else:
            return "Unknown", 0.0, {}

    def predict(self, texts):
        """Predict labels for multiple texts."""
        print(f"[DEBUG] Rule-based classifier predicting {len(texts)} texts")

        predictions = []
        confidences = []
        details = []

        for text in texts:
            label, confidence, detail = self.predict_single(text)
            predictions.append(label)
            confidences.append(confidence)
            details.append(detail)

        return predictions, confidences, details


class FuzzyMatchingClassifier:
    """Fuzzy matching classifier with character and semantic similarity options."""

    def __init__(self, similarity_method='character'):
        """Initialize with similarity method."""
        self.similarity_method = similarity_method
        self.examples = self._get_initial_examples()
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english') if similarity_method == 'semantic' else None
        self.example_vectors = None
        print(f"[DEBUG] FuzzyMatchingClassifier initialized with {similarity_method} similarity")

    def _get_initial_examples(self):
        """Define initial training examples."""
        return {
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

    def _fit_semantic_vectors(self):
        """Fit TF-IDF vectorizer on examples for semantic similarity."""
        if self.similarity_method == 'semantic':
            all_examples = []
            for examples_list in self.examples.values():
                all_examples.extend(examples_list)

            self.vectorizer.fit(all_examples)
            self.example_vectors = {}
            for label, examples_list in self.examples.items():
                vectors = self.vectorizer.transform(examples_list)
                self.example_vectors[label] = vectors

            print(f"[DEBUG] Fitted semantic vectors for {len(all_examples)} examples")

    def _character_similarity(self, text1, text2):
        """Calculate character-level similarity using SequenceMatcher."""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def _semantic_similarity(self, text, label_examples):
        """Calculate semantic similarity using TF-IDF cosine similarity."""
        if self.vectorizer is None:
            return 0.0

        try:
            text_vector = self.vectorizer.transform([text])
            example_vectors = self.example_vectors[label_examples]
            similarities = cosine_similarity(text_vector, example_vectors).flatten()
            return similarities.max()
        except:
            return 0.0

    def predict_single(self, text):
        """Predict label for single text."""
        if self.similarity_method == 'semantic' and self.example_vectors is None:
            self._fit_semantic_vectors()

        best_label = "Unknown"
        best_similarity = 0.0
        best_match_details = {}

        for label, examples_list in self.examples.items():
            if self.similarity_method == 'character':
                similarities = [self._character_similarity(text, example) for example in examples_list]
                max_similarity = max(similarities) if similarities else 0.0
                best_example = examples_list[similarities.index(max_similarity)] if similarities else ""
            else:
                max_similarity = self._semantic_similarity(text, label)
                best_example_idx = 0
                if label in self.example_vectors:
                    text_vector = self.vectorizer.transform([text])
                    similarities = cosine_similarity(text_vector, self.example_vectors[label]).flatten()
                    best_example_idx = similarities.argmax()
                best_example = examples_list[best_example_idx] if examples_list else ""

            if max_similarity > best_similarity:
                best_similarity = max_similarity
                best_label = label
                best_match_details = {
                    'similarity': max_similarity,
                    'best_example': best_example,
                    'method': self.similarity_method
                }

        return best_label, best_similarity, best_match_details

    def predict(self, texts):
        """Predict labels for multiple texts."""
        print(f"[DEBUG] Fuzzy classifier predicting {len(texts)} texts with {self.similarity_method} similarity")

        predictions = []
        confidences = []
        details = []

        for text in texts:
            label, confidence, detail = self.predict_single(text)
            predictions.append(label)
            confidences.append(confidence)
            details.append(detail)

        return predictions, confidences, details

    def set_similarity_method(self, method):
        """Change similarity method."""
        self.similarity_method = method
        if method == 'semantic':
            if self.vectorizer is None:
                self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            self._fit_semantic_vectors()
        print(f"[DEBUG] Changed similarity method to {method}")


class HITLAnalyzer:
    """Analyzer for Human-in-the-Loop performance and improvements."""

    def __init__(self, agreement_df):
        """Initialize with human annotation data."""
        self.agreement_df = agreement_df
        self.human_consensus = self._calculate_human_consensus()
        print(f"[DEBUG] HITLAnalyzer initialized with {len(self.human_consensus)} consensus labels")

    def _calculate_human_consensus(self):
        """Calculate human consensus labels (majority vote)."""
        consensus = {}

        for doc_id in self.agreement_df['id'].unique():
            doc_annotations = self.agreement_df[self.agreement_df['id'] == doc_id]

            if len(doc_annotations) > 0:
                text = doc_annotations['text'].iloc[0]
                labels = doc_annotations['full_label'].tolist()

                label_counts = Counter(labels)
                majority_label = label_counts.most_common(1)[0][0]
                confidence = label_counts.most_common(1)[0][1] / len(labels)

                consensus[doc_id] = {
                    'text': text,
                    'label': majority_label,
                    'confidence': confidence,
                    'n_annotators': len(labels)
                }

        return consensus

    def evaluate_classifier_performance(self, classifier, classifier_name):
        """Evaluate classifier against human consensus."""
        print(f"[DEBUG] Evaluating {classifier_name} performance")

        texts = [data['text'] for data in self.human_consensus.values()]
        doc_ids = list(self.human_consensus.keys())
        true_labels = [data['label'] for data in self.human_consensus.values()]

        pred_labels, confidences, details = classifier.predict(texts)
        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted', zero_division=0)

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': pred_labels,
            'confidences': confidences,
            'true_labels': true_labels,
            'doc_ids': doc_ids,
            'details': details
        }

        print(f"[DEBUG] {classifier_name} accuracy: {accuracy:.3f}")
        return results

    def compare_classifiers(self, rule_classifier, fuzzy_classifier):
        """Compare performance of both classifiers."""
        print(f"[DEBUG] Comparing classifier performance")

        rule_results = self.evaluate_classifier_performance(rule_classifier, "Rule-based")
        fuzzy_results = self.evaluate_classifier_performance(fuzzy_classifier, "Fuzzy Matching")

        comparison = {
            'rule_based': rule_results,
            'fuzzy_matching': fuzzy_results
        }

        return comparison

    def identify_error_patterns(self, classifier_results, classifier_name):
        """Identify systematic error patterns."""
        print(f"[DEBUG] Identifying error patterns for {classifier_name}")

        true_labels = classifier_results['true_labels']
        pred_labels = classifier_results['predictions']
        doc_ids = classifier_results['doc_ids']

        disagreements = []
        for i, (true_label, pred_label, doc_id) in enumerate(zip(true_labels, pred_labels, doc_ids)):
            if true_label != pred_label:
                disagreements.append({
                    'doc_id': doc_id,
                    'text': self.human_consensus[doc_id]['text'],
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'confidence': classifier_results['confidences'][i],
                    'human_confidence': self.human_consensus[doc_id]['confidence']
                })

        error_patterns = Counter([(d['true_label'], d['predicted_label']) for d in disagreements])
        return disagreements, error_patterns

    def suggest_improvements(self, error_patterns, classifier_type):
        """Suggest improvements based on error patterns."""
        print(f"[DEBUG] Generating improvement suggestions for {classifier_type}")

        suggestions = []
        for (true_label, pred_label), count in error_patterns.most_common(5):
            if classifier_type == "rule_based":
                suggestion = {
                    'error_pattern': f"{true_label} → {pred_label}",
                    'frequency': count,
                    'suggestion': f"Add more specific rules to distinguish '{true_label}' from '{pred_label}'",
                    'action': "add_rule",
                    'priority': "high" if count > 2 else "medium"
                }
            else:
                suggestion = {
                    'error_pattern': f"{true_label} → {pred_label}",
                    'frequency': count,
                    'suggestion': f"Add more training examples for '{true_label}' to improve distinction from '{pred_label}'",
                    'action': "add_example",
                    'priority': "high" if count > 2 else "medium"
                }

            suggestions.append(suggestion)

        return suggestions

# ============================================================================
# DATA LOADING
# ============================================================================
print("[INFO] Loading agreement data from CSV...")
agreement_df = pd.read_csv('data.csv')
print(f"[DEBUG] Loaded {len(agreement_df)} annotations")

# ============================================================================
# COMPONENT INITIALIZATION
# ============================================================================
print("[INFO] Initializing HITL Demonstration Components...")

rule_classifier = BaselineRuleClassifier()
fuzzy_classifier = FuzzyMatchingClassifier(similarity_method='character')
hitl_analyzer = HITLAnalyzer(agreement_df)

# ============================================================================
# DASH APP SETUP
# ============================================================================
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server

# Enhanced CSS with purple gradients and proper theme handling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Light Theme */
            .theme-light {
                --bg-primary: #ffffff;
                --bg-secondary: #f8f9fa;
                --text-primary: #212529;
                --text-secondary: #6c757d;
                --border-color: #dee2e6;
                --card-bg: #ffffff;
                --shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
                --purple-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                --purple-gradient-hover: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
                --purple-shadow: rgba(102, 126, 234, 0.4);
            }
            
            /* Dark Theme */
            .theme-dark {
                --bg-primary: #1a1a1a;
                --bg-secondary: #2d2d2d;
                --text-primary: #ffffff;
                --text-secondary: #cbd5e0;
                --border-color: #4a5568;
                --card-bg: #2d2d2d;
                --shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.3);
                --purple-gradient: linear-gradient(135deg, #805ad5 0%, #9f7aea 100%);
                --purple-gradient-hover: linear-gradient(135deg, #6b46c1 0%, #8b5cf6 100%);
                --purple-shadow: rgba(139, 92, 246, 0.4);
            }
            
            /* Apply theme variables */
            body {
                background-color: var(--bg-primary) !important;
                color: var(--text-primary) !important;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
                transition: all 0.3s ease;
            }
            
            #main-container {
                background-color: var(--bg-primary);
                color: var(--text-primary);
            }
            
            #main-title {
                color: var(--text-primary) !important;
                font-size: 2.5rem !important;
            }
            
            #main-subtitle {
                color: var(--text-secondary) !important;
            }
            
            /* Purple Gradient Buttons */
            .btn-primary, .btn.btn-primary {
                background: var(--purple-gradient) !important;
                border: none !important;
                color: white !important;
                font-weight: 600 !important;
                border-radius: 12px !important;
                padding: 12px 24px !important;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2) !important;
                transition: all 0.3s ease !important;
                transform: translateY(0) !important;
            }

            .btn-primary:hover, .btn.btn-primary:hover {
                background: var(--purple-gradient-hover) !important;
                transform: translateY(-3px) !important;
                box-shadow: 0 8px 25px var(--purple-shadow) !important;
                color: white !important;
            }

            .btn-primary:active, .btn.btn-primary:active {
                transform: translateY(-1px) !important;
                box-shadow: 0 4px 15px var(--purple-shadow) !important;
            }

            .btn-success {
                background: linear-gradient(135deg, #48bb78 0%, #38a169 100%) !important;
                border: none !important;
                color: white !important;
                border-radius: 12px !important;
                transition: all 0.3s ease !important;
            }

            .btn-success:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 6px 20px rgba(72, 187, 120, 0.4) !important;
            }
            
            /* Card styling */
            .card {
                background-color: var(--card-bg) !important;
                border: 1px solid var(--border-color) !important;
                box-shadow: var(--shadow) !important;
                border-radius: 15px !important;
                transition: all 0.3s ease !important;
            }

            .card:hover {
                box-shadow: 0 12px 30px rgba(102, 126, 234, 0.15) !important;
                transform: translateY(-2px) !important;
            }
            
            .card-header {
                background: var(--purple-gradient) !important;
                border-bottom: none !important;
                color: white !important;
                font-weight: 600 !important;
                border-radius: 15px 15px 0 0 !important;
            }
            
            .card-body {
                color: var(--text-primary) !important;
                padding: 2rem !important;
            }

            /* Enhanced form controls */
            .Select-control, .dropdown {
                border-radius: 12px !important;
                border: 2px solid rgba(102, 126, 234, 0.2) !important;
                box-shadow: 0 2px 8px rgba(102, 126, 234, 0.1) !important;
                transition: all 0.3s ease !important;
                background-color: var(--card-bg) !important;
                color: var(--text-primary) !important;
            }

            .Select-control:hover, .dropdown:hover {
                border-color: rgba(102, 126, 234, 0.4) !important;
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2) !important;
            }

            /* Input styling */
            .form-control, .form-select, input, textarea {
                background-color: var(--card-bg) !important;
                border: 2px solid var(--border-color) !important;
                color: var(--text-primary) !important;
                border-radius: 12px !important;
                padding: 12px 16px !important;
                transition: all 0.3s ease !important;
            }
            
            .form-control:focus, .form-select:focus, input:focus, textarea:focus {
                background-color: var(--card-bg) !important;
                border-color: #667eea !important;
                color: var(--text-primary) !important;
                box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25) !important;
            }

            /* Text and label colors */
            label, .form-label, p, h1, h2, h3, h4, h5, h6, span, div {
                color: var(--text-primary) !important;
            }

            .text-muted, .text-secondary {
                color: var(--text-secondary) !important;
            }

            /* Tab styling */
            .nav-tabs {
                border-bottom: 2px solid var(--border-color) !important;
                background: transparent !important;
            }

            .nav-tabs .nav-link {
                color: var(--text-secondary) !important;
                border: none !important;
                background: transparent !important;
                font-weight: 600 !important;
                font-size: 1.1rem !important;
                padding: 1rem 2rem !important;
                border-radius: 12px 12px 0 0 !important;
                transition: all 0.3s ease !important;
            }

            .nav-tabs .nav-link:hover {
                background: var(--bg-secondary) !important;
                color: var(--text-primary) !important;
                transform: translateY(-2px) !important;
            }

            .nav-tabs .nav-link.active {
                background: var(--purple-gradient) !important;
                color: white !important;
                border: none !important;
                font-weight: 700 !important;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
            }

            /* Slider styling */
            .rc-slider-track {
                background: var(--purple-gradient) !important;
                height: 8px !important;
            }

            .rc-slider-handle {
                border: 3px solid #667eea !important;
                background: white !important;
                width: 24px !important;
                height: 24px !important;
                margin-top: -8px !important;
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
            }

            .rc-slider-rail {
                background: var(--border-color) !important;
                height: 8px !important;
            }

            /* Checkbox and radio styling */
            input[type="checkbox"], input[type="radio"] {
                accent-color: #667eea !important;
                width: 18px !important;
                height: 18px !important;
            }

            /* Plot containers - BIGGER SIZES */
            .js-plotly-plot {
                border-radius: 15px !important;
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15) !important;
                background: var(--card-bg) !important;
                min-height: 600px !important;
            }

            /* MATRIX PLOTS - EXTRA LARGE */
            .js-plotly-plot[data-title*="Matrix"],
            .js-plotly-plot[data-title*="Confusion"] {
                min-height: 800px !important;
            }
            
            /* Table styling */
            .dash-table-container {
                background-color: var(--card-bg) !important;
                border-radius: 15px !important;
                overflow: hidden !important;
                border: 1px solid var(--border-color) !important;
            }

            .dash-table-container .dash-spreadsheet-container {
                background-color: var(--card-bg) !important;
            }

            .dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner {
                background-color: var(--card-bg) !important;
            }

            .dash-table-container th {
                background: var(--purple-gradient) !important;
                color: white !important;
                font-weight: 600 !important;
            }

            .dash-table-container td {
                background-color: var(--card-bg) !important;
                color: var(--text-primary) !important;
                border-color: var(--border-color) !important;
            }
            
            /* Progress bar */
            .progress {
                background-color: var(--bg-secondary) !important;
                border-radius: 12px !important;
                height: 8px !important;
            }

            .progress-bar {
                background: var(--purple-gradient) !important;
                border-radius: 12px !important;
            }

            /* Alert styling */
            .alert {
                border-radius: 12px !important;
                border: none !important;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
            }

            .alert-info {
                background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%) !important;
                color: white !important;
            }

            .alert-danger {
                background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%) !important;
                color: white !important;
            }

            .alert-warning {
                background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%) !important;
                color: white !important;
            }

            .alert-success {
                background: linear-gradient(135deg, #48bb78 0%, #38a169 100%) !important;
                color: white !important;
            }

            /* Theme toggle buttons */
            .btn-outline-secondary {
                color: var(--text-secondary) !important;
                border: 2px solid var(--border-color) !important;
                background: transparent !important;
                border-radius: 8px !important;
                transition: all 0.3s ease !important;
            }
            
            .btn-outline-secondary:hover, .btn-outline-secondary.active {
                background: var(--purple-gradient) !important;
                color: white !important;
                border-color: transparent !important;
                transform: translateY(-1px) !important;
            }

            /* Responsive adjustments */
            @media (max-width: 768px) {
                .js-plotly-plot {
                    min-height: 400px !important;
                }
                
                .js-plotly-plot[data-title*="Matrix"],
                .js-plotly-plot[data-title*="Confusion"] {
                    min-height: 500px !important;
                }
            }
        </style>
    </head>
    <body class="theme-light">
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# ============================================================================
# APP LAYOUT - FIXED
# ============================================================================
app.layout = dbc.Container([
    # Data stores for persistence across tabs - FIXED: Added missing stores
    dcc.Store(id='theme-store', data='light'),
    dcc.Store(id='analysis-results-store'),
    dcc.Store(id='error-analysis-store'),
    
    # Header Section with Theme Toggle
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([
                    html.H1("Human-in-the-Loop Baseline Model Demonstration", 
                        id="main-title",
                        style={
                            'margin': 0, 
                            'fontWeight': '800', 
                            'letterSpacing': '-1px',
                            'fontSize': '3rem',
                            'textAlign': 'center',
                            'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                            'WebkitBackgroundClip': 'text',
                            'WebkitTextFillColor': 'transparent',
                            'backgroundClip': 'text'
                        }),
                    html.P("Advanced Rule-Based and Fuzzy Matching Classifiers with Interactive Training",
                        id="main-subtitle",
                        style={
                            'margin': 0, 
                            'opacity': 0.7, 
                            'fontSize': '1.2rem',
                            'textAlign': 'center',
                            'fontWeight': '400'
                        })
                ], style={'textAlign': 'center', 'flex': 1}),
                
                html.Div([
                    dbc.ButtonGroup([
                        dbc.Button("Light", id="light-theme-btn", size="sm", outline=True, color="secondary"),
                        dbc.Button("Dark", id="dark-theme-btn", size="sm", outline=True, color="secondary")
                    ], size="sm")
                ], style={'display': 'flex', 'alignItems': 'center'})
            ], style={
                'display': 'flex', 
                'justifyContent': 'space-between', 
                'alignItems': 'center',
                'padding': '2rem 0 1.5rem 0',
                'borderBottom': '1px solid #e9ecef'
            }) 
        ], width=12)
    ]),
    
    # Navigation Tabs - FIXED: Correct tab IDs
    dbc.Row([
        dbc.Col([
            html.Div([
               dbc.Tabs([
                    dbc.Tab(
                        label="Performance Overview", 
                        tab_id="performance-tab",
                        tab_style={'padding': '1rem 2rem', 'border': 'none'},
                        active_tab_style={'border': 'none', 'borderBottom': '3px solid #007bff'}
                    ),
                    dbc.Tab(
                        label="Error Analysis & Opportunities",
                        tab_id="error-tab",
                        tab_style={'padding': '1rem 2rem', 'border': 'none'},
                        active_tab_style={'border': 'none', 'borderBottom': '3px solid #007bff'}
                    ),
                    dbc.Tab(
                        label="HITL Rule Management (CRUD)", 
                        tab_id="crud-tab",
                        tab_style={'padding': '1rem 2rem', 'border': 'none'},
                        active_tab_style={'border': 'none', 'borderBottom': '3px solid #007bff'}
                    )
                ], 
                id="main-hitl-tabs",  # FIXED: Correct ID
                active_tab="performance-tab",  # FIXED: Correct active tab
                style={'borderBottom': '1px solid #dee2e6', 'marginBottom': '0'})
            ], style={'display': 'flex', 'justifyContent': 'center'})
        ])
    ], style={'marginTop': '1rem'}),
    
    # Main Content Area - Full Width
    dbc.Row([
        dbc.Col([
            html.Div(
                id="hitl-tab-content",
                style={
                    'minHeight': '80vh',
                    'padding': '2rem 0'
                }
            )
        ], width=12)
    ], style={'margin': '0'}),

], 
fluid=True, 
id="main-container",
style={
    'padding': '0 3rem',
    'maxWidth': '100%',
    'minHeight': '100vh'
})

# ============================================================================
# HELPER FUNCTIONS - ENHANCED WITH BIGGER PLOTS
# ============================================================================

def create_performance_overview_tab(analysis_data):
    """Create the performance overview tab."""

    config_section = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Fuzzy Matching Configuration", style={'color': 'white'})),
                dbc.CardBody([
                    html.Label("Similarity Method:", style={'fontWeight': '600', 'marginBottom': '1rem'}),
                    dbc.RadioItems(
                        id="similarity-method-radio",
                        options=[
                            {"label": "Character-Level Similarity", "value": "character"},
                            {"label": "Semantic Similarity (TF-IDF)", "value": "semantic"}
                        ],
                        value="character",
                        inline=True,
                        style={'fontSize': '1.1rem'}
                    )
                ])
            ])
        ], md=6),
        dbc.Col([
            dbc.Button("Analyze Performance", id="analyze-performance-btn",
                     color="primary", size="lg", className="mt-3",
                     style={'width': '100%', 'height': '60px', 'fontSize': '1.2rem'})
        ], md=6)
    ], className="mb-4")

    results_section = html.Div(id="performance-results-display")

    if analysis_data:
        results_section = create_performance_results_display(analysis_data)

    return html.Div([config_section, results_section])

def create_error_analysis_tab(error_data):
    """Create the error analysis tab."""

    if not error_data:
        return html.Div([
            html.H4("Error Analysis & Improvement Opportunities", 
                style={
                    'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    'WebkitBackgroundClip': 'text',
                    'WebkitTextFillColor': 'transparent',
                    'backgroundClip': 'text',
                    'fontWeight': '700',
                    'fontSize': '2.5rem',
                    'marginBottom': '2rem',
                    'textAlign': 'center'
                }),
            dbc.Alert([
                html.H5("No Analysis Data Available", className="alert-heading"),
                html.P("Please go to the Performance Overview tab and click 'Analyze Performance' first.", 
                      style={'marginBottom': '0'})
            ], color="info", className="mt-4")
        ])

    return html.Div([
        html.H4("Error Analysis & Improvement Opportunities", 
            style={
                'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                'WebkitBackgroundClip': 'text',
                'WebkitTextFillColor': 'transparent',
                'backgroundClip': 'text',
                'fontWeight': '700',
                'fontSize': '2.5rem',
                'marginBottom': '2rem',
                'textAlign': 'center'
            }),
        create_error_analysis_display(error_data)
    ])

def create_crud_management_tab():
    """Create the CRUD management tab."""
    return html.Div([
        html.H4("HITL Rule Management Interface", 
            style={
                'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                'WebkitBackgroundClip': 'text',
                'WebkitTextFillColor': 'transparent',
                'backgroundClip': 'text',
                'fontWeight': '700',
                'fontSize': '2.5rem',
                'marginBottom': '2rem',
                'textAlign': 'center'
            }),

        dbc.Tabs([
            dbc.Tab(label="Rule-Based Classifier Rules", tab_id="rule-crud-tab"),
            dbc.Tab(label="Fuzzy Matching Examples", tab_id="fuzzy-crud-tab"),
            dbc.Tab(label="Test Environment", tab_id="test-crud-tab")
        ], id="crud-sub-tabs", active_tab="rule-crud-tab"),

        html.Div(id="crud-sub-content", className="mt-4")
    ])

def create_performance_results_display(analysis_data):
    """Create performance results display with BIGGER plots."""
    try:
        comparison_results = analysis_data['comparison_results']

        # BIGGER Performance comparison chart
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        rule_values = [comparison_results['rule_based'][metric] for metric in metrics]
        fuzzy_values = [comparison_results['fuzzy_matching'][metric] for metric in metrics]

        performance_fig = go.Figure()
        performance_fig.add_trace(go.Bar(
            name='Rule-based Classifier',
            x=metrics,
            y=rule_values,
            text=[f'{val:.3f}' for val in rule_values],
            textposition='outside',
            marker_color='#667eea'
        ))
        performance_fig.add_trace(go.Bar(
            name='Fuzzy Matching Classifier',
            x=metrics,
            y=fuzzy_values,
            text=[f'{val:.3f}' for val in fuzzy_values],
            textposition='outside',
            marker_color='#764ba2'
        ))
        performance_fig.update_layout(
            title="Classifier Performance Comparison vs Human Consensus",
            title_font_size=20,
            xaxis_title="Metrics",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1]),
            barmode='group',
            height=600,  # BIGGER
            font=dict(size=14),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        # BIGGER Confidence distribution chart
        rule_conf = comparison_results['rule_based']['confidences']
        fuzzy_conf = comparison_results['fuzzy_matching']['confidences']

        confidence_fig = go.Figure()
        confidence_fig.add_trace(go.Histogram(
            x=rule_conf,
            name='Rule-based Confidence',
            opacity=0.7,
            nbinsx=20,
            marker_color='#667eea'
        ))
        confidence_fig.add_trace(go.Histogram(
            x=fuzzy_conf,
            name='Fuzzy Matching Confidence',
            opacity=0.7,
            nbinsx=20,
            marker_color='#764ba2'
        ))
        confidence_fig.update_layout(
            title="Confidence Score Distributions",
            title_font_size=20,
            xaxis_title="Confidence Score",
            yaxis_title="Frequency",
            barmode='overlay',
            height=600,  # BIGGER
            font=dict(size=14),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        return html.Div([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=performance_fig, style={'height': '650px'})  # BIGGER
                ], md=6),
                dbc.Col([
                    dcc.Graph(figure=confidence_fig, style={'height': '650px'})  # BIGGER
                ], md=6)
            ]),

            # Summary metrics cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H6("Rule-based Classifier", style={'color': 'white', 'fontWeight': '600'})),
                        dbc.CardBody([
                            html.P(f"Accuracy: {comparison_results['rule_based']['accuracy']:.1%}", 
                                  style={'fontSize': '1.2rem', 'fontWeight': '500'}),
                            html.P(f"F1-Score: {comparison_results['rule_based']['f1_score']:.3f}", 
                                  style={'fontSize': '1.2rem', 'fontWeight': '500'}),
                            html.P(f"Avg Confidence: {np.mean(rule_conf):.3f}", 
                                  style={'fontSize': '1.2rem', 'fontWeight': '500'})
                        ])
                    ])
                ], md=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H6("Fuzzy Matching Classifier", style={'color': 'white', 'fontWeight': '600'})),
                        dbc.CardBody([
                            html.P(f"Accuracy: {comparison_results['fuzzy_matching']['accuracy']:.1%}", 
                                  style={'fontSize': '1.2rem', 'fontWeight': '500'}),
                            html.P(f"F1-Score: {comparison_results['fuzzy_matching']['f1_score']:.3f}", 
                                  style={'fontSize': '1.2rem', 'fontWeight': '500'}),
                            html.P(f"Avg Confidence: {np.mean(fuzzy_conf):.3f}", 
                                  style={'fontSize': '1.2rem', 'fontWeight': '500'})
                        ])
                    ])
                ], md=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H6("Human Consensus", style={'color': 'white', 'fontWeight': '600'})),
                        dbc.CardBody([
                            html.P(f"Documents: {len(hitl_analyzer.human_consensus)}", 
                                  style={'fontSize': '1.2rem', 'fontWeight': '500'}),
                            html.P(f"Avg Confidence: {np.mean([d['confidence'] for d in hitl_analyzer.human_consensus.values()]):.1%}", 
                                  style={'fontSize': '1.2rem', 'fontWeight': '500'}),
                            html.P("Gold Standard Baseline", 
                                  style={'fontSize': '1.2rem', 'fontWeight': '500'})
                        ])
                    ])
                ], md=4)
            ], className="mt-4")
        ])

    except Exception as e:
        return dbc.Alert(f"Error displaying results: {str(e)}", color="danger")

def create_error_analysis_display(error_data):
    """Create error analysis display with BIGGER plots."""
    try:
        rule_disagreements = error_data['rule_disagreements']
        fuzzy_disagreements = error_data['fuzzy_disagreements']
        rule_suggestions = error_data['rule_suggestions']
        fuzzy_suggestions = error_data['fuzzy_suggestions']

        # BIGGER Error analysis chart
        rule_errors = Counter([d['true_label'] for d in rule_disagreements])
        fuzzy_errors = Counter([d['true_label'] for d in fuzzy_disagreements])

        all_labels = set(rule_errors.keys()) | set(fuzzy_errors.keys())
        rule_counts = [rule_errors.get(label, 0) for label in all_labels]
        fuzzy_counts = [fuzzy_errors.get(label, 0) for label in all_labels]

        error_fig = go.Figure()
        error_fig.add_trace(go.Bar(
            name='Rule-based Errors',
            x=list(all_labels),
            y=rule_counts,
            text=rule_counts,
            textposition='outside',
            marker_color='#667eea'
        ))
        error_fig.add_trace(go.Bar(
            name='Fuzzy Matching Errors',
            x=list(all_labels),
            y=fuzzy_counts,
            text=fuzzy_counts,
            textposition='outside',
            marker_color='#764ba2'
        ))
        error_fig.update_layout(
            title="Error Count by True Label Category",
            title_font_size=20,
            xaxis_title="True Label",
            yaxis_title="Number of Errors",
            barmode='group',
            height=700,  # BIGGER
            font=dict(size=14),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        return html.Div([
            dcc.Graph(figure=error_fig, style={'height': '750px'}),  # BIGGER

            dbc.Row([
                dbc.Col([
                    html.H5("Rule-based Improvement Suggestions", 
                           style={'fontWeight': '600', 'fontSize': '1.5rem', 'marginBottom': '1.5rem'}),
                    create_suggestions_display(rule_suggestions)
                ], md=6),
                dbc.Col([
                    html.H5("Fuzzy Matching Improvement Suggestions", 
                           style={'fontWeight': '600', 'fontSize': '1.5rem', 'marginBottom': '1.5rem'}),
                    create_suggestions_display(fuzzy_suggestions)
                ], md=6)
            ], className="mt-4")
        ])

    except Exception as e:
        return dbc.Alert(f"Error displaying error analysis: {str(e)}", color="danger")

def create_suggestions_display(suggestions):
    """Create display for improvement suggestions."""
    if not suggestions:
        return html.P("No specific suggestions available.", 
                     style={'fontSize': '1.1rem', 'fontStyle': 'italic'})

    suggestion_cards = []
    for suggestion in suggestions:
        color_map = {"high": "danger", "medium": "warning", "low": "info"}
        color = color_map[suggestion['priority']]

        card = dbc.Card([
            dbc.CardBody([
                html.H6(f"Error Pattern: {suggestion['error_pattern']}", 
                        className="card-title", 
                        style={'fontWeight': '600', 'fontSize': '1.2rem'}),
                html.P(suggestion['suggestion'], 
                      style={'fontSize': '1.1rem', 'marginBottom': '0.5rem'}),
                html.Small(f"Frequency: {suggestion['frequency']} errors", 
                          className="text-muted",
                          style={'fontSize': '1rem'})
            ])
        ], color=color, outline=True, className="mb-3")

        suggestion_cards.append(card)

    return html.Div(suggestion_cards)

def create_rule_crud_interface():
    """Create rule CRUD interface."""
    current_rules = []
    for label, rule_data in rule_classifier.rules.items():
        current_rules.append({
            'Label': label,
            'Rule Expression': rule_data['rule'],
            'Weight': rule_data['weight'],
            'Description': rule_data['description']
        })

    rules_table = dash_table.DataTable(
        data=current_rules,
        columns=[{"name": i, "id": i} for i in current_rules[0].keys()],
        style_cell={'textAlign': 'left', 'fontSize': 14, 'padding': '12px'},
        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold', 'fontSize': 16},
        style_data={'height': 'auto', 'lineHeight': '20px'},
        page_size=10
    )

    return html.Div([
        html.H5("Rule-Based Classifier Management", 
               style={'fontWeight': '600', 'fontSize': '1.8rem', 'marginBottom': '2rem'}),

        html.H6("Current Rules:", 
               style={'fontWeight': '600', 'fontSize': '1.4rem', 'marginBottom': '1rem'}),
        rules_table,

        html.Hr(style={'margin': '2rem 0'}),

        html.H6("Add New Rule:", 
               style={'fontWeight': '600', 'fontSize': '1.4rem', 'marginBottom': '1.5rem'}),
        dbc.Row([
            dbc.Col([
                html.Label("Label:", style={'fontWeight': '600', 'fontSize': '1.1rem'}),
                dcc.Input(id="new-rule-label", type="text", 
                         placeholder="e.g., Technical Issue_Bug Report",
                         style={'width': '100%'})
            ], md=6),
            dbc.Col([
                html.Label("Weight:", style={'fontWeight': '600', 'fontSize': '1.1rem'}),
                dcc.Input(id="new-rule-weight", type="number", value=1.0, step=0.1,
                         style={'width': '100%'})
            ], md=6)
        ]),
        dbc.Row([
            dbc.Col([
                html.Label("Boolean Rule Expression:", style={'fontWeight': '600', 'fontSize': '1.1rem'}),
                dcc.Input(id="new-rule-expression", type="text",
                         placeholder="e.g., (bug OR error) AND NOT login",
                         style={'width': '100%'})
            ], md=12)
        ], className="mt-3"),
        dbc.Row([
            dbc.Col([
                html.Label("Description:", style={'fontWeight': '600', 'fontSize': '1.1rem'}),
                dcc.Input(id="new-rule-description", type="text",
                         placeholder="Brief description of this rule",
                         style={'width': '100%'})
            ], md=8),
            dbc.Col([
                dbc.Button("Add Rule", id="add-rule-btn", color="success", 
                          className="mt-4", style={'width': '100%'})
            ], md=4)
        ], className="mt-3"),

        html.Div(id="rule-crud-feedback", className="mt-3")
    ])

def create_fuzzy_crud_interface():
    """Create fuzzy matching CRUD interface."""
    examples_data = []
    for label, examples_list in fuzzy_classifier.examples.items():
        for example in examples_list:
            examples_data.append({
                'Label': label,
                'Example Text': example[:100] + "..." if len(example) > 100 else example
            })

    examples_table = dash_table.DataTable(
        data=examples_data,
        columns=[{"name": i, "id": i} for i in examples_data[0].keys()],
        style_cell={'textAlign': 'left', 'fontSize': 14, 'padding': '12px'},
        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold', 'fontSize': 16},
        style_data={'height': 'auto', 'lineHeight': '20px'},
        page_size=10
    )

    return html.Div([
        html.H5("Fuzzy Matching Examples Management", 
               style={'fontWeight': '600', 'fontSize': '1.8rem', 'marginBottom': '2rem'}),

        html.H6("Current Examples:", 
               style={'fontWeight': '600', 'fontSize': '1.4rem', 'marginBottom': '1rem'}),
        examples_table,

        html.Hr(style={'margin': '2rem 0'}),

        html.H6("Add New Example:", 
               style={'fontWeight': '600', 'fontSize': '1.4rem', 'marginBottom': '1.5rem'}),
        dbc.Row([
            dbc.Col([
                html.Label("Label:", style={'fontWeight': '600', 'fontSize': '1.1rem'}),
                dcc.Input(id="new-example-label", type="text",
                         placeholder="e.g., Technical Issue_Bug Report",
                         style={'width': '100%'})
            ], md=6),
            dbc.Col([
                dbc.Button("Add Example", id="add-example-btn", color="success", 
                          className="mt-4", style={'width': '100%'})
            ], md=6)
        ]),
        dbc.Row([
            dbc.Col([
                html.Label("Example Text:", style={'fontWeight': '600', 'fontSize': '1.1rem'}),
                dcc.Textarea(id="new-example-text",
                           placeholder="Enter example text that represents this label...",
                           style={'width': '100%', 'height': 120})
            ], md=12)
        ], className="mt-3"),

        html.Div(id="example-crud-feedback", className="mt-3")
    ])

def create_test_interface():
    """Create testing interface."""
    return html.Div([
        html.H5("Test Environment - Sandbox Mode", 
               style={'fontWeight': '600', 'fontSize': '1.8rem', 'marginBottom': '2rem'}),

        dbc.Row([
            dbc.Col([
                html.Label("Test Text:", style={'fontWeight': '600', 'fontSize': '1.2rem'}),
                dcc.Textarea(id="test-text-input",
                           placeholder="Enter text to test classification...",
                           style={'width': '100%', 'height': 150})
            ], md=8),
            dbc.Col([
                dbc.Button("Test Classifications", id="test-classify-btn",
                         color="primary", size="lg", className="mt-4",
                         style={'width': '100%', 'height': '60px'})
            ], md=4)
        ]),

        html.Hr(style={'margin': '2rem 0'}),

        html.Div(id="test-results-display")
    ])

# ============================================================================
# CALLBACKS - FIXED
# ============================================================================

# Theme switching callback - NEW
@app.callback(
    [Output('theme-store', 'data'),
     Output('light-theme-btn', 'className'),
     Output('dark-theme-btn', 'className')],
    [Input('light-theme-btn', 'n_clicks'),
     Input('dark-theme-btn', 'n_clicks')],
    [State('theme-store', 'data')]
)
def toggle_theme(light_clicks, dark_clicks, current_theme):
    """Toggle between light and dark themes."""
    ctx = dash.callback_context
    if not ctx.triggered:
        if current_theme == 'light':
            return 'light', 'btn btn-outline-secondary active', 'btn btn-outline-secondary'
        else:
            return 'dark', 'btn btn-outline-secondary', 'btn btn-outline-secondary active'
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'light-theme-btn':
        return 'light', 'btn btn-outline-secondary active', 'btn btn-outline-secondary'
    elif button_id == 'dark-theme-btn':
        return 'dark', 'btn btn-outline-secondary', 'btn btn-outline-secondary active'
    
    return current_theme, 'btn btn-outline-secondary', 'btn btn-outline-secondary'

# Apply theme to body - NEW
app.clientside_callback(
    """
    function(theme) {
        document.body.className = theme === 'dark' ? 'theme-dark' : 'theme-light';
        return '';
    }
    """,
    Output('main-container', 'className'),
    [Input('theme-store', 'data')]
)

# Main tab content callback - FIXED
@app.callback(
    Output("hitl-tab-content", "children"),
    [Input("main-hitl-tabs", "active_tab"),  # FIXED: Correct tab ID
     Input("analysis-results-store", "data"),
     Input("error-analysis-store", "data")]
)
def render_hitl_tab_content(active_tab, analysis_data, error_data):
    """Render content based on selected tab with persistent data."""

    if active_tab == "performance-tab":
        return create_performance_overview_tab(analysis_data)
    elif active_tab == "error-tab":
        return create_error_analysis_tab(error_data)
    elif active_tab == "crud-tab":
        return create_crud_management_tab()
    else:
        return html.Div("Tab content not found")

# Performance analysis callback
@app.callback(
    [Output("analysis-results-store", "data"),
     Output("error-analysis-store", "data")],
    [Input("analyze-performance-btn", "n_clicks")],
    [State("similarity-method-radio", "value")]
)
def update_performance_analysis(n_clicks, similarity_method):
    """Update performance analysis and store results."""

    if n_clicks is None:
        return None, None

    print(f"[INFO] Starting HITL performance analysis with {similarity_method} similarity")

    try:
        fuzzy_classifier.set_similarity_method(similarity_method)
        comparison_results = hitl_analyzer.compare_classifiers(rule_classifier, fuzzy_classifier)

        rule_disagreements, rule_error_patterns = hitl_analyzer.identify_error_patterns(
            comparison_results['rule_based'], "Rule-based"
        )
        fuzzy_disagreements, fuzzy_error_patterns = hitl_analyzer.identify_error_patterns(
            comparison_results['fuzzy_matching'], "Fuzzy Matching"
        )

        rule_suggestions = hitl_analyzer.suggest_improvements(rule_error_patterns, "rule_based")
        fuzzy_suggestions = hitl_analyzer.suggest_improvements(fuzzy_error_patterns, "fuzzy_matching")

        analysis_data = {
            'comparison_results': comparison_results,
            'similarity_method': similarity_method
        }

        error_data = {
            'rule_disagreements': rule_disagreements,
            'fuzzy_disagreements': fuzzy_disagreements,
            'rule_suggestions': rule_suggestions,
            'fuzzy_suggestions': fuzzy_suggestions
        }

        return analysis_data, error_data

    except Exception as e:
        print(f"[ERROR] Performance analysis failed: {str(e)}")
        return None, None

# CRUD sub-tabs callback
@app.callback(
    Output("crud-sub-content", "children"),
    Input("crud-sub-tabs", "active_tab")
)
def render_crud_sub_content(active_tab):
    """Render CRUD sub-tab content."""

    if active_tab == "rule-crud-tab":
        return create_rule_crud_interface()
    elif active_tab == "fuzzy-crud-tab":
        return create_fuzzy_crud_interface()
    elif active_tab == "test-crud-tab":
        return create_test_interface()
    else:
        return html.Div("CRUD content not found")

# Test classification callback - ENHANCED
@app.callback(
    Output("test-results-display", "children"),
    [Input("test-classify-btn", "n_clicks")],
    [State("test-text-input", "value")]
)
def test_classification(n_clicks, test_text):
    """Test classification on input text."""

    if n_clicks is None or not test_text:
        return html.P("Enter text and click 'Test Classifications' to see results.", 
                     style={'fontSize': '1.1rem', 'textAlign': 'center', 'marginTop': '2rem'})

    try:
        rule_pred, rule_conf, rule_detail = rule_classifier.predict_single(test_text)
        fuzzy_pred, fuzzy_conf, fuzzy_detail = fuzzy_classifier.predict_single(test_text)

        return dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H6("Rule-based Classification", style={'color': 'white', 'fontWeight': '600'})),
                    dbc.CardBody([
                        html.P(f"Predicted Label: {rule_pred}", style={'fontSize': '1.2rem', 'fontWeight': '500'}),
                        html.P(f"Confidence: {rule_conf:.3f}", style={'fontSize': '1.2rem', 'fontWeight': '500'}),
                        html.P(f"Matched Keywords: {', '.join(rule_detail.get('matched_keywords', []))}" if rule_detail else "No keywords matched", 
                              style={'fontSize': '1.1rem'}),
                        html.P(f"Rule Fired: {rule_detail.get('rule_fired', 'None')}" if rule_detail else "No rule fired", 
                              style={'fontSize': '1rem', 'fontStyle': 'italic'})
                    ])
                ])
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H6("Fuzzy Matching Classification", style={'color': 'white', 'fontWeight': '600'})),
                    dbc.CardBody([
                        html.P(f"Predicted Label: {fuzzy_pred}", style={'fontSize': '1.2rem', 'fontWeight': '500'}),
                        html.P(f"Confidence: {fuzzy_conf:.3f}", style={'fontSize': '1.2rem', 'fontWeight': '500'}),
                        html.P(f"Best Match: {fuzzy_detail.get('best_example', 'None')[:100]}..." if fuzzy_detail and fuzzy_detail.get('best_example') else "No match found", 
                              style={'fontSize': '1.1rem'}),
                        html.P(f"Method: {fuzzy_detail.get('method', 'Unknown')}" if fuzzy_detail else "Unknown method", 
                              style={'fontSize': '1rem', 'fontStyle': 'italic'})
                    ])
                ])
            ], md=6)
        ])

    except Exception as e:
        return dbc.Alert(f"Error testing classification: {str(e)}", color="danger")

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8050)