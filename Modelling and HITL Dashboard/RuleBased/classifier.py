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
            # Sort keywords by length (longest first) to avoid partial replacements
            sorted_keywords = sorted(eval_context.items(), key=lambda x: len(x[0]), reverse=True)
            for keyword, present in sorted_keywords:
                eval_expression = re.sub(r'\b' + re.escape(keyword) + r'\b', str(present), eval_expression)

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

