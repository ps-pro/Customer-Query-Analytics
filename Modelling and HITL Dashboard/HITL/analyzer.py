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

