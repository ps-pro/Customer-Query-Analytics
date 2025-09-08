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

