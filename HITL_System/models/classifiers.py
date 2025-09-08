# models/classifiers.py
import re
import numpy as np
from collections import defaultdict
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Any, Optional
import hashlib

class EnhancedRuleClassifier:
    """Enhanced rule-based classifier with uncertainty detection."""

    def __init__(self, rules: Dict = None):
        """Initialize with rules dictionary."""
        self.rules = rules or {}
        print(f"[DEBUG] EnhancedRuleClassifier initialized with {len(self.rules)} rules")

    def _evaluate_boolean_rule(self, rule_expression: str, text_lower: str) -> Tuple[bool, Dict[str, bool], float]:
        """Evaluate boolean rule expression against text with confidence calculation."""
        try:
            expression = rule_expression.replace(" AND ", " and ").replace(" OR ", " or ").replace(" NOT ", " not ")
            keywords = re.findall(r'\b[a-zA-Z]+\b', expression)
            
            eval_context = {}
            for keyword in keywords:
                if keyword.lower() not in ['and', 'or', 'not']:
                    eval_context[keyword] = keyword.lower() in text_lower

            # Sort keywords by length (longest first) to avoid partial replacements
            sorted_keywords = sorted(eval_context.items(), key=lambda x: len(x[0]), reverse=True)
            eval_expression = expression
            for keyword, present in sorted_keywords:
                eval_expression = re.sub(r'\b' + re.escape(keyword) + r'\b', str(present), eval_expression)

            result = eval(eval_expression)
            
            # Calculate rule confidence based on keyword matches
            matched_keywords = sum(eval_context.values())
            total_keywords = len(eval_context)
            keyword_confidence = matched_keywords / max(total_keywords, 1)
            
            # Additional confidence factors
            text_length_factor = min(1.0, len(text_lower.split()) / 10)  # Longer text = more confidence
            match_density = matched_keywords / max(len(text_lower.split()), 1)  # Keyword density
            
            confidence = (keyword_confidence * 0.7 + text_length_factor * 0.2 + match_density * 0.1)
            
            return result, eval_context, confidence

        except Exception as e:
            print(f"[ERROR] Rule evaluation failed for '{rule_expression}': {str(e)}")
            return False, {}, 0.0

    def predict_single_with_uncertainty(self, text: str) -> Tuple[str, float, float, Dict]:
        """Predict label with uncertainty score for single text."""
        text_lower = text.lower()
        predictions = []

        for label, rule_data in self.rules.items():
            rule_expression = rule_data['rule']
            weight = rule_data['weight']

            matches, keyword_matches, rule_confidence = self._evaluate_boolean_rule(rule_expression, text_lower)

            if matches:
                final_confidence = rule_confidence * weight
                predictions.append({
                    'label': label,
                    'confidence': final_confidence,
                    'rule_confidence': rule_confidence,
                    'matched_keywords': [k for k, v in keyword_matches.items() if v],
                    'rule_fired': rule_expression,
                    'weight': weight
                })

        if predictions:
            # Sort by confidence
            predictions.sort(key=lambda x: x['confidence'], reverse=True)
            best_prediction = predictions[0]
            
            # Calculate uncertainty based on prediction distribution
            if len(predictions) > 1:
                top_conf = predictions[0]['confidence']
                second_conf = predictions[1]['confidence']
                uncertainty = 1 - (top_conf - second_conf)  # Low gap = high uncertainty
            else:
                uncertainty = 1 - best_prediction['confidence']  # Low confidence = high uncertainty
            
            uncertainty = max(0.0, min(1.0, uncertainty))  # Clamp to [0,1]
            
            return best_prediction['label'], best_prediction['confidence'], uncertainty, best_prediction
        else:
            return "Unknown", 0.0, 1.0, {}  # No match = maximum uncertainty

    def predict_batch_with_uncertainty(self, texts: List[str]) -> List[Tuple[str, float, float, Dict]]:
        """Predict labels with uncertainty for batch of texts."""
        print(f"[DEBUG] Rule-based classifier predicting {len(texts)} texts with uncertainty")
        
        results = []
        for text in texts:
            label, confidence, uncertainty, details = self.predict_single_with_uncertainty(text)
            results.append((label, confidence, uncertainty, details))
        
        return results

    def get_rule_coverage_analysis(self, texts: List[str]) -> Dict:
        """Analyze rule coverage across text corpus."""
        rule_hits = defaultdict(int)
        uncovered_texts = []
        
        for text in texts:
            label, confidence, uncertainty, details = self.predict_single_with_uncertainty(text)
            if label != "Unknown":
                if 'rule_fired' in details:
                    rule_hits[details['rule_fired']] += 1
            else:
                uncovered_texts.append(text)
        
        return {
            'rule_hits': dict(rule_hits),
            'uncovered_count': len(uncovered_texts),
            'uncovered_texts': uncovered_texts[:10],  # Sample
            'coverage_rate': (len(texts) - len(uncovered_texts)) / len(texts)
        }

class EnhancedFuzzyClassifier:
    """Enhanced fuzzy matching classifier with uncertainty detection."""

    def __init__(self, examples: Dict = None, similarity_method: str = 'character'):
        """Initialize with examples and similarity method."""
        self.examples = examples or {}
        self.similarity_method = similarity_method
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english') if similarity_method == 'semantic' else None
        self.example_vectors = None
        self._fit_vectors_if_needed()
        print(f"[DEBUG] EnhancedFuzzyClassifier initialized with {similarity_method} similarity")

    def _fit_vectors_if_needed(self):
        """Fit TF-IDF vectors if using semantic similarity."""
        if self.similarity_method == 'semantic' and self.vectorizer is not None:
            all_examples = []
            for examples_list in self.examples.values():
                all_examples.extend(examples_list)

            if all_examples:
                self.vectorizer.fit(all_examples)
                self.example_vectors = {}
                for label, examples_list in self.examples.items():
                    vectors = self.vectorizer.transform(examples_list)
                    self.example_vectors[label] = vectors

    def _character_similarity(self, text1: str, text2: str) -> float:
        """Calculate character-level similarity."""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def _semantic_similarity(self, text: str, label: str) -> Tuple[float, int]:
        """Calculate semantic similarity and return best example index."""
        if self.vectorizer is None or label not in self.example_vectors:
            return 0.0, 0

        try:
            text_vector = self.vectorizer.transform([text])
            example_vectors = self.example_vectors[label]
            similarities = cosine_similarity(text_vector, example_vectors).flatten()
            best_idx = similarities.argmax()
            return similarities[best_idx], best_idx
        except:
            return 0.0, 0

    def predict_single_with_uncertainty(self, text: str) -> Tuple[str, float, float, Dict]:
        """Predict label with uncertainty for single text."""
        predictions = []

        for label, examples_list in self.examples.items():
            if not examples_list:
                continue

            if self.similarity_method == 'character':
                similarities = [self._character_similarity(text, example) for example in examples_list]
                max_similarity = max(similarities) if similarities else 0.0
                best_example_idx = similarities.index(max_similarity) if similarities else 0
            else:
                max_similarity, best_example_idx = self._semantic_similarity(text, label)

            if max_similarity > 0:
                predictions.append({
                    'label': label,
                    'similarity': max_similarity,
                    'best_example': examples_list[best_example_idx] if examples_list else "",
                    'best_example_idx': best_example_idx,
                    'method': self.similarity_method,
                    'total_examples': len(examples_list)
                })

        if predictions:
            # Sort by similarity
            predictions.sort(key=lambda x: x['similarity'], reverse=True)
            best_prediction = predictions[0]
            
            # Calculate uncertainty based on similarity distribution
            if len(predictions) > 1:
                top_sim = predictions[0]['similarity']
                second_sim = predictions[1]['similarity']
                uncertainty = 1 - (top_sim - second_sim)
            else:
                uncertainty = 1 - best_prediction['similarity']
            
            # Additional uncertainty factors
            if best_prediction['similarity'] < 0.5:
                uncertainty = min(1.0, uncertainty + 0.3)  # Very low similarity = high uncertainty
            
            uncertainty = max(0.0, min(1.0, uncertainty))
            
            return (best_prediction['label'], best_prediction['similarity'], 
                   uncertainty, best_prediction)
        else:
            return "Unknown", 0.0, 1.0, {}

    def predict_batch_with_uncertainty(self, texts: List[str]) -> List[Tuple[str, float, float, Dict]]:
        """Predict labels with uncertainty for batch of texts."""
        print(f"[DEBUG] Fuzzy classifier predicting {len(texts)} texts with {self.similarity_method} similarity")
        
        results = []
        for text in texts:
            label, confidence, uncertainty, details = self.predict_single_with_uncertainty(text)
            results.append((label, confidence, uncertainty, details))
        
        return results

    def set_similarity_method(self, method: str):
        """Change similarity method and refit vectors if needed."""
        self.similarity_method = method
        if method == 'semantic':
            if self.vectorizer is None:
                self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            self._fit_vectors_if_needed()
        print(f"[DEBUG] Changed similarity method to {method}")

    def get_example_coverage_analysis(self, texts: List[str]) -> Dict:
        """Analyze example coverage and similarity distribution."""
        similarity_distribution = []
        low_similarity_texts = []
        label_performance = defaultdict(list)
        
        for text in texts:
            label, confidence, uncertainty, details = self.predict_single_with_uncertainty(text)
            similarity_distribution.append(confidence)
            
            if confidence < 0.3:
                low_similarity_texts.append((text, confidence))
            
            if label != "Unknown":
                label_performance[label].append(confidence)
        
        # Calculate statistics per label
        label_stats = {}
        for label, similarities in label_performance.items():
            label_stats[label] = {
                'avg_similarity': np.mean(similarities),
                'min_similarity': np.min(similarities),
                'max_similarity': np.max(similarities),
                'count': len(similarities)
            }
        
        return {
            'similarity_distribution': similarity_distribution,
            'avg_similarity': np.mean(similarity_distribution),
            'low_similarity_count': len(low_similarity_texts),
            'low_similarity_samples': low_similarity_texts[:10],
            'label_performance': label_stats
        }

class UncertaintyDetector:
    """Detect uncertain cases that need human annotation."""
    
    def __init__(self, uncertainty_threshold: float = 0.6):
        """Initialize uncertainty detector."""
        self.uncertainty_threshold = uncertainty_threshold
        
    def is_uncertain(self, rule_result: Tuple, fuzzy_result: Tuple) -> Tuple[bool, float, str]:
        """
        Determine if a case is uncertain based on classifier results.
        
        Args:
            rule_result: (label, confidence, uncertainty, details)
            fuzzy_result: (label, confidence, uncertainty, details)
            
        Returns:
            (is_uncertain, uncertainty_score, reason)
        """
        rule_label, rule_conf, rule_unc, _ = rule_result
        fuzzy_label, fuzzy_conf, fuzzy_unc, _ = fuzzy_result
        
        reasons = []
        uncertainty_factors = []
        
        # Factor 1: Individual classifier uncertainty
        if rule_unc > self.uncertainty_threshold:
            reasons.append("high_rule_uncertainty")
            uncertainty_factors.append(rule_unc)
            
        if fuzzy_unc > self.uncertainty_threshold:
            reasons.append("high_fuzzy_uncertainty")
            uncertainty_factors.append(fuzzy_unc)
        
        # Factor 2: Classifier disagreement
        disagreement = rule_label != fuzzy_label
        if disagreement and rule_label != "Unknown" and fuzzy_label != "Unknown":
            reasons.append("classifier_disagreement")
            uncertainty_factors.append(0.8)  # High uncertainty for disagreement
        
        # Factor 3: Low confidence predictions
        if rule_conf < (1 - self.uncertainty_threshold) and rule_label != "Unknown":
            reasons.append("low_rule_confidence")
            uncertainty_factors.append(1 - rule_conf)
            
        if fuzzy_conf < (1 - self.uncertainty_threshold) and fuzzy_label != "Unknown":
            reasons.append("low_fuzzy_confidence")
            uncertainty_factors.append(1 - fuzzy_conf)
        
        # Factor 4: Both classifiers predict Unknown
        if rule_label == "Unknown" and fuzzy_label == "Unknown":
            reasons.append("both_unknown")
            uncertainty_factors.append(1.0)
        
        # Calculate overall uncertainty score
        if uncertainty_factors:
            uncertainty_score = min(1.0, np.mean(uncertainty_factors))
            is_uncertain = uncertainty_score > self.uncertainty_threshold
            reason = "; ".join(reasons)
            return is_uncertain, uncertainty_score, reason
        else:
            return False, 0.0, "confident_prediction"
    
    def update_threshold(self, new_threshold: float):
        """Update uncertainty threshold."""
        self.uncertainty_threshold = max(0.0, min(1.0, new_threshold))
        print(f"[DEBUG] Updated uncertainty threshold to {self.uncertainty_threshold}")
        
    def batch_uncertainty_detection(self, rule_results: List[Tuple], 
                                  fuzzy_results: List[Tuple]) -> List[Dict]:
        """Detect uncertain cases in batch."""
        uncertain_cases = []
        
        for i, (rule_result, fuzzy_result) in enumerate(zip(rule_results, fuzzy_results)):
            is_uncertain, uncertainty_score, reason = self.is_uncertain(rule_result, fuzzy_result)
            
            if is_uncertain:
                case = {
                    'index': i,
                    'rule_result': rule_result,
                    'fuzzy_result': fuzzy_result,
                    'uncertainty_score': uncertainty_score,
                    'reason': reason,
                    'disagreement': rule_result[0] != fuzzy_result[0]
                }
                uncertain_cases.append(case)
        
        # Sort by uncertainty score (highest first)
        uncertain_cases.sort(key=lambda x: x['uncertainty_score'], reverse=True)
        
        print(f"[DEBUG] Detected {len(uncertain_cases)} uncertain cases from {len(rule_results)} predictions")
        return uncertain_cases