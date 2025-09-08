# models/active_learning.py
import numpy as np
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import re

class ActiveLearningManager:
    """Manage active learning strategies for HITL system."""
    
    def __init__(self, db_manager, uncertainty_detector):
        """Initialize with database manager and uncertainty detector."""
        self.db_manager = db_manager
        self.uncertainty_detector = uncertainty_detector
        self.annotation_history = []
        self.performance_tracking = defaultdict(list)
        
    def process_new_texts(self, texts: List[str], rule_classifier, fuzzy_classifier) -> Dict:
        """Process new texts and identify candidates for annotation."""
        print(f"[INFO] Processing {len(texts)} new texts for active learning")
        
        # Get predictions from both classifiers
        rule_results = rule_classifier.predict_batch_with_uncertainty(texts)
        fuzzy_results = fuzzy_classifier.predict_batch_with_uncertainty(texts)
        
        # Detect uncertain cases
        uncertain_cases = self.uncertainty_detector.batch_uncertainty_detection(
            rule_results, fuzzy_results
        )
        
        # Store all predictions for tracking
        self._store_predictions(texts, rule_results, fuzzy_results)
        
        # Add uncertain cases to database queue
        added_cases = []
        for case in uncertain_cases:
            text = texts[case['index']]
            text_hash = self._add_to_queue(text, case)
            if text_hash:
                added_cases.append({
                    'text_hash': text_hash,
                    'text': text,
                    'uncertainty_score': case['uncertainty_score'],
                    'reason': case['reason']
                })
        
        return {
            'total_processed': len(texts),
            'uncertain_cases_found': len(uncertain_cases),
            'cases_added_to_queue': len(added_cases),
            'queue_summary': self._get_queue_summary()
        }
    
    def _store_predictions(self, texts: List[str], rule_results: List[Tuple], 
                          fuzzy_results: List[Tuple]):
        """Store predictions in database for tracking."""
        for i, text in enumerate(texts):
            text_hash = hashlib.md5(text.encode()).hexdigest()
            rule_result = rule_results[i]
            fuzzy_result = fuzzy_results[i]
            
            # Store in prediction log (this would be implemented in database manager)
            # For now, we'll track in memory
            pass
    
    def _add_to_queue(self, text: str, case_info: Dict) -> Optional[str]:
        """Add uncertain case to annotation queue."""
        rule_result = case_info['rule_result']
        fuzzy_result = case_info['fuzzy_result']
        
        text_hash = self.db_manager.add_uncertain_case(
            text=text,
            rule_pred=rule_result[0],
            rule_conf=rule_result[1],
            fuzzy_pred=fuzzy_result[0],
            fuzzy_conf=fuzzy_result[1],
            uncertainty_score=case_info['uncertainty_score'],
            disagreement=case_info['disagreement']
        )
        
        # Log the active learning selection
        self.db_manager.log_active_learning_selection(
            text_hash=text_hash,
            selection_reason=case_info['reason'],
            uncertainty_score=case_info['uncertainty_score']
        )
        
        return text_hash
    
    def get_next_annotation_batch(self, batch_size: int = 15) -> List[Dict]:
        """Get next batch of cases for human annotation."""
        cases = self.db_manager.get_uncertain_cases_batch(batch_size)
        
        if cases:
            print(f"[INFO] Retrieved {len(cases)} cases for annotation")
            # Apply diversity filtering if needed
            cases = self._apply_diversity_filtering(cases)
            
        return cases
    
    def _apply_diversity_filtering(self, cases: List[Dict]) -> List[Dict]:
        """Apply diversity filtering to avoid repetitive cases."""
        if len(cases) <= 5:
            return cases
        
        # Simple diversity: avoid too many cases with same predicted labels
        label_counts = defaultdict(int)
        filtered_cases = []
        
        for case in cases:
            rule_pred = case['rule_prediction']
            fuzzy_pred = case['fuzzy_prediction']
            
            # Limit cases per prediction combination
            key = f"{rule_pred}_{fuzzy_pred}"
            if label_counts[key] < 3:  # Max 3 cases per prediction combo
                filtered_cases.append(case)
                label_counts[key] += 1
        
        return filtered_cases
    
    def process_human_annotation(self, text_hash: str, text: str, human_label: str,
                                confidence_rating: int, annotation_time: float) -> Dict:
        """Process a human annotation and trigger model updates if needed."""
        print(f"[INFO] Processing human annotation for text_hash: {text_hash}")
        
        # Get original predictions for this case
        uncertain_case = self._get_uncertain_case(text_hash)
        if not uncertain_case:
            return {'error': 'Uncertain case not found'}
        
        # Store annotation in database
        annotation_id = self.db_manager.add_human_annotation(
            text_hash=text_hash,
            text=text,
            original_rule_pred=uncertain_case['rule_prediction'],
            original_fuzzy_pred=uncertain_case['fuzzy_prediction'],
            human_label=human_label,
            confidence_rating=confidence_rating,
            annotation_time=annotation_time
        )
        
        # Update active learning log
        rule_agreed = uncertain_case['rule_prediction'] == human_label
        fuzzy_agreed = uncertain_case['fuzzy_prediction'] == human_label
        overall_agreed = rule_agreed or fuzzy_agreed
        
        self.db_manager.update_active_learning_result(
            text_hash=text_hash,
            was_annotated=True,
            human_agreed=overall_agreed
        )
        
        # Check if auto-update should be triggered
        annotation_count = self.db_manager.get_annotation_count()
        auto_update_freq = int(self.db_manager.get_config('auto_update_frequency', '5'))
        
        result = {
            'annotation_id': annotation_id,
            'agreement': {
                'rule_agreed': rule_agreed,
                'fuzzy_agreed': fuzzy_agreed,
                'overall_agreed': overall_agreed
            },
            'total_annotations': annotation_count
        }
        
        if annotation_count % auto_update_freq == 0:
            result['trigger_update'] = True
            print(f"[INFO] Triggering model update after {annotation_count} annotations")
        
        return result
    
    def _get_uncertain_case(self, text_hash: str) -> Optional[Dict]:
        """Get uncertain case details by text hash."""
        # This would query the database - simplified for now
        cases = self.db_manager.get_uncertain_cases_batch(1000)  # Get all pending
        for case in cases:
            if case['text_hash'] == text_hash:
                return case
        return None
    
    def _get_queue_summary(self) -> Dict:
        """Get summary of annotation queue."""
        queue_count = self.db_manager.get_uncertain_cases_count()
        return {
            'pending_cases': queue_count,
            'avg_uncertainty': 0.7,  # Would calculate from actual data
            'top_reasons': ['high_rule_uncertainty', 'classifier_disagreement']
        }
    
    def suggest_model_improvements(self, rule_classifier, fuzzy_classifier) -> Dict:
        """Suggest improvements based on annotation patterns."""
        print("[INFO] Analyzing annotation patterns for improvement suggestions")
        
        # Get recent annotations
        recent_annotations = self.db_manager.get_recent_annotations(100)
        
        if not recent_annotations:
            return {'suggestions': [], 'analysis': 'Insufficient annotation data'}
        
        # Analyze patterns
        rule_suggestions = self._analyze_rule_patterns(recent_annotations)
        example_suggestions = self._analyze_example_patterns(recent_annotations)
        
        return {
            'rule_suggestions': rule_suggestions,
            'example_suggestions': example_suggestions,
            'annotation_stats': self._get_annotation_statistics(recent_annotations)
        }
    
    def _analyze_rule_patterns(self, annotations: List[Dict]) -> List[Dict]:
        """Analyze patterns in annotations to suggest new rules."""
        suggestions = []
        
        # Group by human label
        label_groups = defaultdict(list)
        for ann in annotations:
            if ann['human_label'] != ann['original_rule_pred']:
                label_groups[ann['human_label']].append(ann)
        
        for label, failed_cases in label_groups.items():
            if len(failed_cases) >= 3:  # Need multiple examples to suggest rule
                # Extract common keywords from failed cases
                common_keywords = self._extract_common_keywords([case['text'] for case in failed_cases])
                
                if common_keywords:
                    suggestion = {
                        'label': label,
                        'suggested_keywords': common_keywords[:5],
                        'failed_cases_count': len(failed_cases),
                        'confidence': min(1.0, len(failed_cases) / 10),
                        'sample_texts': [case['text'][:100] for case in failed_cases[:3]]
                    }
                    suggestions.append(suggestion)
        
        return suggestions
    
    def _analyze_example_patterns(self, annotations: List[Dict]) -> List[Dict]:
        """Analyze patterns to suggest new examples."""
        suggestions = []
        
        # Find cases where fuzzy matching failed but rule-based succeeded
        for ann in annotations:
            if (ann['human_label'] == ann['original_rule_pred'] and 
                ann['human_label'] != ann['original_fuzzy_pred']):
                
                suggestion = {
                    'label': ann['human_label'],
                    'suggested_example': ann['text'],
                    'confidence': ann['confidence_rating'] / 5.0,
                    'reason': 'fuzzy_matching_miss'
                }
                suggestions.append(suggestion)
        
        return suggestions[:10]  # Limit suggestions
    
    def _extract_common_keywords(self, texts: List[str]) -> List[str]:
        """Extract common keywords from a list of texts."""
        # Simple keyword extraction
        all_words = []
        for text in texts:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            all_words.extend(words)
        
        # Count and return most common
        word_counts = Counter(all_words)
        return [word for word, count in word_counts.most_common(10) if count >= 2]
    
    def _get_annotation_statistics(self, annotations: List[Dict]) -> Dict:
        """Get statistics about annotations."""
        if not annotations:
            return {}
        
        rule_agreement = sum(1 for ann in annotations 
                           if ann['human_label'] == ann['original_rule_pred']) / len(annotations)
        
        fuzzy_agreement = sum(1 for ann in annotations 
                            if ann['human_label'] == ann['original_fuzzy_pred']) / len(annotations)
        
        avg_confidence = sum(ann['confidence_rating'] for ann in annotations) / len(annotations)
        
        label_distribution = Counter(ann['human_label'] for ann in annotations)
        
        return {
            'total_annotations': len(annotations),
            'rule_agreement_rate': rule_agreement,
            'fuzzy_agreement_rate': fuzzy_agreement,
            'average_confidence': avg_confidence,
            'label_distribution': dict(label_distribution)
        }
    
    def auto_generate_rules(self, annotations: List[Dict], min_confidence: float = 0.8) -> List[Dict]:
        """Automatically generate rules from annotation patterns."""
        print("[INFO] Auto-generating rules from annotation patterns")
        
        generated_rules = []
        
        # Group annotations by label
        label_groups = defaultdict(list)
        for ann in annotations:
            if ann['confidence_rating'] >= 4:  # High confidence annotations only
                label_groups[ann['human_label']].append(ann)
        
        for label, group_annotations in label_groups.items():
            if len(group_annotations) >= 5:  # Need enough examples
                rule = self._generate_rule_from_examples(label, group_annotations)
                if rule and rule['confidence'] >= min_confidence:
                    generated_rules.append(rule)
        
        return generated_rules
    
    def _generate_rule_from_examples(self, label: str, annotations: List[Dict]) -> Optional[Dict]:
        """Generate a rule from a group of annotations."""
        # Extract common patterns from texts
        texts = [ann['text'] for ann in annotations]
        common_keywords = self._extract_common_keywords(texts)
        
        if len(common_keywords) >= 2:
            # Create a simple OR rule with most common keywords
            rule_keywords = common_keywords[:4]
            rule_expression = " OR ".join(rule_keywords)
            
            # Calculate confidence based on keyword frequency and annotation confidence
            avg_annotation_conf = sum(ann['confidence_rating'] for ann in annotations) / len(annotations)
            keyword_coverage = len(rule_keywords) / len(common_keywords) if common_keywords else 0
            
            confidence = (avg_annotation_conf / 5.0) * 0.7 + keyword_coverage * 0.3
            
            return {
                'label': label,
                'rule_expression': f"({rule_expression})",
                'weight': 1.0,
                'description': f"Auto-generated from {len(annotations)} annotations",
                'confidence': confidence,
                'source_annotations': len(annotations),
                'keywords_used': rule_keywords
            }
        
        return None
    
    def update_uncertainty_threshold_adaptive(self) -> float:
        """Adaptively update uncertainty threshold based on annotation patterns."""
        # Get recent annotation effectiveness
        recent_annotations = self.db_manager.get_recent_annotations(50)
        
        if len(recent_annotations) < 10:
            return self.uncertainty_detector.uncertainty_threshold
        
        # Calculate how often annotations provided useful information
        useful_annotations = sum(1 for ann in recent_annotations
                               if ann['human_label'] not in [ann['original_rule_pred'], ann['original_fuzzy_pred']])
        
        usefulness_rate = useful_annotations / len(recent_annotations)
        
        current_threshold = self.uncertainty_detector.uncertainty_threshold
        
        # Adjust threshold based on usefulness
        if usefulness_rate > 0.8:
            # Too many useful annotations - lower threshold to be more selective
            new_threshold = min(0.9, current_threshold + 0.05)
        elif usefulness_rate < 0.4:
            # Too few useful annotations - raise threshold to capture more cases
            new_threshold = max(0.3, current_threshold - 0.05)
        else:
            new_threshold = current_threshold
        
        if new_threshold != current_threshold:
            self.uncertainty_detector.update_threshold(new_threshold)
            self.db_manager.set_config('uncertainty_threshold', str(new_threshold))
            print(f"[INFO] Adaptively updated uncertainty threshold: {current_threshold:.2f} -> {new_threshold:.2f}")
        
        return new_threshold
    
    def get_learning_insights(self) -> Dict:
        """Get insights about the learning process."""
        stats = self.db_manager.get_annotation_stats()
        effectiveness = self.db_manager.get_learning_effectiveness()
        
        # Calculate learning velocity (annotations per day)
        recent_annotations = self.db_manager.get_recent_annotations(100)
        if recent_annotations:
            first_date = datetime.fromisoformat(recent_annotations[-1]['created_at'])
            last_date = datetime.fromisoformat(recent_annotations[0]['created_at'])
            days_span = max(1, (last_date - first_date).days)
            annotation_velocity = len(recent_annotations) / days_span
        else:
            annotation_velocity = 0
        
        return {
            'annotation_stats': stats,
            'learning_effectiveness': effectiveness,
            'annotation_velocity': annotation_velocity,
            'queue_status': self._get_queue_summary(),
            'current_threshold': self.uncertainty_detector.uncertainty_threshold
        }