# utils/evaluation.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import json

class HITLEvaluator:
    """Evaluate and track performance of HITL system."""
    
    def __init__(self, db_manager):
        """Initialize with database manager."""
        self.db_manager = db_manager
        
    def evaluate_classifier_performance(self, classifier, classifier_name: str, 
                                      test_cases: List[Dict]) -> Dict:
        """Evaluate classifier against test cases with human labels."""
        if not test_cases:
            return {'error': 'No test cases provided'}
        
        print(f"[INFO] Evaluating {classifier_name} performance on {len(test_cases)} cases")
        
        texts = [case['text'] for case in test_cases]
        true_labels = [case['human_label'] for case in test_cases]
        
        # Get predictions with uncertainty
        if hasattr(classifier, 'predict_batch_with_uncertainty'):
            results = classifier.predict_batch_with_uncertainty(texts)
            pred_labels = [result[0] for result in results]
            confidences = [result[1] for result in results]
            uncertainties = [result[2] for result in results]
        else:
            # Fallback for basic classifiers
            pred_labels = classifier.predict(texts)
            confidences = [0.5] * len(pred_labels)
            uncertainties = [0.5] * len(pred_labels)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average='weighted', zero_division=0
        )
        
        # Calculate per-class metrics
        unique_labels = list(set(true_labels + pred_labels))
        per_class_metrics = {}
        
        for label in unique_labels:
            if label in true_labels and label in pred_labels:
                label_true = [1 if l == label else 0 for l in true_labels]
                label_pred = [1 if l == label else 0 for l in pred_labels]
                
                if sum(label_true) > 0:  # Only if label exists in true labels
                    p, r, f, _ = precision_recall_fscore_support(
                        label_true, label_pred, average='binary', zero_division=0
                    )
                    per_class_metrics[label] = {
                        'precision': p,
                        'recall': r,
                        'f1_score': f,
                        'support': sum(label_true)
                    }
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)
        
        # Calculate confidence-based metrics
        high_conf_indices = [i for i, conf in enumerate(confidences) if conf > 0.7]
        if high_conf_indices:
            high_conf_accuracy = accuracy_score(
                [true_labels[i] for i in high_conf_indices],
                [pred_labels[i] for i in high_conf_indices]
            )
        else:
            high_conf_accuracy = 0.0
        
        # Store metrics in database
        total_predictions = len(pred_labels)
        correct_predictions = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
        annotation_count = self.db_manager.get_annotation_count()
        
        self.db_manager.save_performance_metrics(
            metric_type='evaluation',
            classifier_type=classifier_name.lower().replace(' ', '_'),
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            total_predictions=total_predictions,
            correct_predictions=correct_predictions,
            annotation_count=annotation_count
        )
        
        return {
            'classifier_name': classifier_name,
            'overall_metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            },
            'per_class_metrics': per_class_metrics,
            'confidence_metrics': {
                'avg_confidence': np.mean(confidences),
                'avg_uncertainty': np.mean(uncertainties),
                'high_confidence_accuracy': high_conf_accuracy,
                'high_confidence_count': len(high_conf_indices)
            },
            'confusion_matrix': {
                'matrix': cm.tolist(),
                'labels': unique_labels
            },
            'prediction_details': {
                'total_predictions': total_predictions,
                'correct_predictions': correct_predictions,
                'annotation_count': annotation_count
            }
        }
    
    def compare_classifiers(self, rule_classifier, fuzzy_classifier, 
                          test_cases: List[Dict]) -> Dict:
        """Compare performance of both classifiers."""
        print(f"[INFO] Comparing classifiers on {len(test_cases)} test cases")
        
        rule_results = self.evaluate_classifier_performance(
            rule_classifier, "Rule-based", test_cases
        )
        fuzzy_results = self.evaluate_classifier_performance(
            fuzzy_classifier, "Fuzzy Matching", test_cases
        )
        
        # Calculate improvement metrics
        improvement = {
            'accuracy_diff': rule_results['overall_metrics']['accuracy'] - fuzzy_results['overall_metrics']['accuracy'],
            'f1_diff': rule_results['overall_metrics']['f1_score'] - fuzzy_results['overall_metrics']['f1_score'],
            'better_classifier': 'rule_based' if rule_results['overall_metrics']['f1_score'] > fuzzy_results['overall_metrics']['f1_score'] else 'fuzzy_matching'
        }
        
        return {
            'rule_based': rule_results,
            'fuzzy_matching': fuzzy_results,
            'comparison': improvement,
            'evaluation_timestamp': datetime.now().isoformat()
        }
    
    def track_learning_progress(self, window_days: int = 30) -> Dict:
        """Track learning progress over time."""
        print(f"[INFO] Tracking learning progress over {window_days} days")
        
        # Get performance history
        performance_history = self.db_manager.get_performance_history(window_days)
        
        if not performance_history:
            return {'error': 'No performance history available'}
        
        # Group by classifier type
        rule_metrics = [m for m in performance_history if m['classifier_type'] == 'rule_based']
        fuzzy_metrics = [m for m in performance_history if m['classifier_type'] == 'fuzzy_matching']
        
        def calculate_trends(metrics_list):
            if len(metrics_list) < 2:
                return {'trend': 'insufficient_data'}
            
            # Sort by timestamp
            sorted_metrics = sorted(metrics_list, key=lambda x: x['created_at'])
            
            accuracies = [m['accuracy'] for m in sorted_metrics]
            f1_scores = [m['f1_macro'] for m in sorted_metrics]
            
            # Calculate trends
            accuracy_trend = (accuracies[-1] - accuracies[0]) if len(accuracies) > 1 else 0
            f1_trend = (f1_scores[-1] - f1_scores[0]) if len(f1_scores) > 1 else 0
            
            return {
                'accuracy_trend': accuracy_trend,
                'f1_trend': f1_trend,
                'latest_accuracy': accuracies[-1],
                'latest_f1': f1_scores[-1],
                'evaluation_count': len(sorted_metrics),
                'trend': 'improving' if f1_trend > 0.01 else 'declining' if f1_trend < -0.01 else 'stable'
            }
        
        rule_trends = calculate_trends(rule_metrics)
        fuzzy_trends = calculate_trends(fuzzy_metrics)
        
        # Overall system trends
        annotation_count = self.db_manager.get_annotation_count()
        
        return {
            'rule_based_trends': rule_trends,
            'fuzzy_matching_trends': fuzzy_trends,
            'system_metrics': {
                'total_annotations': annotation_count,
                'evaluation_period_days': window_days,
                'last_evaluation': performance_history[-1]['created_at'] if performance_history else None
            }
        }
    
    def analyze_annotation_impact(self) -> Dict:
        """Analyze the impact of human annotations on model performance."""
        print("[INFO] Analyzing annotation impact on performance")
        
        # Get annotation history and performance metrics
        annotations = self.db_manager.get_recent_annotations(200)
        performance_history = self.db_manager.get_performance_history(60)
        
        if not annotations or not performance_history:
            return {'error': 'Insufficient data for impact analysis'}
        
        # Group performance by annotation milestones
        annotation_milestones = []
        current_annotations = 0
        
        for metric in sorted(performance_history, key=lambda x: x['created_at']):
            # Find annotations up to this point
            metric_time = datetime.fromisoformat(metric['created_at'])
            annotations_before = [
                a for a in annotations 
                if datetime.fromisoformat(a['created_at']) <= metric_time
            ]
            
            if len(annotations_before) != current_annotations:
                current_annotations = len(annotations_before)
                annotation_milestones.append({
                    'annotation_count': current_annotations,
                    'timestamp': metric['created_at'],
                    'accuracy': metric['accuracy'],
                    'f1_score': metric['f1_macro'],
                    'classifier_type': metric['classifier_type']
                })
        
        # Calculate correlation between annotations and performance
        if len(annotation_milestones) >= 3:
            ann_counts = [m['annotation_count'] for m in annotation_milestones]
            accuracies = [m['accuracy'] for m in annotation_milestones]
            f1_scores = [m['f1_score'] for m in annotation_milestones]
            
            # Simple correlation calculation
            def calculate_correlation(x, y):
                if len(x) != len(y) or len(x) < 2:
                    return 0
                n = len(x)
                sum_x = sum(x)
                sum_y = sum(y)
                sum_xy = sum(a * b for a, b in zip(x, y))
                sum_x2 = sum(a * a for a in x)
                sum_y2 = sum(b * b for b in y)
                
                denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)) ** 0.5
                if denominator == 0:
                    return 0
                return (n * sum_xy - sum_x * sum_y) / denominator
            
            accuracy_correlation = calculate_correlation(ann_counts, accuracies)
            f1_correlation = calculate_correlation(ann_counts, f1_scores)
        else:
            accuracy_correlation = 0
            f1_correlation = 0
        
        return {
            'annotation_impact': {
                'accuracy_correlation': accuracy_correlation,
                'f1_correlation': f1_correlation,
                'interpretation': self._interpret_correlation(f1_correlation)
            },
            'milestones': annotation_milestones,
            'summary': {
                'total_milestones': len(annotation_milestones),
                'annotation_range': [min(ann_counts), max(ann_counts)] if annotation_milestones else [0, 0],
                'performance_improvement': (
                    max(f1_scores) - min(f1_scores) if annotation_milestones else 0
                )
            }
        }
    
    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret correlation coefficient."""
        abs_corr = abs(correlation)
        if abs_corr > 0.7:
            strength = "strong"
        elif abs_corr > 0.4:
            strength = "moderate"
        elif abs_corr > 0.2:
            strength = "weak"
        else:
            strength = "negligible"
        
        direction = "positive" if correlation > 0 else "negative"
        return f"{strength} {direction} correlation"
    
    def generate_error_analysis(self, classifier, classifier_name: str, 
                              test_cases: List[Dict]) -> Dict:
        """Generate detailed error analysis."""
        print(f"[INFO] Generating error analysis for {classifier_name}")
        
        if not test_cases:
            return {'error': 'No test cases provided'}
        
        texts = [case['text'] for case in test_cases]
        true_labels = [case['human_label'] for case in test_cases]
        
        # Get predictions
        if hasattr(classifier, 'predict_batch_with_uncertainty'):
            results = classifier.predict_batch_with_uncertainty(texts)
            pred_labels = [result[0] for result in results]
            confidences = [result[1] for result in results]
        else:
            pred_labels = classifier.predict(texts)
            confidences = [0.5] * len(pred_labels)
        
        # Identify errors
        errors = []
        for i, (true_label, pred_label) in enumerate(zip(true_labels, pred_labels)):
            if true_label != pred_label:
                errors.append({
                    'text': texts[i][:200],  # Truncate for display
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'confidence': confidences[i],
                    'error_type': self._classify_error_type(true_label, pred_label)
                })
        
        # Analyze error patterns
        error_patterns = Counter()
        confidence_by_error = defaultdict(list)
        
        for error in errors:
            pattern = f"{error['true_label']} → {error['predicted_label']}"
            error_patterns[pattern] += 1
            confidence_by_error[pattern].append(error['confidence'])
        
        # Generate suggestions
        suggestions = []
        for pattern, count in error_patterns.most_common(5):
            avg_confidence = np.mean(confidence_by_error[pattern])
            true_label, pred_label = pattern.split(' → ')
            
            suggestion = {
                'error_pattern': pattern,
                'frequency': count,
                'avg_confidence': avg_confidence,
                'suggestion': self._generate_error_suggestion(
                    true_label, pred_label, count, avg_confidence, classifier_name
                ),
                'priority': 'high' if count > 3 else 'medium' if count > 1 else 'low'
            }
            suggestions.append(suggestion)
        
        return {
            'classifier_name': classifier_name,
            'total_errors': len(errors),
            'error_rate': len(errors) / len(test_cases),
            'error_patterns': dict(error_patterns),
            'sample_errors': errors[:10],  # Show first 10 errors
            'suggestions': suggestions,
            'confidence_analysis': {
                'avg_error_confidence': np.mean([e['confidence'] for e in errors]) if errors else 0,
                'high_confidence_errors': sum(1 for e in errors if e['confidence'] > 0.7),
                'low_confidence_errors': sum(1 for e in errors if e['confidence'] < 0.3)
            }
        }
    
    def _classify_error_type(self, true_label: str, pred_label: str) -> str:
        """Classify the type of error."""
        if pred_label == "Unknown":
            return "missed_detection"
        elif true_label == "Unknown":
            return "false_positive"
        else:
            return "misclassification"
    
    def _generate_error_suggestion(self, true_label: str, pred_label: str, 
                                 frequency: int, avg_confidence: float, 
                                 classifier_name: str) -> str:
        """Generate suggestion based on error pattern."""
        if classifier_name.lower() == 'rule-based':
            if pred_label == "Unknown":
                return f"Add rules to detect '{true_label}' patterns - currently missing {frequency} cases"
            else:
                return f"Refine rules to distinguish '{true_label}' from '{pred_label}' - {frequency} confusion cases"
        else:
            if pred_label == "Unknown":
                return f"Add training examples for '{true_label}' - low similarity causing {frequency} misses"
            else:
                return f"Add examples to distinguish '{true_label}' from '{pred_label}' - {frequency} confusions"
    
    def calculate_roi_metrics(self) -> Dict:
        """Calculate return on investment for human annotation effort."""
        print("[INFO] Calculating annotation ROI metrics")
        
        # Get annotation stats and performance trends
        annotation_stats = self.db_manager.get_annotation_stats()
        learning_progress = self.track_learning_progress(30)
        
        if not annotation_stats or annotation_stats['total_annotations'] == 0:
            return {'error': 'No annotations available for ROI calculation'}
        
        # Estimate annotation effort
        annotations = self.db_manager.get_recent_annotations(100)
        if annotations:
            avg_time_per_annotation = np.mean([
                a['annotation_time_seconds'] for a in annotations 
                if a['annotation_time_seconds'] and a['annotation_time_seconds'] > 0
            ])
        else:
            avg_time_per_annotation = 30  # Default estimate
        
        total_annotation_time = annotation_stats['total_annotations'] * avg_time_per_annotation
        
        # Calculate performance gains
        if ('rule_based_trends' in learning_progress and 
            learning_progress['rule_based_trends'].get('accuracy_trend', 0) > 0):
            accuracy_improvement = learning_progress['rule_based_trends']['accuracy_trend']
        else:
            accuracy_improvement = 0
        
        # Calculate efficiency metrics
        annotations_per_improvement = (
            annotation_stats['total_annotations'] / max(0.01, accuracy_improvement * 100)
            if accuracy_improvement > 0 else float('inf')
        )
        
        return {
            'annotation_effort': {
                'total_annotations': annotation_stats['total_annotations'],
                'estimated_total_time_minutes': total_annotation_time / 60,
                'avg_time_per_annotation_seconds': avg_time_per_annotation
            },
            'performance_gains': {
                'accuracy_improvement': accuracy_improvement,
                'annotations_per_percent_improvement': annotations_per_improvement
            },
            'efficiency_metrics': {
                'roi_score': max(0, accuracy_improvement * 100 / max(1, annotation_stats['total_annotations'])),
                'annotation_velocity': annotation_stats.get('annotations_today', 0),
                'learning_trend': learning_progress.get('rule_based_trends', {}).get('trend', 'unknown')
            }
        }
    
    def generate_comprehensive_report(self, rule_classifier, fuzzy_classifier, 
                                    test_cases: List[Dict] = None) -> Dict:
        """Generate comprehensive evaluation report."""
        print("[INFO] Generating comprehensive HITL evaluation report")
        
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'system_overview': {}
        }
        
        # Get test cases from database if not provided
        if not test_cases:
            annotations = self.db_manager.get_annotations_for_training()
            test_cases = [{
                'text': ann['text'],
                'human_label': ann['human_label']
            } for ann in annotations[-100:]]  # Use recent 100 for testing
        
        if test_cases:
            # Classifier comparison
            report['classifier_comparison'] = self.compare_classifiers(
                rule_classifier, fuzzy_classifier, test_cases
            )
            
            # Error analysis
            report['error_analysis'] = {
                'rule_based': self.generate_error_analysis(
                    rule_classifier, 'Rule-based', test_cases
                ),
                'fuzzy_matching': self.generate_error_analysis(
                    fuzzy_classifier, 'Fuzzy Matching', test_cases
                )
            }
        
        # Learning progress
        report['learning_progress'] = self.track_learning_progress()
        
        # Annotation impact
        report['annotation_impact'] = self.analyze_annotation_impact()
        
        # ROI metrics
        report['roi_analysis'] = self.calculate_roi_metrics()
        
        # System statistics
        report['system_statistics'] = {
            'database_stats': self.db_manager.get_database_stats(),
            'annotation_stats': self.db_manager.get_annotation_stats(),
            'learning_effectiveness': self.db_manager.get_learning_effectiveness()
        }
        
        return report