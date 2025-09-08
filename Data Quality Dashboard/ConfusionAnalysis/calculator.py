import pandas as pd
import numpy as np
import itertools
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback, dash_table, State
import dash_bootstrap_components as dbc
from collections import Counter, defaultdict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')


# Core Annotator Confusion Analysis Calculator
class AnnotatorConfusionCalculator:
    """Calculator for analyzing individual annotator performance and confusion patterns."""

    def __init__(self, agreement_df):
        """Initialize with agreement dataframe."""
        self.agreement_df = agreement_df
        self.annotators = sorted(agreement_df['annotator'].unique())
        self.documents = sorted(agreement_df['id'].unique())
        print(f"[DEBUG] AnnotatorConfusionCalculator initialized")
        print(f"[DEBUG] Annotators: {self.annotators}")
        print(f"[DEBUG] Total documents: {len(self.documents)}")

    def calculate_majority_vote(self, df_subset, value_column, exclude_annotator=None):
        """Calculate majority vote for each document, optionally excluding one annotator."""
        print(f"[DEBUG] Calculating majority vote for {value_column}, excluding: {exclude_annotator}")

        # Filter out excluded annotator if specified
        if exclude_annotator:
            analysis_df = df_subset[df_subset['annotator'] != exclude_annotator].copy()
        else:
            analysis_df = df_subset.copy()

        # Create pivot table
        pivot_data = analysis_df.pivot(index='id', columns='annotator', values=value_column)

        majority_votes = {}
        for doc_id, row in pivot_data.iterrows():
            valid_annotations = row.dropna()
            if len(valid_annotations) > 0:
                # Calculate majority vote
                vote_counts = Counter(valid_annotations)
                majority_label = vote_counts.most_common(1)[0][0]
                majority_count = vote_counts.most_common(1)[0][1]
                total_votes = len(valid_annotations)
                confidence = majority_count / total_votes

                majority_votes[doc_id] = {
                    'majority_label': majority_label,
                    'confidence': confidence,
                    'vote_counts': dict(vote_counts),
                    'total_annotators': total_votes
                }

        print(f"[DEBUG] Calculated majority votes for {len(majority_votes)} documents")
        return majority_votes

    def calculate_annotator_vs_majority_confusion(self, df_subset, value_column, target_annotator):
        """Calculate confusion matrix for specific annotator vs majority vote."""
        print(f"[DEBUG] Calculating confusion matrix for {target_annotator} vs majority")

        # Get majority votes excluding the target annotator
        majority_votes = self.calculate_majority_vote(df_subset, value_column, exclude_annotator=target_annotator)

        # Get target annotator's labels
        target_labels = df_subset[df_subset['annotator'] == target_annotator].set_index('id')[value_column]

        # Align data for comparison
        comparison_data = []
        for doc_id in majority_votes.keys():
            if doc_id in target_labels.index:
                comparison_data.append({
                    'document_id': doc_id,
                    'annotator_label': target_labels[doc_id],
                    'majority_label': majority_votes[doc_id]['majority_label'],
                    'majority_confidence': majority_votes[doc_id]['confidence'],
                    'agreement': target_labels[doc_id] == majority_votes[doc_id]['majority_label']
                })

        comparison_df = pd.DataFrame(comparison_data)

        if len(comparison_df) == 0:
            print(f"[WARNING] No comparison data for {target_annotator}")
            return None, None, None

        # Calculate metrics
        accuracy = comparison_df['agreement'].mean()

        # Create confusion matrix
        all_labels = sorted(set(comparison_df['annotator_label'].tolist() + comparison_df['majority_label'].tolist()))
        confusion_mat = confusion_matrix(
            comparison_df['majority_label'],
            comparison_df['annotator_label'],
            labels=all_labels
        )
        confusion_df = pd.DataFrame(confusion_mat, index=all_labels, columns=all_labels)

        # Calculate systematic biases (most common errors)
        errors = comparison_df[~comparison_df['agreement']]
        bias_patterns = Counter(zip(errors['majority_label'], errors['annotator_label']))

        print(f"[DEBUG] {target_annotator} accuracy: {accuracy:.3f}, total comparisons: {len(comparison_df)}")

        return confusion_df, accuracy, bias_patterns

    def calculate_all_annotator_performance(self, df_subset, value_column):
        """Calculate performance metrics for all annotators."""
        print(f"[DEBUG] Calculating performance for all annotators")

        performance_results = {}

        for annotator in self.annotators:
            if annotator not in df_subset['annotator'].values:
                continue

            confusion_df, accuracy, bias_patterns = self.calculate_annotator_vs_majority_confusion(
                df_subset, value_column, annotator
            )

            if confusion_df is not None:
                # Calculate additional metrics
                n_documents = len(df_subset[df_subset['annotator'] == annotator])

                # Most problematic labels (highest error rates)
                if len(bias_patterns) > 0:
                    top_biases = bias_patterns.most_common(3)
                else:
                    top_biases = []

                performance_results[annotator] = {
                    'accuracy': accuracy,
                    'confusion_matrix': confusion_df,
                    'bias_patterns': bias_patterns,
                    'top_biases': top_biases,
                    'n_documents': n_documents,
                    'total_errors': len(bias_patterns)
                }

        print(f"[DEBUG] Completed performance analysis for {len(performance_results)} annotators")
        return performance_results

    def calculate_pairwise_annotator_agreement(self, df_subset, value_column):
        """Calculate pairwise agreement between all annotator pairs."""
        print(f"[DEBUG] Calculating pairwise annotator agreement")

        # Create pivot table
        pivot_data = df_subset.pivot(index='id', columns='annotator', values=value_column)

        # Initialize pairwise agreement matrix
        pairwise_agreement = pd.DataFrame(index=self.annotators, columns=self.annotators, dtype=float)

        # Calculate pairwise agreements
        for i, ann1 in enumerate(self.annotators):
            for j, ann2 in enumerate(self.annotators):
                if ann1 not in pivot_data.columns or ann2 not in pivot_data.columns:
                    pairwise_agreement.loc[ann1, ann2] = np.nan
                    continue

                if i == j:
                    pairwise_agreement.loc[ann1, ann2] = 1.0
                else:
                    # Calculate agreement between two annotators
                    valid_pairs = ~(pivot_data[ann1].isna() | pivot_data[ann2].isna())
                    if valid_pairs.sum() > 0:
                        agreements = (pivot_data.loc[valid_pairs, ann1] == pivot_data.loc[valid_pairs, ann2])
                        agreement_rate = agreements.mean()
                        pairwise_agreement.loc[ann1, ann2] = agreement_rate
                    else:
                        pairwise_agreement.loc[ann1, ann2] = np.nan

        print(f"[DEBUG] Pairwise agreement matrix calculated")
        return pairwise_agreement

    def identify_systematic_biases(self, performance_results):
        """Identify systematic biases across all annotators."""
        print(f"[DEBUG] Identifying systematic biases")

        # Aggregate bias patterns across all annotators
        all_biases = Counter()
        annotator_specific_biases = {}

        for annotator, results in performance_results.items():
            bias_patterns = results['bias_patterns']
            all_biases.update(bias_patterns)

            # Identify annotator-specific systematic errors
            if len(bias_patterns) > 0:
                total_errors = sum(bias_patterns.values())
                systematic_biases = []

                for (true_label, predicted_label), count in bias_patterns.items():
                    error_rate = count / total_errors
                    if error_rate > 0.1:  # More than 10% of errors
                        systematic_biases.append({
                            'true_label': true_label,
                            'predicted_label': predicted_label,
                            'count': count,
                            'error_rate': error_rate
                        })

                annotator_specific_biases[annotator] = systematic_biases

        # Global bias patterns
        global_biases = all_biases.most_common(10)

        print(f"[DEBUG] Identified {len(global_biases)} global bias patterns")
        return global_biases, annotator_specific_biases

    def generate_training_recommendations(self, performance_results, annotator_specific_biases):
        """Generate personalized training recommendations for each annotator."""
        print(f"[DEBUG] Generating training recommendations")

        recommendations = {}

        for annotator, results in performance_results.items():
            annotator_recommendations = []
            accuracy = results['accuracy']
            biases = annotator_specific_biases.get(annotator, [])

            # Overall performance assessment
            if accuracy < 0.6:
                annotator_recommendations.append({
                    'priority': 'high',
                    'category': 'overall_performance',
                    'message': f"Low overall accuracy ({accuracy:.1%}). Requires comprehensive retraining on annotation guidelines."
                })
            elif accuracy < 0.8:
                annotator_recommendations.append({
                    'priority': 'medium',
                    'category': 'overall_performance',
                    'message': f"Moderate accuracy ({accuracy:.1%}). Focus on specific problem areas identified below."
                })

            # Specific bias recommendations
            for bias in biases:
                if bias['error_rate'] > 0.2:  # High error rate
                    annotator_recommendations.append({
                        'priority': 'high',
                        'category': 'systematic_bias',
                        'message': f"Frequently confuses '{bias['true_label']}' with '{bias['predicted_label']}' ({bias['error_rate']:.1%} of errors). Review boundary conditions between these categories."
                    })
                else:
                    annotator_recommendations.append({
                        'priority': 'low',
                        'category': 'systematic_bias',
                        'message': f"Occasional confusion between '{bias['true_label']}' and '{bias['predicted_label']}'. Minor guideline clarification needed."
                    })

            # Performance-based recommendations
            if len(biases) == 0 and accuracy > 0.8:
                annotator_recommendations.append({
                    'priority': 'low',
                    'category': 'performance_good',
                    'message': "Good performance with no systematic biases detected. Continue current practices."
                })

            recommendations[annotator] = annotator_recommendations

        print(f"[DEBUG] Generated recommendations for {len(recommendations)} annotators")
        return recommendations


