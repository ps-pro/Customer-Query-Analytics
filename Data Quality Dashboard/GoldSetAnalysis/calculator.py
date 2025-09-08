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



# Core Gold-Set Analysis Calculator
class GoldSetAnalysisCalculator:
    """Calculator for analyzing and recommending gold-set refresh strategies."""

    def __init__(self, agreement_df):
        """Initialize with agreement dataframe."""
        self.agreement_df = agreement_df
        self.annotators = sorted(agreement_df['annotator'].unique())
        self.documents = sorted(agreement_df['id'].unique())
        print(f"[DEBUG] GoldSetAnalysisCalculator initialized")
        print(f"[DEBUG] Total documents: {len(self.documents)}")
        print(f"[DEBUG] Total annotators: {len(self.annotators)}")

    def calculate_document_agreement_levels(self, df_subset, value_column):
        """Calculate agreement level for each document."""
        print(f"[DEBUG] Calculating document agreement levels for {value_column}")

        document_agreements = []

        # Create pivot table for analysis
        pivot_data = df_subset.pivot(index='id', columns='annotator', values=value_column)

        for doc_id, row in pivot_data.iterrows():
            valid_annotations = row.dropna()

            if len(valid_annotations) < 2:
                continue

            # Calculate agreement metrics
            unique_labels = set(valid_annotations)
            n_annotators = len(valid_annotations)
            n_unique_labels = len(unique_labels)

            # Agreement rate calculation
            if n_unique_labels == 1:
                agreement_rate = 1.0
                majority_label = list(unique_labels)[0]
                majority_count = n_annotators
            else:
                label_counts = Counter(valid_annotations)
                majority_label = label_counts.most_common(1)[0][0]
                majority_count = label_counts.most_common(1)[0][1]
                agreement_rate = majority_count / n_annotators

            # Get document text
            doc_text = df_subset[df_subset['id'] == doc_id]['text'].iloc[0] if 'text' in df_subset.columns else ""

            document_agreements.append({
                'document_id': doc_id,
                'agreement_rate': agreement_rate,
                'majority_label': majority_label,
                'majority_count': majority_count,
                'n_annotators': n_annotators,
                'n_unique_labels': n_unique_labels,
                'unique_labels': list(unique_labels),
                'text': doc_text,
                'text_length': len(doc_text),
                'word_count': len(doc_text.split()) if doc_text else 0
            })

        agreement_df = pd.DataFrame(document_agreements)
        print(f"[DEBUG] Calculated agreement for {len(agreement_df)} documents")

        return agreement_df

    def analyze_label_coverage(self, df_subset, value_column):
        """Analyze current label distribution and coverage."""
        print(f"[DEBUG] Analyzing label coverage for {value_column}")

        # Calculate label frequencies
        label_counts = Counter(df_subset[value_column])
        total_annotations = len(df_subset)

        coverage_data = []
        for label, count in label_counts.items():
            percentage = (count / total_annotations) * 100
            coverage_data.append({
                'label': label,
                'count': count,
                'percentage': percentage
            })

        coverage_df = pd.DataFrame(coverage_data).sort_values('count', ascending=False)

        # Identify under-represented labels (less than expected if uniform)
        n_unique_labels = len(label_counts)
        expected_percentage = 100 / n_unique_labels
        coverage_df['coverage_status'] = coverage_df['percentage'].apply(
            lambda x: 'over_represented' if x > expected_percentage * 1.5
            else 'under_represented' if x < expected_percentage * 0.5
            else 'adequate'
        )

        print(f"[DEBUG] Coverage analysis: {len(coverage_df)} labels")
        print(f"[DEBUG] Under-represented: {sum(coverage_df['coverage_status'] == 'under_represented')}")

        return coverage_df

    def select_high_confidence_candidates(self, agreement_df, coverage_df, min_agreement=0.9,
                                        samples_per_label=5):
        """Select high-confidence documents for gold-set inclusion."""
        print(f"[DEBUG] Selecting high-confidence candidates")
        print(f"[DEBUG] Min agreement: {min_agreement}, Samples per label: {samples_per_label}")

        # Filter for high-agreement documents
        high_confidence_docs = agreement_df[agreement_df['agreement_rate'] >= min_agreement].copy()

        if len(high_confidence_docs) == 0:
            print(f"[WARNING] No documents found with agreement >= {min_agreement}")
            return pd.DataFrame()

        # Stratified sampling by label to ensure coverage
        selected_candidates = []

        for _, label_info in coverage_df.iterrows():
            label = label_info['label']

            # Get high-confidence documents for this label
            label_docs = high_confidence_docs[
                high_confidence_docs['majority_label'] == label
            ].copy()

            if len(label_docs) == 0:
                print(f"[WARNING] No high-confidence documents for label: {label}")
                continue

            # Sort by agreement rate and text diversity (prefer varied text lengths)
            label_docs = label_docs.sort_values(['agreement_rate', 'text_length'],
                                               ascending=[False, True])

            # Select up to samples_per_label documents
            n_select = min(samples_per_label, len(label_docs))
            selected = label_docs.head(n_select)

            for _, doc in selected.iterrows():
                selected_candidates.append({
                    'document_id': doc['document_id'],
                    'label': label,
                    'agreement_rate': doc['agreement_rate'],
                    'selection_reason': 'high_confidence',
                    'priority': 'high',
                    'text_preview': doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text'],
                    'text_length': doc['text_length'],
                    'n_annotators': doc['n_annotators']
                })

        candidates_df = pd.DataFrame(selected_candidates)
        print(f"[DEBUG] Selected {len(candidates_df)} high-confidence candidates")

        return candidates_df

    def select_disagreement_candidates(self, agreement_df, coverage_df, min_disagreement=0.4,
                                     max_disagreement=0.7, samples_per_label=3):
        """Select useful disagreement cases for guideline development."""
        print(f"[DEBUG] Selecting disagreement candidates")
        print(f"[DEBUG] Disagreement range: {min_disagreement} - {max_disagreement}")

        # Filter for useful disagreement level
        disagreement_docs = agreement_df[
            (agreement_df['agreement_rate'] >= min_disagreement) &
            (agreement_df['agreement_rate'] <= max_disagreement)
        ].copy()

        if len(disagreement_docs) == 0:
            print(f"[WARNING] No documents in disagreement range {min_disagreement}-{max_disagreement}")
            return pd.DataFrame()

        selected_candidates = []

        for _, label_info in coverage_df.iterrows():
            label = label_info['label']

            # Get disagreement documents for this label (majority label)
            label_docs = disagreement_docs[
                disagreement_docs['majority_label'] == label
            ].copy()

            if len(label_docs) == 0:
                continue

            # Sort by agreement rate (prefer moderate disagreement) and text complexity
            label_docs = label_docs.sort_values(['agreement_rate', 'n_unique_labels'],
                                               ascending=[True, False])

            # Select up to samples_per_label documents
            n_select = min(samples_per_label, len(label_docs))
            selected = label_docs.head(n_select)

            for _, doc in selected.iterrows():
                selected_candidates.append({
                    'document_id': doc['document_id'],
                    'label': label,
                    'agreement_rate': doc['agreement_rate'],
                    'selection_reason': 'useful_disagreement',
                    'priority': 'medium',
                    'text_preview': doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text'],
                    'text_length': doc['text_length'],
                    'n_annotators': doc['n_annotators'],
                    'unique_labels': ', '.join(doc['unique_labels'])
                })

        candidates_df = pd.DataFrame(selected_candidates)
        print(f"[DEBUG] Selected {len(candidates_df)} disagreement candidates")

        return candidates_df

    def identify_coverage_gaps(self, coverage_df, high_confidence_candidates, disagreement_candidates):
        """Identify labels that need more representation in gold-set."""
        print(f"[DEBUG] Identifying coverage gaps")

        # Count current gold-set candidates by label
        all_candidates = pd.concat([high_confidence_candidates, disagreement_candidates], ignore_index=True)
        candidate_counts = Counter(all_candidates['label']) if len(all_candidates) > 0 else Counter()

        coverage_gaps = []

        for _, label_info in coverage_df.iterrows():
            label = label_info['label']
            current_percentage = label_info['percentage']
            coverage_status = label_info['coverage_status']
            candidates_selected = candidate_counts.get(label, 0)

            # Identify gaps
            needs_more = False
            gap_reason = ""

            if coverage_status == 'under_represented':
                needs_more = True
                gap_reason = f"Under-represented in dataset ({current_percentage:.1f}%)"

            if candidates_selected == 0:
                needs_more = True
                gap_reason += "; No gold-set candidates selected"
            elif candidates_selected < 5:  # CHANGED: Raised threshold from 3 to 5
                needs_more = True
                gap_reason += f"; Only {candidates_selected} candidates selected (recommended: 5+)"

            # ADD: More comprehensive gap detection
            if current_percentage < 5.0:  # Labels with less than 5% representation
                needs_more = True
                gap_reason += f"; Very low representation ({current_percentage:.1f}%)"

            # ADD: Ensure minimum coverage for quality
            if candidates_selected < 8 and coverage_status != 'over_represented':  # Want 8+ for non-over-represented
                needs_more = True
                gap_reason += f"; Insufficient gold-set coverage for quality assurance"

            if needs_more:
                coverage_gaps.append({
                    'label': label,
                    'current_percentage': current_percentage,
                    'coverage_status': coverage_status,
                    'candidates_selected': candidates_selected,
                    'gap_reason': gap_reason.strip('; '),
                    'priority': 'high' if coverage_status == 'under_represented' else 'medium'
                })

        gaps_df = pd.DataFrame(coverage_gaps)
        print(f"[DEBUG] Identified {len(gaps_df)} coverage gaps")

        return gaps_df

    def calculate_gold_set_quality_metrics(self, high_confidence_candidates, disagreement_candidates,
                                         coverage_df):
        """Calculate overall quality metrics for the proposed gold-set."""
        print(f"[DEBUG] Calculating gold-set quality metrics")

        all_candidates = pd.concat([high_confidence_candidates, disagreement_candidates], ignore_index=True)

        if len(all_candidates) == 0:
            return {
                'total_candidates': 0,
                'high_confidence_count': 0,
                'disagreement_count': 0,
                'label_coverage': 0,
                'avg_agreement_rate': 0,
                'quality_score': 0
            }

        # Basic counts
        total_candidates = len(all_candidates)
        high_confidence_count = len(high_confidence_candidates)
        disagreement_count = len(disagreement_candidates)

        # Coverage metrics
        unique_labels_covered = all_candidates['label'].nunique()
        total_labels = len(coverage_df)
        label_coverage_percentage = (unique_labels_covered / total_labels) * 100

        # Quality metrics
        avg_agreement_rate = all_candidates['agreement_rate'].mean()

        # Overall quality score (weighted combination)
        coverage_score = min(label_coverage_percentage / 100, 1.0)  # Cap at 100%
        confidence_score = high_confidence_count / max(total_candidates, 1)
        disagreement_score = min(disagreement_count / max(total_candidates, 1), 0.3)  # Cap contribution

        quality_score = (coverage_score * 0.4 + confidence_score * 0.4 + disagreement_score * 0.2)

        metrics = {
            'total_candidates': total_candidates,
            'high_confidence_count': high_confidence_count,
            'disagreement_count': disagreement_count,
            'unique_labels_covered': unique_labels_covered,
            'total_labels': total_labels,
            'label_coverage_percentage': label_coverage_percentage,
            'avg_agreement_rate': avg_agreement_rate,
            'quality_score': quality_score
        }

        print(f"[DEBUG] Quality metrics calculated: {quality_score:.3f}")
        return metrics

