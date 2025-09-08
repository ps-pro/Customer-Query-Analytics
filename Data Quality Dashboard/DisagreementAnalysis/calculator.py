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



# Core Disagreement Analysis Calculator
class DisagreementAnalysisCalculator:
    """Calculator for identifying and analyzing annotation disagreements."""

    def __init__(self, agreement_df):
        """Initialize with agreement dataframe."""
        self.agreement_df = agreement_df
        self.annotators = sorted(agreement_df['annotator'].unique())
        self.documents = sorted(agreement_df['id'].unique())
        print(f"[DEBUG] DisagreementAnalysisCalculator initialized")
        print(f"[DEBUG] Total documents: {len(self.documents)}")
        print(f"[DEBUG] Total annotators: {len(self.annotators)}")

    def calculate_document_disagreement_scores(self, df_subset, value_column):
        """Calculate disagreement metrics for each document."""
        print(f"[DEBUG] Calculating document disagreement scores for {value_column}")

        disagreement_results = []

        # Create pivot table for analysis
        pivot_data = df_subset.pivot(index='id', columns='annotator', values=value_column)

        for doc_id, row in pivot_data.iterrows():
            valid_annotations = row.dropna()

            if len(valid_annotations) < 2:
                continue

            # Calculate disagreement metrics
            unique_labels = set(valid_annotations)
            n_annotators = len(valid_annotations)
            n_unique_labels = len(unique_labels)

            # Perfect agreement: all annotators agree
            perfect_agreement = n_unique_labels == 1

            # Agreement rate: percentage of annotators who agree with majority
            if n_unique_labels > 1:
                label_counts = Counter(valid_annotations)
                majority_count = max(label_counts.values())
                agreement_rate = majority_count / n_annotators
            else:
                agreement_rate = 1.0

            # Disagreement score: 1 - agreement_rate (higher = more disagreement)
            disagreement_score = 1.0 - agreement_rate

            # Get document text for analysis
            doc_text = df_subset[df_subset['id'] == doc_id]['text'].iloc[0] if 'text' in df_subset.columns else ""

            disagreement_results.append({
                'document_id': doc_id,
                'disagreement_score': disagreement_score,
                'agreement_rate': agreement_rate,
                'n_annotators': n_annotators,
                'n_unique_labels': n_unique_labels,
                'unique_labels': list(unique_labels),
                'annotator_labels': dict(valid_annotations),
                'perfect_agreement': perfect_agreement,
                'text': doc_text,
                'text_length': len(doc_text),
                'word_count': len(doc_text.split()) if doc_text else 0
            })

        disagreement_df = pd.DataFrame(disagreement_results)
        disagreement_df = disagreement_df.sort_values('disagreement_score', ascending=False)

        print(f"[DEBUG] Processed {len(disagreement_df)} documents")
        print(f"[DEBUG] Disagreement score range: {disagreement_df['disagreement_score'].min():.3f} - {disagreement_df['disagreement_score'].max():.3f}")

        return disagreement_df

    def calculate_label_confusion_matrix(self, df_subset, value_column):
        """Calculate which label pairs are most commonly confused."""
        print(f"[DEBUG] Calculating label confusion matrix for {value_column}")

        confusion_pairs = []

        # Group by document to find disagreements
        for doc_id in df_subset['id'].unique():
            doc_annotations = df_subset[df_subset['id'] == doc_id][value_column].tolist()

            if len(set(doc_annotations)) > 1:  # Only if there's disagreement
                # Get all pairs of different labels for this document
                for label1, label2 in itertools.combinations(set(doc_annotations), 2):
                    confusion_pairs.append((label1, label2))
                    confusion_pairs.append((label2, label1))  # Both directions

        # Count confusion frequencies
        confusion_counts = Counter(confusion_pairs)

        # Convert to matrix format
        all_labels = sorted(df_subset[value_column].unique())
        confusion_matrix = pd.DataFrame(0, index=all_labels, columns=all_labels)

        for (label1, label2), count in confusion_counts.items():
            if label1 in all_labels and label2 in all_labels:
                confusion_matrix.loc[label1, label2] = count

        print(f"[DEBUG] Confusion matrix calculated for {len(all_labels)} labels")
        print(f"[DEBUG] Total confusion pairs: {len(confusion_pairs)}")

        return confusion_matrix, confusion_counts

    def analyze_disagreement_patterns(self, disagreement_df):
        """Analyze patterns in disagreement data."""
        print(f"[DEBUG] Analyzing disagreement patterns")

        patterns = {
            'total_documents': len(disagreement_df),
            'perfect_agreement_docs': len(disagreement_df[disagreement_df['perfect_agreement']]),
            'high_disagreement_docs': len(disagreement_df[disagreement_df['disagreement_score'] > 0.5]),
            'avg_disagreement_score': disagreement_df['disagreement_score'].mean(),
            'avg_unique_labels_per_doc': disagreement_df['n_unique_labels'].mean(),
            'text_length_vs_disagreement_corr': disagreement_df['text_length'].corr(disagreement_df['disagreement_score']),
            'word_count_vs_disagreement_corr': disagreement_df['word_count'].corr(disagreement_df['disagreement_score'])
        }

        # Most disagreeable document
        if len(disagreement_df) > 0:
            most_disagreeable = disagreement_df.iloc[0]
            patterns['most_disagreeable_doc'] = {
                'id': most_disagreeable['document_id'],
                'score': most_disagreeable['disagreement_score'],
                'labels': most_disagreeable['unique_labels'],
                'text_preview': most_disagreeable['text'][:200] + "..." if len(most_disagreeable['text']) > 200 else most_disagreeable['text']
            }

        print(f"[DEBUG] Pattern analysis completed")
        print(f"[DEBUG] Perfect agreement: {patterns['perfect_agreement_docs']}/{patterns['total_documents']} documents")

        return patterns

    def filter_by_disagreement_threshold(self, disagreement_df, min_disagreement, max_disagreement):
        """Filter documents by disagreement score range."""
        print(f"[DEBUG] Filtering by disagreement threshold: {min_disagreement:.3f} - {max_disagreement:.3f}")

        filtered_df = disagreement_df[
            (disagreement_df['disagreement_score'] >= min_disagreement) &
            (disagreement_df['disagreement_score'] <= max_disagreement)
        ].copy()

        print(f"[DEBUG] Filtered to {len(filtered_df)} documents")
        return filtered_df

    def get_top_disagreement_documents(self, disagreement_df, top_n=50):
        """Get top N most disagreeable documents with full details."""
        print(f"[DEBUG] Getting top {top_n} disagreement documents")

        top_docs = disagreement_df.head(top_n).copy()

        # Prepare detailed table data
        table_data = []
        for _, doc in top_docs.iterrows():
            # Format annotator labels for display
            annotator_labels_str = "; ".join([f"{ann}: {label}" for ann, label in doc['annotator_labels'].items()])

            table_data.append({
                'Rank': len(table_data) + 1,
                'Sample ID': doc['document_id'],
                'Disagreement Score': f"{doc['disagreement_score']:.3f}",
                'Agreement Rate': f"{doc['agreement_rate']:.1%}",
                'Unique Labels': doc['n_unique_labels'],
                'Annotator Labels': annotator_labels_str,
                'Text Preview': doc['text'][:150] + "..." if len(doc['text']) > 150 else doc['text'],
                'Text Length': doc['text_length'],
                'Word Count': doc['word_count']
            })

        return table_data
