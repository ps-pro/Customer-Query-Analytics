# Dashboard 2: Data Quality Dashboard - Complete app.py
# ================================================================

# PART 1: IMPORTS (COPY FROM YOUR ORIGINAL CODE LINES 1-15)
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
                'Document ID': doc['document_id'],
                'Disagreement Score': f"{doc['disagreement_score']:.3f}",
                'Agreement Rate': f"{doc['agreement_rate']:.1%}",
                'Unique Labels': doc['n_unique_labels'],
                'Annotator Labels': annotator_labels_str,
                'Text Preview': doc['text'][:150] + "..." if len(doc['text']) > 150 else doc['text'],
                'Text Length': doc['text_length'],
                'Word Count': doc['word_count']
            })

        return table_data


# Disagreement Analysis Visualizer
class DisagreementAnalysisVisualizer:
    """Visualizer for disagreement analysis."""

    @staticmethod
    def create_disagreement_distribution_histogram(disagreement_df):
        """Create histogram showing distribution of disagreement scores."""
        print(f"[DEBUG] Creating disagreement distribution histogram")

        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=disagreement_df['disagreement_score'],
            nbinsx=30,
            name="Document Count",
            opacity=0.7,
            marker_color='lightblue'
        ))

        # Add vertical lines for key thresholds
        fig.add_vline(x=0.0, line_dash="solid", line_color="green",
                     annotation_text="Perfect Agreement")
        fig.add_vline(x=0.5, line_dash="dash", line_color="orange",
                     annotation_text="High Disagreement")
        fig.add_vline(x=disagreement_df['disagreement_score'].mean(),
                     line_dash="dot", line_color="red",
                     annotation_text=f"Average: {disagreement_df['disagreement_score'].mean():.3f}")

        fig.update_layout(
            title="Distribution of Disagreement Scores Across All Documents",
            xaxis_title="Disagreement Score (0 = Perfect Agreement, 1 = Maximum Disagreement)",
            yaxis_title="Number of Documents",
            showlegend=False
        )

        return fig

    @staticmethod
    def create_confusion_matrix_heatmap(confusion_matrix, title="Label Confusion Matrix"):
        """Create heatmap showing which labels are most commonly confused."""
        print(f"[DEBUG] Creating confusion matrix heatmap")

        # Remove self-confusion (diagonal) for clarity
        confusion_display = confusion_matrix.copy()
        np.fill_diagonal(confusion_display.values, 0)

        fig = go.Figure(data=go.Heatmap(
            z=confusion_display.values,
            x=confusion_display.columns,
            y=confusion_display.index,
            colorscale='Reds',
            text=confusion_display.values,
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Confusion Count")
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Label (Confused With)",
            yaxis_title="Label (Original)",
            height=max(400, len(confusion_matrix) * 30)
        )

        return fig

    @staticmethod
    def create_text_complexity_vs_disagreement_scatter(disagreement_df):
        """Create scatter plot of text characteristics vs disagreement."""
        print(f"[DEBUG] Creating text complexity vs disagreement scatter plot")

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Text Length vs Disagreement", "Word Count vs Disagreement"]
        )

        # Text length scatter
        fig.add_trace(
            go.Scatter(
                x=disagreement_df['text_length'],
                y=disagreement_df['disagreement_score'],
                mode='markers',
                text=disagreement_df['document_id'],
                hovertemplate='<b>Doc: %{text}</b><br>Length: %{x}<br>Disagreement: %{y:.3f}<extra></extra>',
                marker=dict(size=6, opacity=0.6, color='blue'),
                name="Text Length"
            ),
            row=1, col=1
        )

        # Word count scatter
        fig.add_trace(
            go.Scatter(
                x=disagreement_df['word_count'],
                y=disagreement_df['disagreement_score'],
                mode='markers',
                text=disagreement_df['document_id'],
                hovertemplate='<b>Doc: %{text}</b><br>Words: %{x}<br>Disagreement: %{y:.3f}<extra></extra>',
                marker=dict(size=6, opacity=0.6, color='red'),
                name="Word Count"
            ),
            row=1, col=2
        )

        # Add trend lines if correlation exists
        if len(disagreement_df) > 1:
            # Text length trend
            text_corr = disagreement_df['text_length'].corr(disagreement_df['disagreement_score'])
            if abs(text_corr) > 0.1:
                z1 = np.polyfit(disagreement_df['text_length'], disagreement_df['disagreement_score'], 1)
                p1 = np.poly1d(z1)
                x_trend1 = np.linspace(disagreement_df['text_length'].min(), disagreement_df['text_length'].max(), 100)
                fig.add_trace(
                    go.Scatter(x=x_trend1, y=p1(x_trend1), mode='lines', name=f'Trend (r={text_corr:.3f})',
                              line=dict(dash='dash', color='blue')), row=1, col=1
                )

            # Word count trend
            word_corr = disagreement_df['word_count'].corr(disagreement_df['disagreement_score'])
            if abs(word_corr) > 0.1:
                z2 = np.polyfit(disagreement_df['word_count'], disagreement_df['disagreement_score'], 1)
                p2 = np.poly1d(z2)
                x_trend2 = np.linspace(disagreement_df['word_count'].min(), disagreement_df['word_count'].max(), 100)
                fig.add_trace(
                    go.Scatter(x=x_trend2, y=p2(x_trend2), mode='lines', name=f'Trend (r={word_corr:.3f})',
                              line=dict(dash='dash', color='red')), row=1, col=2
                )

        fig.update_layout(
            title="Text Characteristics vs Disagreement Analysis",
            showlegend=False
        )

        fig.update_yaxes(title_text="Disagreement Score", row=1, col=1)
        fig.update_yaxes(title_text="Disagreement Score", row=1, col=2)
        fig.update_xaxes(title_text="Text Length (characters)", row=1, col=1)
        fig.update_xaxes(title_text="Word Count", row=1, col=2)

        return fig

    @staticmethod
    def create_patterns_summary_cards(patterns):
        """Create summary cards showing disagreement patterns."""
        print(f"[DEBUG] Creating patterns summary cards")

        cards = [
            dbc.Card([
                dbc.CardBody([
                    html.H5("Total Documents", className="card-title"),
                    html.H3(f"{patterns['total_documents']:,}", className="text-primary"),
                    html.P(f"Perfect Agreement: {patterns['perfect_agreement_docs']:,}")
                ])
            ]),
            dbc.Card([
                dbc.CardBody([
                    html.H5("High Disagreement", className="card-title"),
                    html.H3(f"{patterns['high_disagreement_docs']:,}", className="text-warning"),
                    html.P(f"Score > 0.5: {patterns['high_disagreement_docs']/patterns['total_documents']:.1%}")
                ])
            ]),
            dbc.Card([
                dbc.CardBody([
                    html.H5("Average Disagreement", className="card-title"),
                    html.H3(f"{patterns['avg_disagreement_score']:.3f}", className="text-info"),
                    html.P(f"Labels per doc: {patterns['avg_unique_labels_per_doc']:.1f}")
                ])
            ])
        ]

        return dbc.Row([dbc.Col(card, md=4) for card in cards])

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

# Annotator Confusion Visualizer
class AnnotatorConfusionVisualizer:
    """Visualizer for annotator confusion analysis."""

    @staticmethod
    def create_individual_confusion_matrix(confusion_df, annotator_name, accuracy):
        """Create confusion matrix heatmap for individual annotator."""
        print(f"[DEBUG] Creating confusion matrix for {annotator_name}")

        fig = go.Figure(data=go.Heatmap(
            z=confusion_df.values,
            x=confusion_df.columns,
            y=confusion_df.index,
            colorscale='Blues',
            text=confusion_df.values,
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Count")
        ))

        fig.update_layout(
            title=f"{annotator_name} vs Majority Vote<br>Accuracy: {accuracy:.1%}",
            xaxis_title="Annotator Label",
            yaxis_title="Majority Vote (True)",
            height=400,
            width=500
        )

        return fig

    @staticmethod
    def create_annotator_performance_ranking(performance_results):
        """Create bar chart ranking annotator performance."""
        print(f"[DEBUG] Creating annotator performance ranking")

        annotators = list(performance_results.keys())
        accuracies = [performance_results[ann]['accuracy'] for ann in annotators]
        error_counts = [performance_results[ann]['total_errors'] for ann in annotators]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Accuracy vs Majority Vote", "Total Error Count"],
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Accuracy chart
        colors = ['green' if acc > 0.8 else 'orange' if acc > 0.6 else 'red' for acc in accuracies]
        fig.add_trace(
            go.Bar(x=annotators, y=accuracies, name="Accuracy",
                  text=[f'{acc:.1%}' for acc in accuracies],
                  textposition='outside', marker_color=colors),
            row=1, col=1
        )

        # Error count chart
        fig.add_trace(
            go.Bar(x=annotators, y=error_counts, name="Error Count",
                  text=error_counts, textposition='outside', marker_color='lightcoral'),
            row=1, col=2
        )

        # Add benchmark lines
        fig.add_hline(y=0.8, line_dash="dash", line_color="green", row=1, col=1,
                     annotation_text="Good Performance")
        fig.add_hline(y=0.6, line_dash="dash", line_color="orange", row=1, col=1,
                     annotation_text="Needs Improvement")

        fig.update_layout(
            title="Annotator Performance Analysis",
            showlegend=False
        )

        fig.update_yaxes(title_text="Accuracy", range=[0, 1], row=1, col=1)
        fig.update_yaxes(title_text="Error Count", row=1, col=2)

        return fig

    @staticmethod
    def create_pairwise_agreement_heatmap(pairwise_agreement):
        """Create heatmap showing pairwise agreement between annotators."""
        print(f"[DEBUG] Creating pairwise agreement heatmap")

        fig = go.Figure(data=go.Heatmap(
            z=pairwise_agreement.values,
            x=pairwise_agreement.columns,
            y=pairwise_agreement.index,
            colorscale='RdYlBu',
            zmin=0,
            zmax=1,
            text=np.round(pairwise_agreement.values, 3),
            texttemplate='%{text}',
            textfont={"size": 12},
            colorbar=dict(title="Agreement Rate")
        ))

        fig.update_layout(
            title="Pairwise Annotator Agreement Matrix",
            xaxis_title="Annotator",
            yaxis_title="Annotator",
            height=500,
            width=600
        )

        return fig

    @staticmethod
    def create_bias_pattern_visualization(global_biases, title="Top Systematic Bias Patterns"):
        """Create visualization of systematic bias patterns."""
        print(f"[DEBUG] Creating bias pattern visualization")

        if not global_biases:
            fig = go.Figure()
            fig.add_annotation(text="No systematic biases detected",
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig

        # Prepare data
        bias_labels = [f"{true_label} → {pred_label}" for (true_label, pred_label), _ in global_biases]
        bias_counts = [count for _, count in global_biases]

        fig = go.Figure(data=go.Bar(
            x=bias_counts,
            y=bias_labels,
            orientation='h',
            text=bias_counts,
            textposition='outside'
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Error Frequency",
            yaxis_title="Confusion Pattern (True → Predicted)",
            height=max(400, len(global_biases) * 30)
        )

        return fig

    @staticmethod
    def create_training_recommendations_display(recommendations):
        """Create display for training recommendations."""
        print(f"[DEBUG] Creating training recommendations display")

        recommendation_cards = []

        for annotator, recs in recommendations.items():
            # Color based on overall priority
            high_priority_count = sum(1 for rec in recs if rec['priority'] == 'high')
            if high_priority_count > 0:
                card_color = "danger"
            elif any(rec['priority'] == 'medium' for rec in recs):
                card_color = "warning"
            else:
                card_color = "success"

            # Create recommendation list
            rec_items = []
            for rec in recs:
                priority_badge = dbc.Badge(rec['priority'].upper(),
                                         color={"high": "danger", "medium": "warning", "low": "info"}[rec['priority']])
                rec_items.append(
                    html.Li([priority_badge, " ", rec['message']], className="mb-2")
                )

            card = dbc.Card([
                dbc.CardHeader(html.H6(annotator)),
                dbc.CardBody([
                    html.Ul(rec_items, className="mb-0")
                ])
            ], color=card_color, outline=True, className="mb-3")

            recommendation_cards.append(dbc.Col(card, md=6))

        return dbc.Row(recommendation_cards)
    
    

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
            elif candidates_selected < 3:
                needs_more = True
                gap_reason += f"; Only {candidates_selected} candidates selected"

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

# Gold-Set Analysis Visualizer
class GoldSetAnalysisVisualizer:
    """Visualizer for gold-set analysis."""

    @staticmethod
    def create_coverage_comparison_chart(coverage_df, quality_metrics):
        """Create chart comparing current vs ideal label coverage."""
        print(f"[DEBUG] Creating coverage comparison chart")

        labels = coverage_df['label'].tolist()
        current_percentages = coverage_df['percentage'].tolist()

        # Calculate ideal (uniform) distribution
        ideal_percentage = 100 / len(labels)
        ideal_percentages = [ideal_percentage] * len(labels)

        fig = go.Figure()

        # Current distribution
        fig.add_trace(go.Bar(
            name='Current Distribution',
            x=labels,
            y=current_percentages,
            text=[f'{p:.1f}%' for p in current_percentages],
            textposition='outside'
        ))

        # Ideal distribution line
        fig.add_trace(go.Scatter(
            name='Ideal (Uniform)',
            x=labels,
            y=ideal_percentages,
            mode='lines+markers',
            line=dict(dash='dash', color='red'),
            marker=dict(color='red')
        ))

        fig.update_layout(
            title=f"Label Distribution Analysis<br>Coverage: {quality_metrics['unique_labels_covered']}/{quality_metrics['total_labels']} labels ({quality_metrics['label_coverage_percentage']:.1f}%)",
            xaxis_title="Labels",
            yaxis_title="Percentage of Dataset",
            barmode='group',
            height=500
        )

        return fig

    @staticmethod
    def create_candidate_distribution_chart(high_confidence_candidates, disagreement_candidates):
        """Create chart showing distribution of selected candidates."""
        print(f"[DEBUG] Creating candidate distribution chart")

        all_candidates = pd.concat([high_confidence_candidates, disagreement_candidates], ignore_index=True)

        if len(all_candidates) == 0:
            fig = go.Figure()
            fig.add_annotation(text="No candidates selected",
                             xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig

        # Count by label and type
        candidate_counts = all_candidates.groupby(['label', 'selection_reason']).size().reset_index(name='count')

        fig = px.bar(
            candidate_counts,
            x='label',
            y='count',
            color='selection_reason',
            title="Selected Gold-Set Candidates by Label and Type",
            labels={'count': 'Number of Candidates', 'label': 'Label'},
            color_discrete_map={
                'high_confidence': 'green',
                'useful_disagreement': 'orange'
            }
        )

        fig.update_layout(height=400)
        return fig

    @staticmethod
    def create_agreement_distribution_scatter(agreement_df, min_agreement, max_disagreement):
        """Create scatter plot showing agreement distribution with selection boundaries."""
        print(f"[DEBUG] Creating agreement distribution scatter")

        fig = go.Figure()

        # All documents
        fig.add_trace(go.Scatter(
            x=agreement_df['text_length'],
            y=agreement_df['agreement_rate'],
            mode='markers',
            text=agreement_df['document_id'],
            hovertemplate='<b>Doc: %{text}</b><br>Length: %{x}<br>Agreement: %{y:.3f}<extra></extra>',
            marker=dict(size=6, opacity=0.6, color='lightblue'),
            name="All Documents"
        ))

        # Selection boundaries
        fig.add_hline(y=min_agreement, line_dash="dash", line_color="green",
                     annotation_text=f"High Confidence Threshold: {min_agreement}")
        fig.add_hline(y=max_disagreement, line_dash="dash", line_color="orange",
                     annotation_text=f"Max Useful Disagreement: {max_disagreement}")

        # Highlight selection regions
        fig.add_hrect(y0=min_agreement, y1=1.0, fillcolor="green", opacity=0.1,
                     annotation_text="High Confidence Zone", annotation_position="top left")
        fig.add_hrect(y0=0.4, y1=max_disagreement, fillcolor="orange", opacity=0.1,
                     annotation_text="Useful Disagreement Zone", annotation_position="bottom left")

        fig.update_layout(
            title="Document Agreement Distribution with Selection Criteria",
            xaxis_title="Text Length (characters)",
            yaxis_title="Agreement Rate",
            showlegend=True
        )

        return fig

    @staticmethod
    def create_quality_metrics_dashboard(quality_metrics):
        """Create dashboard showing gold-set quality metrics."""
        print(f"[DEBUG] Creating quality metrics dashboard")

        # Quality score color
        score = quality_metrics['quality_score']
        if score >= 0.8:
            score_color = "success"
        elif score >= 0.6:
            score_color = "warning"
        else:
            score_color = "danger"

        cards = [
            dbc.Card([
                dbc.CardBody([
                    html.H5("Total Candidates", className="card-title"),
                    html.H3(f"{quality_metrics['total_candidates']:,}", className="text-primary"),
                    html.P(f"High Conf: {quality_metrics['high_confidence_count']}, Disagreement: {quality_metrics['disagreement_count']}")
                ])
            ]),
            dbc.Card([
                dbc.CardBody([
                    html.H5("Label Coverage", className="card-title"),
                    html.H3(f"{quality_metrics['label_coverage_percentage']:.0f}%", className="text-info"),
                    html.P(f"{quality_metrics['unique_labels_covered']}/{quality_metrics['total_labels']} labels covered")
                ])
            ]),
            dbc.Card([
                dbc.CardBody([
                    html.H5("Quality Score", className="card-title"),
                    html.H3(f"{score:.2f}", className=f"text-{score_color}"),
                    html.P(f"Avg Agreement: {quality_metrics['avg_agreement_rate']:.1%}")
                ])
            ])
        ]

        return dbc.Row([dbc.Col(card, md=4) for card in cards])

# PART 8: DATA LOADING AND INITIALIZATION
# Load data
agreement_df = pd.read_csv('data.csv')
print(f"[INFO] Loaded {len(agreement_df)} annotations from data.csv")

# Initialize calculators
disagreement_calculator = DisagreementAnalysisCalculator(agreement_df)
disagreement_visualizer = DisagreementAnalysisVisualizer()

confusion_calculator = AnnotatorConfusionCalculator(agreement_df)
confusion_visualizer = AnnotatorConfusionVisualizer()

goldset_calculator = GoldSetAnalysisCalculator(agreement_df)
goldset_visualizer = GoldSetAnalysisVisualizer()

# PART 9: DASH APP INITIALIZATION
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# CRITICAL FOR PLOTLY CLOUD DEPLOYMENT
server = app.server


# ========================================
# STYLING CONSTANTS
# ========================================

# Add this right after app initialization
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
            }
            
            /* Dark Theme */
            .theme-dark {
                --bg-primary: #1a1a1a;
                --bg-secondary: #2d2d2d;
                --text-primary: #ffffff;
                --text-secondary: #adb5bd;
                --border-color: #495057;
                --card-bg: #2d2d2d;
                --shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.3);
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
            
            /* Card styling */
            .card {
                background-color: var(--card-bg) !important;
                border: 1px solid var(--border-color) !important;
                box-shadow: var(--shadow) !important;
                border-radius: 0.5rem !important;
            }
            
            .card-header {
                background-color: var(--bg-secondary) !important;
                border-bottom: 1px solid var(--border-color) !important;
                color: var(--text-primary) !important;
            }
            
            .card-body {
                color: var(--text-primary) !important;
            }

            /* Enhanced form controls */
            .Select-control, .dropdown {
                border-radius: 10px !important;
                border: 2px solid rgba(102, 126, 234, 0.2) !important;
                box-shadow: 0 2px 8px rgba(102, 126, 234, 0.1) !important;
                transition: all 0.3s ease !important;
            }

            .Select-control:hover, .dropdown:hover {
                border-color: rgba(102, 126, 234, 0.4) !important;
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2) !important;
            }

            /* Slider styling */
            .rc-slider-track {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                height: 6px !important;
            }

            .rc-slider-handle {
                border: 3px solid #667eea !important;
                background: white !important;
                width: 20px !important;
                height: 20px !important;
                margin-top: -7px !important;
                box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3) !important;
            }

            .rc-slider-rail {
                background: rgba(102, 126, 234, 0.2) !important;
                height: 6px !important;
            }

            /* Checkbox styling */
            input[type="checkbox"] {
                accent-color: #667eea !important;
                width: 18px !important;
                height: 18px !important;
            }

            /* Button hover effect */
            .btn:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
            }            
            /* Tab styling */
            .nav-tabs {
                border-bottom: 1px solid var(--border-color) !important;
            }

            .nav-tabs .nav-link {
                color: var(--text-secondary) !important;
                border: none !important;
                background: transparent !important;
                font-weight: 600 !important;
                fontSize: 1.1rem !important;
            }

            .nav-tabs .nav-link.active {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                -webkit-background-clip: text !important;
                -webkit-text-fill-color: transparent !important;
                background-clip: text !important;
                border: none !important;
                border-bottom: 3px solid transparent !important;
                background-image: linear-gradient(135deg, #667eea 0%, #764ba2 100%), linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                background-size: 100% 100%, 100% 3px !important;
                background-position: 0 0, 0 100% !important;
                background-repeat: no-repeat !important;
                font-weight: 700 !important;
            }
            /* Button styling */
            .btn {
                border-radius: 0.375rem !important;
                font-weight: 500 !important;
                transition: all 0.2s ease !important;
            }
            
            .btn-outline-secondary {
                color: var(--text-secondary) !important;
                border-color: var(--border-color) !important;
            }
            
            .btn-outline-secondary:hover {
                background-color: var(--bg-secondary) !important;
                color: var(--text-primary) !important;
            }
            
            /* Plot container styling */
            .js-plotly-plot {
                border-radius: 0.5rem !important;
                box-shadow: var(--shadow) !important;
            }
            
            /* Table styling */
            .dash-table-container {
                background-color: var(--card-bg) !important;
                border-radius: 0.5rem !important;
                overflow: hidden !important;
            }
            
            /* Progress bar */
            .progress {
                background-color: var(--bg-secondary) !important;
                border-radius: 0.5rem !important;
            }
            
            /* Input styling */
            .form-control, .form-select {
                background-color: var(--card-bg) !important;
                border: 1px solid var(--border-color) !important;
                color: var(--text-primary) !important;
            }
            
            .form-control:focus, .form-select:focus {
                background-color: var(--card-bg) !important;
                border-color: #007bff !important;
                color: var(--text-primary) !important;
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

# PART 10: APP LAYOUT
app.layout = dbc.Container([
    # Store for theme state
    dcc.Store(id='theme-store', data='light'),
    
    # Header Section with Theme Toggle
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div([
                    html.H1("Data Quality Dashboard", 
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
                    html.P("Comprehensive analysis of annotation quality, disagreements, and gold-set recommendations",
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
                        dbc.Button("Light", id="light-theme-btn", size="sm", outline=True),
                        dbc.Button("Dark", id="dark-theme-btn", size="sm", outline=True)
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
    
    # Navigation Tabs - Redesigned
    dbc.Row([
        dbc.Col([
            html.Div([
                dbc.Tabs([
                    dbc.Tab(
                        label="Top Disagreement Items", 
                        tab_id="disagreement-tab",
                        tab_style={'padding': '1rem 2rem', 'border': 'none'},
                        active_tab_style={'border': 'none', 'borderBottom': '3px solid #007bff'}
                    ),
                    dbc.Tab(
                        label="Per-Annotator Confusion", 
                        tab_id="confusion-tab",
                        tab_style={'padding': '1rem 2rem', 'border': 'none'},
                        active_tab_style={'border': 'none', 'borderBottom': '3px solid #007bff'}
                    ),
                    dbc.Tab(
                        label="Suggested Gold-Set Refresh", 
                        tab_id="goldset-tab",
                        tab_style={'padding': '1rem 2rem', 'border': 'none'},
                        active_tab_style={'border': 'none', 'borderBottom': '3px solid #007bff'}
                    )
                ], 
                id="main-tabs", 
                active_tab="disagreement-tab",
                style={'borderBottom': '1px solid #dee2e6', 'marginBottom': '0'})
            ], style={'display': 'flex', 'justifyContent': 'center'})
        ])
    ], style={'marginTop': '1rem'}),
    
    # Main Content Area - Full Width
    dbc.Row([
        dbc.Col([
            html.Div(
                id="tab-content",
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

# PART 11: TAB CONTENT CALLBACK
@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "active_tab")
)
def render_tab_content(active_tab):
    """Render content based on selected tab."""
    
    if active_tab == "disagreement-tab":
        return create_disagreement_tab()
    elif active_tab == "confusion-tab":
        return create_confusion_tab()
    elif active_tab == "goldset-tab":
        return create_goldset_tab()
    else:
        return html.Div("Tab content not found")
    


def create_disagreement_tab():
    """Create the disagreement analysis tab."""
    return html.Div([
        html.H4("Top Disagreement Items - Document Analysis", 
                style={
                    'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    'WebkitBackgroundClip': 'text',
                    'WebkitTextFillColor': 'transparent',
                    'backgroundClip': 'text',
                    'fontWeight': '700',
                    'fontSize': '2rem',
                    'marginBottom': '2rem',
                    'textAlign': 'center'
                }),

        # Controls Section - Redesigned
        dbc.Card([
            dbc.CardHeader([
                html.H5("Disagreement Analysis Configuration", 
                       style={
                           'margin': 0,
                           'fontWeight': '600',
                           'color': '#667eea'
                       })
            ], style={'backgroundColor': 'rgba(102, 126, 234, 0.1)', 'border': 'none'}),
            dbc.CardBody([
                # Row 1: Main Controls
                dbc.Row([
                    dbc.Col([
                        html.Label("Label Type:", 
                                 style={'fontWeight': '600', 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}),
                        dcc.Dropdown(
                            id='disagreement-label-type-selector',
                            options=[
                                {'label': 'Full Hierarchical Labels', 'value': 'full_label'},
                                {'label': 'L1 (Parent) Labels', 'value': 'L1_label'},
                                {'label': 'L2 (Child) Labels', 'value': 'L2_label'}
                            ],
                            value='full_label',
                            style={
                                'fontSize': '1rem',
                                'minHeight': '45px'
                            }
                        )
                    ], md=3),
                    dbc.Col([
                        html.Label("Top N Documents:", 
                                 style={'fontWeight': '600', 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}),
                        dcc.Dropdown(
                            id='top-n-selector',
                            options=[
                                {'label': 'Top 10', 'value': 10},
                                {'label': 'Top 25', 'value': 25},
                                {'label': 'Top 50', 'value': 50},
                                {'label': 'Top 100', 'value': 100}
                            ],
                            value=25,
                            style={
                                'fontSize': '1rem',
                                'minHeight': '45px'
                            }
                        )
                    ], md=3),
                    dbc.Col([
                        html.Label("Min Disagreement Score:", 
                                 style={'fontWeight': '600', 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}),
                        html.Div([
                            dcc.Slider(
                                id='min-disagreement-slider',
                                min=0.0,
                                max=1.0,
                                step=0.1,
                                value=0.0,
                                marks={i/10: {'label': f'{i/10:.1f}', 'style': {'fontSize': '12px'}} for i in range(0, 11, 2)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], style={'padding': '0 15px'})
                    ], md=3),
                    dbc.Col([
                        html.Label("Max Disagreement Score:", 
                                 style={'fontWeight': '600', 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}),
                        html.Div([
                            dcc.Slider(
                                id='max-disagreement-slider',
                                min=0.0,
                                max=1.0,
                                step=0.1,
                                value=1.0,
                                marks={i/10: {'label': f'{i/10:.1f}', 'style': {'fontSize': '12px'}} for i in range(0, 11, 2)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], style={'padding': '0 15px'})
                    ], md=3)
                ], className="mb-4"),
                
                # Row 2: Annotators and Calculate Button
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Annotators:", 
                                 style={'fontWeight': '600', 'marginBottom': '1rem', 'fontSize': '1.1rem'}),
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Checklist(
                                    id='disagreement-annotator-selector',
                                    options=[{'label': ann, 'value': ann} for ann in disagreement_calculator.annotators],
                                    value=disagreement_calculator.annotators,
                                    inline=True,
                                    style={'fontSize': '1rem'},
                                    inputStyle={'marginRight': '8px', 'transform': 'scale(1.2)'},
                                    labelStyle={'marginRight': '20px', 'marginBottom': '10px'}
                                )
                            ], style={'padding': '1.5rem'})
                        ], style={'backgroundColor': 'rgba(102, 126, 234, 0.05)', 'border': '1px solid rgba(102, 126, 234, 0.2)'})
                    ], md=9),
                    dbc.Col([
                        html.Div(style={'height': '3rem'}),  # Spacer
                        dbc.Button("Calculate Disagreement Analysis", 
                                 id="disagreement-calculate-btn",
                                 size="lg", 
                                 style={
                                     'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                                     'border': 'none',
                                     'borderRadius': '12px',
                                     'fontWeight': '600',
                                     'fontSize': '1.1rem',
                                     'padding': '12px 30px',
                                     'boxShadow': '0 4px 15px rgba(102, 126, 234, 0.3)',
                                     'transition': 'all 0.3s ease',
                                     'width': '100%',
                                     'minHeight': '50px'
                                 })
                    ], md=3)
                ])
            ], style={'padding': '2rem'})
        ], style={
            'border': '1px solid rgba(102, 126, 234, 0.3)',
            'borderRadius': '15px',
            'boxShadow': '0 8px 25px rgba(102, 126, 234, 0.15)',
            'marginBottom': '3rem'
        }),

        # Progress indicator
        dbc.Row([
            dbc.Col([
                dbc.Progress(id="disagreement-calculation-progress", 
                           value=0, 
                           style={"visibility": "hidden", "height": "8px", "borderRadius": "10px"},
                           color="info")
            ])
        ], className="mb-4"),

        # Results Section
        html.Div(
            id="disagreement-results-container",
            style={
                'minHeight': '200px',
                'padding': '2rem 0'
            }
        )
    ], style={'padding': '0 1rem'})

def create_confusion_tab():
    """Create the annotator confusion analysis tab."""
    return html.Div([
        html.H4("Per-Annotator Confusion Analysis", 
                style={
                    'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    'WebkitBackgroundClip': 'text',
                    'WebkitTextFillColor': 'transparent',
                    'backgroundClip': 'text',
                    'fontWeight': '700',
                    'fontSize': '2rem',
                    'marginBottom': '2rem',
                    'textAlign': 'center'
                }),

        # Controls Section - Redesigned
        dbc.Card([
            dbc.CardHeader([
                html.H5("Annotator Confusion Analysis Configuration", 
                       style={
                           'margin': 0,
                           'fontWeight': '600',
                           'color': '#667eea'
                       })
            ], style={'backgroundColor': 'rgba(102, 126, 234, 0.1)', 'border': 'none'}),
            dbc.CardBody([
                # Row 1: Main Controls
                dbc.Row([
                    dbc.Col([
                        html.Label("Analysis Mode:", 
                                 style={'fontWeight': '600', 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}),
                        dcc.Dropdown(
                            id='confusion-analysis-mode',
                            options=[
                                {'label': 'Complete Analysis', 'value': 'complete'},
                                {'label': 'Individual Performance Only', 'value': 'individual'},
                                {'label': 'Pairwise Comparison Only', 'value': 'pairwise'}
                            ],
                            value='complete',
                            style={
                                'fontSize': '1rem',
                                'minHeight': '45px'
                            }
                        )
                    ], md=4),
                    dbc.Col([
                        html.Label("Label Type:", 
                                 style={'fontWeight': '600', 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}),
                        dcc.Dropdown(
                            id='confusion-label-type-selector',
                            options=[
                                {'label': 'Full Hierarchical Labels', 'value': 'full_label'},
                                {'label': 'L1 (Parent) Labels', 'value': 'L1_label'},
                                {'label': 'L2 (Child) Labels', 'value': 'L2_label'}
                            ],
                            value='full_label',
                            style={
                                'fontSize': '1rem',
                                'minHeight': '45px'
                            }
                        )
                    ], md=4),
                    dbc.Col([
                        html.Label("Focus Annotator (Optional):", 
                                 style={'fontWeight': '600', 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}),
                        dcc.Dropdown(
                            id='focus-annotator-selector',
                            options=[{'label': 'All Annotators', 'value': 'all'}] +
                                    [{'label': ann, 'value': ann} for ann in confusion_calculator.annotators],
                            value='all',
                            style={
                                'fontSize': '1rem',
                                'minHeight': '45px'
                            }
                        )
                    ], md=4)
                ], className="mb-4"),
                
                # Row 2: Annotators and Calculate Button
                dbc.Row([
                    dbc.Col([
                        html.Label("Include Annotators:", 
                                 style={'fontWeight': '600', 'marginBottom': '1rem', 'fontSize': '1.1rem'}),
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Checklist(
                                    id='confusion-annotator-selector',
                                    options=[{'label': ann, 'value': ann} for ann in confusion_calculator.annotators],
                                    value=confusion_calculator.annotators,
                                    inline=True,
                                    style={'fontSize': '1rem'},
                                    inputStyle={'marginRight': '8px', 'transform': 'scale(1.2)'},
                                    labelStyle={'marginRight': '20px', 'marginBottom': '10px'}
                                )
                            ], style={'padding': '1.5rem'})
                        ], style={'backgroundColor': 'rgba(102, 126, 234, 0.05)', 'border': '1px solid rgba(102, 126, 234, 0.2)'})
                    ], md=9),
                    dbc.Col([
                        html.Div(style={'height': '3rem'}),  # Spacer
                        dbc.Button("Calculate Confusion Analysis", 
                                 id="confusion-calculate-btn",
                                 size="lg", 
                                 style={
                                     'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                                     'border': 'none',
                                     'borderRadius': '12px',
                                     'fontWeight': '600',
                                     'fontSize': '1.1rem',
                                     'padding': '12px 30px',
                                     'boxShadow': '0 4px 15px rgba(102, 126, 234, 0.3)',
                                     'transition': 'all 0.3s ease',
                                     'width': '100%',
                                     'minHeight': '50px'
                                 })
                    ], md=3)
                ])
            ], style={'padding': '2rem'})
        ], style={
            'border': '1px solid rgba(102, 126, 234, 0.3)',
            'borderRadius': '15px',
            'boxShadow': '0 8px 25px rgba(102, 126, 234, 0.15)',
            'marginBottom': '3rem'
        }),

        # Analysis Mode Explanation Card
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.H6("Analysis Mode Guide", 
                           style={'fontWeight': '600', 'color': '#667eea', 'marginBottom': '1rem'}),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Span("Complete Analysis", style={'fontWeight': '600', 'color': '#667eea'}),
                                html.P("Performs both individual performance and pairwise comparison for comprehensive insights", 
                                      style={'color': 'var(--text-secondary)', 'fontSize': '0.9rem', 'margin': '0.25rem 0 0 0'})
                            ])
                        ], md=4),
                        dbc.Col([
                            html.Div([
                                html.Span("Individual Performance", style={'fontWeight': '600', 'color': '#f39c12'}),
                                html.P("Analyzes each annotator's performance against majority vote", 
                                      style={'color': 'var(--text-secondary)', 'fontSize': '0.9rem', 'margin': '0.25rem 0 0 0'})
                            ])
                        ], md=4),
                        dbc.Col([
                            html.Div([
                                html.Span("Pairwise Comparison", style={'fontWeight': '600', 'color': '#27ae60'}),
                                html.P("Compares agreement rates between each pair of annotators", 
                                      style={'color': 'var(--text-secondary)', 'fontSize': '0.9rem', 'margin': '0.25rem 0 0 0'})
                            ])
                        ], md=4)
                    ])
                ])
            ], style={'padding': '1.5rem'})
        ], style={
            'backgroundColor': 'rgba(102, 126, 234, 0.05)',
            'border': '1px solid rgba(102, 126, 234, 0.2)',
            'borderRadius': '12px',
            'marginBottom': '3rem'
        }),

        # Progress indicator
        dbc.Row([
            dbc.Col([
                dbc.Progress(id="confusion-calculation-progress", 
                           value=0, 
                           style={"visibility": "hidden", "height": "8px", "borderRadius": "10px"},
                           color="info")
            ])
        ], className="mb-4"),

        # Results Section
        html.Div(
            id="confusion-results-container",
            style={
                'minHeight': '200px',
                'padding': '2rem 0'
            }
        )
    ], style={'padding': '0 1rem'})
def create_goldset_tab():
    """Create the gold-set refresh analysis tab."""
    return html.Div([
        html.H4("Suggested Gold-Set Refresh Analysis", 
                style={
                    'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                    'WebkitBackgroundClip': 'text',
                    'WebkitTextFillColor': 'transparent',
                    'backgroundClip': 'text',
                    'fontWeight': '700',
                    'fontSize': '2rem',
                    'marginBottom': '2rem',
                    'textAlign': 'center'
                }),

        # Controls Section - Redesigned
        dbc.Card([
            dbc.CardHeader([
                html.H5("Gold-Set Strategy Configuration", 
                       style={
                           'margin': 0,
                           'fontWeight': '600',
                           'color': '#667eea'
                       })
            ], style={'backgroundColor': 'rgba(102, 126, 234, 0.1)', 'border': 'none'}),
            dbc.CardBody([
                # Row 1: Main Strategy Controls
                dbc.Row([
                    dbc.Col([
                        html.Label("Label Type:", 
                                 style={'fontWeight': '600', 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}),
                        dcc.Dropdown(
                            id='goldset-label-type-selector',
                            options=[
                                {'label': 'Full Hierarchical Labels', 'value': 'full_label'},
                                {'label': 'L1 (Parent) Labels', 'value': 'L1_label'},
                                {'label': 'L2 (Child) Labels', 'value': 'L2_label'}
                            ],
                            value='full_label',
                            style={
                                'fontSize': '1rem',
                                'minHeight': '45px'
                            }
                        )
                    ], md=3),
                    dbc.Col([
                        html.Label("Gold-Set Strategy:", 
                                 style={'fontWeight': '600', 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}),
                        dcc.Dropdown(
                            id='goldset-strategy-selector',
                            options=[
                                {'label': 'High Agreement Only', 'value': 'high_only'},
                                {'label': 'Mixed (Recommended)', 'value': 'mixed'},
                                {'label': 'Include More Disagreement', 'value': 'disagreement_focus'}
                            ],
                            value='mixed',
                            style={
                                'fontSize': '1rem',
                                'minHeight': '45px'
                            }
                        )
                    ], md=3),
                    dbc.Col([
                        html.Label("Samples per Label:", 
                                 style={'fontWeight': '600', 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}),
                        html.Div([
                            dcc.Slider(
                                id='samples-per-label-slider',
                                min=2,
                                max=10,
                                step=1,
                                value=5,
                                marks={i: {'label': str(i), 'style': {'fontSize': '12px'}} for i in range(2, 11, 2)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], style={'padding': '0 15px'})
                    ], md=3),
                    dbc.Col([
                        html.Label("High Agreement Threshold:", 
                                 style={'fontWeight': '600', 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}),
                        html.Div([
                            dcc.Slider(
                                id='high-agreement-threshold-slider',
                                min=0.7,
                                max=1.0,
                                step=0.05,
                                value=0.9,
                                marks={i/100: {'label': f'{i/100:.2f}', 'style': {'fontSize': '12px'}} for i in range(70, 101, 10)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], style={'padding': '0 15px'})
                    ], md=3)
                ], className="mb-4"),
                
                # Row 2: Range Controls and Annotators
                dbc.Row([
                    dbc.Col([
                        html.Label("Useful Disagreement Range:", 
                                 style={'fontWeight': '600', 'marginBottom': '0.5rem', 'fontSize': '1.1rem'}),
                        html.Div([
                            dcc.RangeSlider(
                                id='disagreement-range-slider',
                                min=0.3,
                                max=0.8,
                                step=0.05,
                                value=[0.4, 0.7],
                                marks={i/100: {'label': f'{i/100:.2f}', 'style': {'fontSize': '12px'}} for i in range(30, 81, 10)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], style={'padding': '0 15px'})
                    ], md=6),
                    dbc.Col([
                        html.Label("Select Annotators:", 
                                 style={'fontWeight': '600', 'marginBottom': '1rem', 'fontSize': '1.1rem'}),
                        dbc.Card([
                            dbc.CardBody([
                                dcc.Checklist(
                                    id='goldset-annotator-selector',
                                    options=[{'label': ann, 'value': ann} for ann in goldset_calculator.annotators],
                                    value=goldset_calculator.annotators,
                                    inline=True,
                                    style={'fontSize': '1rem'},
                                    inputStyle={'marginRight': '8px', 'transform': 'scale(1.2)'},
                                    labelStyle={'marginRight': '20px', 'marginBottom': '10px'}
                                )
                            ], style={'padding': '1.5rem'})
                        ], style={'backgroundColor': 'rgba(102, 126, 234, 0.05)', 'border': '1px solid rgba(102, 126, 234, 0.2)'})
                    ], md=6)
                ], className="mb-4"),
                
                # Row 3: Calculate Button
                dbc.Row([
                    dbc.Col([
                        dbc.Button("Generate Gold-Set Recommendations", 
                                 id="goldset-calculate-btn",
                                 size="lg", 
                                 style={
                                     'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                                     'border': 'none',
                                     'borderRadius': '12px',
                                     'fontWeight': '600',
                                     'fontSize': '1.1rem',
                                     'padding': '12px 30px',
                                     'boxShadow': '0 4px 15px rgba(102, 126, 234, 0.3)',
                                     'transition': 'all 0.3s ease',
                                     'width': '100%',
                                     'maxWidth': '400px',
                                     'margin': '0 auto',
                                     'display': 'block'
                                 })
                    ], md=12, style={'textAlign': 'center'})
                ])
            ], style={'padding': '2rem'})
        ], style={
            'border': '1px solid rgba(102, 126, 234, 0.3)',
            'borderRadius': '15px',
            'boxShadow': '0 8px 25px rgba(102, 126, 234, 0.15)',
            'marginBottom': '3rem'
        }),

        # Strategy Explanation Card
        dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.H6("Gold-Set Strategy Guide", 
                           style={'fontWeight': '600', 'color': '#667eea', 'marginBottom': '1rem'}),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.Span("High Agreement Only", style={'fontWeight': '600', 'color': '#27ae60'}),
                                html.P("Selects only documents with high annotator agreement for training stability", 
                                      style={'color': 'var(--text-secondary)', 'fontSize': '0.9rem', 'margin': '0.25rem 0 0 0'})
                            ])
                        ], md=4),
                        dbc.Col([
                            html.Div([
                                html.Span("Mixed (Recommended)", style={'fontWeight': '600', 'color': '#667eea'}),
                                html.P("Combines high-agreement documents with useful edge cases for comprehensive coverage", 
                                      style={'color': 'var(--text-secondary)', 'fontSize': '0.9rem', 'margin': '0.25rem 0 0 0'})
                            ])
                        ], md=4),
                        dbc.Col([
                            html.Div([
                                html.Span("Include More Disagreement", style={'fontWeight': '600', 'color': '#f39c12'}),
                                html.P("Focuses on edge cases and disagreement examples for guideline development", 
                                      style={'color': 'var(--text-secondary)', 'fontSize': '0.9rem', 'margin': '0.25rem 0 0 0'})
                            ])
                        ], md=4)
                    ])
                ])
            ], style={'padding': '1.5rem'})
        ], style={
            'backgroundColor': 'rgba(102, 126, 234, 0.05)',
            'border': '1px solid rgba(102, 126, 234, 0.2)',
            'borderRadius': '12px',
            'marginBottom': '3rem'
        }),

        # Progress indicator
        dbc.Row([
            dbc.Col([
                dbc.Progress(id="goldset-calculation-progress", 
                           value=0, 
                           style={"visibility": "hidden", "height": "8px", "borderRadius": "10px"},
                           color="info")
            ])
        ], className="mb-4"),

        # Results Section
        html.Div(
            id="goldset-results-container",
            style={
                'minHeight': '200px',
                'padding': '2rem 0'
            }
        )
    ], style={'padding': '0 1rem'})

# PART 13: CALLBACKS
# HERE YOU NEED TO COPY THE SPECIFIC CALLBACKS FROM YOUR ORIGINAL CODE:


# Validation callback for sliders
@app.callback(
    Output('max-disagreement-slider', 'min'),
    Input('min-disagreement-slider', 'value')
)
def update_max_slider_min(min_value):
    """Ensure max disagreement is always >= min disagreement."""
    return min_value

# Main analysis callback
@app.callback(
    [Output("disagreement-results-container", "children"),
     Output("disagreement-calculation-progress", "style"),
     Output("disagreement-calculate-btn", "disabled"),
     Output("disagreement-calculate-btn", "children")],
    [Input("disagreement-calculate-btn", "n_clicks")],
    [State("disagreement-annotator-selector", "value"),
     State("disagreement-label-type-selector", "value"),
     State("top-n-selector", "value"),
     State("min-disagreement-slider", "value"),
     State("max-disagreement-slider", "value")]
)
def update_disagreement_analysis(n_clicks, selected_annotators, label_type, top_n,
                                min_disagreement, max_disagreement):
    """Update disagreement analysis based on user selections."""

    if n_clicks is None:
        return (html.Div("Click 'Calculate Disagreement Analysis' to begin"),
                {"visibility": "hidden"}, False, "Calculate Disagreement Analysis")

    print(f"[INFO] Starting disagreement analysis")
    print(f"[INFO] Annotators: {len(selected_annotators)}, Label type: {label_type}")
    print(f"[INFO] Top N: {top_n}, Disagreement range: {min_disagreement:.3f} - {max_disagreement:.3f}")

    # Validate inputs
    if not selected_annotators:
        error_msg = dbc.Alert("Please select at least one annotator", color="warning")
        return error_msg, {"visibility": "hidden"}, False, "Calculate Disagreement Analysis"

    if max_disagreement < min_disagreement:
        error_msg = dbc.Alert("Maximum disagreement must be >= minimum disagreement", color="danger")
        return error_msg, {"visibility": "hidden"}, False, "Calculate Disagreement Analysis"

    try:
        # Show progress and disable button
        progress_style = {"visibility": "visible"}

        # Filter data
        filtered_df = disagreement_calculator.agreement_df[
            disagreement_calculator.agreement_df['annotator'].isin(selected_annotators)
        ].copy()

        print(f"[DEBUG] Filtered data shape: {filtered_df.shape}")

        # Calculate document disagreement scores
        disagreement_df = disagreement_calculator.calculate_document_disagreement_scores(
            filtered_df, label_type
        )

        # Filter by disagreement threshold
        threshold_filtered_df = disagreement_calculator.filter_by_disagreement_threshold(
            disagreement_df, min_disagreement, max_disagreement
        )

        # Calculate label confusion matrix
        confusion_matrix, confusion_counts = disagreement_calculator.calculate_label_confusion_matrix(
            filtered_df, label_type
        )

        # Analyze patterns
        patterns = disagreement_calculator.analyze_disagreement_patterns(disagreement_df)

        # Get top disagreement documents
        top_disagreement_data = disagreement_calculator.get_top_disagreement_documents(
            threshold_filtered_df, top_n
        )

        # Create visualizations
        distribution_fig = disagreement_visualizer.create_disagreement_distribution_histogram(disagreement_df)
        confusion_fig = disagreement_visualizer.create_confusion_matrix_heatmap(confusion_matrix)
        complexity_fig = disagreement_visualizer.create_text_complexity_vs_disagreement_scatter(disagreement_df)
        patterns_cards = disagreement_visualizer.create_patterns_summary_cards(patterns)

        # Create top disagreement table
        if top_disagreement_data:
            disagreement_table = dash_table.DataTable(
                data=top_disagreement_data,
                columns=[{"name": i, "id": i} for i in top_disagreement_data[0].keys()],
                style_cell={
                    'textAlign': 'left',
                    'fontSize': 12,
                    'font_family': 'Arial'
                },
                style_cell_conditional=[
                    {'if': {'column_id': 'Text Preview'}, 'width': '40%'},
                    {'if': {'column_id': 'Annotator Labels'}, 'width': '30%'},
                ],
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                style_data={'whiteSpace': 'normal', 'height': 'auto'},
                page_size=10,
                sort_action='native',
                filter_action='native'
            )
        else:
            disagreement_table = html.P("No documents found matching the criteria.")

        # Create results layout
        results = html.Div([
            # Summary Section
            html.H4("Disagreement Analysis Summary"),
            patterns_cards,
            html.Hr(),

            # Visualizations
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=distribution_fig)
                ], md=6),
                dbc.Col([
                    dcc.Graph(figure=confusion_fig)
                ], md=6)
            ], className="mb-4"),

            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=complexity_fig)
                ], md=12)
            ], className="mb-4"),

            # Top disagreement documents
            html.H4(f"Top {min(top_n, len(top_disagreement_data))} Disagreement Documents"),
            html.P(f"Showing documents with disagreement scores between {min_disagreement:.3f} and {max_disagreement:.3f}"),
            disagreement_table
        ])

        return results, {"visibility": "hidden"}, False, "Calculate Disagreement Analysis"

    except Exception as e:
        print(f"[ERROR] Disagreement analysis failed: {str(e)}")
        error_message = html.Div([
            dbc.Alert(f"Analysis failed: {str(e)}", color="danger")
        ])
        return error_message, {"visibility": "hidden"}, False, "Calculate Disagreement Analysis"


# Main analysis callback
@app.callback(
    [Output("confusion-results-container", "children"),
     Output("confusion-calculation-progress", "style"),
     Output("confusion-calculate-btn", "disabled"),
     Output("confusion-calculate-btn", "children")],
    [Input("confusion-calculate-btn", "n_clicks")],
    [State("confusion-annotator-selector", "value"),
     State("confusion-label-type-selector", "value"),
     State("confusion-analysis-mode", "value"),
     State("focus-annotator-selector", "value")]
)
def update_confusion_analysis(n_clicks, selected_annotators, label_type, analysis_mode, focus_annotator):
    """Update confusion analysis based on user selections."""

    if n_clicks is None:
        return (html.Div("Click 'Calculate Confusion Analysis' to begin"),
                {"visibility": "hidden"}, False, "Calculate Confusion Analysis")

    print(f"[INFO] Starting confusion analysis")
    print(f"[INFO] Annotators: {len(selected_annotators)}, Mode: {analysis_mode}")
    print(f"[INFO] Label type: {label_type}, Focus: {focus_annotator}")

    if not selected_annotators:
        error_msg = dbc.Alert("Please select at least two annotators", color="warning")
        return error_msg, {"visibility": "hidden"}, False, "Calculate Confusion Analysis"

    try:
        # Show progress and disable button
        progress_style = {"visibility": "visible"}

        # Filter data
        filtered_df = confusion_calculator.agreement_df[
            confusion_calculator.agreement_df['annotator'].isin(selected_annotators)
        ].copy()

        print(f"[DEBUG] Filtered data shape: {filtered_df.shape}")

        results_components = []

        # Individual performance analysis
        if analysis_mode in ['complete', 'individual']:
            performance_results = confusion_calculator.calculate_all_annotator_performance(
                filtered_df, label_type
            )

            if performance_results:
                # Performance ranking chart
                performance_fig = confusion_visualizer.create_annotator_performance_ranking(performance_results)
                results_components.extend([
                    html.H4("Individual Annotator Performance"),
                    dcc.Graph(figure=performance_fig),
                ])

                # Individual confusion matrices
                if focus_annotator != 'all' and focus_annotator in performance_results:
                    # Show detailed analysis for focused annotator
                    confusion_matrix = performance_results[focus_annotator]['confusion_matrix']
                    accuracy = performance_results[focus_annotator]['accuracy']
                    confusion_fig = confusion_visualizer.create_individual_confusion_matrix(
                        confusion_matrix, focus_annotator, accuracy
                    )
                    results_components.extend([
                        html.H5(f"Detailed Analysis: {focus_annotator}"),
                        dcc.Graph(figure=confusion_fig)
                    ])
                else:
                    # Show confusion matrices for all annotators
                    confusion_figs = []
                    for annotator, results in list(performance_results.items())[:4]:  # Limit to 4 for space
                        confusion_fig = confusion_visualizer.create_individual_confusion_matrix(
                            results['confusion_matrix'], annotator, results['accuracy']
                        )
                        confusion_figs.append(dbc.Col([dcc.Graph(figure=confusion_fig)], md=6))

                    if confusion_figs:
                        results_components.extend([
                            html.H5("Individual Confusion Matrices (Top 4)"),
                            dbc.Row(confusion_figs[:2]),
                            dbc.Row(confusion_figs[2:4] if len(confusion_figs) > 2 else [])
                        ])

                # Systematic bias analysis
                global_biases, annotator_biases = confusion_calculator.identify_systematic_biases(performance_results)
                bias_fig = confusion_visualizer.create_bias_pattern_visualization(global_biases)

                # Training recommendations
                recommendations = confusion_calculator.generate_training_recommendations(
                    performance_results, annotator_biases
                )
                recommendation_display = confusion_visualizer.create_training_recommendations_display(recommendations)

                results_components.extend([
                    html.Hr(),
                    html.H4("Systematic Bias Analysis"),
                    dcc.Graph(figure=bias_fig),
                    html.Hr(),
                    html.H4("Training Recommendations"),
                    recommendation_display
                ])

        # Pairwise analysis
        if analysis_mode in ['complete', 'pairwise']:
            pairwise_agreement = confusion_calculator.calculate_pairwise_annotator_agreement(
                filtered_df, label_type
            )
            pairwise_fig = confusion_visualizer.create_pairwise_agreement_heatmap(pairwise_agreement)

            results_components.extend([
                html.Hr() if analysis_mode == 'complete' else html.Div(),
                html.H4("Pairwise Annotator Agreement"),
                dcc.Graph(figure=pairwise_fig)
            ])

        # Wrap results
        final_results = html.Div(results_components)

        return final_results, {"visibility": "hidden"}, False, "Calculate Confusion Analysis"

    except Exception as e:
        print(f"[ERROR] Confusion analysis failed: {str(e)}")
        error_message = html.Div([
            dbc.Alert(f"Analysis failed: {str(e)}", color="danger")
        ])
        return error_message, {"visibility": "hidden"}, False, "Calculate Confusion Analysis"



# Main analysis callback
@app.callback(
    [Output("goldset-results-container", "children"),
     Output("goldset-calculation-progress", "style"),
     Output("goldset-calculate-btn", "disabled"),
     Output("goldset-calculate-btn", "children")],
    [Input("goldset-calculate-btn", "n_clicks")],
    [State("goldset-annotator-selector", "value"),
     State("goldset-label-type-selector", "value"),
     State("goldset-strategy-selector", "value"),
     State("samples-per-label-slider", "value"),
     State("high-agreement-threshold-slider", "value"),
     State("disagreement-range-slider", "value")]
)
def update_goldset_analysis(n_clicks, selected_annotators, label_type, strategy,
                           samples_per_label, high_agreement_threshold, disagreement_range):
    """Update gold-set analysis based on user selections."""

    if n_clicks is None:
        return (html.Div("Click 'Generate Gold-Set Recommendations' to begin"),
                {"visibility": "hidden"}, False, "Generate Gold-Set Recommendations")

    print(f"[INFO] Starting gold-set analysis")
    print(f"[INFO] Strategy: {strategy}, Samples per label: {samples_per_label}")
    print(f"[INFO] High agreement threshold: {high_agreement_threshold}")
    print(f"[INFO] Disagreement range: {disagreement_range}")

    if not selected_annotators:
        error_msg = dbc.Alert("Please select at least two annotators", color="warning")
        return error_msg, {"visibility": "hidden"}, False, "Generate Gold-Set Recommendations"

    try:
        # Filter data
        filtered_df = goldset_calculator.agreement_df[
            goldset_calculator.agreement_df['annotator'].isin(selected_annotators)
        ].copy()

        print(f"[DEBUG] Filtered data shape: {filtered_df.shape}")

        # Calculate document agreement levels
        agreement_df = goldset_calculator.calculate_document_agreement_levels(
            filtered_df, label_type
        )

        # Analyze label coverage
        coverage_df = goldset_calculator.analyze_label_coverage(filtered_df, label_type)

        # Select candidates based on strategy
        high_confidence_candidates = pd.DataFrame()
        disagreement_candidates = pd.DataFrame()

        if strategy in ['high_only', 'mixed']:
            high_confidence_candidates = goldset_calculator.select_high_confidence_candidates(
                agreement_df, coverage_df, high_agreement_threshold, samples_per_label
            )

        if strategy in ['mixed', 'disagreement_focus']:
            disagreement_samples = samples_per_label if strategy == 'disagreement_focus' else max(2, samples_per_label // 2)
            disagreement_candidates = goldset_calculator.select_disagreement_candidates(
                agreement_df, coverage_df, disagreement_range[0], disagreement_range[1], disagreement_samples
            )

        # Identify coverage gaps
        coverage_gaps = goldset_calculator.identify_coverage_gaps(
            coverage_df, high_confidence_candidates, disagreement_candidates
        )

        # Calculate quality metrics
        quality_metrics = goldset_calculator.calculate_gold_set_quality_metrics(
            high_confidence_candidates, disagreement_candidates, coverage_df
        )

        # Create visualizations
        coverage_fig = goldset_visualizer.create_coverage_comparison_chart(coverage_df, quality_metrics)
        candidate_fig = goldset_visualizer.create_candidate_distribution_chart(
            high_confidence_candidates, disagreement_candidates
        )
        agreement_fig = goldset_visualizer.create_agreement_distribution_scatter(
            agreement_df, high_agreement_threshold, disagreement_range[1]
        )
        quality_dashboard = goldset_visualizer.create_quality_metrics_dashboard(quality_metrics)

        # Create candidate tables
        all_candidates = pd.concat([high_confidence_candidates, disagreement_candidates], ignore_index=True)

        if len(all_candidates) > 0:
            # Prepare table data
            table_data = []
            for _, candidate in all_candidates.iterrows():
                table_data.append({
                    'Document ID': candidate['document_id'],
                    'Label': candidate['label'],
                    'Agreement Rate': f"{candidate['agreement_rate']:.1%}",
                    'Selection Reason': candidate['selection_reason'].replace('_', ' ').title(),
                    'Priority': candidate['priority'].title(),
                    'Text Preview': candidate['text_preview'],
                    'Text Length': candidate['text_length'],
                    'Annotators': candidate['n_annotators']
                })

            candidates_table = dash_table.DataTable(
                data=table_data,
                columns=[{"name": i, "id": i} for i in table_data[0].keys()],
                style_cell={
                    'textAlign': 'left',
                    'fontSize': 12,
                    'font_family': 'Arial'
                },
                style_cell_conditional=[
                    {'if': {'column_id': 'Text Preview'}, 'width': '40%'},
                ],
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                style_data={'whiteSpace': 'normal', 'height': 'auto'},
                page_size=10,
                sort_action='native',
                filter_action='native'
            )
        else:
            candidates_table = html.P("No candidates selected with current criteria.")

        # Coverage gaps table
        if len(coverage_gaps) > 0:
            gaps_table_data = coverage_gaps.to_dict('records')
            gaps_table = dash_table.DataTable(
                data=gaps_table_data,
                columns=[{"name": i.replace('_', ' ').title(), "id": i} for i in coverage_gaps.columns],
                style_cell={'textAlign': 'left'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                page_size=5
            )
        else:
            gaps_table = html.P("No significant coverage gaps identified.")

        # Create results layout
        results = html.Div([
            # Quality Dashboard
            html.H4("Gold-Set Quality Overview"),
            quality_dashboard,
            html.Hr(),

            # Visualizations
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=coverage_fig)
                ], md=6),
                dbc.Col([
                    dcc.Graph(figure=candidate_fig)
                ], md=6)
            ], className="mb-4"),

            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=agreement_fig)
                ], md=12)
            ], className="mb-4"),

            # Recommended candidates
            html.H4(f"Recommended Gold-Set Candidates ({len(all_candidates)} documents)"),
            html.P(f"Strategy: {strategy.replace('_', ' ').title()}, High Agreement ≥ {high_agreement_threshold:.0%}, Useful Disagreement: {disagreement_range[0]:.0%}-{disagreement_range[1]:.0%}"),
            candidates_table,

            html.Hr(),

            # Coverage gaps
            html.H4("Coverage Gap Analysis"),
            html.P("Labels requiring additional attention for comprehensive gold-set coverage:"),
            gaps_table
        ])

        return results, {"visibility": "hidden"}, False, "Generate Gold-Set Recommendations"

    except Exception as e:
        print(f"[ERROR] Gold-set analysis failed: {str(e)}")
        error_message = html.Div([
            dbc.Alert(f"Analysis failed: {str(e)}", color="danger")
        ])
        return error_message, {"visibility": "hidden"}, False, "Generate Gold-Set Recommendations"
    
@app.callback(
    [Output('theme-store', 'data'),
     Output('light-theme-btn', 'color'),
     Output('dark-theme-btn', 'color')],
    [Input('light-theme-btn', 'n_clicks'),
     Input('dark-theme-btn', 'n_clicks')],
    [State('theme-store', 'data')]
)
def toggle_theme(light_clicks, dark_clicks, current_theme):
    ctx = dash.callback_context
    if not ctx.triggered:
        return 'light', 'primary', 'outline-secondary'
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'light-theme-btn':
        return 'light', 'primary', 'outline-secondary'
    elif trigger_id == 'dark-theme-btn':
        return 'dark', 'outline-secondary', 'primary'
    
    return current_theme, 'primary' if current_theme == 'light' else 'outline-secondary', 'primary' if current_theme == 'dark' else 'outline-secondary'

# Apply theme to body
app.clientside_callback(
    """
    function(theme) {
        if (theme === 'dark') {
            document.body.className = 'theme-dark';
        } else {
            document.body.className = 'theme-light';
        }
        return theme;
    }
    """,
    Output('theme-store', 'data', allow_duplicate=True),
    Input('theme-store', 'data'),
    prevent_initial_call=True
)


# PART 14: HELPER FUNCTIONS (IF ANY)
# Copy any helper functions that the callbacks use

# ========================================
# END OF FILE
# ========================================

# ###### THE BELOW CODE IS ONLY FOR LOCAL TESTING ######
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8050)