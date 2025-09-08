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
            title={
                'text': f"Label Distribution Analysis<br>Coverage: {quality_metrics['unique_labels_covered']}/{quality_metrics['total_labels']} labels ({quality_metrics['label_coverage_percentage']:.1f}%)",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial, sans-serif', 'color': 'black'},
                'font_weight': 'bold'
            },
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
            name="All Samples"
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
            title={
                'text': "Sample Agreement Distribution with Selection Criteria",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial, sans-serif', 'color': 'black'},
                'font_weight': 'bold'
            },
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


